import torch
from torch import nn
from torch.nn import functional as F
from openebm.elm.nanolightning.torchlightning_module import LightningModule
# import torch.optim as optim
# from torchmetrics import Accuracy
# from transformers import AutoTokenizer

import math
import random
import os
from openebm.elm.utils import setup_ebt, init_whole_model_weights
from openebm.elm.utils import MLP, Memory_Augmented_MLP, Memory_Gating_MLP, mask_q_tokens
from openebm.elm.replay_buffer import CausalReplayBuffer
from openebm.elm.metrics import calculate_bpb_score

import ipdb

class EBT_NLP(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):#passed in from model ckpt
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))
        
        # tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, clean_up_tokenization_spaces = False)
        # Use tokenizer_obj if available (set by ModelTrainer), otherwise use tokenizer directly
        self.tokenizer = self.hparams.tokenizer_obj if hasattr(self.hparams, 'tokenizer_obj') else self.hparams.tokenizer
        self.tokenizer_pad_token_id = None # Nanochat doesn't have <|pad|> or <|eos|> # self.tokenizer.eos_token_id

        self.vocab_size = self.tokenizer.get_vocab_size() # len(self.tokenizer) # self.vocab_size = self.tokenizer.vocab_size caused errors since is smaller than len(self.tokenizer), is 50254 for neox-20b, len tokenizer is 50277 so decided to use that
        self.hparams.vocab_size = self.vocab_size
        
        if self.hparams.mcmc_step_size_learnable and getattr(self.hparams, 'mcmc_step_size_per_step', False):
            self.alpha = nn.ParameterList([
                nn.Parameter(torch.tensor(float(self.hparams.mcmc_step_size), dtype=torch.float32))
                for _ in range(self.hparams.mcmc_num_steps)
            ])
        else:
            self.alpha = nn.Parameter(torch.tensor(float(self.hparams.mcmc_step_size), dtype=torch.float32),
                                      requires_grad=self.hparams.mcmc_step_size_learnable)
        self.langevin_dynamics_noise_std = nn.Parameter(torch.tensor(float(self.hparams.langevin_dynamics_noise)), requires_grad=False) # if using self.hparams.langevin_dynamics_noise_learnable this will be turned on in warm_up_finished func

        self.embeddings = nn.Embedding(self.vocab_size, self.hparams.embedding_dim)
        init_whole_model_weights(self.embeddings, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)
        
        self.log_softmax = nn.LogSoftmax(dim = -1)
        self.softmax = nn.Softmax(dim = -1)
        
        if not self.hparams.vocab_to_embed_uses_prob_dist: # if are not using the prob dist * embed as vocab to embed
            if 'learnable_process_memory' in self.hparams and self.hparams.learnable_process_memory and self.hparams.process_memory_type != None:
                self.vocab_to_embed = Memory_Gating_MLP(self.vocab_size, self.hparams.embedding_dim, self.hparams.process_memory_type, self.hparams.process_memory_linear_layer)
            elif 'learnable_process_memory' in self.hparams and self.hparams.learnable_process_memory:
                assert self.hparams.num_modality_processing_mlp_layers > 1, "must set self.hparams.num_modality_processing_mlp_layers > 1 if not using self.hparams.process_memory_type"
                self.vocab_to_embed = Memory_Augmented_MLP(self.vocab_size, self.hparams.embedding_dim, self.hparams.embedding_dim, self.hparams.embedding_dim, dropout_rate=0, layer_norm=True, num_hidden_layers = self.hparams.num_modality_processing_mlp_layers)
            elif self.hparams.num_modality_processing_mlp_layers != 1:
                self.vocab_to_embed = MLP(self.vocab_size, self.hparams.embedding_dim, self.hparams.embedding_dim, dropout_rate=0, layer_norm=True, num_hidden_layers = self.hparams.num_modality_processing_mlp_layers - 2)
            else:
                self.vocab_to_embed = nn.Linear(self.vocab_size, self.hparams.embedding_dim, bias = False, device = self.device) #NOTE this is ebt special, since we want to input a prob dist and pred this prob dist but the transformer needs an embedding as input
            init_whole_model_weights(self.vocab_to_embed, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)

        self.transformer = setup_ebt(self.hparams)
        
        self.finished_warming_up = False

        self.mcmc_replay_buffer = 'mcmc_replay_buffer' in self.hparams and self.hparams.mcmc_replay_buffer and self.hparams.execution_mode != "inference"
        if self.mcmc_replay_buffer:
            replay_buffer_max_size = self.hparams.mcmc_replay_buffer_size
            self.replay_buffer_samples = self.hparams.batch_size_per_device * self.hparams.mcmc_replay_buffer_sample_bs_percent
            self.replay_buffer = CausalReplayBuffer(max_size=replay_buffer_max_size, sample_size=self.replay_buffer_samples)

        self._alpha_debug_step = 0  # counter for alpha diagnostic prints
        if self.hparams.debug_unused_parameters:
            self.used_parameters = set()
            self.parameters_not_to_check = set() # dont check these since may be frozen or dont want them to update

    def _apply(self, fn):
        """Override to ensure alpha always stays in float32 regardless of model dtype.

        _apply is the lowest-level method called by all dtype/device conversions
        (to, cuda, float, etc.). Lightning's to() bypasses nn.Module.to() when a
        trainer is present, so overriding to() is insufficient.
        """
        result = super()._apply(fn)
        result.alpha.data = result.alpha.data.to(dtype=torch.float32)
        return result

    @torch.compiler.disable
    def _mcmc_step_excluded(self, predicted_tokens, real_embeddings_input, mcmc_step, i, num_mcmc_steps,
                      langevin_dynamics_noise_std, alpha, start_pos, learning, return_raw_logits,
                      real_token_ids=None):
        batch_size = predicted_tokens.shape[0]
        seq_length = predicted_tokens.shape[1]
        
        if self.hparams.no_mcmc_detach:
            predicted_tokens.requires_grad_().reshape(batch_size, seq_length, self.vocab_size) # B, S, V
        else: # default, do detach
            predicted_tokens = predicted_tokens.detach().requires_grad_().reshape(batch_size, seq_length, self.vocab_size) # B, S, V

        if self.hparams.langevin_dynamics_noise != 0:
            ld_noise = torch.randn_like(predicted_tokens.detach()) * langevin_dynamics_noise_std # langevin dynamics
            predicted_tokens = predicted_tokens + ld_noise

        if self.hparams.normalize_initial_condition:
            if getattr(self.hparams, 'float_precision', '') == "bf16-true":
                if self.hparams.normalize_initial_condition_only_first_step:
                    if mcmc_step == 0:
                        predicted_tokens = self.softmax(predicted_tokens)
                else:
                    predicted_tokens = self.softmax(predicted_tokens)
            else:
                if self.hparams.normalize_initial_condition_only_first_step:
                    if mcmc_step == 0:
                        predicted_tokens = self.softmax(predicted_tokens.float()).to(predicted_tokens.dtype)
                else:
                    predicted_tokens = self.softmax(predicted_tokens.float()).to(predicted_tokens.dtype)
                
            if self.hparams.vocab_to_embed_uses_prob_dist: # predicted_embeds is B, S, V; embed is V, D
                predicted_embeddings = torch.matmul(predicted_tokens, self.embeddings.weight) #BS, S, D
            else:
                predicted_embeddings = self.vocab_to_embed(predicted_tokens) #BS, S, D
        else:
            predicted_embeddings = self.vocab_to_embed(predicted_tokens) #BS, S, D
        
        all_embeddings = torch.cat((real_embeddings_input.detach(), predicted_embeddings), dim = 1) # B, 2*S, D
        
        # create_graph=True 的 autograd.grad 与 compiled graph 不兼容
        # 所以若 transformer 已被 torch.compile 编译，MCMC 中需要用 eager 版本
        transformer = getattr(self, 'transformer_eager', self.transformer)
        energy_preds = transformer(
            all_embeddings,
            start_pos=start_pos,
            mcmc_step=mcmc_step,
            real_token_ids=real_token_ids,
            predicted_tokens=predicted_tokens,
        ) # is B, 2*S, D; checked and there are no in place ops; mcmc_step only applies to when using certain types of ebt
        energy_preds = energy_preds.reshape(-1, 1)
        
        with torch.amp.autocast(device_type='cuda', enabled=False):
            energy_f32 = energy_preds.float()
            mcmc_gradient_mode = getattr(self.hparams, "mcmc_gradient_mode", "second_order")
            create_graph_for_mcmc = learning and mcmc_gradient_mode == "second_order" and not getattr(self.hparams, "fsdp_first_order_mcmc_debug", False)
            if self.hparams.truncate_mcmc:  #retain_graph defaults to create_graph value here; if learning is true then create_graph else dont (inference)
                if i == (num_mcmc_steps - 1):
                    predicted_tokens_grad = torch.autograd.grad([energy_f32.sum()], [predicted_tokens], create_graph=create_graph_for_mcmc)[0]
                else:
                    predicted_tokens_grad = torch.autograd.grad([energy_f32.sum()], [predicted_tokens], create_graph=False)[0]
            else:
                predicted_tokens_grad = torch.autograd.grad([energy_f32.sum()], [predicted_tokens], create_graph=create_graph_for_mcmc)[0]
        # predicted_tokens_grad has shape B, S, V
        
        if self.hparams.clamp_futures_grad:
            _alpha_for_clamp = self.alpha[mcmc_step] if isinstance(self.alpha, nn.ParameterList) else self.alpha
            min_and_max = self.hparams.clamp_futures_grad_max_change / torch.clamp(_alpha_for_clamp.float(), min=0.0001)
            # predicted_tokens_grad = scale_clamp(predicted_tokens_grad, -min_and_max, min_and_max)
            predicted_tokens_grad = torch.clamp(predicted_tokens_grad, min = -min_and_max, max = min_and_max)
            
        if torch.isnan(predicted_tokens_grad).any() or torch.isinf(predicted_tokens_grad).any():
            raise ValueError("NaN or Inf gradients detected during MCMC.")
        
        predicted_tokens = predicted_tokens - alpha * predicted_tokens_grad # do this to tokens will be unnormalize prob dist convert to prob dist after
        
        if self.hparams.absolute_clamp != 0.0:
            predicted_tokens = torch.clamp(predicted_tokens, min = -self.hparams.absolute_clamp, max = self.hparams.absolute_clamp)
        
        if self.hparams.sharpen_predicted_distribution != 0.0:
            predicted_tokens = predicted_tokens / self.hparams.sharpen_predicted_distribution

        if return_raw_logits:
            predicted_tokens_for_loss = predicted_tokens # BS, S, V
        else:
            predicted_tokens_for_loss = self.log_softmax(predicted_tokens).reshape(-1, self.vocab_size) # BS*S, V
            
        return predicted_tokens, energy_preds, predicted_tokens_for_loss

    def forward(self, x, start_pos = 0, learning = True, return_raw_logits = False, replay_buffer_logits = None, no_randomness = True): # accepts input_ids as input; a lot of the logic here is just for S2 params, see pseudocode in paper for a more concise view of how this works. it can be < 10 LOC
        predicted_distributions = []
        predicted_energies = []

        real_embeddings_input = self.embeddings(x)
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        
        model_dtype = self.embeddings.weight.dtype
        if not isinstance(self.alpha, nn.ParameterList):
            alpha = torch.clamp(self.alpha, min=0.0001).float()
            if not no_randomness and self.hparams.randomize_mcmc_step_size_scale != 1:
                expanded_alpha = alpha.expand(batch_size, seq_length, 1)

                scale = self.hparams.randomize_mcmc_step_size_scale
                low = alpha / scale
                high = alpha * scale
                alpha = low + torch.rand_like(expanded_alpha) * (high - low)
        else:
            alpha = None  # will be resolved per step below

        langevin_dynamics_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)

        predicted_tokens = self.corrupt_embeddings(real_embeddings_input) # B, S, V
        if replay_buffer_logits is not None: # using replay buffer, use the logits instead of corruption
            predicted_tokens[batch_size - replay_buffer_logits.shape[0]:] = replay_buffer_logits # NOTE this assumes the fresh data is concatted first
                
        
        mcmc_steps = [] # in the general case of no randomize_mcmc_num_steps then this has len == self.hparams.randomize_mcmc_num_steps
        for step in range(self.hparams.mcmc_num_steps):
            if not no_randomness and hasattr(self.hparams, 'randomize_mcmc_num_steps') and self.hparams.randomize_mcmc_num_steps > 0:
                if self.hparams.randomize_mcmc_num_steps_final_landscape: # makes so only applies rand steps to final landscape
                    if step == (self.hparams.mcmc_num_steps - 1):
                        min_steps = 1 if self.hparams.randomize_mcmc_num_steps_min == 0 else self.hparams.randomize_mcmc_num_steps_min
                        repeats = torch.randint(min_steps, self.hparams.randomize_mcmc_num_steps + 2, (1,)).item()
                        mcmc_steps.extend([step] * repeats)
                    else:
                        mcmc_steps.append(step)
                else:
                    min_steps = 1 if self.hparams.randomize_mcmc_num_steps_min == 0 else self.hparams.randomize_mcmc_num_steps_min
                    repeats = torch.randint(min_steps, self.hparams.randomize_mcmc_num_steps + 2, (1,)).item()
                    mcmc_steps.extend([step] * repeats)
            elif no_randomness and hasattr(self.hparams, 'randomize_mcmc_num_steps') and self.hparams.randomize_mcmc_num_steps > 0: # use max steps
                if step == (self.hparams.mcmc_num_steps - 1): # i found this was a better pretraining metric and was more stable, only do several steps on final energy landscape instead of over all energy landscapes
                    mcmc_steps.extend([step] * (self.hparams.randomize_mcmc_num_steps + 1))
                else:
                    mcmc_steps.append(step)
            else:
                mcmc_steps.append(step)

        with torch.set_grad_enabled(True):
            for i, mcmc_step in enumerate(mcmc_steps):
                if isinstance(self.alpha, nn.ParameterList):
                    step_alpha = torch.clamp(self.alpha[mcmc_step], min=0.0001)
                else:
                    step_alpha = alpha

                predicted_tokens, energy_preds, predicted_tokens_for_loss = self._mcmc_step_excluded(
                    predicted_tokens, real_embeddings_input, mcmc_step, i, len(mcmc_steps),
                    langevin_dynamics_noise_std, step_alpha, start_pos, learning, return_raw_logits,
                    real_token_ids=x
                )
                if getattr(self.hparams, "mcmc_gradient_mode", "second_order") in {"first_order_cd", "first_order_nce", "proposal_aware_nce"}:
                    # These first-order objectives recompute positive and
                    # negative energies below. Keep only detached sampler
                    # outputs here so MCMC does not retain transformer graphs.
                    predicted_energies.append(energy_preds.detach())
                    predicted_distributions.append(predicted_tokens_for_loss.detach())
                else:
                    predicted_energies.append(energy_preds)
                    predicted_distributions.append(predicted_tokens_for_loss)
                del energy_preds, predicted_tokens_for_loss  # release references to help GC

        return predicted_distributions, predicted_energies

    def forward_loss_wrapper(self, x, phase="train", token_bytes=None):
        no_randomness = False if phase == "train" else True
        learning = phase == "train"
        mcmc_gradient_mode = getattr(self.hparams, "mcmc_gradient_mode", "second_order")
        proposal_mode = getattr(self.hparams, "proposal_aware_nce_proposal", "uniform")
        proposal_aware_no_mcmc = (
            mcmc_gradient_mode == "proposal_aware_nce"
            and proposal_mode != "mcmc_final"
        )

        if proposal_aware_no_mcmc:
            input_ids = x[0].squeeze(dim=0)
            next_token_indices = x[1].squeeze(dim=0)
            if self.hparams.execution_mode == "finetune":
                next_token_indices = mask_q_tokens(next_token_indices, self.tokenizer)
            next_token_indices = next_token_indices.reshape(-1)
            contrastive_loss, proposal_metrics = self.calculate_proposal_aware_nce_loss(
                input_ids,
                next_token_indices,
                proposal_logits=None,
            )
            loss_coeff = getattr(self.hparams, "proposal_aware_nce_loss_coeff", 1.0)
            total_loss = loss_coeff * contrastive_loss
            zero = total_loss.detach() * 0.0
            log_dict = {
                'loss': total_loss,
                'initial_loss' : zero,
                'reconstruction_loss': zero,
                'final_step_loss': zero,
                'contrastive_loss' : contrastive_loss.detach(),
                'initial_final_pred_energies_gap': proposal_metrics['proposal_aware_nce_energy_gap'],
                'perplexity': zero,
                'bpb': zero,
                'bpb_nats': zero,
                'bpb_bytes': zero,
                'bpb_tokens': zero,
                'bpb_nats_per_token': zero,
                'bpb_bytes_per_token': zero,
                'objective_loss': total_loss.detach(),
                'initial_step_ce': zero,
                'final_step_ce': zero,
                'mcmc_ce_improvement': zero,
                'perplexity_is_finite': zero,
            }
            log_dict.update(proposal_metrics)
            return log_dict

        if not no_randomness and self.mcmc_replay_buffer: # dont do this when doing val/testing
            # all_tokens = x['input_ids'].squeeze(dim=1)
            all_tokens = x[0].squeeze(dim=0)
            input_ids, replay_buffer_logits, next_token_indices = self.replay_buffer.get_batch(all_tokens) # this automatically does indexing for input ids and next token indices while also passing back the logits
            predicted_distributions, predicted_energies = self(input_ids, return_raw_logits = True, replay_buffer_logits = replay_buffer_logits, no_randomness = no_randomness, learning = learning)
            self.replay_buffer.update(all_tokens.detach(), predicted_distributions[-1].detach()) # update using the final predicted distributions
        else:
            input_ids = x[0].squeeze(dim=0)
            next_token_indices = x[1].squeeze(dim=0)
            predicted_distributions, predicted_energies = self(input_ids, return_raw_logits = True, no_randomness = no_randomness, learning = learning)

            # input_ids = x['input_ids'].squeeze(dim=1)[:, :-1]
            # predicted_distributions, predicted_energies = self(input_ids, return_raw_logits = True, no_randomness = no_randomness)
            # next_token_indices = x['input_ids'].squeeze(dim=1)[:, 1:] # squeeze was to remove 1 on 2nd dim

        if self.hparams.execution_mode == "finetune": # Only tokens after "[[Answer]]: " will be calculated in finetune
            next_token_indices = mask_q_tokens(next_token_indices, self.tokenizer)
        next_token_indices = next_token_indices.reshape(-1) # BS * S; reshape since targets are supposed to be 1D

        reconstruction_loss = 0
        total_mcmc_steps = len(predicted_energies) # in general this equals self.hparams.mcmc_num_steps, isnt in case of rand number
        for mcmc_step, (predicted_distribution, predicted_energy) in enumerate(zip(predicted_distributions, predicted_energies)):
            if self.hparams.soften_target_prob_dist != 0.0:
                if total_mcmc_steps <= 1:
                    label_smoothing = 0.0
                else:
                    label_smoothing = ((total_mcmc_steps - 1) - mcmc_step) / (total_mcmc_steps - 1) * self.hparams.soften_target_prob_dist
                predicted_distribution = predicted_distribution.reshape(-1, self.vocab_size)
                cce_loss = F.cross_entropy(predicted_distribution, next_token_indices, label_smoothing=label_smoothing, ignore_index=-1)
            else:
                predicted_distribution = self.log_softmax(predicted_distribution).reshape(-1, self.vocab_size)
                cce_loss = F.nll_loss(predicted_distribution, next_token_indices, ignore_index=-1)
            
            if self.hparams.truncate_mcmc:
                if mcmc_step == (total_mcmc_steps - 1):
                    reconstruction_loss = cce_loss
                    ppl_loss = torch.exp(cce_loss).detach()
                    final_reconstruction_loss = cce_loss.detach()
            else:
                reconstruction_loss += cce_loss
                if mcmc_step == (total_mcmc_steps - 1):
                    ppl_loss = torch.exp(cce_loss).detach()
                    final_reconstruction_loss = cce_loss.detach()
                    reconstruction_loss = reconstruction_loss / total_mcmc_steps # normalize so is indifferent to number of mcmc steps
                
            #pure logging things (no function for training)
            if mcmc_step == 0:
                initial_loss = cce_loss.detach()
                initial_pred_energies = predicted_energy.squeeze().mean().detach()
            if mcmc_step == (total_mcmc_steps - 1):
                final_pred_energies = predicted_energy.squeeze().mean().detach()
        
        initial_final_pred_energies_gap = initial_pred_energies - final_pred_energies

        use_first_order_surrogate = mcmc_gradient_mode in {"first_order_cd", "first_order_nce"}
        proposal_metrics = {}
        if mcmc_gradient_mode == "proposal_aware_nce":
            contrastive_loss, proposal_metrics = self.calculate_proposal_aware_nce_loss(
                input_ids,
                next_token_indices,
                proposal_logits=predicted_distributions[-1].detach(),
            )
            loss_coeff = getattr(self.hparams, "proposal_aware_nce_loss_coeff", 1.0)
            total_loss = loss_coeff * contrastive_loss
            contrastive_loss = contrastive_loss.detach()
        elif use_first_order_surrogate:
            # In first-order surrogate modes, the MCMC sampler is detached from model
            # parameters. Recompute positive/negative energies at detached token
            # states so the optimization and validation losses use the same
            # first-order EBM objective. CE is still logged separately below.
            use_nce_loss = mcmc_gradient_mode == "first_order_nce"
            contrastive_loss, surrogate_metrics = self.calculate_contrastive_loss(
                predicted_energies,
                input_ids,
                next_token_indices,
                fake_pred_tokens=predicted_distributions[-1].detach(),
                recompute_fake_energy=True,
                combine_recomputed_energies=mcmc_gradient_mode in {"first_order_cd", "first_order_nce"},
                loss_mode="nce" if use_nce_loss else None,
                return_metrics=True,
            )
            proposal_metrics.update(surrogate_metrics)
            loss_coeff = (
                getattr(self.hparams, "first_order_nce_loss_coeff", 1.0)
                if use_nce_loss
                else getattr(self.hparams, "first_order_cd_loss_coeff", 1.0)
            )
            total_loss = loss_coeff * contrastive_loss
            local_cd_coeff = float(getattr(self.hparams, "first_order_local_cd_coeff", 0.0) or 0.0)
            if mcmc_gradient_mode == "first_order_cd" and local_cd_coeff != 0.0:
                local_cd_loss, local_cd_metrics = self.calculate_trajectory_local_cd_loss(
                    input_ids,
                    next_token_indices,
                    predicted_distributions,
                )
                total_loss = total_loss + local_cd_coeff * local_cd_loss
                proposal_metrics.update(local_cd_metrics)
            contrastive_loss = contrastive_loss.detach()
        elif self.hparams.contrastive_loss: # works by pushing up on energies model predicted and pushing down on energy of true samples
            contrastive_loss = self.calculate_contrastive_loss(predicted_energies, input_ids, next_token_indices)
            total_loss = self.hparams.reconstruction_coeff * reconstruction_loss + self.hparams.contrastive_loss_coeff * contrastive_loss
            contrastive_loss = contrastive_loss.detach()
        else:
            total_loss = self.hparams.reconstruction_coeff * reconstruction_loss
            contrastive_loss = 0.0
        
        if token_bytes is not None:
            # Compute per-token loss (reduction='none') for accurate BPB.
            # Passing scalar mean loss (cce_loss) is incorrect because
            # calculate_bpb_score multiplies loss by (num_bytes > 0) mask and sums,
            # yielding mean_loss × count rather than sum(per_token_loss[valid]).
            # predicted_distribution from the last MCMC step is already (-1, vocab_size).
            if self.hparams.soften_target_prob_dist != 0.0:
                per_token_ce = F.cross_entropy(predicted_distribution, next_token_indices,
                                                label_smoothing=label_smoothing, ignore_index=-1, reduction='none')
            else:
                per_token_ce = F.nll_loss(predicted_distribution, next_token_indices, ignore_index=-1, reduction='none')
            bpb_loss, bpb_nats, bpb_bytes, bpb_tokens = calculate_bpb_score(
                next_token_indices,
                per_token_ce.detach(),
                token_bytes,
            )
            bpb_nats_per_token = bpb_nats / bpb_tokens if bpb_tokens > 0 else 0.0
            bpb_bytes_per_token = bpb_bytes / bpb_tokens if bpb_tokens > 0 else 0.0
        else:
            bpb_loss = 0
            bpb_nats = 0
            bpb_bytes = 0
            bpb_tokens = 0
            bpb_nats_per_token = 0.0
            bpb_bytes_per_token = 0.0

        log_dict = {
            'loss': total_loss,
            'initial_loss' : initial_loss,
            'reconstruction_loss': reconstruction_loss.detach(),
            'final_step_loss': final_reconstruction_loss,
            'contrastive_loss' : contrastive_loss,
            'initial_final_pred_energies_gap': initial_final_pred_energies_gap,
            'perplexity': ppl_loss,
            'bpb': bpb_loss,
            'bpb_nats': bpb_nats,    # accumulated nats for epoch-level BPB
            'bpb_bytes': bpb_bytes,  # accumulated bytes for epoch-level BPB
            'bpb_tokens': bpb_tokens,
            'bpb_nats_per_token': bpb_nats_per_token,
            'bpb_bytes_per_token': bpb_bytes_per_token,
            'objective_loss': total_loss.detach(),
            'initial_step_ce': initial_loss,
            'final_step_ce': final_reconstruction_loss,
            'mcmc_ce_improvement': initial_loss - final_reconstruction_loss,
            'perplexity_is_finite': torch.isfinite(ppl_loss).to(ppl_loss.dtype),
        }
        log_dict.update(proposal_metrics)
        return log_dict
    

    def corrupt_embeddings(self, embeddings):
        if self.hparams.vocab_to_embed_uses_prob_dist:
            predicted_tokens_dtype = self.embeddings.weight.dtype
        else:
            predicted_tokens_dtype = self.vocab_to_embed.weight.dtype
        predicted_tokens_device = embeddings.device

        if self.hparams.denoising_initial_condition == "most_recent_embedding":
            raise NotImplementedError(f"most_recent_embedding denoising_initial_condition not supported for NLP yet")
        elif self.hparams.denoising_initial_condition == "random_noise":
            predicted_tokens = torch.randn(size=(embeddings.shape[0], embeddings.shape[1], self.vocab_size), dtype=predicted_tokens_dtype, device=predicted_tokens_device) * self.hparams.gaussian_random_noise_scaling
        elif self.hparams.denoising_initial_condition == "zeros":
            predicted_tokens = torch.zeros(size=(embeddings.shape[0], embeddings.shape[1], self.vocab_size), dtype=predicted_tokens_dtype, device=predicted_tokens_device)
        else:
            raise NotImplementedError(f"{self.hparams.denoising_initial_condition} denoising_initial_condition not yet supported")
        
        return predicted_tokens
    
    def calculate_contrastive_loss(self, predicted_energies, input_ids, next_token_indices,
                                   fake_pred_tokens=None, recompute_fake_energy=False,
                                   combine_recomputed_energies=False,
                                   loss_mode=None,
                                   return_metrics=False):
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        real_embeddings_input = self.embeddings(input_ids)
        
        next_token_indices_2d = next_token_indices.reshape(batch_size, seq_length)
        valid_positions = next_token_indices_2d != -1
        safe_next_token_indices_2d = next_token_indices_2d.clamp(min=0)
        
        if self.hparams.discrete_contrastive_loss_true_logit_val != 0: # NOTE from experience this doesnt work very well and it not recommended compared to just one hot encoding
            true_logit_value = self.hparams.discrete_contrastive_loss_true_logit_val
            false_logit_value = -1 * true_logit_value
            true_token_logits = torch.full((batch_size, seq_length, self.vocab_size), false_logit_value, device=next_token_indices.device)
            
            batch_idx = torch.arange(batch_size, device=next_token_indices.device).view(-1, 1).expand(-1, seq_length)
            seq_idx = torch.arange(seq_length, device=next_token_indices.device).view(1, -1).expand(batch_size, -1)
            true_token_logits[batch_idx, seq_idx, safe_next_token_indices_2d] = true_logit_value
            true_token_logits = true_token_logits.masked_fill(~valid_positions.unsqueeze(-1), 0.0)
            
            if self.hparams.normalize_initial_condition:
                true_token_logits = self.softmax(true_token_logits)
                    
                if self.hparams.vocab_to_embed_uses_prob_dist:
                    true_embeddings = torch.matmul(true_token_logits, self.embeddings.weight)
                else:
                    true_embeddings = self.vocab_to_embed(true_token_logits)
            else:
                true_embeddings = self.vocab_to_embed(true_token_logits)
        else:
            assert self.hparams.normalize_initial_condition, "if not using normalize initial condition must set logit val"
            true_token_one_hot = torch.zeros((batch_size, seq_length, self.vocab_size), device=next_token_indices.device)
            batch_idx = torch.arange(batch_size, device=next_token_indices.device).view(-1, 1).expand(-1, seq_length)
            seq_idx = torch.arange(seq_length, device=next_token_indices.device).view(1, -1).expand(batch_size, -1)
            true_token_one_hot[batch_idx, seq_idx, safe_next_token_indices_2d] = 1.0
            true_token_one_hot = true_token_one_hot.masked_fill(~valid_positions.unsqueeze(-1), 0.0)
            
            if self.hparams.vocab_to_embed_uses_prob_dist:
                true_embeddings = torch.matmul(true_token_one_hot, self.embeddings.weight)
            else:
                true_embeddings = self.vocab_to_embed(true_token_one_hot)

        true_pred_tokens = true_token_logits if self.hparams.discrete_contrastive_loss_true_logit_val != 0 else true_token_one_hot
        if recompute_fake_energy:
            if fake_pred_tokens is None:
                raise ValueError("recompute_fake_energy=True requires fake_pred_tokens.")
            fake_pred_tokens = fake_pred_tokens.detach()
            if self.hparams.normalize_initial_condition:
                if getattr(self.hparams, 'float_precision', '') == "bf16-true":
                    fake_model_tokens = self.softmax(fake_pred_tokens)
                else:
                    fake_model_tokens = self.softmax(fake_pred_tokens.float()).to(fake_pred_tokens.dtype)
                if self.hparams.vocab_to_embed_uses_prob_dist:
                    fake_embeddings = torch.matmul(fake_model_tokens, self.embeddings.weight)
                else:
                    fake_embeddings = self.vocab_to_embed(fake_model_tokens)
            else:
                fake_model_tokens = fake_pred_tokens
                fake_embeddings = self.vocab_to_embed(fake_pred_tokens)

            if combine_recomputed_energies:
                # Used by graph-safe first-order modes: one larger transformer call avoids
                # two separate ZeRO-3/FSDP parameter materialization cycles.
                all_true_embeddings = torch.cat((real_embeddings_input, true_embeddings), dim=1)
                all_fake_embeddings = torch.cat((real_embeddings_input, fake_embeddings), dim=1)
                combined_embeddings = torch.cat((all_true_embeddings, all_fake_embeddings), dim=0)
                combined_input_ids = torch.cat((input_ids, input_ids), dim=0)
                combined_pred_tokens = torch.cat((true_pred_tokens, fake_model_tokens), dim=0)
                combined_energies = self.transformer(
                    combined_embeddings,
                    start_pos=0,
                    mcmc_step=self.hparams.mcmc_num_steps - 1,
                    real_token_ids=combined_input_ids,
                    predicted_tokens=combined_pred_tokens,
                ).reshape(2, batch_size * seq_length, 1)
                real_energies = combined_energies[0]
                fake_energies = combined_energies[1]
            else:
                all_true_embeddings = torch.cat((real_embeddings_input, true_embeddings), dim=1)
                real_energies = self.transformer(
                    all_true_embeddings,
                    start_pos=0,
                    mcmc_step=self.hparams.mcmc_num_steps - 1,
                    real_token_ids=input_ids,
                    predicted_tokens=true_pred_tokens,
                ) # NOTE if want to use this maybe check in better detail what ired does
                real_energies = real_energies.reshape(-1, 1) # BS, 1
                all_fake_embeddings = torch.cat((real_embeddings_input, fake_embeddings), dim=1)
                fake_energies = self.transformer(
                    all_fake_embeddings,
                    start_pos=0,
                    mcmc_step=self.hparams.mcmc_num_steps - 1,
                    real_token_ids=input_ids,
                    predicted_tokens=fake_model_tokens,
                )
                fake_energies = fake_energies.reshape(-1, 1)
        else:
            all_true_embeddings = torch.cat((real_embeddings_input, true_embeddings), dim=1)
            real_energies = self.transformer(
                all_true_embeddings,
                start_pos=0,
                mcmc_step=self.hparams.mcmc_num_steps - 1,
                real_token_ids=input_ids,
                predicted_tokens=true_pred_tokens,
            ) # NOTE if want to use this maybe check in better detail what ired does
            real_energies = real_energies.reshape(-1, 1) # BS, 1
            fake_energies = predicted_energies[-1] # B*S, 1
        energy_stack = torch.cat([real_energies, fake_energies], dim=1)
        energy_targets = torch.zeros(real_energies.shape[0], dtype=torch.long, device=fake_energies.device)
        padding_positions = ~valid_positions.reshape(-1)
        energy_targets[padding_positions] = -100 # prevents nans instead of using self.tokenizer_pad_token_id, as setting this to 0 leads to issues
        loss_mode = loss_mode or getattr(self.hparams, "first_order_cd_loss_type", "ce")
        if loss_mode == "nce":
            per_token_loss = F.softplus(real_energies.squeeze(-1)) + F.softplus(-fake_energies.squeeze(-1))
            valid_flat = valid_positions.reshape(-1)
            if valid_flat.any():
                contrastive_loss = per_token_loss[valid_flat].mean()
            else:
                contrastive_loss = per_token_loss.sum() * 0.0
        elif loss_mode == "margin":
            margin = getattr(self.hparams, "first_order_cd_margin", 1.0)
            per_token_loss = F.relu(margin + real_energies.squeeze(-1) - fake_energies.squeeze(-1))
            valid_flat = valid_positions.reshape(-1)
            if valid_flat.any():
                contrastive_loss = per_token_loss[valid_flat].mean()
            else:
                contrastive_loss = per_token_loss.sum() * 0.0
        else:
            contrastive_loss = F.cross_entropy(-1 * energy_stack, energy_targets, ignore_index=-100)
        if return_metrics:
            valid_flat = valid_positions.reshape(-1)
            real_energy_flat = real_energies.squeeze(-1).float()
            fake_energy_flat = fake_energies.squeeze(-1).float()
            if valid_flat.any():
                real_energy_mean = real_energy_flat[valid_flat].mean().detach()
                fake_energy_mean = fake_energy_flat[valid_flat].mean().detach()
                energy_gap = (fake_energy_flat - real_energy_flat)[valid_flat].mean().detach()
            else:
                real_energy_mean = contrastive_loss.detach() * 0.0
                fake_energy_mean = contrastive_loss.detach() * 0.0
                energy_gap = contrastive_loss.detach() * 0.0
            return contrastive_loss, {
                'first_order_real_energy': real_energy_mean,
                'first_order_fake_energy': fake_energy_mean,
                'first_order_energy_gap': energy_gap,
            }
        return contrastive_loss

    def calculate_trajectory_local_cd_loss(self, input_ids, next_token_indices, predicted_distributions):
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        device = next_token_indices.device
        dtype = self.embeddings.weight.dtype

        valid_positions = next_token_indices.reshape(batch_size, seq_length) != -1
        stride = int(getattr(self.hparams, "first_order_local_cd_pair_stride", 1) or 1)
        max_pairs = int(getattr(self.hparams, "first_order_local_cd_num_pairs", 1) or 1)
        stride = max(stride, 1)
        max_pairs = max(max_pairs, 1)
        available_pairs = max(len(predicted_distributions) - stride, 0)
        pair_count = min(max_pairs, available_pairs)

        if pair_count <= 0:
            zero = self.embeddings.weight.sum() * 0.0
            return zero, {
                'first_order_local_cd_loss': zero.detach(),
                'first_order_local_cd_energy_delta': zero.detach(),
                'first_order_local_cd_pairs_used': zero.detach(),
            }

        pair_logits = []
        for pair_idx in range(pair_count):
            earlier = predicted_distributions[pair_idx].detach().reshape(batch_size, seq_length, self.vocab_size)
            later = predicted_distributions[pair_idx + stride].detach().reshape(batch_size, seq_length, self.vocab_size)
            pair_logits.extend([earlier, later])

        local_logits = torch.cat(pair_logits, dim=0)
        local_valid_positions = valid_positions.repeat(pair_count * 2, 1)
        local_tokens, local_embeddings = self._relaxed_logits_to_tokens_and_embeddings(
            local_logits,
            valid_positions=local_valid_positions,
        )

        real_embeddings_input = self.embeddings(input_ids)
        repeated_real_embeddings = real_embeddings_input.repeat(pair_count * 2, 1, 1)
        combined_embeddings = torch.cat((repeated_real_embeddings, local_embeddings), dim=1)
        repeated_input_ids = input_ids.repeat(pair_count * 2, 1)
        local_energies = self.transformer(
            combined_embeddings,
            start_pos=0,
            mcmc_step=self.hparams.mcmc_num_steps - 1,
            real_token_ids=repeated_input_ids,
            predicted_tokens=local_tokens,
        ).reshape(pair_count, 2, batch_size, seq_length)

        earlier_energies = local_energies[:, 0].float()
        later_energies = local_energies[:, 1].float()
        energy_delta = later_energies - earlier_energies
        local_valid = valid_positions.unsqueeze(0).expand(pair_count, -1, -1)

        if local_valid.any():
            valid_delta = energy_delta[local_valid]
            loss_type = getattr(self.hparams, "first_order_local_cd_loss_type", "raw")
            if loss_type == "softplus":
                margin = float(getattr(self.hparams, "first_order_local_cd_margin", 0.0) or 0.0)
                local_cd_loss = F.softplus(valid_delta + margin).mean()
            else:
                local_cd_loss = valid_delta.mean()
            mean_delta = valid_delta.mean().detach()
        else:
            local_cd_loss = energy_delta.sum() * 0.0
            mean_delta = local_cd_loss.detach()

        pairs_used = torch.tensor(float(pair_count), device=device, dtype=dtype)
        return local_cd_loss, {
            'first_order_local_cd_loss': local_cd_loss.detach(),
            'first_order_local_cd_energy_delta': mean_delta,
            'first_order_local_cd_pairs_used': pairs_used,
        }

    def _one_hot_token_probs(self, token_indices, valid_positions):
        safe_token_indices = token_indices.clamp(min=0)
        token_probs = torch.zeros(
            (*safe_token_indices.shape, self.vocab_size),
            device=token_indices.device,
            dtype=self.embeddings.weight.dtype,
        )
        token_probs.scatter_(-1, safe_token_indices.unsqueeze(-1), 1.0)
        token_probs = token_probs.masked_fill(~valid_positions.unsqueeze(-1), 0.0)
        return token_probs

    def _token_probs_to_embeddings(self, token_probs):
        if self.hparams.vocab_to_embed_uses_prob_dist:
            return torch.matmul(token_probs, self.embeddings.weight)
        return self.vocab_to_embed(token_probs)

    def _relaxed_logits_to_tokens_and_embeddings(self, relaxed_logits, valid_positions=None):
        relaxed_logits = relaxed_logits.detach()
        if self.hparams.normalize_initial_condition:
            if getattr(self.hparams, 'float_precision', '') == "bf16-true":
                relaxed_tokens = self.softmax(relaxed_logits)
            else:
                relaxed_tokens = self.softmax(relaxed_logits.float()).to(relaxed_logits.dtype)
        else:
            relaxed_tokens = relaxed_logits
        if valid_positions is not None:
            relaxed_tokens = relaxed_tokens.masked_fill(~valid_positions.unsqueeze(-1), 0.0)
        return relaxed_tokens, self._token_probs_to_embeddings(relaxed_tokens)

    def _proposal_aware_logz_offset(self, proposal_mode, reference):
        configured = getattr(self.hparams, "proposal_aware_nce_logz_offset", None)
        if configured is not None:
            return reference.new_tensor(float(configured))
        if proposal_mode == "uniform":
            return reference.new_tensor(math.log(float(self.vocab_size)))
        return reference.new_tensor(0.0)

    def _sample_proposal_negatives(self, next_token_indices_2d, valid_positions, proposal_logits=None):
        proposal_mode = getattr(self.hparams, "proposal_aware_nce_proposal", "uniform")
        k = int(getattr(self.hparams, "proposal_aware_nce_k", 1))
        if k < 1:
            raise ValueError("--proposal_aware_nce_k must be >= 1.")

        batch_size, seq_length = next_token_indices_2d.shape
        safe_next = next_token_indices_2d.clamp(min=0)
        exclude_positive = bool(getattr(self.hparams, "proposal_aware_nce_exclude_positive_negatives", False))

        if proposal_mode == "uniform":
            if exclude_positive:
                if self.vocab_size <= 1:
                    raise ValueError("Cannot exclude positive negatives with vocab_size <= 1.")
                neg_indices = torch.randint(
                    low=0,
                    high=self.vocab_size - 1,
                    size=(batch_size, k, seq_length),
                    device=next_token_indices_2d.device,
                )
                neg_indices = neg_indices + (neg_indices >= safe_next[:, None, :]).long()
                logq_neg_value = -math.log(float(self.vocab_size - 1))
            else:
                neg_indices = torch.randint(
                    low=0,
                    high=self.vocab_size,
                    size=(batch_size, k, seq_length),
                    device=next_token_indices_2d.device,
                )
                logq_neg_value = -math.log(float(self.vocab_size))
            logq_value = -math.log(float(self.vocab_size))
            logq_pos = torch.full(
                (batch_size, seq_length),
                logq_value,
                device=next_token_indices_2d.device,
                dtype=torch.float32,
            )
            logq_neg = torch.full(
                (batch_size, k, seq_length),
                logq_neg_value,
                device=next_token_indices_2d.device,
                dtype=torch.float32,
            )
            return neg_indices, logq_pos, logq_neg

        if proposal_mode == "mcmc_final":
            if proposal_logits is None:
                raise ValueError("proposal_aware_nce_proposal=mcmc_final requires proposal_logits.")
            proposal_logits = proposal_logits.detach().reshape(batch_size, seq_length, self.vocab_size)
            log_probs = F.log_softmax(proposal_logits.float(), dim=-1)
            probs_3d = log_probs.exp()
            if exclude_positive:
                if self.vocab_size <= 1:
                    raise ValueError("Cannot exclude positive negatives with vocab_size <= 1.")
                probs_excluding_positive = probs_3d.scatter(
                    -1,
                    safe_next.unsqueeze(-1),
                    0.0,
                )
                row_sums = probs_excluding_positive.sum(dim=-1, keepdim=True)
                fallback = torch.ones_like(probs_excluding_positive)
                fallback = fallback.scatter(-1, safe_next.unsqueeze(-1), 0.0)
                fallback = fallback / float(self.vocab_size - 1)
                probs_3d = torch.where(
                    row_sums > 1e-12,
                    probs_excluding_positive / row_sums.clamp_min(1e-12),
                    fallback,
                )
            probs = probs_3d.reshape(-1, self.vocab_size)
            sampled = torch.multinomial(probs, num_samples=k, replacement=True)
            neg_indices = sampled.view(batch_size, seq_length, k).permute(0, 2, 1).contiguous()
            logq_pos = log_probs.gather(-1, safe_next.unsqueeze(-1)).squeeze(-1)
            logq_neg_source = torch.log(probs_3d.clamp_min(1e-30)) if exclude_positive else log_probs
            logq_neg = logq_neg_source.gather(
                -1,
                neg_indices.permute(0, 2, 1).reshape(batch_size, seq_length, k)
            ).permute(0, 2, 1).contiguous()
            return neg_indices, logq_pos, logq_neg

        raise ValueError(
            f"Unknown proposal_aware_nce_proposal={proposal_mode}. "
            "Supported values: uniform, mcmc_final."
        )

    def calculate_proposal_aware_nce_loss(self, input_ids, next_token_indices, proposal_logits=None):
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        next_token_indices_2d = next_token_indices.reshape(batch_size, seq_length)
        valid_positions = next_token_indices_2d != -1
        valid_flat = valid_positions.reshape(-1)

        proposal_mode = getattr(self.hparams, "proposal_aware_nce_proposal", "uniform")
        k = int(getattr(self.hparams, "proposal_aware_nce_k", 1))
        nce_base_coeff = float(getattr(self.hparams, "proposal_aware_nce_base_coeff", 1.0))
        relaxed_cd_coeff = float(getattr(self.hparams, "proposal_aware_nce_relaxed_cd_coeff", 0.0))
        use_relaxed_cd = relaxed_cd_coeff != 0.0
        if use_relaxed_cd and proposal_logits is None:
            raise ValueError(
                "--proposal_aware_nce_relaxed_cd_coeff requires a final MCMC relaxed state. "
                "Use --proposal_aware_nce_proposal mcmc_final."
            )
        neg_indices, logq_pos, logq_neg = self._sample_proposal_negatives(
            next_token_indices_2d,
            valid_positions,
            proposal_logits=proposal_logits,
        )

        real_embeddings_input = self.embeddings(input_ids)
        true_probs = self._one_hot_token_probs(next_token_indices_2d, valid_positions)
        true_embeddings = self._token_probs_to_embeddings(true_probs)

        neg_valid_positions = valid_positions[:, None, :].expand(-1, k, -1)
        neg_probs = self._one_hot_token_probs(neg_indices, neg_valid_positions)
        fake_embeddings = self._token_probs_to_embeddings(neg_probs)
        if use_relaxed_cd:
            relaxed_tokens = proposal_logits.detach().reshape(batch_size, seq_length, self.vocab_size)
            relaxed_tokens, relaxed_embeddings = self._relaxed_logits_to_tokens_and_embeddings(
                relaxed_tokens,
                valid_positions=valid_positions,
            )

        expanded_real_embeddings = real_embeddings_input[:, None, :, :].expand(-1, k, -1, -1)
        expanded_input_ids = input_ids[:, None, :].expand(-1, k, -1)

        all_true_embeddings = torch.cat((real_embeddings_input, true_embeddings), dim=1)
        all_fake_embeddings = torch.cat(
            (
                expanded_real_embeddings.reshape(batch_size * k, seq_length, -1),
                fake_embeddings.reshape(batch_size * k, seq_length, -1),
            ),
            dim=1,
        )
        combined_embeddings_parts = [all_true_embeddings, all_fake_embeddings]
        combined_input_ids_parts = [input_ids, expanded_input_ids.reshape(batch_size * k, seq_length)]
        combined_pred_tokens_parts = [
            true_probs,
            neg_probs.reshape(batch_size * k, seq_length, self.vocab_size),
        ]
        if use_relaxed_cd:
            all_relaxed_embeddings = torch.cat((real_embeddings_input, relaxed_embeddings), dim=1)
            combined_embeddings_parts.append(all_relaxed_embeddings)
            combined_input_ids_parts.append(input_ids)
            combined_pred_tokens_parts.append(relaxed_tokens)

        combined_embeddings = torch.cat(combined_embeddings_parts, dim=0)
        combined_input_ids = torch.cat(combined_input_ids_parts, dim=0)
        combined_pred_tokens = torch.cat(combined_pred_tokens_parts, dim=0)

        combined_energies = self.transformer(
            combined_embeddings,
            start_pos=0,
            mcmc_step=self.hparams.mcmc_num_steps - 1,
            real_token_ids=combined_input_ids,
            predicted_tokens=combined_pred_tokens,
        ).reshape(-1, seq_length, 1)
        real_energies = combined_energies[:batch_size].squeeze(-1)
        fake_end = batch_size + batch_size * k
        fake_energies = combined_energies[batch_size:fake_end].reshape(batch_size, k, seq_length)
        relaxed_energies = None
        if use_relaxed_cd:
            relaxed_energies = combined_energies[fake_end:fake_end + batch_size].squeeze(-1)

        log_k = math.log(float(k))
        logz_offset = self._proposal_aware_logz_offset(proposal_mode, real_energies)
        r_pos = -real_energies.float() - logq_pos.float() - logz_offset - log_k
        r_neg = -fake_energies.float() - logq_neg.float() - logz_offset - log_k

        if valid_flat.any():
            pos_loss = F.softplus(-r_pos).reshape(-1)[valid_flat].mean()
            neg_loss = F.softplus(r_neg).permute(0, 2, 1).reshape(batch_size * seq_length, k)
            neg_loss = neg_loss[valid_flat].mean()
            nce_loss = pos_loss + neg_loss
        else:
            nce_loss = (r_pos.sum() + r_neg.sum()) * 0.0
        weighted_nce_loss = nce_base_coeff * nce_loss
        proposal_loss = weighted_nce_loss

        rank_coeff = float(getattr(self.hparams, "proposal_aware_nce_rank_coeff", 0.0))
        rank_loss = proposal_loss.detach() * 0.0
        if rank_coeff != 0.0:
            margin = float(getattr(self.hparams, "proposal_aware_nce_rank_margin", 1.0))
            hard_fake_energies = fake_energies.min(dim=1).values.float()
            per_token_rank = F.softplus(real_energies.float() - hard_fake_energies + margin)
            if valid_flat.any():
                rank_loss = per_token_rank.reshape(-1)[valid_flat].mean()
            else:
                rank_loss = per_token_rank.sum() * 0.0
            proposal_loss = proposal_loss + rank_coeff * rank_loss

        relaxed_cd_loss = proposal_loss.detach() * 0.0
        if use_relaxed_cd:
            relaxed_cd_margin = float(getattr(self.hparams, "proposal_aware_nce_relaxed_cd_margin", 0.0))
            per_token_relaxed_cd = F.softplus(
                real_energies.float() - relaxed_energies.float() + relaxed_cd_margin
            )
            if valid_flat.any():
                relaxed_cd_loss = per_token_relaxed_cd.reshape(-1)[valid_flat].mean()
            else:
                relaxed_cd_loss = per_token_relaxed_cd.sum() * 0.0
            proposal_loss = proposal_loss + relaxed_cd_coeff * relaxed_cd_loss

        if valid_flat.any():
            energy_gap = (fake_energies.mean(dim=1) - real_energies).reshape(-1)[valid_flat].mean().detach()
            if relaxed_energies is not None:
                relaxed_energy_gap = (relaxed_energies - real_energies).reshape(-1)[valid_flat].mean().detach()
            else:
                relaxed_energy_gap = proposal_loss.detach() * 0.0
            mean_logq_pos = logq_pos.reshape(-1)[valid_flat].mean().detach()
            mean_logq_neg = logq_neg.permute(0, 2, 1).reshape(batch_size * seq_length, k)[valid_flat].mean().detach()
        else:
            energy_gap = proposal_loss.detach() * 0.0
            relaxed_energy_gap = proposal_loss.detach() * 0.0
            mean_logq_pos = proposal_loss.detach() * 0.0
            mean_logq_neg = proposal_loss.detach() * 0.0

        metrics = {
            'proposal_aware_nce_loss': proposal_loss.detach(),
            'proposal_aware_nce_base_loss': nce_loss.detach(),
            'proposal_aware_nce_weighted_base_loss': weighted_nce_loss.detach(),
            'proposal_aware_nce_rank_loss': rank_loss.detach(),
            'proposal_aware_nce_relaxed_cd_loss': relaxed_cd_loss.detach(),
            'proposal_aware_nce_energy_gap': energy_gap,
            'proposal_aware_nce_relaxed_energy_gap': relaxed_energy_gap,
            'proposal_aware_nce_logq_pos': mean_logq_pos,
            'proposal_aware_nce_logq_neg': mean_logq_neg,
            'proposal_aware_nce_logz_offset': logz_offset.detach(),
            'proposal_aware_nce_base_coeff': real_energies.new_tensor(nce_base_coeff).detach(),
            'proposal_aware_nce_relaxed_cd_coeff': real_energies.new_tensor(relaxed_cd_coeff).detach(),
        }
        return proposal_loss, metrics
    
    def warm_up_finished(self):
        if self.hparams.clamp_max_after_warm_up != 0.0:
            print(f"changing clamp value after warming up from {self.hparams.clamp_futures_grad_max_change} (see next line)")
            self.hparams.clamp_futures_grad_max_change = self.hparams.clamp_max_after_warm_up
            print(f"to the value {self.hparams.clamp_futures_grad_max_change}")
        self.finished_warming_up = True
        self.langevin_dynamics_noise_std.requires_grad = self.hparams.langevin_dynamics_noise_learnable


    def ebt_advanced_inference(self, original_real_input_ids, start_pos=0, learning=True): # code was written with help from AI
        real_embeddings_input = self.embeddings(original_real_input_ids)  # (B, S, D)
        original_predicted_tokens = self.corrupt_embeddings(real_embeddings_input)  # (B, S, V)

        alpha = self.alpha * self.hparams.infer_ebt_override_alpha if 0 < self.hparams.infer_ebt_override_alpha < 1 else (
            torch.tensor(self.hparams.infer_ebt_override_alpha, device=self.device) if self.hparams.infer_ebt_override_alpha >= 1 else self.alpha
        )

        noise = (torch.tensor(
            self.hparams.infer_langevin_dynamics_noise,
            dtype=self.langevin_dynamics_noise_std.dtype,
            device=self.langevin_dynamics_noise_std.device
        ) if self.hparams.infer_langevin_dynamics_noise != 0 else self.langevin_dynamics_noise_std)

        B, S, V = original_predicted_tokens.shape
        G = self.hparams.infer_generated_samples

        if G > 1:
            repeated_pred = original_predicted_tokens.repeat_interleave(G, dim=0)
            # Optionally corrupt again so each copy starts differently
            repeated_pred = self.corrupt_embeddings(real_embeddings_input.repeat_interleave(G, dim=0))
            repeated_real_embeds = real_embeddings_input.repeat_interleave(G, dim=0)
            repeated_bs = B * G
        else:
            repeated_pred = original_predicted_tokens
            repeated_real_embeds = real_embeddings_input
            repeated_bs = B

        all_final_pred = torch.zeros_like(repeated_pred)
        energies_list_accum = None
        predicted_distributions_accum = None

        chunk_size = B  # or another chunk size if you prefer
        for start in range(0, repeated_bs, chunk_size):
            end = min(start + chunk_size, repeated_bs)

            chunk_pred = repeated_pred[start:end]           # shape: (chunk_size, S, V)
            chunk_real_embeds = repeated_real_embeds[start:end]  # shape: (chunk_size, S, D)
            chunk_real_ids = original_real_input_ids.repeat_interleave(G, dim=0)[start:end] if G > 1 else original_real_input_ids[start:end]

            final_pred_chunk, energies_list_chunk, predicted_distributions_chunk = self._run_ebt_inference_steps(
                chunk_pred, chunk_real_embeds,
                alpha, noise, start_pos, learning,
                real_token_ids=chunk_real_ids
            )
            all_final_pred[start:end] = final_pred_chunk


            energies_list_chunk = [
                e.reshape(chunk_size, -1) for e in energies_list_chunk
            ]

            if energies_list_accum is None:
                energies_list_accum = [e for e in energies_list_chunk]
                predicted_distributions_accum = [p.detach() for p in predicted_distributions_chunk]
            else:
                for i in range(len(energies_list_accum)):
                    energies_list_accum[i] = torch.cat(
                        [energies_list_accum[i], energies_list_chunk[i]], dim=0
                    )
                for i in range(len(predicted_distributions_accum)):
                    if i < len(predicted_distributions_chunk):
                        predicted_distributions_accum[i] = torch.cat(
                            [predicted_distributions_accum[i], predicted_distributions_chunk[i].detach()], dim=0
                        )
        # energies_list_accum is a list of length total_mcmc_steps, each shape (B*G, S)

        if G > 1:
            final_energies_3d = energies_list_accum[-1].reshape(B, G, S)
            
            if self.hparams.infer_debug_sample_distances: # to print the distances between samples if are generating many. good to know if model's samples are diverse or if should add more noise to initial condition
                all_final_pred_4d = all_final_pred.reshape(B, G, S, V)
                softmaxed_preds = self.softmax(all_final_pred_4d)
                for b in range(min(B, 2)):  # Only show first 2 batches to avoid excessive output
                    for s in range(min(S, 5)):  # Only show first 5 sequence positions
                        print(f"Batch {b}, Seq pos {s} - Sample distances:")
                        for i in range(G):
                            for j in range(i+1, G):
                                p_i = softmaxed_preds[b, i, s]
                                p_j = softmaxed_preds[b, j, s]
                                # KL divergence
                                # Add small value to avoid log(0)
                                kl_div = F.kl_div(
                                    (p_i + 1e-10).log(), 
                                    p_j + 1e-10, 
                                    reduction='sum'
                                )
                                # L2 distance
                                l2_dist = torch.norm(p_i - p_j, p=2)
                                print(f"  Sample {i} vs {j}: KL={kl_div.item():.4f}, L2={l2_dist.item():.4f}")
            if self.hparams.infer_energy_sampling_technique == "min":
                best_indices_2d = final_energies_3d.argmin(dim=1)  # shape: (B, S)
            elif self.hparams.infer_energy_sampling_technique == "max":
                best_indices_2d = final_energies_3d.argmax(dim=1)  # shape: (B, S)
            elif self.hparams.infer_energy_sampling_technique == "max_gap":
                initial_energies_3d = energies_list_accum[0].reshape(B, G, S)
                gap_3d = initial_energies_3d - final_energies_3d
                best_indices_2d = gap_3d.argmax(dim=1)             # shape: (B, S)
            else:
                raise ValueError(f"Unknown infer_energy_sampling_technique: {self.hparams.infer_energy_sampling_technique}")

            all_final_pred_4d = all_final_pred.reshape(B, G, S, V)

            b_arange = torch.arange(B, device=all_final_pred.device).unsqueeze(-1)  # shape: (B, 1)
            s_arange = torch.arange(S, device=all_final_pred.device).unsqueeze(0)   # shape: (1, S)

            
            final_output = all_final_pred_4d[b_arange, best_indices_2d, s_arange, :]
        else:
            final_output = all_final_pred

        # final_output shape (B, S, V), energies_list_accum (at each index for original num_mcmc_steps len) shape (B*G, S)
        return final_output, energies_list_accum, predicted_distributions_accum

    def _run_ebt_inference_steps(
        self,
        initial_pred_tokens,
        real_embeds,
        adjusted_alpha,
        noise,
        start_pos,
        learning,
        real_token_ids=None
    ):
        energies_list = []
        pred_states_list = []
        pred_states_list.append(initial_pred_tokens)

        def do_mcmc_step(step_idx, cur_pred_tokens, alpha):
            with torch.set_grad_enabled(True):
                cur_pred_tokens = cur_pred_tokens.detach().requires_grad_()

                # Add noise if set
                if not self.hparams.infer_langevin_first_step: # default
                    cur_pred_tokens = cur_pred_tokens + noise * torch.randn_like(cur_pred_tokens)
                else:
                    if step_idx == 0: # only do langevin on first step
                        cur_pred_tokens = cur_pred_tokens + noise * torch.randn_like(cur_pred_tokens)

                # Convert logits -> embeddings
                if self.hparams.normalize_initial_condition:
                    if self.hparams.normalize_initial_condition_only_first_step:
                        if step_idx == 0:
                            cur_pred_tokens = self.softmax(cur_pred_tokens)
                    else:
                        cur_pred_tokens = self.softmax(cur_pred_tokens)
                            
                    if self.hparams.vocab_to_embed_uses_prob_dist: # predicted_embeds is B, S, V; embed is V, D
                        pred_embeds = torch.matmul(cur_pred_tokens, self.embeddings.weight) #BS, S, D
                    else:
                        pred_embeds = self.vocab_to_embed(cur_pred_tokens) #BS, S, D
                else:
                    pred_embeds = self.vocab_to_embed(cur_pred_tokens)

                combined_embeddings = torch.cat([real_embeds, pred_embeds], dim=1)  # (chunk_size, 2S, D)
                energies = self.transformer(
                    combined_embeddings,
                    start_pos=start_pos,
                    mcmc_step=step_idx,
                    real_token_ids=real_token_ids,
                    predicted_tokens=cur_pred_tokens,
                )
                energies = energies.reshape(-1)
                energies_list.append(energies.detach())

                grad = torch.autograd.grad(energies.sum(), [cur_pred_tokens], create_graph=learning)[0]

                if self.hparams.clamp_futures_grad:
                    min_and_max = self.hparams.clamp_futures_grad_max_change / (alpha)
                    grad = torch.clamp(grad, -min_and_max, min_and_max)

                if self.hparams.infer_accept_lower_energies: # have to get energy to determine if should decrease
                    old_energies = energies.reshape(cur_pred_tokens.shape[:2])
                    proposed_tokens = cur_pred_tokens - alpha * grad
                    new_energies = get_energy(step_idx, proposed_tokens).reshape(cur_pred_tokens.shape[:2])
                    accept_mask = (new_energies < old_energies).float().unsqueeze(-1)
                    updated_tokens = accept_mask * proposed_tokens + (1 - accept_mask) * cur_pred_tokens

                else:
                    updated_tokens = cur_pred_tokens - alpha * grad
                return updated_tokens.detach()
            
        def get_energy(step_idx, cur_pred_tokens): # for if just want to get energy of currently predicted tokens
            with torch.no_grad():
                cur_pred_tokens = cur_pred_tokens.detach().requires_grad_()

                # Convert logits -> embeddings
                if self.hparams.normalize_initial_condition:
                    if self.hparams.normalize_initial_condition_only_first_step:
                        if step_idx == 0:
                            cur_pred_tokens = self.softmax(cur_pred_tokens)
                    else:
                        cur_pred_tokens = self.softmax(cur_pred_tokens)
                            
                    if self.hparams.vocab_to_embed_uses_prob_dist: # predicted_embeds is B, S, V; embed is V, D
                        pred_embeds = torch.matmul(cur_pred_tokens, self.embeddings.weight) #BS, S, D
                    else:
                        pred_embeds = self.vocab_to_embed(cur_pred_tokens) #BS, S, D
                else:
                    pred_embeds = self.vocab_to_embed(cur_pred_tokens)
                combined_embeddings = torch.cat([real_embeds, pred_embeds], dim=1)  # (chunk_size, 2S, D)
                energies = self.transformer(
                    combined_embeddings,
                    start_pos=start_pos,
                    mcmc_step=step_idx,
                    real_token_ids=real_token_ids,
                    predicted_tokens=cur_pred_tokens,
                )
                energies = energies.reshape(-1)
                return energies

        # ebt_type
        if self.hparams.ebt_type == "default" or (self.hparams.ebt_type == "time_embed" and not getattr(self.hparams, 'use_mcmc_time_embed', False)):
            # default mode or time_embed without time embedding: shared transition kernel, arbitrary step count
            total_steps = self.hparams.infer_ebt_num_steps if self.hparams.infer_ebt_num_steps > 1 else self.hparams.mcmc_num_steps
            pred_state = initial_pred_tokens
            for step_idx in range(total_steps):
                pred_state = do_mcmc_step(step_idx, pred_state, adjusted_alpha)
                pred_states_list.append(pred_state)
        else:
            # alternative ebt_type i.e. adaln or time embed
            pred_state = initial_pred_tokens
            for step_idx in range(self.hparams.mcmc_num_steps):
                if self.hparams.infer_steps_final_landscape and step_idx != (self.hparams.mcmc_num_steps - 1):
                    alpha = self.alpha if self.hparams.infer_alpha_final_landscape else adjusted_alpha
                    pred_state = do_mcmc_step(step_idx, pred_state, alpha)
                    pred_states_list.append(pred_state)
                else:
                    inner_steps = self.hparams.infer_ebt_num_steps if self.hparams.infer_ebt_num_steps != 1 else (self.hparams.randomize_mcmc_num_steps_min if self.hparams.randomize_mcmc_num_steps_min != 0 else 1)
                    for _ in range(inner_steps):
                        alpha = self.alpha if (self.hparams.infer_alpha_final_landscape and step_idx != (self.hparams.mcmc_num_steps - 1)) else adjusted_alpha
                        pred_state = do_mcmc_step(step_idx, pred_state, alpha)
                        pred_states_list.append(pred_state)


        final_pred_state_energies = get_energy((self.hparams.mcmc_num_steps - 1), pred_state)
        energies_list.append(final_pred_state_energies)
        return pred_state, energies_list, pred_states_list
