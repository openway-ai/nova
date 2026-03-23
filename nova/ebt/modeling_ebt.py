import torch
from torch import nn
from torch.func import grad, functional_call
from torch.nn import functional as F
from nanolightning.torchlightning_module import LightningModule
# import torch.optim as optim
# from torchmetrics import Accuracy
# from transformers import AutoTokenizer

import math
import random
import os
from utils import setup_ebt, init_whole_model_weights
from utils import MLP, Memory_Augmented_MLP, Memory_Gating_MLP, mask_q_tokens
from replay_buffer import CausalReplayBuffer
from metrics import calculate_bpb_score

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
        
        self.alpha = nn.Parameter(torch.tensor(float(self.hparams.mcmc_step_size)), requires_grad=self.hparams.mcmc_step_size_learnable)
        self.langevin_dynamics_noise_std = nn.Parameter(torch.tensor(float(self.hparams.langevin_dynamics_noise)), requires_grad=False) # if using self.hparams.langevin_dynamics_noise_learnable this will be turned on in warm_up_finished func

        self.embeddings = nn.Embedding(self.vocab_size, self.hparams.embedding_dim)
        init_whole_model_weights(self.embeddings, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)
        
        self.log_softmax = nn.LogSoftmax(dim = -1)
        self.softmax = nn.Softmax(dim = -1)
        
        # Vocab to Embed Projection
        self._init_vocab_to_embed()

        self.transformer = setup_ebt(self.hparams)
        # Store raw reference for torch.compile/torch.func compatibility
        self.transformer_raw_ref = [self.transformer]
        
        self.finished_warming_up = False

        self.mcmc_replay_buffer = 'mcmc_replay_buffer' in self.hparams and self.hparams.mcmc_replay_buffer and self.hparams.execution_mode != "inference"
        if self.mcmc_replay_buffer:
            replay_buffer_max_size = self.hparams.mcmc_replay_buffer_size
            self.replay_buffer_samples = self.hparams.batch_size_per_device * self.hparams.mcmc_replay_buffer_sample_bs_percent
            self.replay_buffer = CausalReplayBuffer(max_size=replay_buffer_max_size, sample_size=self.replay_buffer_samples)

        # Compilation for speed optimization
        # Compiling grad_fn avoids repeated definition and recompilation in the forward loop
        compile_mode = getattr(self.hparams, 'compile_mode', 'full')
        
        if self.hparams.compile_model and compile_mode == 'full':
            print("[EBT] Compiling grad_fn with mode='max-autotune'...")
            self.grad_fn = torch.compile(grad(self._energy_fn_impl, argnums=2), mode="max-autotune")
        else:
            self.grad_fn = grad(self._energy_fn_impl, argnums=2)

        # DEBUGGING CODE ################################################################################################################################################
        if self.hparams.debug_unused_parameters:
            self.used_parameters = set()
            self.parameters_not_to_check = set() # dont check these since may be frozen or dont want them to update
        
    def _init_vocab_to_embed(self):
        if not self.hparams.vocab_to_embed_uses_prob_dist:
            if getattr(self.hparams, 'learnable_process_memory', False):
                if self.hparams.process_memory_type:
                    self.vocab_to_embed = Memory_Gating_MLP(self.vocab_size, self.hparams.embedding_dim, self.hparams.process_memory_type, self.hparams.process_memory_linear_layer)
                else:
                    assert self.hparams.num_modality_processing_mlp_layers > 1, "num_modality_processing_mlp_layers > 1 required if process_memory_type is None"
                    self.vocab_to_embed = Memory_Augmented_MLP(self.vocab_size, self.hparams.embedding_dim, self.hparams.embedding_dim, self.hparams.embedding_dim, dropout_rate=0, layer_norm=True, num_hidden_layers=self.hparams.num_modality_processing_mlp_layers)
            elif self.hparams.num_modality_processing_mlp_layers != 1:
                self.vocab_to_embed = MLP(self.vocab_size, self.hparams.embedding_dim, self.hparams.embedding_dim, dropout_rate=0, layer_norm=True, num_hidden_layers=self.hparams.num_modality_processing_mlp_layers - 2)
            else:
                self.vocab_to_embed = nn.Linear(self.vocab_size, self.hparams.embedding_dim, bias=False, device=self.device)
            
            init_whole_model_weights(self.vocab_to_embed, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)

    def _get_probs(self, tokens, step):
        if self.hparams.normalize_initial_condition:
            if self.hparams.normalize_initial_condition_only_first_step and step != 0:
                return tokens
            return torch.softmax(tokens, dim=-1)
        return tokens

    def _energy_fn_impl(self, params, buffers, current_predicted_tokens, real_embeddings_input, start_pos, mcmc_step):
        # Functional energy calculation for gradient computation
        probs = self._get_probs(current_predicted_tokens, mcmc_step)

        if self.hparams.vocab_to_embed_uses_prob_dist and self.hparams.normalize_initial_condition:
            pred_embeds = torch.matmul(probs, self.embeddings.weight)
        else:
            pred_embeds = self.vocab_to_embed(probs)

        all_embeddings = torch.cat((real_embeddings_input, pred_embeds), dim=1)
        
        # Use functional_call with raw transformer reference to support torch.func.grad
        transformer_raw = self.transformer_raw_ref[0]
        energy_preds = functional_call(
            transformer_raw, 
            {**params, **buffers}, 
            (all_embeddings,), 
            kwargs={'start_pos': start_pos, 'mcmc_step': mcmc_step}
        )
        return energy_preds.sum()

    def forward(self, x, start_pos = 0, learning = True, return_raw_logits = False, replay_buffer_logits = None, no_randomness = True): # accepts input_ids as input; a lot of the logic here is just for S2 params, see pseudocode in paper for a more concise view of how this works. it can be < 10 LOC
        predicted_distributions = []
        predicted_energies = []

        real_embeddings_input = self.embeddings(x)
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        
        # 1. Initialize Tokens
        predicted_tokens = self.corrupt_embeddings(real_embeddings_input)
        if replay_buffer_logits is not None:
            predicted_tokens[batch_size - replay_buffer_logits.shape[0]:] = replay_buffer_logits

        # 2. Determine Alpha
        alpha = torch.clamp(self.alpha, min=0.0001)
        if not no_randomness and self.hparams.randomize_mcmc_step_size_scale != 1:
            scale = self.hparams.randomize_mcmc_step_size_scale
            expanded_alpha = alpha.expand(batch_size, seq_length, 1)
            alpha = (alpha / scale) + torch.rand_like(expanded_alpha) * (alpha * scale - alpha / scale)

        # 3. Determine MCMC Steps
        mcmc_steps = self._get_mcmc_steps(no_randomness)

        # 4. Prepare Parameters for Functional Call
        transformer_raw = self.transformer_raw_ref[0]
        params = dict(transformer_raw.named_parameters())
        buffers = dict(transformer_raw.named_buffers())

        langevin_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)

        with torch.set_grad_enabled(True):
            for step in mcmc_steps:
                # Detach & Require Grad
                if not self.hparams.no_mcmc_detach:
                    predicted_tokens = predicted_tokens.detach()
                predicted_tokens = predicted_tokens.requires_grad_().reshape(batch_size, seq_length, self.vocab_size)

                # Langevin Dynamics
                if self.hparams.langevin_dynamics_noise != 0:
                    predicted_tokens = predicted_tokens + torch.randn_like(predicted_tokens) * langevin_std

                # Compute Gradient
                predicted_tokens_grad = self.grad_fn(params, buffers, predicted_tokens, real_embeddings_input, start_pos, step)

                # Update Tokens
                # Removed clamp_futures_grad as it is marked deprecated
                predicted_tokens = predicted_tokens - alpha * predicted_tokens_grad

                # Re-calculate Energy for Logging (since grad_fn returns sum)
                with torch.no_grad():
                    probs = self._get_probs(predicted_tokens, step)
                    
                    if self.hparams.vocab_to_embed_uses_prob_dist and self.hparams.normalize_initial_condition:
                        pred_embeds = torch.matmul(probs, self.embeddings.weight)
                    else:
                        pred_embeds = self.vocab_to_embed(probs)

                    all_embeddings = torch.cat((real_embeddings_input, pred_embeds), dim=1)
                    # Use standard forward call here
                    energy_preds = self.transformer(all_embeddings, start_pos=start_pos, mcmc_step=step)
                    predicted_energies.append(energy_preds.reshape(-1, 1))

                # Prepare Output
                if return_raw_logits:
                    predicted_distributions.append(predicted_tokens)
                else:
                    predicted_distributions.append(self.log_softmax(predicted_tokens).reshape(-1, self.vocab_size))

        return predicted_distributions, predicted_energies

    def _get_mcmc_steps(self, no_randomness):
        steps = []
        base_steps = self.hparams.mcmc_num_steps
        randomize_steps = getattr(self.hparams, 'randomize_mcmc_num_steps', 0)

        for step in range(base_steps):
            if randomize_steps > 0:
                is_final = (step == base_steps - 1)
                should_randomize = (not self.hparams.randomize_mcmc_num_steps_final_landscape) or is_final
                
                if should_randomize:
                    if no_randomness:
                        # Use max steps for consistency during eval/determinism
                        repeats = randomize_steps + 1 if is_final else 1 # logic copied from original: only extend final step
                    else:
                        min_steps = max(1, self.hparams.randomize_mcmc_num_steps_min)
                        repeats = torch.randint(min_steps, randomize_steps + 2, (1,)).item()
                    
                    steps.extend([step] * repeats)
                    continue
            
            steps.append(step)
        return steps

    def forward_loss_wrapper(self, x, phase="train", token_bytes=None):
        no_randomness = False if phase == "train" else True
        if not no_randomness and self.mcmc_replay_buffer: # dont do this when doing val/testing
            # all_tokens = x['input_ids'].squeeze(dim=1)
            all_tokens = x[0].squeeze(dim=0)
            input_ids, replay_buffer_logits, next_token_indices = self.replay_buffer.get_batch(all_tokens) # this automatically does indexing for input ids and next token indices while also passing back the logits
            predicted_distributions, predicted_energies = self(input_ids, return_raw_logits = True, replay_buffer_logits = replay_buffer_logits, no_randomness = no_randomness)
            self.replay_buffer.update(all_tokens.detach(), predicted_distributions[-1].detach()) # update using the final predicted distributions
        else:
            input_ids = x[0].squeeze(dim=0)
            next_token_indices = x[1].squeeze(dim=0)
            predicted_distributions, predicted_energies = self(input_ids, return_raw_logits = True, no_randomness = no_randomness)

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
                cce_loss = F.cross_entropy(predicted_distribution, next_token_indices, label_smoothing=label_smoothing) # , ignore_index=self.tokenizer_pad_token_id
            else:
                predicted_distribution = self.log_softmax(predicted_distribution).reshape(-1, self.vocab_size)
                cce_loss = F.nll_loss(predicted_distribution, next_token_indices) # , ignore_index=self.tokenizer_pad_token_id)
            
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

        if self.hparams.contrastive_loss: # works by pushing up on energies model predicted and pushing down on energy of true samples
            contrastive_loss = self.calculate_contrastive_loss(predicted_energies, input_ids, next_token_indices)
            total_loss = self.hparams.reconstruction_coeff * reconstruction_loss + self.hparams.contrastive_loss_coeff * contrastive_loss
            contrastive_loss = contrastive_loss.detach()
        else:
            total_loss = self.hparams.reconstruction_coeff * reconstruction_loss
            contrastive_loss = 0.0
        
        if token_bytes is not None:
            bpb_loss = calculate_bpb_score(next_token_indices, cce_loss.detach(), token_bytes)
        else:
            bpb_loss = 0

        log_dict = {
            'loss': total_loss,
            'initial_loss' : initial_loss,
            'final_step_loss': final_reconstruction_loss,
            'contrastive_loss' : contrastive_loss,
            'initial_final_pred_energies_gap': initial_final_pred_energies_gap,
            'perplexity': ppl_loss,
            'bpb': bpb_loss  # Renamed from 'bpb_loss' for cleaner logging
        }
        return log_dict
    

    def corrupt_embeddings(self, embeddings):
        if self.hparams.denoising_initial_condition == "most_recent_embedding":
            raise NotImplementedError(f"most_recent_embedding denoising_initial_condition not supported for NLP yet")
        elif self.hparams.denoising_initial_condition == "random_noise":
            predicted_tokens = torch.randn(size=(embeddings.shape[0], embeddings.shape[1], self.vocab_size), device = self.device) * self.hparams.gaussian_random_noise_scaling
        elif self.hparams.denoising_initial_condition == "zeros":
            predicted_tokens = torch.zeros(size=(embeddings.shape[0], embeddings.shape[1], self.vocab_size), device = self.device)
        else:
            raise NotImplementedError(f"{self.hparams.denoising_initial_condition} denoising_initial_condition not yet supported")
        
        return predicted_tokens
    
    def calculate_contrastive_loss(self, predicted_energies, input_ids, next_token_indices):
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        real_embeddings_input = self.embeddings(input_ids)
        
        next_token_indices_2d = next_token_indices.reshape(batch_size, seq_length)
        
        if self.hparams.discrete_contrastive_loss_true_logit_val != 0: # NOTE from experience this doesnt work very well and it not recommended compared to just one hot encoding
            true_logit_value = self.hparams.discrete_contrastive_loss_true_logit_val
            false_logit_value = -1 * true_logit_value
            true_token_logits = torch.full((batch_size, seq_length, self.vocab_size), false_logit_value, device=next_token_indices.device)
            
            batch_idx = torch.arange(batch_size, device=next_token_indices.device).view(-1, 1).expand(-1, seq_length)
            seq_idx = torch.arange(seq_length, device=next_token_indices.device).view(1, -1).expand(batch_size, -1)
            true_token_logits[batch_idx, seq_idx, next_token_indices_2d] = true_logit_value
            
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
            true_token_one_hot[batch_idx, seq_idx, next_token_indices_2d] = 1.0
            
            if self.hparams.vocab_to_embed_uses_prob_dist:
                true_embeddings = torch.matmul(true_token_one_hot, self.embeddings.weight)
            else:
                true_embeddings = self.vocab_to_embed(true_token_one_hot)

        all_true_embeddings = torch.cat((real_embeddings_input, true_embeddings), dim=1)
        
        real_energies = self.transformer(all_true_embeddings, start_pos=0, mcmc_step=self.hparams.mcmc_num_steps - 1) # NOTE if want to use this maybe check in better detail what ired does
        real_energies = real_energies.reshape(-1, 1) # BS, 1
        fake_energies = predicted_energies[-1] # B*S, 1
        energy_stack = torch.cat([real_energies, fake_energies], dim=1)
        energy_targets = torch.zeros(real_energies.shape[0], dtype=torch.long, device=fake_energies.device)
        padding_positions = None # (next_token_indices == self.tokenizer_pad_token_id).reshape(-1)
        energy_targets[padding_positions] = -100 # prevents nans instead of using self.tokenizer_pad_token_id, as setting this to 0 leads to issues
        contrastive_loss = F.cross_entropy(-1 * energy_stack, energy_targets, ignore_index=-100)
        return contrastive_loss
    
    def warm_up_finished(self):
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

            final_pred_chunk, energies_list_chunk, predicted_distributions_chunk = self._run_ebt_inference_steps(
                chunk_pred, chunk_real_embeds,
                alpha, noise, start_pos, learning
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

    def _run_ebt_inference_steps(self, initial_pred_tokens, real_embeds, adjusted_alpha, noise, start_pos, learning):
        energies_list = []
        pred_states_list = [initial_pred_tokens]
        cur_pred_tokens = initial_pred_tokens

        total_steps = self.hparams.infer_ebt_num_steps if self.hparams.ebt_type == "default" and self.hparams.infer_ebt_num_steps > 1 else self.hparams.mcmc_num_steps

        # Flattened loop logic for simplicity
        steps_config = []
        if self.hparams.ebt_type == "default":
            steps_config = [(i, adjusted_alpha) for i in range(total_steps)]
        else:
            # Complex scheduling for time_embed/adaln
            for step_idx in range(self.hparams.mcmc_num_steps):
                is_final_landscape = (step_idx == self.hparams.mcmc_num_steps - 1)
                
                if self.hparams.infer_steps_final_landscape and not is_final_landscape:
                    alpha = self.alpha if self.hparams.infer_alpha_final_landscape else adjusted_alpha
                    steps_config.append((step_idx, alpha))
                else:
                    inner_steps = self.hparams.infer_ebt_num_steps if self.hparams.infer_ebt_num_steps != 1 else max(1, self.hparams.randomize_mcmc_num_steps_min)
                    for _ in range(inner_steps):
                        alpha = self.alpha if (self.hparams.infer_alpha_final_landscape and not is_final_landscape) else adjusted_alpha
                        steps_config.append((step_idx, alpha))

        # Functional energy helper for inference gradient
        def compute_energy_sum(tokens, step_idx):
            probs = self._get_probs(tokens, step_idx)
            if self.hparams.vocab_to_embed_uses_prob_dist and self.hparams.normalize_initial_condition:
                embeds = torch.matmul(probs, self.embeddings.weight)
            else:
                embeds = self.vocab_to_embed(probs)
            
            combined = torch.cat([real_embeds, embeds], dim=1)
            return self.transformer(combined, start_pos=start_pos, mcmc_step=step_idx).sum()

        for i, (mcmc_step_idx, alpha) in enumerate(steps_config):
            with torch.set_grad_enabled(True):
                cur_pred_tokens = cur_pred_tokens.detach().requires_grad_()
                
                # Noise
                if not self.hparams.infer_langevin_first_step or i == 0:
                     cur_pred_tokens = cur_pred_tokens + noise * torch.randn_like(cur_pred_tokens)

                # Gradient Step
                # Recalculate energy for gradient
                energy_sum = compute_energy_sum(cur_pred_tokens, mcmc_step_idx)
                grad_val = torch.autograd.grad(energy_sum, [cur_pred_tokens], create_graph=learning)[0]
                
                # Update
                cur_pred_tokens = cur_pred_tokens - alpha * grad_val
                
                # Store energy for logging (detached)
                # Note: this recalculates forward pass, but it's inference so it's okay
                with torch.no_grad():
                     probs = self._get_probs(cur_pred_tokens, mcmc_step_idx)
                     if self.hparams.vocab_to_embed_uses_prob_dist and self.hparams.normalize_initial_condition:
                         embeds = torch.matmul(probs, self.embeddings.weight)
                     else:
                         embeds = self.vocab_to_embed(probs)
                     combined = torch.cat([real_embeds, embeds], dim=1)
                     energies = self.transformer(combined, start_pos=start_pos, mcmc_step=mcmc_step_idx)
                     energies_list.append(energies.reshape(-1).detach())

                pred_states_list.append(cur_pred_tokens.detach())

        return cur_pred_tokens, energies_list, pred_states_list