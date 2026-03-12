import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule
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
        self.tokenizer = self.hparams.tokenizer
        self.tokenizer_pad_token_id = None # Nanochat doesn't have <|pad|> or <|eos|> # self.tokenizer.eos_token_id 
        
        self.vocab_size = self.tokenizer.get_vocab_size() # len(self.tokenizer) # self.vocab_size = self.tokenizer.vocab_size caused errors since is smaller than len(self.tokenizer), is 50254 for neox-20b, len tokenizer is 50277 so decided to use that
        self.hparams.vocab_size = self.vocab_size

        self.alpha = nn.Parameter(torch.tensor(float(self.hparams.mcmc_step_size)), requires_grad=self.hparams.mcmc_step_size_learnable)
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

    def forward(self, x, start_pos = 0, learning = True, return_raw_logits = False, replay_buffer_logits = None, no_randomness = True): # accepts input_ids as input; a lot of the logic here is just for S2 params, see pseudocode in paper for a more concise view of how this works. it can be < 10 LOC
        predicted_distributions = []
        predicted_energies = []

        real_embeddings_input = self.embeddings(x)
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        
        alpha = torch.clamp(self.alpha, min=0.0001)

        langevin_dynamics_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)

        predicted_tokens = self.corrupt_embeddings(real_embeddings_input) # B, S, V

        mcmc_steps = [] # in the general case of no randomize_mcmc_num_steps then this has len == self.hparams.randomize_mcmc_num_steps
        for step in range(self.hparams.mcmc_num_steps):
            mcmc_steps.append(step)

        with torch.set_grad_enabled(True):
            for i, mcmc_step in enumerate(mcmc_steps):
                predicted_tokens = predicted_tokens.detach().requires_grad_().reshape(batch_size, seq_length, self.vocab_size) # B, S, V
                predicted_tokens = self.softmax(predicted_tokens)
                predicted_embeddings = self.vocab_to_embed(predicted_tokens) #BS, S, D
                all_embeddings = torch.cat((real_embeddings_input, predicted_embeddings), dim = 1) # B, 2*S, D
                
                energy_preds = self.transformer(all_embeddings, start_pos = start_pos, mcmc_step=mcmc_step) # is B, 2*S, D; checked and there are no in place ops; mcmc_step only applies to when using certain types of ebt
                energy_preds = energy_preds.reshape(-1, 1)
                predicted_energies.append(energy_preds)
                predicted_tokens_grad = torch.autograd.grad([energy_preds.sum()], [predicted_tokens], create_graph=learning)[0]
                predicted_tokens = predicted_tokens - alpha * predicted_tokens_grad # do this to tokens will be unnormalize prob dist convert to prob dist after  

                if return_raw_logits:
                    predicted_tokens_for_loss = predicted_tokens # BS, S, V
                else:
                    predicted_tokens_for_loss = self.log_softmax(predicted_tokens).reshape(-1, self.vocab_size) # BS*S, V
                predicted_distributions.append(predicted_tokens_for_loss)        

        return predicted_distributions, predicted_energies

    def forward_loss_wrapper(self, x, phase="train", token_bytes=None):
        no_randomness = False if phase == "train" else True
        input_ids = x[0].squeeze(dim=0)
        next_token_indices = x[1].squeeze(dim=0)
        predicted_distributions, predicted_energies = self(input_ids, return_raw_logits = True, no_randomness = no_randomness)

        if self.hparams.execution_mode == "finetune": # Only tokens after "[[Answer]]: " will be calculated in finetune
            next_token_indices = mask_q_tokens(next_token_indices, self.tokenizer)
        next_token_indices = next_token_indices.reshape(-1) # BS * S; reshape since targets are supposed to be 1D

        reconstruction_loss = 0
        total_mcmc_steps = len(predicted_energies) # in general this equals self.hparams.mcmc_num_steps, isnt in case of rand number
        for mcmc_step, (predicted_distribution, predicted_energy) in enumerate(zip(predicted_distributions, predicted_energies)):

            predicted_distribution = self.log_softmax(predicted_distribution).reshape(-1, self.vocab_size)
            cce_loss = F.nll_loss(predicted_distribution, next_token_indices) # , ignore_index=self.tokenizer_pad_token_id)
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
            'bpb_loss': bpb_loss
        }
        return log_dict
    

    def corrupt_embeddings(self, embeddings):

        predicted_tokens = torch.randn(size=(embeddings.shape[0], embeddings.shape[1], self.vocab_size), device = self.device) * self.hparams.gaussian_random_noise_scaling

        return predicted_tokens
    
    def warm_up_finished(self):

        self.finished_warming_up = True
        self.langevin_dynamics_noise_std.requires_grad = self.hparams.langevin_dynamics_noise_learnable


