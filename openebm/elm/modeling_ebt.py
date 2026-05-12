"""Energy-Based Transformer (EBT) language model.

This module defines :class:`EBT_NLP`, a LightningModule-based language model
that refines predicted token distributions through MCMC-style iterative
energy minimization. At each step the model takes the gradient of a scalar
energy with respect to the predicted logits and updates them, optionally
with Langevin-style noise and gradient clamping. The module also exposes a
contrastive energy loss and a multi-sample advanced inference loop with
energy-based best-sample selection.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from openebm.elm.nanolightning.torchlightning_module import LightningModule

import math
import random
import os
from openebm.elm.utils import setup_ebt, init_whole_model_weights
from openebm.elm.utils import MLP, Memory_Augmented_MLP, Memory_Gating_MLP, mask_q_tokens
from openebm.elm.replay_buffer import CausalReplayBuffer
from openebm.elm.metrics import calculate_bpb_score

import ipdb

class EBT_NLP(LightningModule):
    """Energy-Based Transformer for next-token prediction.

    The model predicts a distribution over the vocabulary for each position
    by iteratively minimizing a scalar energy produced by an internal
    transformer. MCMC-style updates alternate between computing the
    gradient of the energy with respect to the current predicted logits and
    taking a step in the descent direction, optionally with Langevin noise.

    :param hparams: hyperparameters container, either a ``dict`` loaded from
        a checkpoint or an object whose attributes are copied via ``vars``.
    :type hparams: Any
    """

    def __init__(self, hparams: Any) -> None:
        """Initialize embeddings, transformer, replay buffer, and MCMC state.

        :param hparams: hyperparameters, as dict (from ckpt) or namespace.
        :type hparams: Any
        """
        super().__init__()
        if isinstance(hparams, dict):
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))

        # NOTE: prefer tokenizer_obj when set by ModelTrainer, otherwise fall back to raw tokenizer
        self.tokenizer = self.hparams.tokenizer_obj if hasattr(self.hparams, 'tokenizer_obj') else self.hparams.tokenizer
        self.tokenizer_pad_token_id = None # NOTE: Nanochat tokenizer has no explicit <|pad|>/<|eos|> id

        # NOTE: use tokenizer.get_vocab_size() rather than vocab_size attr; the attr underreports
        # (e.g. neox-20b: 50254 vs len(tokenizer) 50277) so we consistently use the full size.
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.hparams.vocab_size = self.vocab_size

        self.alpha = nn.Parameter(torch.tensor(float(self.hparams.mcmc_step_size)), requires_grad=self.hparams.mcmc_step_size_learnable)
        # NOTE: langevin noise std starts frozen; warm_up_finished() flips requires_grad if configured learnable
        self.langevin_dynamics_noise_std = nn.Parameter(torch.tensor(float(self.hparams.langevin_dynamics_noise)), requires_grad=False)

        self.embeddings = nn.Embedding(self.vocab_size, self.hparams.embedding_dim)
        init_whole_model_weights(self.embeddings, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)

        self.log_softmax = nn.LogSoftmax(dim = -1)
        self.softmax = nn.Softmax(dim = -1)

        if not self.hparams.vocab_to_embed_uses_prob_dist:
            if 'learnable_process_memory' in self.hparams and self.hparams.learnable_process_memory and self.hparams.process_memory_type != None:
                self.vocab_to_embed = Memory_Gating_MLP(self.vocab_size, self.hparams.embedding_dim, self.hparams.process_memory_type, self.hparams.process_memory_linear_layer)
            elif 'learnable_process_memory' in self.hparams and self.hparams.learnable_process_memory:
                assert self.hparams.num_modality_processing_mlp_layers > 1, "must set self.hparams.num_modality_processing_mlp_layers > 1 if not using self.hparams.process_memory_type"
                self.vocab_to_embed = Memory_Augmented_MLP(self.vocab_size, self.hparams.embedding_dim, self.hparams.embedding_dim, self.hparams.embedding_dim, dropout_rate=0, layer_norm=True, num_hidden_layers = self.hparams.num_modality_processing_mlp_layers)
            elif self.hparams.num_modality_processing_mlp_layers != 1:
                self.vocab_to_embed = MLP(self.vocab_size, self.hparams.embedding_dim, self.hparams.embedding_dim, dropout_rate=0, layer_norm=True, num_hidden_layers = self.hparams.num_modality_processing_mlp_layers - 2)
            else:
                # NOTE: EBT-specific: we feed a prob dist through a linear projection instead of an embedding lookup
                self.vocab_to_embed = nn.Linear(self.vocab_size, self.hparams.embedding_dim, bias = False, device = self.device)
            init_whole_model_weights(self.vocab_to_embed, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)

        self.transformer = setup_ebt(self.hparams)

        self.finished_warming_up = False

        self.mcmc_replay_buffer = 'mcmc_replay_buffer' in self.hparams and self.hparams.mcmc_replay_buffer and self.hparams.execution_mode != "inference"
        if self.mcmc_replay_buffer:
            replay_buffer_max_size = self.hparams.mcmc_replay_buffer_size
            self.replay_buffer_samples = self.hparams.batch_size_per_device * self.hparams.mcmc_replay_buffer_sample_bs_percent
            self.replay_buffer = CausalReplayBuffer(max_size=replay_buffer_max_size, sample_size=self.replay_buffer_samples)

        if self.hparams.debug_unused_parameters:
            self.used_parameters = set()
            # NOTE: parameters_not_to_check holds params that are frozen or otherwise excluded from the unused-param check
            self.parameters_not_to_check = set()
# __PART2__
