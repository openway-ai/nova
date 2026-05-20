import torch
from torch import nn
import numpy as np
import traceback
import math
from typing import Optional, Tuple
from dataclasses import dataclass

# from torch.nn import functional as F
# import pytorch_lightning as L
# import torch.optim as optim
# from torchmetrics import Accuracy
# from torchvision.transforms import functional as TF
# import torchvision.models as models
# from diffusers import AutoencoderKL
# import random
# from functools import partial
# from PIL import Image
# import torchvision
# from torchvision.transforms import ToPILImage
# from datetime import datetime
# import torch.distributed as dist
# import os
# from torchvision.utils import make_grid, save_image
# import json


BLOCK_MODE_CHOICES = (
    "dense_token",
    "mtp_mcmc",
    "future_latent_non_causal",
    "blockwise",
    "future_latent_bidirectional",
)

# Block modes that use the new "K future latents per source position" semantic
# with a single-token head (latent_token_head : D -> V).
#
# Intra-block visibility differs across modes:
#   * future_latent_non_causal : z_{t,j} sees only itself (within block).
#   * blockwise                : z_{t,j} sees z_{t,1..j}        (causal in j).
#   * future_latent_bidirectional : z_{t,j} sees z_{t,*}        (full intra-
#                                   block attention; "true joint MCMC plane").
EXPLICIT_BLOCK_LATENT_MODES = (
    "future_latent_non_causal",
    "blockwise",
    "future_latent_bidirectional",
)


def build_explicit_block_latent_mask(
    *,
    context_len: int,
    block_size: int,
    time_len: int,
    block_mode: str,
    device,
    dtype,
):
    """Build the additive attention mask for the explicit-block-latent modes.

    These modes (``future_latent_non_causal`` and ``blockwise``) use a sequence
    layout that is fundamentally different from the legacy ``mtp_mcmc`` /
    ``dense_token`` symmetric layout, so they get their own mask helper. This
    helper does NOT touch any code path used by ``mtp_mcmc``.

    Layout (token positions inside the returned ``(T, T)`` mask, 0-indexed):
      ``[0, time_len)``                          → time tokens (only present
                                                    when ``time_len > 0``,
                                                    e.g. ``time_embed`` trunk)
      ``[time_len, time_len + S)``               → context ``c_1 .. c_S``
                                                    (S = ``context_len``)
      ``[time_len + S, time_len + S + S * K)``   → predicted future latents in
                                                    **position-major** order:
            ``z_{1,1}, z_{1,2}, ..., z_{1,K},``
            ``z_{2,1}, z_{2,2}, ..., z_{2,K},``
            ``...,``
            ``z_{S,1}, z_{S,2}, ..., z_{S,K}``
        i.e. for source position ``t`` (1-indexed) and block offset
        ``j`` (1-indexed), ``z_{t,j}`` lives at index
        ``time_len + S + (t - 1) * K + (j - 1)``.

    Visibility rules (queries -> keys):
      * Time / context queries are causal among themselves and CANNOT see any
        predicted latent. This preserves the standard "context is causal"
        invariant.
      * Pred query ``z_{t,j}`` always sees:
            - all ``time_len`` time tokens
            - context tokens ``c_1 .. c_t`` (i.e. context up to and including
              the source position it is predicting from)
        and is forbidden from seeing ``c_{t+1} .. c_S``.
      * Pred -> pred visibility:
            - ``future_latent_non_causal``: ``z_{t,j}`` only sees itself (no
              communication between latents within the same block, and no
              cross source position visibility).
            - ``blockwise``: ``z_{t,j}`` sees ``z_{t,1} .. z_{t,j}`` (causal
              within the block, still no cross source position visibility).
        In particular, ``z_{u,*}`` is invisible to ``z_{t,*}`` whenever
        ``u != t`` for both modes.

    Returns:
        Additive mask of shape ``(T, T)`` with values in ``{0, -inf}``,
        ``dtype=dtype`` and on ``device``. Add it to attention scores before
        softmax.
    """
    if block_mode not in EXPLICIT_BLOCK_LATENT_MODES:
        raise ValueError(
            "build_explicit_block_latent_mask only supports block_mode in "
            f"{EXPLICIT_BLOCK_LATENT_MODES}; got block_mode={block_mode!r}"
        )
    S = int(context_len)
    K = int(block_size)
    T_time = int(time_len)
    if S <= 0 or K <= 0:
        raise ValueError(
            f"context_len and block_size must be positive; got context_len={S}, block_size={K}"
        )
    if T_time < 0:
        raise ValueError(f"time_len must be >= 0; got time_len={T_time}")

    T = T_time + S + S * K
    ctx_total = T_time + S

    allow = torch.zeros((T, T), dtype=torch.bool, device=device)

    # 1. context queries (time + c_*) are causal among themselves and cannot
    #    see pred latents. This matches the legacy "context never attends to
    #    pred" invariant.
    ctx_indices = torch.arange(ctx_total, device=device)
    causal_ctx = ctx_indices.unsqueeze(1) >= ctx_indices.unsqueeze(0)
    allow[:ctx_total, :ctx_total] = causal_ctx
    # allow[:ctx_total, ctx_total:] is left False.

    # 2. pred queries.
    # pred_t_idx: 0-indexed source position of each pred latent (length S*K).
    # pred_j_idx: 0-indexed block offset of each pred latent.
    pred_t_idx = (
        torch.arange(S, device=device).unsqueeze(1).expand(S, K).reshape(-1)
    )
    pred_j_idx = (
        torch.arange(K, device=device).unsqueeze(0).expand(S, K).reshape(-1)
    )

    # 2a. pred -> time: always allowed (only present when time_len > 0).
    if T_time > 0:
        allow[ctx_total:, :T_time] = True

    # 2b. pred -> context: z_{t,j} sees c_1..c_t ⇔ context_offset_idx <= t-1
    #     where t is 1-indexed (so 0-indexed source_pos == t-1 == pred_t_idx).
    ctx_offset = torch.arange(S, device=device)  # 0..S-1
    pred_to_ctx = ctx_offset.unsqueeze(0) <= pred_t_idx.unsqueeze(1)
    allow[ctx_total:, T_time:ctx_total] = pred_to_ctx

    # 2c. pred -> pred:
    same_t = pred_t_idx.unsqueeze(1) == pred_t_idx.unsqueeze(0)
    if block_mode == "future_latent_non_causal":
        # z_{t,j} sees only itself (same source position AND same block offset).
        same_j = pred_j_idx.unsqueeze(1) == pred_j_idx.unsqueeze(0)
        pred_to_pred = same_t & same_j
    elif block_mode == "future_latent_bidirectional":
        # z_{t,j} sees all z_{t,*}: same source position only, no j-causality.
        # Realises the "true joint MCMC plane": every pair of latents in the
        # same block can exchange information in both directions, so the MCMC
        # update on z_{t,1} sees ∂e_{t,*}/∂z_{t,1} from every offset, and the
        # MCMC update on z_{t,k} sees ∂e_{t,*}/∂z_{t,k} likewise. Cross
        # source-position visibility is still blocked (same_t).
        pred_to_pred = same_t
    else:  # block_mode == "blockwise"
        # z_{t,j} sees z_{t,1..j}: same source position AND key block offset
        # <= query block offset.
        causal_j = pred_j_idx.unsqueeze(0) <= pred_j_idx.unsqueeze(1)
        pred_to_pred = same_t & causal_j
    allow[ctx_total:, ctx_total:] = pred_to_pred

    mask = torch.zeros((T, T), dtype=dtype, device=device)
    mask = mask.masked_fill(~allow, float("-inf"))
    return mask


def build_explicit_block_latent_inference_mask(
    *,
    context_len: int,
    block_size: int,
    time_len: int,
    block_mode: str,
    device,
    dtype,
):
    """Inference-time additive mask for the explicit-block-latent modes.

    Use this when you only need to predict K future tokens at the *end* of
    a context of length C (i.e. a single anchor source position ``t = C``).
    The training mask
    (``build_explicit_block_latent_mask``) creates one block of K latents
    for each of the S source positions, which is unnecessary at inference
    and would multiply attention cost by O(S). For a well-trained model the
    training mask masks out cross source position attention anyway, so the
    output of the K latents at source position ``t = C`` is mathematically
    identical between the two layouts. This helper builds the smaller
    layout directly.

    Layout (token positions inside the returned ``(T, T)`` mask):
      ``[0, time_len)``                          → time tokens
      ``[time_len, time_len + C)``               → context ``c_1 .. c_C``
                                                    (C = ``context_len``)
      ``[time_len + C, time_len + C + K)``       → pred latents ``z_1..z_K``
                                                    (= ``z_{C, 1..K}`` in
                                                    training notation)

    Visibility rules (queries -> keys):
      * Time / context queries are causal among themselves and CANNOT see
        any pred latent (same invariant as training).
      * Pred query ``z_j`` sees:
            - all time tokens
            - ALL context tokens (since the anchor is the end of context).
      * Pred -> pred visibility:
            - ``future_latent_non_causal``: ``z_j`` sees only itself.
            - ``blockwise``: ``z_j`` sees ``z_1..z_j`` (causal in offset).

    Returns:
        Additive mask of shape ``(T, T)`` with values in ``{0, -inf}``.
    """
    if block_mode not in EXPLICIT_BLOCK_LATENT_MODES:
        raise ValueError(
            "build_explicit_block_latent_inference_mask only supports "
            f"block_mode in {EXPLICIT_BLOCK_LATENT_MODES}; got {block_mode!r}"
        )
    C = int(context_len)
    K = int(block_size)
    T_time = int(time_len)
    if C <= 0 or K <= 0:
        raise ValueError(
            f"context_len and block_size must be positive; got C={C}, K={K}"
        )
    if T_time < 0:
        raise ValueError(f"time_len must be >= 0; got time_len={T_time}")

    T = T_time + C + K
    ctx_total = T_time + C

    allow = torch.zeros((T, T), dtype=torch.bool, device=device)

    ctx_indices = torch.arange(ctx_total, device=device)
    causal_ctx = ctx_indices.unsqueeze(1) >= ctx_indices.unsqueeze(0)
    allow[:ctx_total, :ctx_total] = causal_ctx

    if T_time > 0:
        allow[ctx_total:, :T_time] = True
    # All context visible to every pred latent (anchor is "end of context").
    allow[ctx_total:, T_time:ctx_total] = True

    block_idx = torch.arange(K, device=device)
    if block_mode == "future_latent_non_causal":
        # Only self.
        pred_to_pred = block_idx.unsqueeze(1) == block_idx.unsqueeze(0)
    elif block_mode == "future_latent_bidirectional":
        # Full intra-block attention: every latent sees every other latent in
        # the same anchor block (mirrors the training mask's same_t-only rule
        # with S=1 at inference).
        pred_to_pred = torch.ones((K, K), dtype=torch.bool, device=device)
    else:  # blockwise
        # Causal within the block.
        pred_to_pred = block_idx.unsqueeze(0) <= block_idx.unsqueeze(1)
    allow[ctx_total:, ctx_total:] = pred_to_pred

    mask = torch.zeros((T, T), dtype=dtype, device=device)
    mask = mask.masked_fill(~allow, float("-inf"))
    return mask


def build_explicit_block_latent_freq_indices(
    *,
    context_len: int,
    block_size: int,
    time_len: int,
    start_pos: int = 0,
):
    """Rotary position indices for the explicit-block-latent layout.

    Used by ``ar_ebt_time_embed`` (``time_len=1``) and ``ar_ebt_default``
    (``time_len=0``) to gather the appropriate rows from the precomputed
    ``freqs_cis`` table. Returns a Python ``list[int]`` so the caller can
    pass it to ``torch.tensor`` or directly index into ``freqs_cis``.

    Rotary positions follow the natural "predicted token would live here"
    convention so the new modes inherit standard RoPE relative-position
    behavior:

      * ``time_embed`` trunk (``time_len = 1``):
          - time      → position ``start_pos + 0``
          - ``c_t``   → position ``start_pos + t``       (t in [1..S])
          - ``z_{t,j}`` → position ``start_pos + t + j`` (t in [1..S],
                                                          j in [1..K])
        Maximum position used: ``start_pos + S + K``.

      * ``default`` trunk (``time_len = 0``):
          - ``c_t``   → position ``start_pos + (t - 1)``
          - ``z_{t,j}`` → position ``start_pos + (t - 1) + j``
        Maximum position used: ``start_pos + S + K - 1``.

    These match the semantics used by the legacy paths (time at 0; context at
    1..S for time_embed and 0..S-1 for default; pred latents shifted forward
    by their offset).
    """
    S = int(context_len)
    K = int(block_size)
    T_time = int(time_len)
    if S <= 0 or K <= 0:
        raise ValueError(
            f"context_len and block_size must be positive; got context_len={S}, block_size={K}"
        )
    if T_time < 0:
        raise ValueError(f"time_len must be >= 0; got time_len={T_time}")

    sp = int(start_pos)
    indices = []
    if T_time > 0:
        # All time tokens map to rotary position 0 (relative).
        indices.extend([sp] * T_time)
        # Context c_t at position t (1-indexed t).
        indices.extend(sp + t for t in range(1, S + 1))
        # Pred z_{t,j} at position t + j (position-major: t outer, j inner).
        indices.extend(
            sp + t + j for t in range(1, S + 1) for j in range(1, K + 1)
        )
    else:
        # Context c_t at position t-1 (matches legacy default trunk).
        indices.extend(sp + (t - 1) for t in range(1, S + 1))
        # Pred z_{t,j} at position (t-1) + j.
        indices.extend(
            sp + (t - 1) + j
            for t in range(1, S + 1)
            for j in range(1, K + 1)
        )
    return indices


def build_explicit_block_latent_inference_freq_indices(
    *,
    context_len: int,
    block_size: int,
    time_len: int,
    start_pos: int = 0,
):
    """Rotary position indices for the inference layout.

    Inference layout has C context tokens followed by K pred latents anchored
    at the end of context, so positions are:

      * ``time_embed`` trunk (``time_len = 1``):
          - time      → ``start_pos + 0``
          - ``c_t``   → ``start_pos + t``       (t in [1..C])
          - ``z_j``   → ``start_pos + C + j``   (j in [1..K]); equivalently
                        ``z_j`` is treated as ``z_{C, j}`` with rotary
                        position ``C + j``.
      * ``default`` trunk (``time_len = 0``):
          - ``c_t``   → ``start_pos + (t - 1)``
          - ``z_j``   → ``start_pos + (C - 1) + j``

    Maximum position used: ``start_pos + C + K`` (or ``... + K - 1`` for
    default).
    """
    C = int(context_len)
    K = int(block_size)
    T_time = int(time_len)
    if C <= 0 or K <= 0:
        raise ValueError(
            f"context_len and block_size must be positive; got C={C}, K={K}"
        )
    if T_time < 0:
        raise ValueError(f"time_len must be >= 0; got time_len={T_time}")

    sp = int(start_pos)
    indices = []
    if T_time > 0:
        indices.extend([sp] * T_time)
        indices.extend(sp + t for t in range(1, C + 1))
        indices.extend(sp + C + j for j in range(1, K + 1))
    else:
        indices.extend(sp + (t - 1) for t in range(1, C + 1))
        indices.extend(sp + (C - 1) + j for j in range(1, K + 1))
    return indices


@dataclass
class EBTModelArgs:
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    dyt_alpha_init: float = 0.5
    max_batch_size: int = 64
    max_seq_len: int = 16
    weight_initialization: str = "xavier"
    adaln_zero_init: bool = True
    ebt_norm: str = "rms"
    ebt_act_func: str = "silu"
    weight_initialization_gain: float = 1.0
    debug_blockwise_shapes: bool = False
    # Explicit attention / training semantic. Replaces the previous implicit
    # "legacy_symmetric = (pred_len == context_len)" branching. See
    # nova/ebt/ar_ebt_time_embed.py and ar_ebt_default.py for the exact
    # mapping from block_mode -> attention implementation.
    block_mode: str = "dense_token"

# nanochat depth-based auto-scaling
# base_dim = depth * aspect_ratio # aspect_ratio = 64
# head_dim = 128
# embedding_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
# num_heads = model_dim // head_dim

NANOCHAT_HEAD_DIM = 128

def nanochat_depth_scaling(depth, aspect_ratio=64):
    base_dim = depth * aspect_ratio
    head_dim = NANOCHAT_HEAD_DIM
    embedding_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = embedding_dim // head_dim
    return {
        "num_transformer_blocks": depth,
        "embedding_dim": embedding_dim, 
        "multiheaded_attention_heads": num_heads,
    }


model_sizes = { # small -> xl same as mamba https://arxiv.org/pdf/2312.00752; all others estimated empirically. LRs based off mamba where applicable

    "4xs": { # LR 0.0024 recommended
        "num_transformer_blocks": 2,
        "multiheaded_attention_heads": 2,
        "embedding_dim": 128,
    },
    "3xs": { # LR 0.0018
        "num_transformer_blocks": 4,
        "multiheaded_attention_heads": 4,
        "embedding_dim": 256,
    },
    "xxs": { # LR 0.0012
        "num_transformer_blocks": 6,
        "multiheaded_attention_heads": 6,
        "embedding_dim": 384,
    },
    "2xs": { # LR 0.0012
        "num_transformer_blocks": 6,
        "multiheaded_attention_heads": 6,
        "embedding_dim": 384,
    },
    "xs": { # LR 0.0009
        "num_transformer_blocks": 12,
        "multiheaded_attention_heads": 6,
        "embedding_dim": 384,
    },
    "small": { # LR 0.0006
        "num_transformer_blocks": 12,
        "multiheaded_attention_heads": 12,
        "embedding_dim": 768,
    },
    "medium": { # 0.0003
        "num_transformer_blocks": 24,
        "multiheaded_attention_heads": 16,
        "embedding_dim": 1024,
    },
    "large": { # 0.00025
        "num_transformer_blocks": 24,
        "multiheaded_attention_heads": 16,
        "embedding_dim": 1536,
    },
    "xl": { # 0.0002
        "num_transformer_blocks": 24,
        "multiheaded_attention_heads": 32,
        "embedding_dim": 2048,
    },

    "d26": nanochat_depth_scaling(26),
}




class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        out = self.linear(x)
        out = self.relu(out)
        out = self.dropout(out)
        return x + out  # Add the residual connection

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, final_size, dropout_rate, layer_norm, num_hidden_layers=1):
        super(MLP, self).__init__()
        self.add_residual_connections = True  # Residual connections are always on by default
        self.layers = nn.ModuleList()

        # Initial layer
        self.layers.append(nn.Linear(input_size, hidden_size, bias=False))
        if layer_norm:
            self.layers.append(nn.LayerNorm(hidden_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for i in range(1, num_hidden_layers - 1):
            add_residual = self.add_residual_connections and i % 2 == 0

            if add_residual:
                self.layers.append(ResidualBlock(hidden_size, dropout_rate))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
                self.layers.append(nn.ReLU())

            self.layers.append(nn.Dropout(dropout_rate))

        # Last layer
        if final_size == hidden_size and self.add_residual_connections and (num_hidden_layers - 1) % 2 == 0:
            self.layers.append(ResidualBlock(hidden_size, dropout_rate))
        else:
            self.layers.append(nn.Linear(hidden_size, final_size, bias=False))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class Memory_Augmented_MLP(nn.Module):
    def __init__(self, input_modality_size, memory_size, input_modality_hidden_size, final_size, dropout_rate, layer_norm, num_hidden_layers=1):
        super(Memory_Augmented_MLP, self).__init__()
        self.add_residual_connections = True  # Residual connections are always on by default
        self.input_layers = nn.ModuleList()
        self.memory_size = memory_size

        # Initial layer
        self.input_layers.append(nn.Linear(input_modality_size, input_modality_hidden_size, bias=False))
        if layer_norm:
            self.input_layers.append(nn.LayerNorm(input_modality_hidden_size))
        self.input_layers.append(nn.ReLU())
        self.input_layers.append(nn.Dropout(dropout_rate))

        self.memory = nn.Embedding(1, memory_size)

        self.fusion_layers = nn.ModuleList()

        self.fusion_hidden_size = input_modality_hidden_size + memory_size

        # Hidden layers
        for i in range(1, num_hidden_layers - 1):
            add_residual = self.add_residual_connections and i % 2 == 0

            if add_residual:
                self.fusion_layers.append(ResidualBlock(self.fusion_hidden_size, dropout_rate))
            else:
                self.fusion_layers.append(nn.Linear(self.fusion_hidden_size, self.fusion_hidden_size, bias=False))
                self.fusion_layers.append(nn.ReLU())

            self.fusion_layers.append(nn.Dropout(dropout_rate))

        # Last layer
        if final_size == self.fusion_hidden_size and self.add_residual_connections and (num_hidden_layers - 1) % 2 == 0:
            self.fusion_layers.append(ResidualBlock(self.fusion_hidden_size, dropout_rate))
        else:
            self.fusion_layers.append(nn.Linear(self.fusion_hidden_size, final_size, bias=False))

    def forward(self, x):
        batch_size, sequence_length = x.shape[0], x.shape[1]
        for layer in self.input_layers:
            x = layer(x)
        
        memory_embedding = self.memory.weight[0]
        memory_embedding = memory_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, sequence_length, self.memory_size)
        fused_x = torch.cat((x, memory_embedding), dim=-1) # B, S, H + M (where H is hidden size M is memory size)
        
        for layer in self.fusion_layers:
            fused_x = layer(fused_x)
        
        return fused_x
    
class Memory_Gating_MLP(nn.Module):
    def __init__(self, input_modality_size, embedding_dim, process_memory_type = "add", process_memory_linear_layer = False):
        super(Memory_Gating_MLP, self).__init__()
        self.vocab_to_embed = nn.Linear(input_modality_size, embedding_dim, bias = False)
        self.memory = nn.Embedding(1, embedding_dim)
        self.process_memory_linear_layer = process_memory_linear_layer
        if self.process_memory_linear_layer:
            self.memory_linear_layer = nn.Linear(embedding_dim, embedding_dim, bias = False)
        self.process_memory_type = process_memory_type

    def forward(self, x):
        vocab_embeds = self.vocab_to_embed(x)
        memory_embedding = self.memory.weight[0]
        memory_embedding = memory_embedding.unsqueeze(0).unsqueeze(0).expand_as(vocab_embeds)
        if self.process_memory_linear_layer:
            memory_embedding = self.memory_linear_layer(memory_embedding)
        
        if self.process_memory_type == "add":
            final_embeddings = vocab_embeds + memory_embedding
        elif self.process_memory_type == "gate":
            final_embeddings = vocab_embeds * memory_embedding
        elif self.process_memory_type == "residual_gate":
            final_embeddings = vocab_embeds + vocab_embeds * memory_embedding
        else:
            raise NotImplementedError(f"self.process_memory_type {self.process_memory_type} not yet implemented")

        return final_embeddings


def calc_out_of_bounds_loss(energy): # gives loss for < 0 or > 1
    lower_bound_loss = torch.abs(energy)
    upper_bound_loss = torch.abs(energy - 1)
    loss = torch.where(energy < 0, lower_bound_loss, 
                    torch.where(energy > 1, upper_bound_loss, torch.zeros_like(energy)))
    loss = torch.mean(loss)
    
    return loss

# def log_pred_futures(futures, device, dataset_name, i, denormalize):
#     denormalized_futures = denormalize(futures.clone(), dataset_name, device = device)

#     to_pil = ToPILImage()
#     for b in range(denormalized_futures.shape[0]):  # Loop over the batch size
#         if b % 16 == 0:
#             for s in range(denormalized_futures.shape[1]):  # Loop over the sequence length
#                 frame_to_save = to_pil(denormalized_futures[b, s].cpu())  # Extract a frame (C x W x H)
                
#                 # Save the image
#                 current_time = datetime.now().strftime("%H_%M_%S")
#                 frame_to_save.save(f"./logs/debug/mcmc_futures/{current_time}_batch_{b}_seq_{s}_dev_{device}_iter_{i}.png")

def denormalize(tensor, dataset_name, device, custom_normalization, vae_normalization=False):
    tensor = tensor.clone().detach()

    # Define default normalization values
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]
    default_mean = torch.tensor(default_mean, device=device).view(1, 1, 3, 1, 1)
    default_std = torch.tensor(default_std, device=device).view(1, 1, 3, 1, 1)
    # Dataset-specific normalization lookup
    if custom_normalization:
        normal_lookup = {
            "ucf101": ([1.04731617, 1.04372056, 1.02795228], [-0.40689788, -0.36098219, -0.25687788]),
            "k400": ([1.00370078, 0.99871626, 0.97407404], [-0.24295556, -0.24931058, -0.13959686]),
            "smth": ([0.90832217, 0.93885971, 0.93745849], [-0.06761328, -0.12692231, -0.01916805]),
            "ImageNet": ([1, 1, 1], [0, 0, 0]),
            "something": ([0.90832217, 0.93885971, 0.93745849], [-0.06761328, -0.12692231, -0.01916805]),
            "ImageNet1k": ([1, 1, 1], [0, 0, 0])
        }
        dataset_std, dataset_mean = normal_lookup.get(dataset_name, ([1, 1, 1], [0, 0, 0]))

        # Convert means and stds to tensors and reshape for broadcast compatibility
        dataset_mean = torch.tensor(dataset_mean, device=device).view(1, 1, 3, 1, 1)
        dataset_std = torch.tensor(dataset_std, device=device).view(1, 1, 3, 1, 1)
        

        # Perform denormalization
        # First reverse the dataset-specific normalization
        tensor = tensor * dataset_std + dataset_mean
    
    # Then reverse the default normalization
    if vae_normalization:
        default_mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 1, 3, 1, 1) 
        default_std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 1, 3, 1, 1)
        tensor = tensor * default_std + default_mean
        return tensor
    else:
        return tensor * default_std + default_mean

def scale_clamp(tensor, min_value, max_value):
    scale_factor = torch.ones_like(tensor)
    scale_factor = torch.where(tensor > max_value, tensor / max_value, scale_factor)
    scale_factor = torch.where(tensor < min_value, tensor / min_value, scale_factor)
    
    scaled_tensor = tensor / scale_factor
    return scaled_tensor

# def load_trained_pl_model(ckpt_path, new_hparams, for_inference = False):
#     from base_model_trainer import ModelTrainer
#     checkpoint = torch.load(ckpt_path, weights_only=False)
#     model = ModelTrainer(new_hparams)
#     model.load_state_dict(checkpoint['state_dict'])
#     if for_inference:
#         model.cuda().eval()
#         model.model.eval()
#     return model.model

def print_model_layers_and_status(model):
    for name, module in model.named_modules():
        print(f'Layer: {name}, Type: {type(module).__name__}, Training Mode: {module.training}')

def init_whole_model_weights(model, weight_initialization_method, nonlinearity='linear', weight_initialization_gain=1.0):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            if weight_initialization_method == "he":
                valid_nonlinearities = ['linear', 'relu', 'leaky_relu', 'selu', 'tanh']
                if nonlinearity not in valid_nonlinearities:
                    raise ValueError(f"Unsupported nonlinearity: {nonlinearity}. Must be one of {valid_nonlinearities}")
                
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
                if weight_initialization_gain != 1.0:
                    m.weight.data *= weight_initialization_gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif weight_initialization_method == "xavier":
                nn.init.xavier_normal_(m.weight)
                if weight_initialization_gain != 1.0:
                    m.weight.data *= weight_initialization_gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            else:
                raise ValueError(f"Unknown weight init method: {weight_initialization_method}")
    
    model.apply(init_weights)


# def load_image_encoder(backbone_type, backbone_size, device=None, use_ema = False):
#     vit_backbone_archs = {
#         "small": "vits14",
#         "base": "vitb14",
#         "large": "vitl14",
#         "giant": "vitg14",
#     }
        
#     if backbone_type == 'dinov2':
#         backbone_name = vit_backbone_archs[backbone_size]
#         model = torch.hub.load('facebookresearch/dinov2', model=f"dinov2_{backbone_name}")
#         del model._parameters['mask_token'] # this is done as this param was unused and was causing pl ddp unused param issues
#     elif backbone_type == "vae": # all have same encoder just different decoder
#         if use_ema:
#             model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
#         else:
#             model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
#     else:
#         raise NotImplementedError(f"Unspported backbone type: {backbone_type}")
#     if device is not None:
#         model = model.to(device)
#     return model
    
# def get_encoded_images(batch, backbone_type, image_encoder, sdxl_vae_standardization = False):
#     with torch.no_grad():
#         if backbone_type == 'dinov2':
#             return image_encoder(batch)
#         elif backbone_type == "vae":
#             if not sdxl_vae_standardization:
#                 return image_encoder.encode(batch).latent_dist.mean
#             elif sdxl_vae_standardization:
#                 return image_encoder.encode(batch).latent_dist.mean.mul_(0.18215) # constant following SDXL VAE https://github.com/CompVis/stable-diffusion

    
def hinged_mse_loss(predictions, targets, margin=0.1):
    """
    Compute the Hinged MSE loss between predictions and targets.
    :param predictions: Predicted values.
    :param targets: Ground truth values.
    :param margin: The threshold below which errors are ignored.
    :return: Hinged MSE loss.
    """
    errors = torch.abs(predictions - targets)
    hinged_errors = torch.where(errors > margin, errors, torch.zeros_like(errors))
    loss = torch.mean(hinged_errors ** 2)
    return loss

def find_subsequences(input_tensor, sub_seq):
    sub_seq_len = len(sub_seq)
    batch_size, seq_len = input_tensor.shape
    sub_seq_tensor = torch.tensor(sub_seq, device=input_tensor.device)
    sub_seq_tensor = sub_seq_tensor.view(1, -1)
    windows = input_tensor.unfold(1, sub_seq_len, 1)
    matches = (windows == sub_seq_tensor).all(dim=2).long()
    
    if not matches.any(dim=1).all():
        raise ValueError("Sub-sequence not found in one or more sequences.")
    
    start_positions = matches.argmax(dim=1)
    return start_positions

def mask_q_tokens(input_tensor, tokenizer):
    '''
    input_tensor = [batch size, seq len]
    '''
    batch_size = input_tensor.shape[0]
    seq_length = input_tensor.shape[1]
    answer_tag = tokenizer.encode("[[Answer]]:", add_special_tokens=True)
    
    answer_start_pos = find_subsequences(input_tensor, answer_tag)
    answer_start_pos += len(answer_tag)
    mask = torch.arange(seq_length, device=input_tensor.device).expand(batch_size, seq_length)
    mask = mask < answer_start_pos.unsqueeze(1)
    input_tensor = torch.where(mask, tokenizer.pad_token_id, input_tensor)
    
    return input_tensor

def analyse_tokens(input_tensor, tokenizer):
    '''for debugging only'''
    decode = tokenizer.batch_decode(input_tensor, skip_special_tokens=True)
    for i in range(input_tensor.shape[0]):
        print(input_tensor[i].tolist())
        print(decode[i])
        print('-'*60)

def setup_ebt(hparams): # specifically for EBT not for baseline transformer
    # to prevent circular import

    max_seq_len = hparams.context_length+1 # for next pred in context 
    max_seq_len = max_seq_len + 1 if hparams.ebt_type == "time_embed" else max_seq_len # need +1 since cat time embed on sequence dim

    adaln_zero_init = True if hparams.ebt_type == "adaln_zero" else False
    block_mode = getattr(hparams, "block_mode", "dense_token")
    if block_mode not in BLOCK_MODE_CHOICES:
        raise ValueError(
            f"Unknown block_mode={block_mode!r}; must be one of {BLOCK_MODE_CHOICES}"
        )
    transformer_args = EBTModelArgs(dim = hparams.embedding_dim, n_layers = hparams.num_transformer_blocks, n_heads = hparams.multiheaded_attention_heads, max_batch_size = hparams.batch_size_per_device, max_seq_len=max_seq_len, weight_initialization = hparams.weight_initialization_method, adaln_zero_init=adaln_zero_init, ebt_norm=hparams.ebt_norm, ffn_dim_multiplier=hparams.ffn_dim_multiplier, ebt_act_func=hparams.ebt_act_func, weight_initialization_gain=hparams.weight_initialization_gain, dyt_alpha_init=hparams.dyt_alpha_init, debug_blockwise_shapes=getattr(hparams, "debug_blockwise_shapes", False), block_mode=block_mode)
    
    if hparams.ebt_type == "default": # causal decoder trans for ebm https://arxiv.org/abs/2406.08862
        from ar_ebt_default import EBTDefault
        ebt = EBTDefault(params=transformer_args)
    elif hparams.ebt_type == "time_embed": # time embed
        from ar_ebt_time_embed import EBTTimeConcat
        ebt = EBTTimeConcat(params=transformer_args, max_mcmc_steps = hparams.mcmc_num_steps)
    else: # adaln or adaln_zero
        from ar_ebt_adaln import EBTAdaLN
        ebt = EBTAdaLN(params=transformer_args, max_mcmc_steps = hparams.mcmc_num_steps)

    return ebt

def setup_transformer(hparams): # specifically for baseline transformer
    from ar_transformer import Transformer, TransformerModelArgs
    transformer_args = TransformerModelArgs(dim = hparams.embedding_dim, n_layers = hparams.num_transformer_blocks, n_heads = hparams.multiheaded_attention_heads, max_batch_size = hparams.batch_size_per_device, max_seq_len=hparams.context_length, weight_initialization = hparams.weight_initialization_method, ffn_dim_multiplier=hparams.ffn_dim_multiplier, weight_initialization_gain=hparams.weight_initialization_gain)
    transformer = Transformer(params=transformer_args)
    return transformer

def has_layer_norm(model):
    return any(isinstance(module, nn.LayerNorm) for _, module in model.named_modules())

def init_wandb_watch(wandb_logger, model_trainer, wandb_watch_log_freq, wandb_watch_level="parameters"):
    """
    wandb_watch_level controls what is logged:
      - "parameters": only log parameter histograms (low overhead)
      - "gradients": only log gradient histograms
      - "all": log parameters + gradients (high overhead, debug only)
    """
    log_mode = wandb_watch_level  # "parameters", "gradients", or "all"

    if not has_layer_norm(model_trainer.model):
        wandb_logger.watch(model_trainer.model, log=log_mode, log_freq=wandb_watch_log_freq)

    else: # all of complex below code is to get around the issue where wandb watch with layer norm has 'AttributeError: 'NoneType' object has no attribute 'data'' when logging gradients...
        non_layernorm_container = nn.Module()
        layernorm_container = nn.Module()

        non_ln_modules = {}
        ln_modules = {}

        for name, module in model_trainer.model.named_modules():
            if name == "": # skips top level model
                continue
            safe_name = name.replace(".", "_") # model cant contain '.' in name

            if isinstance(module, nn.LayerNorm):
                ln_modules[safe_name] = module
            else:
                # Only add modules that don't contain LayerNorm as submodules
                has_ln_child = any(isinstance(child, nn.LayerNorm)
                                for child in module.modules())
                if not has_ln_child:
                    non_ln_modules[safe_name] = module

        for name, module in non_ln_modules.items():
            non_layernorm_container.add_module(name, module)

        for name, module in ln_modules.items():
            layernorm_container.add_module(name, module)

        wandb_logger.watch(non_layernorm_container, log=log_mode, log_freq=wandb_watch_log_freq)
        # LayerNorm always uses "parameters" only to avoid gradient logging bug
        ln_log_mode = "parameters" if log_mode in ("gradients", "all") else log_mode
        wandb_logger.watch(layernorm_container, log=ln_log_mode, log_freq=wandb_watch_log_freq)

# def save_frames(tensor, root_dir, subfolder, start_index=0):
#     to_pil = ToPILImage()
#     subfolder_path = os.path.join(root_dir, subfolder)
#     os.makedirs(subfolder_path, exist_ok=True)

#     all_frames = [] if subfolder == 'debug' else None

#     for i, video in enumerate(tensor):
#         if subfolder != 'debug':
#             video_dir = os.path.join(subfolder_path, f'video_{start_index + i}')
#             os.makedirs(video_dir, exist_ok=True)

#         for j, frame in enumerate(video):
#             frame_img = to_pil(frame.cpu().detach())

#             if subfolder != 'debug':
#                 frame_img.save(os.path.join(video_dir, f'frame_{j}.png'))
#             else:
#                 all_frames.append(frame_img)

#     if subfolder == 'debug':
#         frame_tensors = torch.stack([torchvision.transforms.functional.to_tensor(img) for img in all_frames])
#         grid = make_grid(frame_tensors, nrow=int(len(frame_tensors)**0.5))
#         save_image(grid, os.path.join(subfolder_path, f'{start_index}.png'))

def call_style_gan_fvd(args):
    # this uses style-gan-V code: https://github.com/universome/stylegan-v/tree/master?tab=readme-ov-file
    
    import subprocess
    import os
    from pathlib import Path

    assert args.context_length == 16, "StyleGAN-V metrics currently only support context length of 16 frames. Other lengths not implemented here (see repo for how to add more: https://github.com/universome/stylegan-v/tree/master?tab=readme-ov-file)."

    stylegan_path = Path("../stylegan-v")
    assert stylegan_path.exists(), "StyleGAN-V repository must exist at ../stylegan-v"
    
    metrics_script = stylegan_path / "src/scripts/calc_metrics_for_dataset.py"
    assert metrics_script.exists(), f"Metrics script not found at {metrics_script}"

    env_path = stylegan_path / "env"
    assert env_path.exists(), f"Conda environment not found at {env_path}. Please create it (e.g. by going inside the stylegan-v repo and running `conda env create -f environment.yaml -p env` if you are using a non-Ampere GPU, or `conda env create -f environment-ampere.yaml -p env` if you are using an Ampere or newer GPU (see inference/vid/README.md))."

    real_path = os.path.abspath(os.path.join(args.save_generation_logs_dir, "real"))
    fake_path = os.path.abspath(os.path.join(args.save_generation_logs_dir, "fake"))
    
    # Run the metrics script in a different conda env, e.g. "styleganv_env"
    cmd = [
        "conda", "run", "--prefix", str(env_path),
        "python",
        str(metrics_script),
        "--real_data_path", str(real_path),
        "--fake_data_path", str(fake_path),
        "--mirror", "0", #NOTE can set to 1 if want horizontal flip
        "--gpus", str(args.num_gpus),
        "--resolution", str(args.image_dims[0]),
        "--metrics", "fvd2048_16f,fid50k_full",
        "--verbose", "1",
        "--use_cache", "0"
    ]
    print("cmd being used for FVD and FID calculations", cmd)

    try:
        result = subprocess.run(cmd, cwd=str(stylegan_path), capture_output=True, text=True, check=True)
        fvd = float('inf')
        fid = float('inf')
        for line in result.stdout.split('\n'):
            if "fvd2048_16f" in line:
                try:
                    fvd = float(line.split(':')[1].strip())
                except:
                    print("Failed to parse FVD value from output")
            elif "fid50k_full" in line:
                try:
                    fid = float(line.split(':')[1].strip())
                except:
                    print("Failed to parse FID value from output")
    except subprocess.CalledProcessError as e:
        print("CalledProcessError:", e)
        print("Command:", cmd)
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        print(traceback.format_exc())
        fvd, fid = float('inf'), float('inf')
    except Exception as e:
        print("Unexpected error:", e)
        print("Command:", cmd)
        print(traceback.format_exc())
        fvd, fid = float('inf'), float('inf')

    return fvd, fid

# def center_crop_arr(pil_image, image_size):
#     """
#     Center cropping implementation from ADM.
#     https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
#     """
#     while min(*pil_image.size) >= 2 * image_size:
#         pil_image = pil_image.resize(
#             tuple(x // 2 for x in pil_image.size), resample=Image.BOX
#         )

#     scale = image_size / min(*pil_image.size)
#     pil_image = pil_image.resize(
#         tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
#     )

#     arr = np.array(pil_image)
#     crop_y = (arr.shape[0] - image_size) // 2
#     crop_x = (arr.shape[1] - image_size) // 2
#     return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

# def setup_diffusion_transformer(hparams):
#     from model.diffusion_transformer import DiT
#     assert hparams.image_dims[0] == hparams.image_dims[1], "need to use square image with current implementation"
    
#     if hparams.image_task == "denoising":
#         # For denoising task, use raw image dimensions (no VAE)
#         input_size = hparams.image_dims[0]
#         in_channels = 3  # RGB channels for raw images
#     else:
#         # For other tasks using VAE
#         assert hparams.image_dims[0] % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
#         input_size = hparams.image_dims[0] // 8
#         in_channels = 4

#     dit = DiT(input_size=input_size, patch_size=hparams.patch_size, in_channels=in_channels, hidden_size=hparams.embedding_dim, depth=hparams.num_transformer_blocks, num_heads=hparams.multiheaded_attention_heads, mlp_ratio=hparams.ffn_dim_multiplier)
#     return dit

# def setup_bidirectional_ebt(hparams):
#     from model.bi_ebt_adaln import EBT
#     assert hparams.image_dims[0] == hparams.image_dims[1], "need to use square image with current implementation"
    
#     if hparams.image_task == "denoising":
#         # For denoising task, use raw image dimensions (no VAE)
#         input_size = hparams.image_dims[0]
#         in_channels = 3  # RGB channels for raw images
#     else:
#         # For other tasks using VAE
#         assert hparams.image_dims[0] % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
#         input_size = hparams.image_dims[0] // 8
#         in_channels = 4
    
#     ebt = EBT(
#         input_size=input_size,
#         patch_size=hparams.patch_size,
#         in_channels=in_channels,
#         hidden_size=hparams.embedding_dim,
#         depth=hparams.num_transformer_blocks,
#         num_heads=hparams.multiheaded_attention_heads,
#         mlp_ratio=hparams.ffn_dim_multiplier
#     )
    
#     return ebt

# def get_clip_text_encoder(size):
#     from transformers import CLIPTextModel, CLIPTokenizer
    
#     model_name_mapping = {
#         "small": "openai/clip-vit-base-patch32",
#         "base": "openai/clip-vit-base-patch16",
#         "large": "openai/clip-vit-large-patch14",
#         "xl": "openai/clip-vit-large-patch14-336",
#     }
    
#     if size not in model_name_mapping:
#         raise ValueError(f"Invalid size: {size}. Available sizes: {list(model_name_mapping.keys())}")
    
#     model_name = model_name_mapping[size]
#     text_encoder = CLIPTextModel.from_pretrained(model_name)
#     clip_hidden_size = text_encoder.config.hidden_size
    
#     return text_encoder, clip_hidden_size

def get_text_embeddings(text_encoder, captions):
    with torch.no_grad():
        batch_outputs = text_encoder(**captions)
        caption_embeddings = batch_outputs.pooler_output
        return caption_embeddings

# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

# note this is unused for text conditional instead of class conditional https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
    
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



# all pos enc functions from below are from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py, from DiT codebase
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb