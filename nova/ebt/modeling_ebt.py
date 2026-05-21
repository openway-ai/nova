import torch
from torch import nn
from torch.nn import functional as F
from nanolightning.torchlightning_module import LightningModule
# import torch.optim as optim
# from torchmetrics import Accuracy
# from transformers import AutoTokenizer

import math
import random
import os
import inspect
from utils import setup_ebt, init_whole_model_weights
from utils import MLP, Memory_Augmented_MLP, Memory_Gating_MLP, mask_q_tokens
from utils import EXPLICIT_BLOCK_LATENT_MODES
from replay_buffer import CausalReplayBuffer
from metrics import calculate_bpb_score
# RMSNorm is defined in the AR EBT trunk modules; reuse it for the new
# per-offset blockwise heads so they match trunk normalization semantics.
from ar_ebt_time_embed import RMSNorm as _RMSNorm

try:
    import ipdb  # type: ignore
except ImportError:
    ipdb = None


# ============================================================================
# E2: Per-offset blockwise head modules (used by explicit-block-latent modes
# `future_latent_non_causal` / `blockwise`). The head receives a single
# offset's pred_hidden `[B, S, D]` and returns logits `[B, S, V]`. For
# inference, S is just 1 and the same modules handle that degenerate case.
#
# Selection is via the `--block_latent_head_type` CLI flag, which is read by
# `_build_block_latent_head()` below. Default `linear` preserves the original
# shared single Linear(D, V) head (byte-identical to pre-E2 behavior).
# ============================================================================

# Public registry of supported head types. Importable for tests / CLI choices.
BLOCK_LATENT_HEAD_TYPES = (
    "linear",                       # E2 baseline: shared single Linear(D, V)
    "per_offset_linear",            # E2.1: K independent Linear(D, V)
    "per_offset_mlp",               # E2.2: K independent Linear-GELU-Linear
    "per_offset_transformer",       # E2.3: K independent (num_layers x causal AR block) + Linear
    "per_offset_tf_linear",         # E4 (Gemma-style TF): teacher-forced compressed Linear
    "per_offset_tf_transformer",    # E4 (Gemma-style TF): teacher-forced transformer head
)

# Head types that require a "previous-offset realized token embedding" as a
# second forward argument: head_j(pred_hidden, prev_token_embed). These are
# the Gemma-style teacher-forced heads — each offset's prediction is
# conditioned on the embedding of the previous offset's token (ground truth
# during training, sampled/argmax during inference).
TF_HEAD_TYPES = frozenset({"per_offset_tf_linear", "per_offset_tf_transformer"})


class _BlockwiseLinearHead(nn.Module):
    """Per-offset linear readout: [B, S, D] -> [B, S, V]."""

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        return self.proj(x)


class _BlockwiseMLPHead(nn.Module):
    """Per-offset 2-layer MLP readout: RMSNorm -> Linear -> GELU -> Linear.

    Used by E2.2. Cheaper than a transformer block; tests whether non-linear
    readout alone is enough.
    """

    def __init__(self, dim: int, vocab_size: int, ffn_mult: float):
        super().__init__()
        hidden = int(dim * ffn_mult)
        hidden = ((hidden + 63) // 64) * 64  # round up to multiple of 64 (matches llama FFN)
        self.norm = _RMSNorm(dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(self.norm(x))))


class _BlockwiseHeadBlock(nn.Module):
    """Minimal causal AR transformer block used inside per-offset transformer heads.

    Pre-norm + causal SDPA + SwiGLU FFN. Independent of EBT trunk's
    ``block_mode`` / MCMC semantics — operates as a plain causal AR block
    along the S (source-position) dim. At inference S=1, the attention
    collapses to a per-token identity (still a valid forward pass).
    """

    def __init__(self, dim: int, n_heads: int, ffn_mult: float, norm_eps: float = 1e-5):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        # Attention
        self.attn_norm = _RMSNorm(dim, eps=norm_eps)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        # FFN (SwiGLU, llama-style)
        ffn_dim = int(dim * ffn_mult)
        ffn_dim = ((ffn_dim + 63) // 64) * 64
        self.ffn_norm = _RMSNorm(dim, eps=norm_eps)
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)

    def forward(self, x):
        # x: [B, S, D]
        B, S, D = x.shape
        # Pre-norm causal self-attention
        h = self.attn_norm(x)
        q = self.wq(h).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        if S > 1:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Single token: self-attention is identity on v
            attn = v
        attn = attn.transpose(1, 2).reshape(B, S, D)
        x = x + self.wo(attn)
        # Pre-norm SwiGLU FFN
        h = self.ffn_norm(x)
        x = x + self.w2(F.silu(self.w1(h)) * self.w3(h))
        return x


class _BlockwiseTransformerHead(nn.Module):
    """Per-offset stack of causal AR transformer blocks + Linear readout.

    DeepSeek-MTP-style (E2.3): each offset's pred_hidden `[B, S, D]` is
    refined by its own small causal AR transformer (``num_layers`` blocks),
    then projected to `[B, S, V]` via a per-offset Linear. The trunk's
    pred_hidden already encodes context up to position t, so this head
    adds *future-offset-specific* capacity without disturbing the trunk.
    """

    def __init__(self, dim: int, vocab_size: int, num_layers: int, n_heads: int,
                 ffn_mult: float, norm_eps: float = 1e-5):
        super().__init__()
        self.blocks = nn.ModuleList([
            _BlockwiseHeadBlock(dim, n_heads, ffn_mult, norm_eps=norm_eps)
            for _ in range(num_layers)
        ])
        self.final_norm = _RMSNorm(dim, eps=norm_eps)
        self.proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.proj(self.final_norm(x))


class _BlockwiseTFLinearHead(nn.Module):
    """Teacher-forced compressed linear head (Gemma drafter, linear variant).

    Conditions each offset's prediction on the embedding of the realized
    previous-offset token. Implements the "concat target_hidden + token
    embedding then down-project" pattern, with a final Linear(D, V) output
    projection.

    Parameter count: 2D*D (down-proj) + D*V (out-proj) ~= D*V (the dominant
    D*V term matches a vanilla per-offset linear head). The factorization
    avoids the 2D*V blow-up of a direct Linear(2D, V).
    """

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.down_proj = nn.Linear(2 * dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, pred_hidden, prev_token_embed):
        # pred_hidden: [..., D], prev_token_embed: [..., D] (same trailing dims)
        x = torch.cat([pred_hidden, prev_token_embed], dim=-1)  # [..., 2D]
        return self.out_proj(self.down_proj(x))


class _BlockwiseTFTransformerHead(nn.Module):
    """Teacher-forced transformer head (Gemma drafter, transformer variant).

    Equivalent to ``_BlockwiseTransformerHead`` (E2.3) but with an additional
    "concat + down-project" front-end that injects ``embed(prev_offset_token)``
    into the head's input. The transformer blocks then operate on the
    (compressed, conditioned) hidden state with the same causal AR mask over
    the S dim as E2.3.

    Mirrors Gemma 4 drafter structure: concat target hidden + token embedding,
    down-project to drafter dim, run a small transformer, project to vocab.
    """

    def __init__(self, dim: int, vocab_size: int, num_layers: int, n_heads: int,
                 ffn_mult: float, norm_eps: float = 1e-5):
        super().__init__()
        self.down_proj = nn.Linear(2 * dim, dim, bias=False)
        self.blocks = nn.ModuleList([
            _BlockwiseHeadBlock(dim, n_heads, ffn_mult, norm_eps=norm_eps)
            for _ in range(num_layers)
        ])
        self.final_norm = _RMSNorm(dim, eps=norm_eps)
        self.proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, pred_hidden, prev_token_embed):
        x = torch.cat([pred_hidden, prev_token_embed], dim=-1)
        x = self.down_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.proj(self.final_norm(x))


def _build_block_latent_head(head_type: str, K: int, dim: int, vocab_size: int, hparams):
    """Construct the block-latent head module(s) for explicit-block-latent modes.

    Returns either an ``nn.Linear`` (for ``linear`` head type, byte-identical
    to pre-E2 behavior) or an ``nn.ModuleList`` with K independent per-offset
    heads (for any ``per_offset_*`` type). Each per-offset head must be
    callable on ``[B, S, D]`` and return ``[B, S, V]``.
    """
    if head_type == "linear":
        return nn.Linear(dim, vocab_size, bias=False)
    if head_type == "per_offset_linear":
        return nn.ModuleList([
            _BlockwiseLinearHead(dim, vocab_size) for _ in range(K)
        ])
    if head_type == "per_offset_mlp":
        ffn_mult = float(getattr(hparams, "block_latent_head_ffn_mult", 4.0))
        return nn.ModuleList([
            _BlockwiseMLPHead(dim, vocab_size, ffn_mult=ffn_mult) for _ in range(K)
        ])
    if head_type == "per_offset_transformer":
        ffn_mult = float(getattr(hparams, "block_latent_head_ffn_mult", 4.0))
        num_layers = int(getattr(hparams, "block_latent_head_layers", 1))
        n_heads_default = int(getattr(hparams, "multiheaded_attention_heads", 2))
        n_heads_override = int(getattr(hparams, "block_latent_head_n_heads", 0))
        n_heads = n_heads_override if n_heads_override > 0 else n_heads_default
        return nn.ModuleList([
            _BlockwiseTransformerHead(
                dim=dim,
                vocab_size=vocab_size,
                num_layers=num_layers,
                n_heads=n_heads,
                ffn_mult=ffn_mult,
            ) for _ in range(K)
        ])
    if head_type == "per_offset_tf_linear":
        return nn.ModuleList([
            _BlockwiseTFLinearHead(dim, vocab_size) for _ in range(K)
        ])
    if head_type == "per_offset_tf_transformer":
        ffn_mult = float(getattr(hparams, "block_latent_head_ffn_mult", 4.0))
        num_layers = int(getattr(hparams, "block_latent_head_layers", 1))
        n_heads_default = int(getattr(hparams, "multiheaded_attention_heads", 2))
        n_heads_override = int(getattr(hparams, "block_latent_head_n_heads", 0))
        n_heads = n_heads_override if n_heads_override > 0 else n_heads_default
        return nn.ModuleList([
            _BlockwiseTFTransformerHead(
                dim=dim,
                vocab_size=vocab_size,
                num_layers=num_layers,
                n_heads=n_heads,
                ffn_mult=ffn_mult,
            ) for _ in range(K)
        ])
    raise ValueError(
        f"Unknown block_latent_head_type={head_type!r}; "
        f"expected one of {BLOCK_LATENT_HEAD_TYPES}"
    )


def _parse_offset_loss_weights(raw, K: int):
    """Parse the ``--offset_loss_weights`` CLI value into a normalized [K]-list.

    Empty / None -> uniform 1/K (which makes the weighted aggregate equal to
    the original ``mean over flattened (B*K*S)`` loss, preserving pre-E1
    behavior).
    """
    if raw is None or str(raw).strip() == "":
        return [1.0 / K] * K
    try:
        values = [float(v) for v in str(raw).split(",") if str(v).strip()]
    except ValueError as e:
        raise ValueError(
            f"--offset_loss_weights must be comma-separated floats, got {raw!r}"
        ) from e
    if len(values) != K:
        raise ValueError(
            f"--offset_loss_weights length {len(values)} != K={K}; got {values}"
        )
    if any(v < 0 for v in values):
        raise ValueError(f"--offset_loss_weights must be non-negative, got {values}")
    s = sum(values)
    if s <= 0:
        raise ValueError(f"--offset_loss_weights must have positive sum, got {values}")
    return [v / s for v in values]

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
        
        self.alpha = nn.Parameter(torch.tensor(float(self.hparams.mcmc_step_size), dtype=torch.float32), requires_grad=self.hparams.mcmc_step_size_learnable)
        self.langevin_dynamics_noise_std = nn.Parameter(torch.tensor(float(self.hparams.langevin_dynamics_noise)), requires_grad=False) # if using self.hparams.langevin_dynamics_noise_learnable this will be turned on in warm_up_finished func

        self.embeddings = nn.Embedding(self.vocab_size, self.hparams.embedding_dim)
        init_whole_model_weights(self.embeddings, self.hparams.weight_initialization_method, weight_initialization_gain=self.hparams.weight_initialization_gain)

        max_blockwise_offsets = max(1, int(getattr(self.hparams, "train_block_size", 1)))
        # Dense blockwise joint head:
        # shared trunk features [B, S, D] -> [B, S, K_max * V], then reshape to [B, K, S, V].
        self.blockwise_joint_head = nn.Linear(
            self.hparams.embedding_dim,
            max_blockwise_offsets * self.vocab_size,
            bias=False,
        )
        init_whole_model_weights(
            self.blockwise_joint_head,
            self.hparams.weight_initialization_method,
            weight_initialization_gain=self.hparams.weight_initialization_gain,
        )

        # Single-token head used by the explicit-block-latent modes
        # (future_latent_non_causal / blockwise). Each future latent's hidden
        # state is mapped through this head to produce a single token's logits;
        # there is no joint multi-offset projection here. Kept independent of
        # `blockwise_joint_head` so mtp_mcmc behavior is byte-identical.
        #
        # E2: the head construction is now driven by `--block_latent_head_type`:
        #   * `linear`                 (default; byte-identical to pre-E2)
        #   * `per_offset_linear`      (E2.1)
        #   * `per_offset_mlp`         (E2.2)
        #   * `per_offset_transformer` (E2.3, DeepSeek-MTP-style)
        # For `linear`, `self.block_latent_token_head` is a single `nn.Linear`;
        # for `per_offset_*` modes it is an `nn.ModuleList` of K heads, one per
        # offset (selected by index in `_apply_block_latent_head_per_offset`).
        self._block_latent_head_type = str(
            getattr(self.hparams, "block_latent_head_type", "linear")
        )
        if self._block_latent_head_type not in BLOCK_LATENT_HEAD_TYPES:
            raise ValueError(
                f"--block_latent_head_type={self._block_latent_head_type!r} "
                f"is not in {BLOCK_LATENT_HEAD_TYPES}"
            )
        self.block_latent_token_head = _build_block_latent_head(
            head_type=self._block_latent_head_type,
            K=max_blockwise_offsets,
            dim=self.hparams.embedding_dim,
            vocab_size=self.vocab_size,
            hparams=self.hparams,
        )
        init_whole_model_weights(
            self.block_latent_token_head,
            self.hparams.weight_initialization_method,
            weight_initialization_gain=self.hparams.weight_initialization_gain,
        )

        # E1: parse `--offset_loss_weights` once at __init__ so per-step loss
        # construction is cheap. Stored as a python list (length K) of floats
        # normalized to sum to 1, so default uniform weights reproduce the
        # original `mean over flattened (B*K*S)` cross-entropy.
        self._offset_loss_weights = _parse_offset_loss_weights(
            getattr(self.hparams, "offset_loss_weights", ""),
            K=max_blockwise_offsets,
        )

        # E3.1: causal-detach in MCMC for explicit-block-latent modes. Default
        # off (no-op, preserves pre-E3 behavior). When on, each per-offset
        # latent z_{t,j} only receives gradient from its own energy e_{t,j}
        # during MCMC updates; the forward causal attention path z_{t,k>=j}
        # ← z_{t,j} is kept so z_{t,k} can still leverage z_{t,j}'s content.
        # See ``_run_explicit_block_latent_mcmc`` for the diagonal-gradient
        # implementation. Has no effect on `mtp_mcmc` (different MCMC path)
        # and is a near no-op on `future_latent_non_causal` (cross-offset
        # gradient is already zero there).
        self._mcmc_causal_detach = bool(
            getattr(self.hparams, "mcmc_causal_detach", False)
        )

        # E3.2: staggered / wavefront MCMC. At step i, only the first
        # ``active_K(i) = min(i+1, K)`` offsets are actually updated; later
        # offsets stay at their initial values until enough steps have passed.
        # Solves a different problem than E3.1: rather than cleaning the
        # backward gradient, it controls *when* each latent starts moving so
        # that z_1 can settle in step 0 before z_2 starts being optimized in
        # step 1. The forward energy is still computed jointly so that z_2's
        # initial value still contributes to e_2 (and through the causal mask
        # to z_2's eventual prediction quality). Default off; compatible with
        # E3.1 (both flags can be on at the same time).
        self._mcmc_staggered = bool(
            getattr(self.hparams, "mcmc_staggered", False)
        )
        
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
        self._transformer_accepts_return_pred_hidden = (
            "return_pred_hidden" in inspect.signature(self.transformer.forward).parameters
        )
        self._transformer_accepts_return_context_hidden = (
            "return_context_hidden" in inspect.signature(self.transformer.forward).parameters
        )
        # Single source of truth for the attention / training semantic the
        # trunk and all downstream paths (train, MCMC, blockwise dense, block
        # refine, inference) must consistently use. If not set on hparams
        # (e.g. a legacy checkpoint), we default to dense_token which matches
        # original main-branch EBT semantics.
        self._block_mode = getattr(self.hparams, "block_mode", "dense_token")
        
        self.finished_warming_up = False

        self.mcmc_replay_buffer = 'mcmc_replay_buffer' in self.hparams and self.hparams.mcmc_replay_buffer and self.hparams.execution_mode != "inference"
        if self.mcmc_replay_buffer:
            replay_buffer_max_size = self.hparams.mcmc_replay_buffer_size
            self.replay_buffer_samples = self.hparams.batch_size_per_device * self.hparams.mcmc_replay_buffer_sample_bs_percent
            self.replay_buffer = CausalReplayBuffer(max_size=replay_buffer_max_size, sample_size=self.replay_buffer_samples)

        # DEBUGGING CODE ################################################################################################################################################
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
                      langevin_dynamics_noise_std, alpha, start_pos, learning, return_raw_logits):
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
            if self.hparams.normalize_initial_condition_only_first_step:
                if mcmc_step == 0:
                    predicted_tokens = self.softmax(predicted_tokens)
            else:
                predicted_tokens = self.softmax(predicted_tokens)
                
            if self.hparams.vocab_to_embed_uses_prob_dist: # predicted_embeds is B, S, V; embed is V, D
                predicted_embeddings = torch.matmul(predicted_tokens, self.embeddings.weight) #BS, S, D
            else:
                predicted_embeddings = self.vocab_to_embed(predicted_tokens) #BS, S, D
        else:
            predicted_embeddings = self.vocab_to_embed(predicted_tokens) #BS, S, D
        
        all_embeddings = torch.cat((real_embeddings_input.detach(), predicted_embeddings), dim = 1) # B, 2*S, D
        context_len = real_embeddings_input.shape[1]
        pred_len = seq_length
        # create_graph=True 的 autograd.grad 与 compiled graph 不兼容
        transformer = getattr(self, 'transformer_eager', self.transformer)
        energy_preds = transformer(
            all_embeddings,
            start_pos=start_pos,
            mcmc_step=mcmc_step,
            context_len=context_len,
            pred_len=pred_len,
            block_mode=self._block_mode,
        )
        energy_preds = energy_preds.reshape(-1, 1)
        
        with torch.amp.autocast(device_type='cuda', enabled=False):
            energy_f32 = energy_preds.float()
            if self.hparams.truncate_mcmc:  #retain_graph defaults to create_graph value here; if learning is true then create_graph else dont (inference)
                if i == (num_mcmc_steps - 1):
                    predicted_tokens_grad = torch.autograd.grad([energy_f32.sum()], [predicted_tokens], create_graph=learning)[0]
                else:
                    predicted_tokens_grad = torch.autograd.grad([energy_f32.sum()], [predicted_tokens], create_graph=False)[0]
            else:
                predicted_tokens_grad = torch.autograd.grad([energy_f32.sum()], [predicted_tokens], create_graph=learning)[0]
        # predicted_tokens_grad has shape B, S, V
        
        if self.hparams.clamp_futures_grad:
            min_and_max = self.hparams.clamp_futures_grad_max_change / (self.alpha.float()) # use self.alpha and not random alpha to clamp
            # predicted_tokens_grad = scale_clamp(predicted_tokens_grad, -min_and_max, min_and_max)
            predicted_tokens_grad = torch.clamp(predicted_tokens_grad, min = -min_and_max, max = min_and_max)
            
        if torch.isnan(predicted_tokens_grad).any() or torch.isinf(predicted_tokens_grad).any():
            raise ValueError("NaN or Inf gradients detected during MCMC.")
        
        predicted_tokens = predicted_tokens - alpha * predicted_tokens_grad # do this to tokens will be unnormalize prob dist convert to prob dist after

        # [DEBUG] check alpha
        # if mcmc_step == 0 and self.training and self._alpha_debug_step <= 5:
        #     print(
        #         f"[ALPHA_DEBUG] _mcmc_step_excluded mcmc_step={mcmc_step} | "
        #         f"alpha dtype={alpha.dtype} alpha_min={alpha.item() if alpha.numel() == 1 else alpha.min().item():.8f} | "
        #         f"predicted_tokens_grad dtype={predicted_tokens_grad.dtype} grad_abs_mean={predicted_tokens_grad.abs().mean().item():.8f} | "
        #         f"predicted_tokens dtype={predicted_tokens.dtype} | "
        #         f"create_graph={learning}",
        #         flush=True,
        #     )
        
        if self.hparams.absolute_clamp != 0.0:
            predicted_tokens = torch.clamp(predicted_tokens, min = -self.hparams.absolute_clamp, max = self.hparams.absolute_clamp)
        
        if self.hparams.sharpen_predicted_distribution != 0.0:
            predicted_tokens = predicted_tokens / self.hparams.sharpen_predicted_distribution

        if return_raw_logits:
            predicted_tokens_for_loss = predicted_tokens # BS, S, V
        else:
            predicted_tokens_for_loss = self.log_softmax(predicted_tokens).reshape(-1, self.vocab_size) # BS*S, V
            
        return predicted_tokens, energy_preds, predicted_tokens_for_loss

    def forward(self, x, start_pos = 0, learning = True, return_raw_logits = False, replay_buffer_logits = None, no_randomness = True, block_size = None): # accepts input_ids as input; a lot of the logic here is just for S2 params, see pseudocode in paper for a more concise view of how this works. it can be < 10 LOC
        real_embeddings_input = self.embeddings(x)
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        model_dtype = self.embeddings.weight.dtype
        if block_size is None:
            # Default block_size differs by block_mode:
            #   * dense_token / mtp_mcmc: legacy symmetric layout, defaults
            #     to seq_length (S=K, the historical contract).
            #   * future_latent_non_causal / blockwise: this entry point is
            #     the inference C+K layout (training uses
            #     forward_explicit_block_latent_logits directly), so the
            #     natural default is K=1 (sequential decoding).
            if self._block_mode in EXPLICIT_BLOCK_LATENT_MODES:
                block_size = getattr(self.hparams, "block_size", 1)
            else:
                block_size = getattr(self.hparams, "block_size", seq_length)
        block_size = int(block_size)
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")

        # block_mode dispatch:
        #   * dense_token / mtp_mcmc: legacy symmetric path. Requires
        #     block_size == seq_length inside the trunk. This path is byte-
        #     identical to the previous mtp_mcmc behavior.
        #   * future_latent_non_causal / blockwise: new modes. We always use
        #     the inference layout here (C context tokens + K = block_size
        #     latents) so this entry point can be used directly for sequential
        #     (K=1) and direct_block (K>1) inference. Training does NOT call
        #     forward(); it goes through forward_explicit_block_latent_logits.
        if self._block_mode in ("dense_token", "mtp_mcmc"):
            if block_size != seq_length:
                raise NotImplementedError(
                    f"EBT_NLP.forward with block_size != seq_length is not supported under "
                    f"block_mode={self._block_mode!r}; this path requires the 'blockwise' "
                    f"block_mode which is not implemented yet. Use sequential inference, "
                    f"or train a blockwise-mode checkpoint to enable non-symmetric block "
                    f"prediction. Got block_size={block_size}, seq_length={seq_length}."
                )
        elif self._block_mode in EXPLICIT_BLOCK_LATENT_MODES:
            ebt_type = getattr(self.hparams, "ebt_type", "default")
            if ebt_type not in ("default", "time_embed"):
                raise NotImplementedError(
                    f"block_mode={self._block_mode!r} is currently only supported for "
                    f"ebt_type in [default, time_embed]; got ebt_type={ebt_type}."
                )
            return self._forward_explicit_block_latent_inference(
                real_embeddings_input=real_embeddings_input,
                block_size=block_size,
                start_pos=start_pos,
                learning=learning,
                return_raw_logits=return_raw_logits,
                no_randomness=no_randomness,
            )
        else:
            raise ValueError(f"Unknown block_mode={self._block_mode!r} on EBT_NLP")

        if getattr(self.hparams, "ebt_type", "default") not in ("default", "time_embed") and block_size != seq_length:
            raise NotImplementedError(
                f"block_size != seq_length is currently only supported for ebt_type in [default, time_embed]; got ebt_type={self.hparams.ebt_type}, block_size={block_size}, seq_length={seq_length}"
            )

        alpha = torch.clamp(self.alpha, min=0.0001).float()
        if not no_randomness and self.hparams.randomize_mcmc_step_size_scale != 1:
            expanded_alpha = alpha.expand(batch_size, block_size, 1)
            scale = self.hparams.randomize_mcmc_step_size_scale
            low = alpha / scale
            high = alpha * scale
            alpha = low + torch.rand_like(expanded_alpha) * (high - low)

        # noise is intentionally detached and cast to model_dtype to avoid inserting
        # a float32 node into the create_graph=True autograd graph.
        langevin_dynamics_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001).detach().to(model_dtype)

        predicted_tokens = self.corrupt_embeddings(real_embeddings_input, target_length=block_size) # B, K, V
        if replay_buffer_logits is not None: # using replay buffer, use the logits instead of corruption
            if block_size != seq_length:
                raise NotImplementedError("replay_buffer_logits path only supports block_size == seq_length")
            if replay_buffer_logits.shape[1] != block_size:
                raise ValueError(
                    f"replay_buffer_logits shape mismatch: expected second dim {block_size}, got {replay_buffer_logits.shape[1]}"
                )
            predicted_tokens[batch_size - replay_buffer_logits.shape[0]:] = replay_buffer_logits # NOTE this assumes the fresh data is concatted first

        _, predicted_distributions, predicted_energies = self._run_mcmc_on_given_pred_tokens(
            real_embeddings_input=real_embeddings_input,
            predicted_tokens=predicted_tokens,
            start_pos=start_pos,
            learning=learning,
            return_raw_logits=return_raw_logits,
            no_randomness=no_randomness,
            alpha=alpha,
            langevin_dynamics_noise_std=langevin_dynamics_noise_std,
        )
        return predicted_distributions, predicted_energies

    def _build_mcmc_steps(self, no_randomness=True):
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
        return mcmc_steps

    def _run_mcmc_on_given_pred_tokens(
        self,
        real_embeddings_input,
        predicted_tokens,
        start_pos=0,
        learning=True,
        return_raw_logits=False,
        no_randomness=True,
        alpha=None,
        langevin_dynamics_noise_std=None,
        optimize_mask=None,
        mcmc_steps=None,
        return_pred_hidden=False,
        return_context_hidden=False,
    ):
        predicted_distributions = []
        predicted_energies = []
        predicted_hiddens = []
        context_hiddens = []
        batch_size, seq_length, _ = predicted_tokens.shape
        if alpha is None:
            alpha = torch.clamp(self.alpha, min=0.0001)
        if langevin_dynamics_noise_std is None:
            langevin_dynamics_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)
        if mcmc_steps is None:
            mcmc_steps = self._build_mcmc_steps(no_randomness=no_randomness)

        need_post_update_hidden = return_context_hidden or return_pred_hidden
        if return_context_hidden and not self._transformer_accepts_return_context_hidden:
            raise NotImplementedError(
                f"Transformer {type(self.transformer).__name__} does not support return_context_hidden=True"
            )
        if return_pred_hidden and not self._transformer_accepts_return_pred_hidden:
            raise NotImplementedError(
                f"Transformer {type(self.transformer).__name__} does not support return_pred_hidden=True"
            )

        optimize_mask_bool = None
        optimize_mask_float = None
        fixed_logits = None
        if optimize_mask is not None:
            optimize_mask_bool = optimize_mask.to(dtype=torch.bool)
            optimize_mask_float = optimize_mask.to(dtype=predicted_tokens.dtype)
            fixed_logits = predicted_tokens.detach()

        transformer_body = getattr(self, "transformer_eager", self.transformer)

        with torch.set_grad_enabled(True):
            for i, mcmc_step in enumerate(mcmc_steps):
                if self.hparams.no_mcmc_detach:
                    predicted_tokens.requires_grad_().reshape(batch_size, seq_length, self.vocab_size) # B, S, V
                else: # default, do detach
                    predicted_tokens = predicted_tokens.detach().requires_grad_().reshape(batch_size, seq_length, self.vocab_size) # B, S, V

                if self.hparams.langevin_dynamics_noise != 0:
                    ld_noise = torch.randn_like(predicted_tokens.detach()) * langevin_dynamics_noise_std # langevin dynamics
                    if optimize_mask_float is not None:
                        ld_noise = ld_noise * optimize_mask_float
                    predicted_tokens = predicted_tokens + ld_noise

                if self.hparams.normalize_initial_condition:
                    if self.hparams.normalize_initial_condition_only_first_step:
                        if mcmc_step == 0:
                            predicted_tokens = self.softmax(predicted_tokens)
                    else:
                        predicted_tokens = self.softmax(predicted_tokens)

                    if self.hparams.vocab_to_embed_uses_prob_dist: # predicted_embeds is B, S, V; embed is V, D
                        predicted_embeddings = torch.matmul(predicted_tokens, self.embeddings.weight) #BS, S, D
                    else:
                        predicted_embeddings = self.vocab_to_embed(predicted_tokens) #BS, S, D
                else:
                    predicted_embeddings = self.vocab_to_embed(predicted_tokens) #BS, S, D

                all_embeddings = torch.cat((real_embeddings_input, predicted_embeddings), dim = 1) # B, S+K, D
                base_transformer_kwargs = dict(
                    start_pos=start_pos,
                    mcmc_step=mcmc_step,
                    context_len=real_embeddings_input.shape[1],
                    pred_len=predicted_embeddings.shape[1],
                    block_mode=self._block_mode,
                )
                energy_preds = transformer_body(
                    all_embeddings,
                    **base_transformer_kwargs,
                ) # checked and there are no in place ops; mcmc_step only applies to when using certain types of ebt
                energy_preds = energy_preds.reshape(-1, 1)
                predicted_energies.append(energy_preds)

                with torch.amp.autocast(device_type='cuda', enabled=False):
                    energy_f32 = energy_preds.float()
                    if self.hparams.truncate_mcmc:  #retain_graph defaults to create_graph value here; if learning is true then create_graph else dont (inference)
                        if i == (len(mcmc_steps) - 1):
                            predicted_tokens_grad = torch.autograd.grad([energy_f32.sum()], [predicted_tokens], create_graph=learning)[0]
                        else:
                            predicted_tokens_grad = torch.autograd.grad([energy_f32.sum()], [predicted_tokens], create_graph=False)[0]
                    else:
                        predicted_tokens_grad = torch.autograd.grad([energy_f32.sum()], [predicted_tokens], create_graph=learning)[0]
                # predicted_tokens_grad has shape B, S, V

                if optimize_mask_float is not None:
                    predicted_tokens_grad = predicted_tokens_grad * optimize_mask_float

                if self.hparams.clamp_futures_grad:
                    min_and_max = self.hparams.clamp_futures_grad_max_change / (self.alpha.float())
                    predicted_tokens_grad = torch.clamp(predicted_tokens_grad, min = -min_and_max, max = min_and_max)

                if torch.isnan(predicted_tokens_grad).any() or torch.isinf(predicted_tokens_grad).any():
                    raise ValueError("NaN or Inf gradients detected during MCMC.")

                predicted_tokens = predicted_tokens - alpha * predicted_tokens_grad # do this to tokens will be unnormalize prob dist convert to prob dist after
                if self.hparams.absolute_clamp != 0.0:
                    predicted_tokens = torch.clamp(predicted_tokens, min = -self.hparams.absolute_clamp, max = self.hparams.absolute_clamp)
                if self.hparams.sharpen_predicted_distribution != 0.0:
                    predicted_tokens = predicted_tokens / self.hparams.sharpen_predicted_distribution

                if optimize_mask_bool is not None:
                    predicted_tokens = torch.where(optimize_mask_bool, predicted_tokens, fixed_logits)

                if need_post_update_hidden:
                    post_update_pred_embeddings = self._logits_to_pred_embeddings(predicted_tokens, mcmc_step)
                    post_update_all_embeddings = torch.cat((real_embeddings_input, post_update_pred_embeddings), dim=1)
                    post_transformer_kwargs = dict(base_transformer_kwargs)
                    post_transformer_kwargs["return_context_hidden"] = return_context_hidden
                    post_transformer_kwargs["return_pred_hidden"] = return_pred_hidden
                    post_transformer_out = transformer_body(
                        post_update_all_embeddings,
                        **post_transformer_kwargs,
                    )
                    if return_context_hidden and return_pred_hidden:
                        _, context_hidden, pred_hidden = post_transformer_out
                        context_hiddens.append(context_hidden)
                        predicted_hiddens.append(pred_hidden)
                    elif return_context_hidden:
                        _, context_hidden = post_transformer_out
                        context_hiddens.append(context_hidden)
                    elif return_pred_hidden:
                        _, pred_hidden = post_transformer_out
                        predicted_hiddens.append(pred_hidden)

                if return_raw_logits:
                    predicted_tokens_for_loss = predicted_tokens # BS, S, V
                else:
                    predicted_tokens_for_loss = self.log_softmax(predicted_tokens).reshape(-1, self.vocab_size) # BS*S, V
                predicted_distributions.append(predicted_tokens_for_loss)

        if return_context_hidden and return_pred_hidden:
            return predicted_tokens, predicted_distributions, predicted_energies, context_hiddens, predicted_hiddens
        if return_context_hidden:
            return predicted_tokens, predicted_distributions, predicted_energies, context_hiddens
        if return_pred_hidden:
            return predicted_tokens, predicted_distributions, predicted_energies, predicted_hiddens
        return predicted_tokens, predicted_distributions, predicted_energies

    def _discrete_token_ids_to_initial_logits(self, token_ids, init_logit_scale=8.0):
        logits = torch.zeros(
            token_ids.shape[0],
            token_ids.shape[1],
            self.vocab_size,
            device=token_ids.device,
            dtype=self.embeddings.weight.dtype,
        )
        return logits.scatter_(-1, token_ids.unsqueeze(-1), float(init_logit_scale))

    def _draft_block_ids_to_initial_logits(self, draft_block_ids, init_logit_scale=8.0):
        return self._discrete_token_ids_to_initial_logits(draft_block_ids, init_logit_scale=init_logit_scale)

    def _logits_to_pred_embeddings(self, logits, step_idx):
        if self.hparams.normalize_initial_condition:
            if self.hparams.normalize_initial_condition_only_first_step:
                if step_idx == 0:
                    logits = self.softmax(logits)
            else:
                logits = self.softmax(logits)

            if self.hparams.vocab_to_embed_uses_prob_dist:
                return torch.matmul(logits, self.embeddings.weight)
            return self.vocab_to_embed(logits)
        return self.vocab_to_embed(logits)

    def ebt_refine_block_fast(self, context_ids, draft_block_ids, refine_steps=None, init_logit_scale=8.0, start_pos=0, learning=False):
        """
        Fast block refinement: only block logits require grad.
        Returns:
            refined_block_logits: [B, K, V]
            refined_block_ids: [B, K]
        """
        if draft_block_ids.shape[1] == 0:
            empty_logits = torch.empty(
                draft_block_ids.shape[0], 0, self.vocab_size, device=draft_block_ids.device, dtype=self.embeddings.weight.dtype
            )
            return empty_logits, draft_block_ids

        if context_ids.shape[1] == 0:
            raise ValueError("ebt_refine_block_fast requires at least one context token")

        # EBT-compatible pair:
        # real_input_ids: [context, draft[:-1]] length = C + K - 1
        # predicted targets: [context[1:], draft] where only draft logits are optimized
        real_input_ids = torch.cat([context_ids, draft_block_ids[:, :-1]], dim=1)
        block_len = draft_block_ids.shape[1]
        real_embeddings_input = self.embeddings(real_input_ids)
        context_target_ids = context_ids[:, 1:]
        fixed_prefix_logits = self._discrete_token_ids_to_initial_logits(
            context_target_ids, init_logit_scale=init_logit_scale
        ).detach()
        block_logits = self._draft_block_ids_to_initial_logits(
            draft_block_ids, init_logit_scale=init_logit_scale
        ).detach()
        init_block_logits = block_logits.detach().clone()

        alpha = torch.clamp(self.alpha, min=0.0001)
        noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)

        requested_steps = int(refine_steps) if refine_steps is not None else int(self.hparams.mcmc_num_steps)
        if requested_steps <= 0:
            refined_block_ids = torch.argmax(block_logits, dim=-1)
            return block_logits, refined_block_ids
        effective_steps = min(requested_steps, int(self.hparams.mcmc_num_steps))
        mcmc_steps = list(range(effective_steps))
        diagnose = bool(getattr(self.hparams, "infer_block_diagnose", False))

        def _block_energy_mean(logits, step_idx):
            with torch.no_grad():
                prefix_embeds = self._logits_to_pred_embeddings(fixed_prefix_logits, step_idx)
                block_embeds = self._logits_to_pred_embeddings(logits, step_idx)
                pred_embeddings = torch.cat([prefix_embeds, block_embeds], dim=1)
                all_embeddings = torch.cat((real_embeddings_input, pred_embeddings), dim=1)
                energy_preds = self.transformer(all_embeddings, start_pos=start_pos, mcmc_step=step_idx, block_mode=self._block_mode)
                return energy_preds[:, -block_len:].mean().item()

        initial_energy_mean = _block_energy_mean(init_block_logits, mcmc_steps[0])
        last_grad_norm = 0.0

        with torch.set_grad_enabled(True):
            for i, mcmc_step in enumerate(mcmc_steps):
                block_logits = block_logits.detach().requires_grad_()
                cur_block_logits = block_logits
                if self.hparams.langevin_dynamics_noise != 0:
                    ld_noise = torch.randn_like(cur_block_logits.detach()) * noise_std
                    cur_block_logits = cur_block_logits + ld_noise

                prefix_embeds = self._logits_to_pred_embeddings(fixed_prefix_logits, mcmc_step)
                block_embeds = self._logits_to_pred_embeddings(cur_block_logits, mcmc_step)
                pred_embeddings = torch.cat([prefix_embeds, block_embeds], dim=1)
                all_embeddings = torch.cat((real_embeddings_input, pred_embeddings), dim=1)
                energy_preds = self.transformer(
                    all_embeddings,
                    start_pos=start_pos,
                    mcmc_step=mcmc_step,
                    context_len=real_embeddings_input.shape[1],
                    pred_len=pred_embeddings.shape[1],
                    block_mode=self._block_mode,
                )
                energy_block = energy_preds[:, -block_len:].reshape(-1, 1)

                block_grad = torch.autograd.grad([energy_block.sum()], [cur_block_logits], create_graph=learning)[0]
                last_grad_norm = block_grad.norm().item()
                if self.hparams.clamp_futures_grad:
                    min_and_max = self.hparams.clamp_futures_grad_max_change / (self.alpha)
                    block_grad = torch.clamp(block_grad, min=-min_and_max, max=min_and_max)
                if torch.isnan(block_grad).any() or torch.isinf(block_grad).any():
                    raise ValueError("NaN or Inf gradients detected during block refinement.")

                block_logits = cur_block_logits - alpha * block_grad
                if self.hparams.absolute_clamp != 0.0:
                    block_logits = torch.clamp(block_logits, min=-self.hparams.absolute_clamp, max=self.hparams.absolute_clamp)
                if self.hparams.sharpen_predicted_distribution != 0.0:
                    block_logits = block_logits / self.hparams.sharpen_predicted_distribution
                block_logits = block_logits.detach()

        final_energy_mean = _block_energy_mean(block_logits, mcmc_steps[-1])
        refined_block_logits = block_logits
        refined_block_ids = torch.argmax(refined_block_logits, dim=-1)

        if diagnose:
            delta_norm = (refined_block_logits - init_block_logits).norm().item()
            max_show = min(2, draft_block_ids.shape[0])
            for b in range(max_show):
                print(f"draft_block_ids[{b}]: {draft_block_ids[b].tolist()}", flush=True)
                print(f"refined_block_ids[{b}]: {refined_block_ids[b].tolist()}", flush=True)
            print(f"||refined_logits - init_logits||: {delta_norm:.6f}", flush=True)
            print(f"block 初始 energy: {initial_energy_mean:.6f}", flush=True)
            print(f"block 最终 energy: {final_energy_mean:.6f}", flush=True)
            print(f"block logits 梯度范数 grad.norm(): {last_grad_norm:.6f}", flush=True)

        return refined_block_logits, refined_block_ids

    def ebt_refine_block(self, context_ids, draft_block_ids, refine_steps=None, init_logit_scale=8.0, start_pos=0, learning=False):
        # Backward-compatible wrapper for older callsites.
        return self.ebt_refine_block_fast(
            context_ids=context_ids,
            draft_block_ids=draft_block_ids,
            refine_steps=refine_steps,
            init_logit_scale=init_logit_scale,
            start_pos=start_pos,
            learning=learning,
        )

    def forward_blockwise_dense_hidden(self, input_ids, no_randomness):
        """
        Shared-trunk blockwise dense hidden extraction for the dense blockwise head.
        IMPORTANT: use post-update pred_hidden, not context_hidden.
        In the current attention layout, context_hidden does not attend to the pred branch,
        so even post-update context_hidden stays independent of alpha. post-update pred_hidden
        does depend on x' = x - alpha * grad(E), which restores a differentiable alpha path.
        Returns:
            pred_hiddens_per_step: list[[B, S_eff, D]]
            predicted_energies: list[[B*S_eff, 1]]
        """
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids [B, S_eff], got shape {tuple(input_ids.shape)}")

        real_embeddings_input = self.embeddings(input_ids)
        seq_eff = input_ids.shape[1]
        alpha = torch.clamp(self.alpha, min=0.0001)
        langevin_dynamics_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)
        predicted_tokens = self.corrupt_embeddings(real_embeddings_input, target_length=seq_eff)  # [B, S_eff, V]

        _, _, predicted_energies, pred_hiddens_per_step = self._run_mcmc_on_given_pred_tokens(
            real_embeddings_input=real_embeddings_input,
            predicted_tokens=predicted_tokens,
            start_pos=0,
            learning=True,
            return_raw_logits=True,
            no_randomness=no_randomness,
            alpha=alpha,
            langevin_dynamics_noise_std=langevin_dynamics_noise_std,
            return_pred_hidden=True,
        )
        return pred_hiddens_per_step, predicted_energies

    def forward_blockwise_dense_logits(self, input_ids, num_offsets, no_randomness, return_hidden=False):
        """
        Shared-trunk blockwise dense prediction.
        Returns:
            multi_offset_logits_per_step: list[[B, K, S_eff, V]]
            predicted_energies: list[[B*S_eff, 1]]
        """
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids [B, S_eff], got shape {tuple(input_ids.shape)}")
        if num_offsets <= 0:
            raise ValueError(f"num_offsets must be > 0, got {num_offsets}")
        max_offsets = self.blockwise_joint_head.out_features // self.vocab_size
        if num_offsets > max_offsets:
            raise ValueError(
                f"Requested K={num_offsets} offsets but model only initialized K_max={max_offsets}. "
                "Increase --train_block_size and reinitialize model."
            )

        batch_size = input_ids.shape[0]
        seq_eff = input_ids.shape[1]
        pred_hiddens_per_step, predicted_energies = self.forward_blockwise_dense_hidden(
            input_ids=input_ids,
            no_randomness=no_randomness,
        )

        multi_offset_logits_per_step = []
        for pred_hidden in pred_hiddens_per_step:
            projected = self.blockwise_joint_head(pred_hidden)  # [B, S_eff, K_max*V]
            projected = projected.reshape(batch_size, seq_eff, max_offsets, self.vocab_size)
            projected = projected[:, :, :num_offsets, :]  # [B, S_eff, K, V]
            projected = projected.permute(0, 2, 1, 3).contiguous()  # [B, K, S_eff, V]
            multi_offset_logits_per_step.append(projected)

        if return_hidden:
            return multi_offset_logits_per_step, predicted_energies, pred_hiddens_per_step
        return multi_offset_logits_per_step, predicted_energies

    # ------------------------------------------------------------------
    # Explicit-block-latent paths (future_latent_non_causal / blockwise).
    # ------------------------------------------------------------------
    # These paths are intentionally kept independent of the dense_token /
    # mtp_mcmc paths above so that any change here is guaranteed to leave
    # mtp_mcmc behavior byte-identical. New helpers, new MCMC loop, new
    # head, new inference entry points.
    #
    # Latent shape convention (logical):
    #     predicted_tokens / pred_hidden : [B, S, K, V] / [B, S, K, D]
    # Internally we flatten to position-major [B, S*K, *] for the trunk:
    #     flatten order: outer source position t (0..S-1), inner block
    #     offset j (0..K-1). i.e. for position-major index p, t = p // K,
    #     j = p % K. This MUST match `build_explicit_block_latent_mask`
    #     and `build_explicit_block_latent_freq_indices` in utils.py.
    #
    # Each future latent z_{t,j} predicts a single token corresponding to
    # block_targets[:, j, t] (using the dataset convention that
    # block_targets has shape [B, K, S]).
    # ------------------------------------------------------------------

    def _apply_block_latent_head_per_offset(self, pred_hidden, S, K, prev_token_embeds=None):
        """Apply a per-offset head (E2.1/E2.2/E2.3 or E4 TF variants) to
        position-major hidden.

        Args:
            pred_hidden: ``[B, S*K, D]`` position-major flatten (outer t,
                inner j), or ``[B, K, D]`` at inference (S=1).
            S: number of source positions (S=1 for the inference layout).
            K: number of future offsets.
            prev_token_embeds: required if and only if the head type is in
                ``TF_HEAD_TYPES``. Shape ``[B, S, K, D]``: for each (t, j),
                the embedding of the realized PREVIOUS-offset token (i.e.
                x_{t+j-1}). During training this is teacher-forced from the
                ground-truth tokens; during inference the caller is
                responsible for filling it in sequentially (see
                ``_apply_block_latent_head_sequential_inference``).

        Returns:
            ``[B, S*K, V]`` position-major logits, so the caller can do the
            same `reshape -> permute` as the shared-Linear path.
        """
        B, P, D = pred_hidden.shape
        if P != S * K:
            raise ValueError(
                f"pred_hidden length {P} != S*K={S*K} (S={S}, K={K})"
            )
        if not isinstance(self.block_latent_token_head, nn.ModuleList):
            raise RuntimeError(
                "_apply_block_latent_head_per_offset called but "
                "block_latent_token_head is not a ModuleList "
                f"(head_type={self._block_latent_head_type!r})"
            )
        if len(self.block_latent_token_head) < K:
            raise RuntimeError(
                f"block_latent_token_head has {len(self.block_latent_token_head)} entries; "
                f"need at least K={K}. (Was the model initialized with a smaller "
                f"train_block_size than the current request?)"
            )
        is_tf = self._block_latent_head_type in TF_HEAD_TYPES
        if is_tf:
            if prev_token_embeds is None:
                raise ValueError(
                    f"head_type={self._block_latent_head_type!r} requires "
                    f"prev_token_embeds (teacher-forced previous-offset token "
                    f"embedding) but got None."
                )
            if prev_token_embeds.shape != (B, S, K, D):
                raise ValueError(
                    f"prev_token_embeds shape {tuple(prev_token_embeds.shape)} "
                    f"!= expected (B={B}, S={S}, K={K}, D={D})."
                )
        # Reshape position-major -> [B, S, K, D] -> permute -> [B, K, S, D]
        h_kstd = pred_hidden.reshape(B, S, K, D).permute(0, 2, 1, 3).contiguous()
        if is_tf:
            pte_kstd = prev_token_embeds.permute(0, 2, 1, 3).contiguous()  # [B, K, S, D]
        per_offset_logits = []
        for j in range(K):
            h_j = h_kstd[:, j]  # [B, S, D]
            if is_tf:
                pte_j = pte_kstd[:, j]  # [B, S, D]
                logits_j = self.block_latent_token_head[j](h_j, pte_j)  # [B, S, V]
            else:
                logits_j = self.block_latent_token_head[j](h_j)  # [B, S, V]
            per_offset_logits.append(logits_j)
        # Stack -> [B, K, S, V], permute -> [B, S, K, V], flatten -> [B, S*K, V]
        logits_kstv = torch.stack(per_offset_logits, dim=1)
        return logits_kstv.permute(0, 2, 1, 3).contiguous().reshape(B, S * K, self.vocab_size)

    def _build_prev_token_embeds_training(self, input_ids, block_targets, K):
        """Build the teacher-forced ``prev_token_embeds`` tensor for the TF
        head path during training.

        For each (t, j) the "previous-offset token" is x_{t+j-1}:
          * j == 0 : prev = input_ids[:, t]                (current input token)
          * j >= 1 : prev = block_targets[:, j-1, :]       (ground-truth offset target one slot earlier)

        Args:
            input_ids: ``[B, S]``
            block_targets: ``[B, K, S]`` (may contain -1 ignore_index; masked
                to a safe id 0 before embedding so the embedding lookup
                doesn't blow up — downstream loss already ignores those
                positions via ignore_index=-1, so the leaked embedding value
                doesn't contribute to gradient at supervised positions).
            K: number of offsets.

        Returns:
            ``[B, S, K, D]`` tensor of per-(t, j) previous-offset embeddings.
        """
        B, S = input_ids.shape
        per_offset_prev = []
        for j in range(K):
            if j == 0:
                prev_j = input_ids
            else:
                # block_targets[:, j-1, :] is x_{t+j} (0-indexed j-1 = 1-indexed
                # offset j-1, which targets x_{t+j-1+1} = x_{t+j}).
                # Wait — block_targets[:, k, :] = x_{t+k+1} for 0-indexed k.
                # offset j (0-indexed) predicts x_{t+j+1}; its "prev" is x_{t+j}.
                # block_targets[:, j-1, :] for 0-indexed j-1 = x_{t+(j-1)+1} = x_{t+j}.
                # So this is correct.
                prev_j = block_targets[:, j - 1, :]
                # Mask out ignore_index (-1) so embedding lookup is safe.
                prev_j = torch.where(prev_j >= 0, prev_j, torch.zeros_like(prev_j))
            per_offset_prev.append(self.embeddings(prev_j))  # [B, S, D]
        return torch.stack(per_offset_prev, dim=2)  # [B, S, K, D]

    def _apply_block_latent_head_sequential_inference(self, pred_hidden, K, anchor_embed):
        """Sequential application of TF heads at inference (S=1 layout).

        Args:
            pred_hidden: ``[B, K, D]`` (inference layout — K future latents
                anchored at the end of context).
            K: number of offsets.
            anchor_embed: ``[B, 1, D]`` — embedding of the last context token,
                used as the "previous-offset" embedding for offset 0.

        Returns:
            ``[B, K, V]`` per-offset logits. Argmax of offset j's logits is
            embedded and fed as the "previous-offset" embedding for offset
            j+1 (greedy autoregressive within block). Caller can re-sample
            the returned logits stochastically; the conditional structure is
            already baked in via the sequential argmax path.
        """
        if pred_hidden.shape[1] != K:
            raise ValueError(
                f"pred_hidden length mismatch: expected K={K}, got {pred_hidden.shape[1]}"
            )
        if anchor_embed.dim() != 3 or anchor_embed.shape[1] != 1:
            raise ValueError(
                f"anchor_embed shape must be [B, 1, D]; got {tuple(anchor_embed.shape)}"
            )
        B = pred_hidden.shape[0]
        prev_embed = anchor_embed  # [B, 1, D]
        logits_per_offset = []
        for j in range(K):
            h_j = pred_hidden[:, j:j + 1, :]  # [B, 1, D]
            if self._block_latent_head_type in TF_HEAD_TYPES:
                logits_j = self.block_latent_token_head[j](h_j, prev_embed)  # [B, 1, V]
            else:
                logits_j = self.block_latent_token_head[j](h_j)  # [B, 1, V]
            logits_per_offset.append(logits_j.squeeze(1))  # [B, V]
            if j < K - 1:
                # Greedy: feed argmax of this offset's logits into next offset.
                next_token = logits_j.argmax(dim=-1)  # [B, 1]
                prev_embed = self.embeddings(next_token)  # [B, 1, D]
        return torch.stack(logits_per_offset, dim=1)  # [B, K, V]

    def _explicit_block_latent_pred_hidden_to_logits(self, pred_hidden, S, K, prev_token_embeds=None):
        """Map a position-major pred_hidden ``[B, S*K, D]`` to logits
        ``[B, K, S, V]`` aligned with ``block_targets [B, K, S]``.

        Dispatches on ``self._block_latent_head_type``:
          * ``linear`` (default): apply shared ``Linear(D, V)`` to all
            positions at once; byte-identical to pre-E2 behavior.
          * ``per_offset_*``: route each offset's hidden through its own
            head (see ``_apply_block_latent_head_per_offset``).
          * ``per_offset_tf_*`` (E4 TF heads): additionally require
            ``prev_token_embeds`` of shape ``[B, S, K, D]`` so each offset
            j's head can condition on x_{t+j-1}'s embedding.

        Steps after head application are unchanged: ``[B, S*K, V]`` ->
        ``[B, S, K, V]`` (position-major) -> permute -> ``[B, K, S, V]``.
        """
        B, P, _ = pred_hidden.shape
        if P != S * K:
            raise ValueError(
                f"pred_hidden length mismatch: expected S*K={S*K}, got {P}"
            )
        if self._block_latent_head_type == "linear":
            logits = self.block_latent_token_head(pred_hidden)  # [B, S*K, V]
        else:
            logits = self._apply_block_latent_head_per_offset(
                pred_hidden, S=S, K=K, prev_token_embeds=prev_token_embeds,
            )
        logits = logits.reshape(B, S, K, self.vocab_size)   # [B, S, K, V]
        logits = logits.permute(0, 2, 1, 3).contiguous()    # [B, K, S, V]
        return logits

    def _run_explicit_block_latent_mcmc(
        self,
        real_embeddings_input,
        predicted_tokens,
        block_size,
        start_pos=0,
        learning=True,
        return_raw_logits=False,
        no_randomness=True,
        alpha=None,
        langevin_dynamics_noise_std=None,
        mcmc_steps=None,
        return_pred_hidden=False,
    ):
        """MCMC loop dedicated to the explicit-block-latent modes.

        Operates on a flattened [B, P, V] latent where ``P`` is either
        ``S * K`` (training layout, with S = real_embeddings_input.shape[1])
        or ``K`` (inference layout). The trunk picks the correct sequence
        layout by inspecting ``pred_len``; see
        ``EBTTimeConcat._forward_explicit_block_latent`` /
        ``EBTDefault._forward_explicit_block_latent``.

        Energies of all P pred latents are summed before back-propagation,
        consistent with EBT/MCMC where all pred-token energies contribute
        to the gradient on the latents.

        IMPORTANT: this function is NOT used by mtp_mcmc; mtp_mcmc still
        uses ``_run_mcmc_on_given_pred_tokens`` which is left untouched.
        """
        if self._block_mode not in EXPLICIT_BLOCK_LATENT_MODES:
            raise ValueError(
                "_run_explicit_block_latent_mcmc must only be called under "
                f"block_mode in {EXPLICIT_BLOCK_LATENT_MODES}; got {self._block_mode!r}"
            )
        if predicted_tokens.dim() != 3 or predicted_tokens.shape[-1] != self.vocab_size:
            raise ValueError(
                "predicted_tokens must be [B, P, V] with V == vocab_size; "
                f"got shape={tuple(predicted_tokens.shape)}, vocab_size={self.vocab_size}"
            )
        K = int(block_size)
        if K <= 0:
            raise ValueError(f"block_size must be > 0, got {K}")
        B, P, V = predicted_tokens.shape
        S = real_embeddings_input.shape[1]
        model_dtype = self.embeddings.weight.dtype
        if P == S * K:
            layout = "training"
        elif P == K:
            layout = "inference"
        else:
            raise ValueError(
                "explicit-block-latent MCMC requires pred_len in {S*K, K}; "
                f"got pred_len={P}, S={S}, K={K}."
            )

        if alpha is None:
            alpha = torch.clamp(self.alpha, min=0.0001).float()
        if langevin_dynamics_noise_std is None:
            # Keep noise detached / model-typed so we do not inject a float32
            # node into the create_graph=True MCMC graph.
            langevin_dynamics_noise_std = torch.clamp(
                self.langevin_dynamics_noise_std, min=0.000001
            ).detach().to(model_dtype)
        if mcmc_steps is None:
            mcmc_steps = self._build_mcmc_steps(no_randomness=no_randomness)

        if return_pred_hidden and not self._transformer_accepts_return_pred_hidden:
            raise NotImplementedError(
                f"Transformer {type(self.transformer).__name__} does not support return_pred_hidden=True"
            )

        transformer_body = getattr(self, "transformer_eager", self.transformer)
        predicted_distributions = []
        predicted_energies = []
        predicted_hiddens = []

        with torch.set_grad_enabled(True):
            for i, mcmc_step in enumerate(mcmc_steps):
                if self.hparams.no_mcmc_detach:
                    predicted_tokens.requires_grad_().reshape(B, P, V)
                else:
                    predicted_tokens = predicted_tokens.detach().requires_grad_().reshape(B, P, V)

                if self.hparams.langevin_dynamics_noise != 0:
                    ld_noise = torch.randn_like(predicted_tokens.detach()) * langevin_dynamics_noise_std
                    predicted_tokens = predicted_tokens + ld_noise

                if self.hparams.normalize_initial_condition:
                    if self.hparams.normalize_initial_condition_only_first_step:
                        if mcmc_step == 0:
                            predicted_tokens = self.softmax(predicted_tokens)
                    else:
                        predicted_tokens = self.softmax(predicted_tokens)

                    if self.hparams.vocab_to_embed_uses_prob_dist:
                        predicted_embeddings = torch.matmul(predicted_tokens, self.embeddings.weight)
                    else:
                        predicted_embeddings = self.vocab_to_embed(predicted_tokens)
                else:
                    predicted_embeddings = self.vocab_to_embed(predicted_tokens)

                # all_embeddings: [B, S + P, D]
                all_embeddings = torch.cat((real_embeddings_input, predicted_embeddings), dim=1)
                trunk_kwargs = dict(
                    start_pos=start_pos,
                    mcmc_step=mcmc_step,
                    context_len=S,
                    pred_len=P,
                    block_size=K,
                    block_mode=self._block_mode,
                )
                energy_preds = transformer_body(all_embeddings, **trunk_kwargs)
                # energy_preds: [B, P, 1]
                energy_preds_flat = energy_preds.reshape(-1, 1)
                predicted_energies.append(energy_preds_flat)

                with torch.amp.autocast(device_type="cuda", enabled=False):
                    # bf16 safety (from origin): force the gradient computation
                    # to run in fp32 to avoid numerical issues with bf16.
                    energy_f32 = energy_preds_flat.float()
                    # create_graph for THIS MCMC step's gradient. Matches the
                    # original logic: truncate_mcmc → only last step has graph;
                    # otherwise → always (when learning).
                    if self.hparams.truncate_mcmc:
                        create_graph_step = learning and (i == (len(mcmc_steps) - 1))
                    else:
                        create_graph_step = learning

                    if getattr(self, "_mcmc_causal_detach", False):
                        # E3.1: per-offset DIAGONAL gradient.
                        #
                        # In blockwise (causal intra-block) mode, the joint
                        # ``energy.sum().backward()`` gives z_{t,j} a gradient
                        # contribution from EVERY e_{t,k>=j} — because z_{t,k>=j}
                        # attends to z_{t,j} via the causal intra-block mask.
                        # This branch keeps the FORWARD causal attention intact
                        # but cuts the BACKWARD gradient: for each offset j we
                        # run a separate backward on ``e_{*,j}.sum()`` and keep
                        # only the j-th row of the resulting gradient.
                        # Cost: K backward passes per MCMC step instead of 1.
                        # Uses the fp32 energy_f32 reshape so bf16 safety is
                        # preserved end-to-end.
                        energy_per_pos_f32 = energy_f32.squeeze(-1).reshape(B, P // K, K)
                        per_offset_diag_slices = []
                        for j in range(K):
                            is_last_j = (j == K - 1)
                            retain_graph_arg = True if (not is_last_j) else create_graph_step
                            g_j = torch.autograd.grad(
                                [energy_per_pos_f32[:, :, j].sum()],
                                [predicted_tokens],
                                create_graph=create_graph_step,
                                retain_graph=retain_graph_arg,
                            )[0]  # [B, P, V]
                            g_j_v = g_j.reshape(B, P // K, K, V)
                            # Diagonal: keep only the j-th offset's row of g_j.
                            per_offset_diag_slices.append(g_j_v[:, :, j, :])  # [B, S, V]
                        # Stack to [B, S, K, V] then flatten back to [B, S*K, V].
                        predicted_tokens_grad = torch.stack(
                            per_offset_diag_slices, dim=2
                        ).reshape(B, P, V)
                    else:
                        predicted_tokens_grad = torch.autograd.grad(
                            [energy_f32.sum()], [predicted_tokens],
                            create_graph=create_graph_step,
                        )[0]

                if self.hparams.clamp_futures_grad:
                    min_and_max = self.hparams.clamp_futures_grad_max_change / (self.alpha.float())
                    predicted_tokens_grad = torch.clamp(predicted_tokens_grad, min=-min_and_max, max=min_and_max)

                if torch.isnan(predicted_tokens_grad).any() or torch.isinf(predicted_tokens_grad).any():
                    raise ValueError("NaN or Inf gradients detected during explicit-block-latent MCMC.")

                if getattr(self, "_mcmc_staggered", False):
                    # E3.2: staggered MCMC. At step i, only the first
                    # ``active_K = min(i+1, K)`` offsets actually receive an
                    # update; the rest are frozen at their previous-step value
                    # (== initial value for steps before they're activated).
                    # We implement this by zeroing the gradient on inactive
                    # offset slots. The forward joint-energy compute is
                    # unchanged so that the inactive z_{t,k>=active_K}'s
                    # initial value still informs e_{t,k} and the causal
                    # forward path z_{t,active_K..} ← z_{t,<active_K}.
                    active_K = min(i + 1, K)
                    if active_K < K:
                        # Mask of shape [1, 1, K, 1], broadcasts over [B, S, K, V]
                        mask = torch.ones(K, device=predicted_tokens.device,
                                          dtype=predicted_tokens_grad.dtype)
                        mask[active_K:] = 0.0
                        grad_v = predicted_tokens_grad.reshape(B, P // K, K, V)
                        predicted_tokens_grad = (grad_v * mask.view(1, 1, K, 1)).reshape(B, P, V)

                predicted_tokens = predicted_tokens - alpha * predicted_tokens_grad
                if self.hparams.absolute_clamp != 0.0:
                    predicted_tokens = torch.clamp(
                        predicted_tokens, min=-self.hparams.absolute_clamp, max=self.hparams.absolute_clamp
                    )
                if self.hparams.sharpen_predicted_distribution != 0.0:
                    predicted_tokens = predicted_tokens / self.hparams.sharpen_predicted_distribution

                if return_pred_hidden:
                    # Re-run trunk on the post-update latent so pred_hidden
                    # depends on alpha (mirrors the dense-blockwise pattern
                    # used by mtp_mcmc).
                    post_pred_embeddings = self._logits_to_pred_embeddings(predicted_tokens, mcmc_step)
                    post_all = torch.cat((real_embeddings_input, post_pred_embeddings), dim=1)
                    post_kwargs = dict(trunk_kwargs)
                    post_kwargs["return_pred_hidden"] = True
                    post_out = transformer_body(post_all, **post_kwargs)
                    if isinstance(post_out, tuple) and len(post_out) == 2:
                        _, pred_hidden = post_out
                    else:
                        raise RuntimeError(
                            "Trunk did not return (energies, pred_hidden) for explicit-block-latent path"
                        )
                    predicted_hiddens.append(pred_hidden)

                if return_raw_logits:
                    predicted_tokens_for_loss = predicted_tokens
                else:
                    predicted_tokens_for_loss = self.log_softmax(predicted_tokens).reshape(-1, self.vocab_size)
                predicted_distributions.append(predicted_tokens_for_loss)

        if return_pred_hidden:
            return predicted_tokens, predicted_distributions, predicted_energies, predicted_hiddens
        return predicted_tokens, predicted_distributions, predicted_energies

    def forward_explicit_block_latent_hidden(self, input_ids, block_size, no_randomness):
        """Training-time pred-hidden extraction for the new modes.

        Returns:
            pred_hiddens_per_step: list[Tensor] each [B, S*K, D] in
                position-major order.
            predicted_energies: list[Tensor] each [B*S*K, 1].
        """
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids [B, S], got shape {tuple(input_ids.shape)}")
        if self._block_mode not in EXPLICIT_BLOCK_LATENT_MODES:
            raise ValueError(
                "forward_explicit_block_latent_hidden only valid for block_mode in "
                f"{EXPLICIT_BLOCK_LATENT_MODES}; got {self._block_mode!r}"
            )
        K = int(block_size)
        if K <= 0:
            raise ValueError(f"block_size must be > 0, got {K}")
        S = int(input_ids.shape[1])

        real_embeddings_input = self.embeddings(input_ids)  # [B, S, D]
        alpha = torch.clamp(self.alpha, min=0.0001)
        langevin_dynamics_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)
        # Initialize S*K latents. Position-major flattening matches the trunk
        # mask / freq-index helpers in utils.py.
        predicted_tokens = self.corrupt_embeddings(real_embeddings_input, target_length=S * K)

        _, _, predicted_energies, pred_hiddens_per_step = self._run_explicit_block_latent_mcmc(
            real_embeddings_input=real_embeddings_input,
            predicted_tokens=predicted_tokens,
            block_size=K,
            start_pos=0,
            learning=True,
            return_raw_logits=True,
            no_randomness=no_randomness,
            alpha=alpha,
            langevin_dynamics_noise_std=langevin_dynamics_noise_std,
            return_pred_hidden=True,
        )
        return pred_hiddens_per_step, predicted_energies

    def forward_explicit_block_latent_logits(
        self, input_ids, block_size, no_randomness, return_hidden=False,
        prev_token_embeds=None,
    ):
        """Training-time per-MCMC-step logits for the new modes.

        Returns:
            logits_per_step: list[Tensor] each [B, K, S, V] aligned with
                ``block_targets [B, K, S]`` (so the last axis is the vocab
                axis). Each latent z_{t,j} is mapped through
                ``block_latent_token_head`` independently.
            predicted_energies: list[Tensor] each [B*S*K, 1].
            pred_hiddens_per_step (only if return_hidden=True): list of
                [B, S*K, D].

        ``prev_token_embeds`` is required when the head type is in
        ``TF_HEAD_TYPES`` and is forwarded unchanged to every MCMC step's
        head application (the teacher-forced previous-offset embedding is
        independent of the MCMC iterate so we build it once and reuse).
        """
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids [B, S], got shape {tuple(input_ids.shape)}")
        K = int(block_size)
        if K <= 0:
            raise ValueError(f"block_size must be > 0, got {K}")
        S = int(input_ids.shape[1])

        pred_hiddens_per_step, predicted_energies = self.forward_explicit_block_latent_hidden(
            input_ids=input_ids,
            block_size=K,
            no_randomness=no_randomness,
        )

        logits_per_step = []
        for pred_hidden in pred_hiddens_per_step:
            logits = self._explicit_block_latent_pred_hidden_to_logits(
                pred_hidden, S=S, K=K, prev_token_embeds=prev_token_embeds,
            )
            logits_per_step.append(logits)

        if return_hidden:
            return logits_per_step, predicted_energies, pred_hiddens_per_step
        return logits_per_step, predicted_energies

    def _forward_explicit_block_latent_inference(
        self,
        real_embeddings_input,
        block_size,
        start_pos=0,
        learning=False,
        return_raw_logits=False,
        no_randomness=True,
    ):
        """Inference-time forward for new modes via the C+K trunk layout.

        Builds K future latents anchored at the end of context, runs MCMC,
        and produces per-step predicted distributions of shape ``[B, K, V]``
        (one prediction per latent), matching the legacy ``(predicted_distributions,
        predicted_energies)`` return shape used by ``call_model_forward_decode``.

        After MCMC, the K latent hidden states are mapped through the single-
        token head ``block_latent_token_head`` to yield the *real* per-latent
        logits (since MCMC operates on the prob-dist surrogate, the trunk's
        pred_hidden is the natural output for downstream sampling).
        """
        K = int(block_size)
        if K <= 0:
            raise ValueError(f"block_size must be > 0, got {K}")
        if real_embeddings_input.dim() != 3:
            raise ValueError(
                f"Expected real_embeddings_input [B, C, D], got shape {tuple(real_embeddings_input.shape)}"
            )

        alpha = torch.clamp(self.alpha, min=0.0001)
        langevin_dynamics_noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)
        # Initialize exactly K latents (one per future block offset).
        predicted_tokens = self.corrupt_embeddings(real_embeddings_input, target_length=K)

        _, _, predicted_energies, pred_hiddens_per_step = self._run_explicit_block_latent_mcmc(
            real_embeddings_input=real_embeddings_input,
            predicted_tokens=predicted_tokens,
            block_size=K,
            start_pos=start_pos,
            learning=learning,
            return_raw_logits=return_raw_logits,
            no_randomness=no_randomness,
            alpha=alpha,
            langevin_dynamics_noise_std=langevin_dynamics_noise_std,
            return_pred_hidden=True,
        )

        # Map each step's [B, K, D] pred_hidden through the block-latent head
        # to produce [B, K, V] logits. These are the inference logits used
        # by sampling code; they are consistent with training where each
        # latent is supervised by exactly one target token.
        #
        # E2/E4: dispatch on head type.
        #   * `linear` (default): shared Linear over [B, K, D].
        #   * `per_offset_*` (non-TF): S=1 degenerate position-major application.
        #   * `per_offset_tf_*` (TF heads): each offset's head conditions on
        #     the realized previous-offset token. At inference we don't have
        #     ground truth, so we run a sequential greedy-argmax loop within
        #     the block (anchor = last context embedding), and apply the
        #     SAME sequential head to every MCMC step (each step gets fresh
        #     argmaxes from its own pred_hidden).
        anchor_embed = real_embeddings_input[:, -1:, :]  # [B, 1, D]
        logits_per_step = []
        for pred_hidden in pred_hiddens_per_step:
            if pred_hidden.shape[1] != K:
                raise RuntimeError(
                    f"pred_hidden length mismatch in inference: expected K={K}, got {pred_hidden.shape[1]}"
                )
            if self._block_latent_head_type == "linear":
                logits_per_step.append(self.block_latent_token_head(pred_hidden))  # [B, K, V]
            elif self._block_latent_head_type in TF_HEAD_TYPES:
                logits_per_step.append(
                    self._apply_block_latent_head_sequential_inference(
                        pred_hidden, K=K, anchor_embed=anchor_embed,
                    )
                )  # [B, K, V]
            else:
                # pred_hidden is [B, K, D] == position-major flatten of [B, S=1, K, D].
                logits_per_step.append(
                    self._apply_block_latent_head_per_offset(pred_hidden, S=1, K=K)
                )  # [B, K, V]

        return logits_per_step, predicted_energies

    def ebt_refine_block_explicit_block_latent(
        self,
        context_ids,
        draft_block_ids,
        refine_steps=None,
        init_logit_scale=8.0,
        start_pos=0,
        learning=False,
    ):
        """Block refinement entry point for the new explicit-block-latent
        modes. Mirrors :meth:`ebt_refine_block_fast` but uses the C+K trunk
        layout instead of the symmetric ``[ctx, draft[:-1]]`` prefix trick.

        Inputs:
            context_ids:     [B, C]  context token ids (real, not optimized)
            draft_block_ids: [B, K]  draft token ids used to seed the K
                                     future latents.

        Returns:
            refined_block_logits: [B, K, V]
            refined_block_ids:    [B, K]
        """
        if self._block_mode not in EXPLICIT_BLOCK_LATENT_MODES:
            raise ValueError(
                "ebt_refine_block_explicit_block_latent only valid for block_mode in "
                f"{EXPLICIT_BLOCK_LATENT_MODES}; got {self._block_mode!r}"
            )
        if draft_block_ids.shape[1] == 0:
            empty_logits = torch.empty(
                draft_block_ids.shape[0], 0, self.vocab_size,
                device=draft_block_ids.device, dtype=self.embeddings.weight.dtype,
            )
            return empty_logits, draft_block_ids
        if context_ids.shape[1] == 0:
            raise ValueError("ebt_refine_block_explicit_block_latent requires at least one context token")

        K = int(draft_block_ids.shape[1])
        real_embeddings_input = self.embeddings(context_ids)  # [B, C, D]
        # Seed K latents from the draft tokens: peaked logits then run MCMC.
        block_logits = self._draft_block_ids_to_initial_logits(
            draft_block_ids, init_logit_scale=init_logit_scale,
        ).detach()
        init_block_logits = block_logits.detach().clone()

        alpha = torch.clamp(self.alpha, min=0.0001)
        noise_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)

        requested_steps = int(refine_steps) if refine_steps is not None else int(self.hparams.mcmc_num_steps)
        if requested_steps <= 0:
            refined_block_ids = torch.argmax(block_logits, dim=-1)
            return block_logits, refined_block_ids
        effective_steps = min(requested_steps, int(self.hparams.mcmc_num_steps))
        mcmc_steps = list(range(effective_steps))
        diagnose = bool(getattr(self.hparams, "infer_block_diagnose", False))

        def _block_energy_mean(logits, step_idx):
            with torch.no_grad():
                block_embeds = self._logits_to_pred_embeddings(logits, step_idx)
                all_embeddings = torch.cat((real_embeddings_input, block_embeds), dim=1)
                energy_preds = self.transformer(
                    all_embeddings,
                    start_pos=start_pos,
                    mcmc_step=step_idx,
                    context_len=real_embeddings_input.shape[1],
                    pred_len=block_embeds.shape[1],
                    block_size=K,
                    block_mode=self._block_mode,
                )
                return energy_preds.mean().item()

        initial_energy_mean = _block_energy_mean(init_block_logits, mcmc_steps[0])
        last_grad_norm = 0.0

        with torch.set_grad_enabled(True):
            for i, mcmc_step in enumerate(mcmc_steps):
                block_logits = block_logits.detach().requires_grad_()
                cur_block_logits = block_logits
                if self.hparams.langevin_dynamics_noise != 0:
                    ld_noise = torch.randn_like(cur_block_logits.detach()) * noise_std
                    cur_block_logits = cur_block_logits + ld_noise

                block_embeds = self._logits_to_pred_embeddings(cur_block_logits, mcmc_step)
                all_embeddings = torch.cat((real_embeddings_input, block_embeds), dim=1)
                energy_preds = self.transformer(
                    all_embeddings,
                    start_pos=start_pos,
                    mcmc_step=mcmc_step,
                    context_len=real_embeddings_input.shape[1],
                    pred_len=block_embeds.shape[1],
                    block_size=K,
                    block_mode=self._block_mode,
                )
                energy_block = energy_preds.reshape(-1, 1)

                block_grad = torch.autograd.grad(
                    [energy_block.sum()], [cur_block_logits], create_graph=learning
                )[0]
                last_grad_norm = block_grad.norm().item()
                if self.hparams.clamp_futures_grad:
                    min_and_max = self.hparams.clamp_futures_grad_max_change / (self.alpha)
                    block_grad = torch.clamp(block_grad, min=-min_and_max, max=min_and_max)
                if torch.isnan(block_grad).any() or torch.isinf(block_grad).any():
                    raise ValueError("NaN or Inf gradients detected during explicit-block-latent block refinement.")

                block_logits = cur_block_logits - alpha * block_grad
                if self.hparams.absolute_clamp != 0.0:
                    block_logits = torch.clamp(
                        block_logits, min=-self.hparams.absolute_clamp, max=self.hparams.absolute_clamp,
                    )
                if self.hparams.sharpen_predicted_distribution != 0.0:
                    block_logits = block_logits / self.hparams.sharpen_predicted_distribution
                block_logits = block_logits.detach()

        # After MCMC, reconstruct the *real* per-latent token logits from
        # the post-update pred_hidden (consistent with training where each
        # latent is decoded by `block_latent_token_head`).
        with torch.no_grad():
            block_embeds = self._logits_to_pred_embeddings(block_logits, mcmc_steps[-1])
            all_embeddings = torch.cat((real_embeddings_input, block_embeds), dim=1)
            _, pred_hidden = self.transformer(
                all_embeddings,
                start_pos=start_pos,
                mcmc_step=mcmc_steps[-1],
                context_len=real_embeddings_input.shape[1],
                pred_len=block_embeds.shape[1],
                block_size=K,
                block_mode=self._block_mode,
                return_pred_hidden=True,
            )
            # E2/E4: dispatch on head type. Same S=1 inference layout as
            # `_forward_explicit_block_latent_inference` above. For TF heads
            # we use the sequential greedy-argmax helper with the last context
            # token as the anchor previous-offset embedding.
            if self._block_latent_head_type == "linear":
                refined_block_logits = self.block_latent_token_head(pred_hidden)  # [B, K, V]
            elif self._block_latent_head_type in TF_HEAD_TYPES:
                refined_block_logits = self._apply_block_latent_head_sequential_inference(
                    pred_hidden, K=K, anchor_embed=real_embeddings_input[:, -1:, :],
                )  # [B, K, V]
            else:
                refined_block_logits = self._apply_block_latent_head_per_offset(
                    pred_hidden, S=1, K=K,
                )  # [B, K, V]
            refined_block_ids = torch.argmax(refined_block_logits, dim=-1)

        if diagnose:
            final_energy_mean = _block_energy_mean(block_logits, mcmc_steps[-1])
            delta_norm = (block_logits - init_block_logits).norm().item()
            max_show = min(2, draft_block_ids.shape[0])
            for b in range(max_show):
                print(f"draft_block_ids[{b}]: {draft_block_ids[b].tolist()}", flush=True)
                print(f"refined_block_ids[{b}]: {refined_block_ids[b].tolist()}", flush=True)
            print(f"||refined_logits - init_logits||: {delta_norm:.6f}", flush=True)
            print(f"block initial energy: {initial_energy_mean:.6f}", flush=True)
            print(f"block final energy: {final_energy_mean:.6f}", flush=True)
            print(f"block logits grad.norm(): {last_grad_norm:.6f}", flush=True)

        return refined_block_logits, refined_block_ids

    def _forward_loss_wrapper_explicit_block_latent(self, x, phase, token_bytes, no_randomness):
        """Blockwise loss path for the new explicit-block-latent modes.

        Constraints versus mtp_mcmc:
          * Each future latent z_{t,j} corresponds to exactly one target
            token: ``block_targets[:, j-1, t-1]`` (1-indexed t,j; equivalently
            the (j, t) entry of the [B, K, S] tensor).
          * Logits come from ``block_latent_token_head`` applied to each
            latent independently — there is no joint multi-offset projection.
          * Attention semantics inside the trunk are governed by ``self._block_mode``
            (future_latent_non_causal vs blockwise) via the new mask helpers.

        The mtp_mcmc loss path below is left completely untouched.
        """
        if not isinstance(x, dict):
            raise ValueError("Expected dense blockwise batch to be a dict with keys: input_ids and block_targets")

        def _maybe_squeeze_loader_dim(tensor):
            if tensor is None:
                return None
            if isinstance(tensor, torch.Tensor) and tensor.dim() > 1 and tensor.shape[0] == 1:
                return tensor.squeeze(dim=0)
            return tensor

        input_ids = _maybe_squeeze_loader_dim(x["input_ids"])
        block_targets = _maybe_squeeze_loader_dim(x["block_targets"])
        target_offsets = _maybe_squeeze_loader_dim(x.get("target_offsets"))

        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids to be 2D [B, S_eff], got shape {tuple(input_ids.shape)}")
        if block_targets.dim() != 3:
            raise ValueError(f"Expected block_targets to be 3D [B, K, S_eff], got shape {tuple(block_targets.shape)}")
        if block_targets.shape[0] != input_ids.shape[0]:
            raise ValueError(
                f"Batch mismatch between input_ids and block_targets: {tuple(input_ids.shape)} vs {tuple(block_targets.shape)}"
            )
        if block_targets.shape[2] != input_ids.shape[1]:
            raise ValueError(
                f"S_eff mismatch between input_ids and block_targets: input_ids.shape[1]={input_ids.shape[1]}, "
                f"block_targets.shape[2]={block_targets.shape[2]}"
            )

        num_offsets = int(block_targets.shape[1])  # = K
        S_eff = int(input_ids.shape[1])
        if target_offsets is None:
            target_offsets = torch.arange(1, num_offsets + 1, device=input_ids.device, dtype=torch.long)

        # E4: teacher-forced (TF) heads condition each offset j>=1's head on
        # the embedding of the realized previous-offset target token. Build
        # once outside the MCMC loop since the embedding is independent of
        # the MCMC iterate; the MCMC dynamics still act on the latent (z)
        # surrogate as in non-TF heads.
        if self._block_latent_head_type in TF_HEAD_TYPES:
            prev_token_embeds = self._build_prev_token_embeds_training(
                input_ids=input_ids, block_targets=block_targets, K=num_offsets,
            )
        else:
            prev_token_embeds = None

        logits_per_step, predicted_energies, pred_hiddens_per_step = self.forward_explicit_block_latent_logits(
            input_ids=input_ids,
            block_size=num_offsets,
            no_randomness=no_randomness,
            return_hidden=True,
            prev_token_embeds=prev_token_embeds,
        )

        # E1.1: per-offset loss weighting. `self._offset_loss_weights` is parsed
        # once at __init__ (normalized to sum to 1). For default uniform
        # weights, ``sum_j (1/K) * ce_j == mean_{b,j,t} ce`` so the loss-driving
        # value matches the pre-E1 ``F.cross_entropy(flat_logits, flat_targets)``
        # exactly (modulo per-offset ignore_index masking, which our dataloader
        # currently never triggers).
        if (
            self._offset_loss_weights is None
            or len(self._offset_loss_weights) != num_offsets
        ):
            # Robust to a checkpoint trained with K' != current K: fall back to
            # uniform 1/K so behavior is sensible without forcing a re-parse.
            weights = [1.0 / num_offsets] * num_offsets
        else:
            weights = self._offset_loss_weights

        # Flatten targets to match the [B, K, S, V] -> [B*K*S, V] layout.
        # Only used downstream for BPB token-count bookkeeping.
        next_token_indices = block_targets.reshape(-1)
        reconstruction_loss = 0
        total_mcmc_steps = len(predicted_energies)
        final_cce_loss_mean = None  # unweighted mean over (b, j, t) — for BPB

        # E1.2: per-offset diagnostics captured at initial / final MCMC step.
        per_offset_ce_initial = [None] * num_offsets
        per_offset_ce_final = [None] * num_offsets
        per_offset_energy_initial = [None] * num_offsets
        per_offset_energy_final = [None] * num_offsets

        for mcmc_step, (predicted_distribution, predicted_energy) in enumerate(
            zip(logits_per_step, predicted_energies)
        ):
            # predicted_distribution: [B, K, S_eff, V]
            # Compute per-offset CE so that (a) loss is per-offset weighted and
            # (b) we can log offset-resolved diagnostics with no extra forward.
            if self.hparams.soften_target_prob_dist != 0.0:
                if total_mcmc_steps <= 1:
                    label_smoothing = 0.0
                else:
                    label_smoothing = (
                        ((total_mcmc_steps - 1) - mcmc_step)
                        / (total_mcmc_steps - 1)
                        * self.hparams.soften_target_prob_dist
                    )
            else:
                label_smoothing = 0.0

            step_per_offset_ce = []
            for j in range(num_offsets):
                logits_j = predicted_distribution[:, j, :, :].reshape(-1, self.vocab_size)
                targets_j = block_targets[:, j, :].reshape(-1)
                if self.hparams.soften_target_prob_dist != 0.0:
                    ce_j = F.cross_entropy(
                        logits_j, targets_j,
                        label_smoothing=label_smoothing, ignore_index=-1,
                    )
                else:
                    ce_j = F.nll_loss(
                        self.log_softmax(logits_j), targets_j, ignore_index=-1,
                    )
                step_per_offset_ce.append(ce_j)

            cce_loss_weighted = sum(weights[j] * step_per_offset_ce[j] for j in range(num_offsets))
            # Backward-compat "mean over flattened tokens" scalar, used for
            # PPL / BPB so those metrics stay comparable to the pre-E1 runs.
            cce_loss_mean = sum(step_per_offset_ce) / num_offsets

            if self.hparams.truncate_mcmc:
                if mcmc_step == (total_mcmc_steps - 1):
                    reconstruction_loss = cce_loss_weighted
                    final_reconstruction_loss = cce_loss_weighted.detach()
                    final_cce_loss_mean = cce_loss_mean.detach()
            else:
                reconstruction_loss = reconstruction_loss + cce_loss_weighted
                if mcmc_step == (total_mcmc_steps - 1):
                    final_reconstruction_loss = cce_loss_weighted.detach()
                    final_cce_loss_mean = cce_loss_mean.detach()
                    reconstruction_loss = reconstruction_loss / total_mcmc_steps

            if mcmc_step == 0:
                initial_loss = cce_loss_weighted.detach()
                initial_pred_energies = predicted_energy.squeeze().mean().detach()
                for j in range(num_offsets):
                    per_offset_ce_initial[j] = step_per_offset_ce[j].detach()
            if mcmc_step == (total_mcmc_steps - 1):
                final_pred_energies = predicted_energy.squeeze().mean().detach()
                for j in range(num_offsets):
                    per_offset_ce_final[j] = step_per_offset_ce[j].detach()

            # Per-offset energy diagnostics: predicted_energy is [B*S*K, 1] in
            # position-major order, so reshape to [B, S, K] for per-offset
            # mean. (At the inference layout the same code path is not used.)
            if mcmc_step == 0 or mcmc_step == (total_mcmc_steps - 1):
                try:
                    e_bsk = predicted_energy.reshape(-1, S_eff, num_offsets)
                except RuntimeError:
                    e_bsk = None
                if e_bsk is not None:
                    for j in range(num_offsets):
                        e_j_mean = e_bsk[:, :, j].mean().detach()
                        if mcmc_step == 0:
                            per_offset_energy_initial[j] = e_j_mean
                        else:
                            per_offset_energy_final[j] = e_j_mean

        initial_final_pred_energies_gap = initial_pred_energies - final_pred_energies
        ppl_loss = torch.exp(final_cce_loss_mean).detach()  # PPL based on unweighted mean (BC)
        total_loss = self.hparams.reconstruction_coeff * reconstruction_loss
        contrastive_loss = 0.0

        if token_bytes is not None:
            # Origin's calculate_bpb_score now returns (bpb, nats, bytes) and is
            # DDP-aware. We still pass the scalar `final_cce_loss_mean` (HEAD
            # variable name) — the backward-compat scalar branch inside
            # calculate_bpb_score multiplies by num_valid_tokens which gives
            # the same per-rank result as before. For more accurate DDP
            # aggregation we'd need to thread a per-token loss tensor through
            # the MCMC loop; leaving as TODO since current XXS runs are 1-GPU.
            bpb_loss, bpb_nats, bpb_bytes = calculate_bpb_score(
                next_token_indices, final_cce_loss_mean, token_bytes,
            )
        else:
            bpb_loss = 0
            bpb_nats = 0
            bpb_bytes = 0

        # Per-offset loss / PPL / BPB logging on the FINAL MCMC step.
        # `offset_{j}_loss` keeps the unweighted per-offset CE (kept identical
        # to mtp_mcmc dashboard keys). `offset_{j}_ppl` / `offset_{j}_bpb` are
        # NEW: they let us directly compare offset 1 of a blockwise run
        # against a non-blockwise `val/bpb` (which only predicts offset 1),
        # which is the apples-to-apples comparison the user wants. The model
        # still does the parallel K-offset forward — we just slice the j-th
        # latent's logits and the j-th target slice from `block_targets`.
        offset_loss_log_dict = {}
        for offset_idx in range(num_offsets):
            offset_value = int(target_offsets[offset_idx].item())
            # Per-offset loss + ppl (HEAD) + bpb via origin's 3-tuple. We
            # use the scalar ce_j we already accumulated in the MCMC loop
            # (per_offset_ce_final) rather than re-running cross_entropy on
            # final_step_logits (which doesn't exist in the explicit-block-
            # latent path); calculate_bpb_score handles scalar via its
            # backward-compat branch.
            ce_j = per_offset_ce_final[offset_idx]
            offset_loss_log_dict[f"offset_{offset_value}_loss"] = ce_j
            offset_loss_log_dict[f"offset_{offset_value}_ppl"] = torch.exp(ce_j).detach()
            if token_bytes is not None:
                offset_targets_flat = block_targets[:, offset_idx, :].reshape(-1)
                offset_bpb, offset_bpb_nats, offset_bpb_bytes = calculate_bpb_score(
                    offset_targets_flat, ce_j, token_bytes,
                )
                offset_loss_log_dict[f"offset_{offset_value}_bpb"] = offset_bpb
                offset_loss_log_dict[f"offset_{offset_value}_bpb_nats"] = offset_bpb_nats
                offset_loss_log_dict[f"offset_{offset_value}_bpb_bytes"] = offset_bpb_bytes

        # E1.2: extra diagnostics — initial CE, initial/final energy, energy
        # gap per offset, and the effective loss weight applied to each offset.
        # These are NEW keys; they coexist with existing offset_*_loss keys.
        offset_diag_log_dict = {}
        device = block_targets.device
        for offset_idx in range(num_offsets):
            offset_value = int(target_offsets[offset_idx].item())
            if per_offset_ce_initial[offset_idx] is not None:
                offset_diag_log_dict[f"offset_{offset_value}_initial_ce"] = per_offset_ce_initial[offset_idx]
            if per_offset_energy_initial[offset_idx] is not None:
                offset_diag_log_dict[f"offset_{offset_value}_initial_energy"] = per_offset_energy_initial[offset_idx]
            if per_offset_energy_final[offset_idx] is not None:
                offset_diag_log_dict[f"offset_{offset_value}_final_energy"] = per_offset_energy_final[offset_idx]
            if (
                per_offset_energy_initial[offset_idx] is not None
                and per_offset_energy_final[offset_idx] is not None
            ):
                offset_diag_log_dict[f"offset_{offset_value}_energy_gap"] = (
                    per_offset_energy_initial[offset_idx] - per_offset_energy_final[offset_idx]
                )
            offset_diag_log_dict[f"offset_{offset_value}_loss_weight"] = torch.tensor(
                float(weights[offset_idx]), device=device,
            )

        if getattr(self.hparams, "debug_blockwise_shapes", False):
            print(
                f"[blockwise-debug][phase={phase}][block_mode={self._block_mode}] "
                f"explicit_block_latent=True, input_ids.shape={tuple(input_ids.shape)}, "
                f"block_targets.shape={tuple(block_targets.shape)}, "
                f"offsets={target_offsets.tolist()}, "
                f"logits_last_step.shape={tuple(logits_per_step[-1].shape)}, "
                f"alpha={self.alpha.detach().item():.6f}",
                flush=True,
            )
            print(
                f"[blockwise-debug][phase={phase}][block_mode={self._block_mode}] "
                f"post_update_pred_hidden_last_step.shape={tuple(pred_hiddens_per_step[-1].shape)} "
                f"(position-major flatten of [B, S_eff={S_eff}, K={num_offsets}, D])",
                flush=True,
            )
            print(
                f"[blockwise-debug][phase={phase}][block_mode={self._block_mode}] "
                f"head_type={self._block_latent_head_type} "
                f"offset_weights={[round(w, 6) for w in weights]} "
                f"aggregated_loss={total_loss.detach().item():.6f}",
                flush=True,
            )
            print(
                f"[blockwise-debug][phase={phase}][block_mode={self._block_mode}] offset_losses="
                f"{ {k: round(v.item(), 6) for k, v in offset_loss_log_dict.items()} }",
                flush=True,
            )

        log_dict = {
            'loss': total_loss,
            'initial_loss': initial_loss,
            'final_step_loss': final_reconstruction_loss,
            'contrastive_loss': contrastive_loss,
            'initial_final_pred_energies_gap': initial_final_pred_energies_gap,
            'perplexity': ppl_loss,
            'bpb': bpb_loss,
            'bpb_nats': bpb_nats,
            'bpb_bytes': bpb_bytes,
        }
        log_dict.update(offset_loss_log_dict)
        log_dict.update(offset_diag_log_dict)
        return log_dict

    def forward_loss_wrapper(self, x, phase="train", token_bytes=None):
        no_randomness = False if phase == "train" else True
        training_objective = getattr(self.hparams, "training_objective", "dense_next_token")

        if training_objective == "blockwise":
            if self.mcmc_replay_buffer:
                raise NotImplementedError("mcmc_replay_buffer is not supported for training_objective=blockwise")
            if self.hparams.contrastive_loss:
                raise NotImplementedError("contrastive_loss is not supported for training_objective=blockwise")
            if self.hparams.execution_mode == "finetune":
                raise NotImplementedError("execution_mode=finetune is not supported for training_objective=blockwise yet")

            # block_mode dispatch within the blockwise training_objective:
            #   * mtp_mcmc        -> existing dense-blockwise + joint head path
            #     (kept byte-identical below).
            #   * future_latent_non_causal / blockwise -> dedicated path that
            #     uses block_latent_token_head and the K-future-latents-per-
            #     source-position semantic (see
            #     forward_explicit_block_latent_logits above).
            if self._block_mode in EXPLICIT_BLOCK_LATENT_MODES:
                return self._forward_loss_wrapper_explicit_block_latent(
                    x=x, phase=phase, token_bytes=token_bytes, no_randomness=no_randomness,
                )

            if not isinstance(x, dict):
                raise ValueError("Expected dense blockwise batch to be a dict with keys: input_ids and block_targets")

            def _maybe_squeeze_loader_dim(tensor):
                if tensor is None:
                    return None
                if isinstance(tensor, torch.Tensor) and tensor.dim() > 1 and tensor.shape[0] == 1:
                    return tensor.squeeze(dim=0)
                return tensor

            input_ids = _maybe_squeeze_loader_dim(x["input_ids"])
            block_targets = _maybe_squeeze_loader_dim(x["block_targets"])
            target_offsets = _maybe_squeeze_loader_dim(x.get("target_offsets"))

            if input_ids.dim() != 2:
                raise ValueError(f"Expected input_ids to be 2D [B, S_eff], got shape {tuple(input_ids.shape)}")
            if block_targets.dim() != 3:
                raise ValueError(f"Expected block_targets to be 3D [B, K, S_eff], got shape {tuple(block_targets.shape)}")
            if block_targets.shape[0] != input_ids.shape[0]:
                raise ValueError(
                    f"Batch mismatch between input_ids and block_targets: {tuple(input_ids.shape)} vs {tuple(block_targets.shape)}"
                )
            if block_targets.shape[2] != input_ids.shape[1]:
                raise ValueError(
                    f"S_eff mismatch between input_ids and block_targets: input_ids.shape[1]={input_ids.shape[1]}, block_targets.shape[2]={block_targets.shape[2]}"
                )

            num_offsets = int(block_targets.shape[1])
            seq_eff = int(input_ids.shape[1])
            if target_offsets is None:
                target_offsets = torch.arange(1, num_offsets + 1, device=input_ids.device, dtype=torch.long)

            multi_offset_logits_per_step, predicted_energies, pred_hiddens_per_step = self.forward_blockwise_dense_logits(
                input_ids,
                num_offsets=num_offsets,
                no_randomness=no_randomness,
                return_hidden=True,
            )
            next_token_indices = block_targets.reshape(-1)
            reconstruction_loss = 0
            total_mcmc_steps = len(predicted_energies)
            final_cce_loss = None

            for mcmc_step, (predicted_distribution, predicted_energy) in enumerate(zip(multi_offset_logits_per_step, predicted_energies)):
                predicted_distribution = predicted_distribution.reshape(-1, self.vocab_size)

                if self.hparams.soften_target_prob_dist != 0.0:
                    if total_mcmc_steps <= 1:
                        label_smoothing = 0.0
                    else:
                        label_smoothing = ((total_mcmc_steps - 1) - mcmc_step) / (total_mcmc_steps - 1) * self.hparams.soften_target_prob_dist
                    cce_loss = F.cross_entropy(
                        predicted_distribution,
                        next_token_indices,
                        label_smoothing=label_smoothing,
                        ignore_index=-1,
                    )
                else:
                    predicted_distribution = self.log_softmax(predicted_distribution)
                    cce_loss = F.nll_loss(predicted_distribution, next_token_indices, ignore_index=-1)

                if self.hparams.truncate_mcmc:
                    if mcmc_step == (total_mcmc_steps - 1):
                        reconstruction_loss = cce_loss
                        final_reconstruction_loss = cce_loss.detach()
                        final_cce_loss = cce_loss.detach()
                else:
                    reconstruction_loss += cce_loss
                    if mcmc_step == (total_mcmc_steps - 1):
                        final_reconstruction_loss = cce_loss.detach()
                        final_cce_loss = cce_loss.detach()
                        reconstruction_loss = reconstruction_loss / total_mcmc_steps

                if mcmc_step == 0:
                    initial_loss = cce_loss.detach()
                    initial_pred_energies = predicted_energy.squeeze().mean().detach()
                if mcmc_step == (total_mcmc_steps - 1):
                    final_pred_energies = predicted_energy.squeeze().mean().detach()

            initial_final_pred_energies_gap = initial_pred_energies - final_pred_energies
            ppl_loss = torch.exp(final_reconstruction_loss).detach()
            total_loss = self.hparams.reconstruction_coeff * reconstruction_loss
            contrastive_loss = 0.0

            if token_bytes is not None:
                bpb_loss, bpb_nats, bpb_bytes = calculate_bpb_score(next_token_indices, final_cce_loss, token_bytes)
            else:
                bpb_loss = 0
                bpb_nats = 0
                bpb_bytes = 0

            offset_loss_log_dict = {}
            final_step_logits = multi_offset_logits_per_step[-1].detach()  # [B, K, S_eff, V]
            final_step_targets = block_targets.detach()  # [B, K, S_eff]
            for offset_idx in range(num_offsets):
                offset_value = int(target_offsets[offset_idx].item())
                offset_logits = final_step_logits[:, offset_idx, :, :].reshape(-1, self.vocab_size)
                offset_targets = final_step_targets[:, offset_idx, :].reshape(-1)
                offset_loss_per_token = F.cross_entropy(
                    offset_logits,
                    offset_targets,
                    ignore_index=-1,
                    reduction="none",
                )
                offset_loss = F.cross_entropy(
                    offset_logits,
                    offset_targets,
                    ignore_index=-1,
                )
                offset_loss_log_dict[f"offset_{offset_value}_loss"] = offset_loss
                # Per-offset PPL (HEAD addition) — same intent as in the
                # explicit-block-latent path. Lets a blockwise mtp_mcmc run's
                # offset_1_ppl be compared directly against a non-blockwise PPL.
                offset_loss_log_dict[f"offset_{offset_value}_ppl"] = torch.exp(offset_loss).detach()
                if token_bytes is not None:
                    # Origin: per-token loss + 3-tuple (DDP-aware nats/bytes).
                    offset_bpb, offset_bpb_nats, offset_bpb_bytes = calculate_bpb_score(
                        offset_targets,
                        offset_loss_per_token,
                        token_bytes,
                    )
                    offset_loss_log_dict[f"offset_{offset_value}_bpb"] = offset_bpb
                    offset_loss_log_dict[f"offset_{offset_value}_bpb_nats"] = offset_bpb_nats
                    offset_loss_log_dict[f"offset_{offset_value}_bpb_bytes"] = offset_bpb_bytes

            if getattr(self.hparams, "debug_blockwise_shapes", False):
                print(
                    f"[blockwise-debug][phase={phase}] dense_mode=True, "
                    f"input_ids.shape={tuple(input_ids.shape)}, block_targets.shape={tuple(block_targets.shape)}, "
                    f"offsets={target_offsets.tolist()}, logits_last_step.shape={tuple(multi_offset_logits_per_step[-1].shape)}, "
                    f"alpha={self.alpha.detach().item():.6f}",
                    flush=True,
                )
                print(
                    f"[blockwise-debug][phase={phase}] post_update_pred_hidden_last_step.shape={tuple(pred_hiddens_per_step[-1].shape)}",
                    flush=True,
                )
                if num_offsets >= 1:
                    print(
                        f"[blockwise-debug][phase={phase}] offset_1_logits.shape={tuple(multi_offset_logits_per_step[-1][:, 0, :, :].shape)}",
                        flush=True,
                    )
                if num_offsets >= 2:
                    print(
                        f"[blockwise-debug][phase={phase}] offset_2_logits.shape={tuple(multi_offset_logits_per_step[-1][:, 1, :, :].shape)}",
                        flush=True,
                    )
                print(
                    f"[blockwise-debug][phase={phase}] aggregated_loss={total_loss.detach().item():.6f}",
                    flush=True,
                )
                print(
                    f"[blockwise-debug][phase={phase}] offset_losses="
                    f"{ {k: round(v.item(), 6) for k, v in offset_loss_log_dict.items()} }",
                    flush=True,
                )

            log_dict = {
                'loss': total_loss,
                'initial_loss' : initial_loss,
                'final_step_loss': final_reconstruction_loss,
                'contrastive_loss' : contrastive_loss,
                'initial_final_pred_energies_gap': initial_final_pred_energies_gap,
                'perplexity': ppl_loss,
                'bpb': bpb_loss,
                'bpb_nats': bpb_nats,
                'bpb_bytes': bpb_bytes,
            }
            log_dict.update(offset_loss_log_dict)
            return log_dict
        else:
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

        if self.hparams.contrastive_loss: # works by pushing up on energies model predicted and pushing down on energy of true samples
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
            bpb_loss, bpb_nats, bpb_bytes = calculate_bpb_score(next_token_indices, per_token_ce.detach(), token_bytes)
        else:
            bpb_loss = 0
            bpb_nats = 0
            bpb_bytes = 0

        log_dict = {
            'loss': total_loss,
            'initial_loss' : initial_loss,
            'final_step_loss': final_reconstruction_loss,
            'contrastive_loss' : contrastive_loss,
            'initial_final_pred_energies_gap': initial_final_pred_energies_gap,
            'perplexity': ppl_loss,
            'bpb': bpb_loss,
            'bpb_nats': bpb_nats,    # accumulated nats for epoch-level BPB
            'bpb_bytes': bpb_bytes,  # accumulated bytes for epoch-level BPB
        }
        return log_dict
    

    def corrupt_embeddings(self, embeddings, target_length=None):
        if target_length is None:
            target_length = embeddings.shape[1]
        if self.hparams.denoising_initial_condition == "most_recent_embedding":
            raise NotImplementedError(f"most_recent_embedding denoising_initial_condition not supported for NLP yet")
        elif self.hparams.denoising_initial_condition == "random_noise":
            predicted_tokens = torch.randn(
                size=(embeddings.shape[0], target_length, self.vocab_size),
                dtype=embeddings.dtype,
                device=self.device,
            ) * self.hparams.gaussian_random_noise_scaling
        elif self.hparams.denoising_initial_condition == "zeros":
            predicted_tokens = torch.zeros(
                size=(embeddings.shape[0], target_length, self.vocab_size),
                dtype=embeddings.dtype,
                device=self.device,
            )
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
        
        real_energies = self.transformer(all_true_embeddings, start_pos=0, mcmc_step=self.hparams.mcmc_num_steps - 1, block_mode=self._block_mode) # NOTE if want to use this maybe check in better detail what ired does
        real_energies = real_energies.reshape(-1, 1) # BS, 1
        fake_energies = predicted_energies[-1] # B*S, 1
        energy_stack = torch.cat([real_energies, fake_energies], dim=1)
        energy_targets = torch.zeros(real_energies.shape[0], dtype=torch.long, device=fake_energies.device)
        padding_positions = None # (next_token_indices == self.tokenizer_pad_token_id).reshape(-1)
        energy_targets[padding_positions] = -100 # prevents nans instead of using self.tokenizer_pad_token_id, as setting this to 0 leads to issues
        contrastive_loss = F.cross_entropy(-1 * energy_stack, energy_targets, ignore_index=-100)
        return contrastive_loss
    
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

    def _run_ebt_inference_steps(
        self,
        initial_pred_tokens,
        real_embeds,
        adjusted_alpha,
        noise,
        start_pos,
        learning
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
                energies = self.transformer(combined_embeddings, start_pos=start_pos, mcmc_step=step_idx, block_mode=self._block_mode)
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
                energies = self.transformer(combined_embeddings, start_pos=start_pos, mcmc_step=step_idx, block_mode=self._block_mode)
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
        return pred_state, energies_list, pred_states_list
