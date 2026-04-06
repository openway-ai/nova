import sys
sys.path.append("../../")

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig, Block, CausalSelfAttention, norm, apply_rotary_emb, has_ve
from utils import EBTModelArgs


def _build_ebt_mask(T: int, S: int, device: torch.device) -> torch.Tensor:
    """
    Build the additive float bias mask for EBT attention.

    Sequence layout: [time(0), real₁(1), ..., real_{S-1}(S-1), pred₁(S), ..., pred_{S-1}(T-1)]
      T = 2*(S-1) + 1  (total length)
      S = (T+1)//2     (real-side count, including time)

    Mask rules:
      real positions (rows 0..S-1): standard lower-triangular causal
      pred_k (1-indexed, row = S+k-1):
        allow columns 0..k   (time + real₁..real_k)
        allow self   (column S+k-1)
        block everything else

    Returns float tensor of shape (T, T), values 0.0 or -inf.
    """
    mask = torch.full((T, T), float('-inf'), device=device)
    # Real positions: standard causal
    for i in range(S):
        mask[i, :i + 1] = 0.0
    # Pred positions
    for k in range(1, S):
        row = S + k - 1
        mask[row, :k + 1] = 0.0   # time + real₁..real_k
        mask[row, row] = 0.0      # self
    return mask


class EBTSelfAttention(CausalSelfAttention):
    """
    Drops flash_attn_func(causal=True) in favour of SDPA with an explicit EBT mask.
    All weight shapes and QK-norm/RoPE logic are identical to CausalSelfAttention.
    """

    def forward(self, x, ve, cos_sin, window_size, kv_cache, attn_mask=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm
        # SDPA expects (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Combine sliding window constraint with EBT mask when window is limited.
        # Both masks are additive floats (0.0=allowed, -inf=blocked); summing ANDs them.
        if window_size is not None and window_size[0] > 0:
            rows = torch.arange(T, device=q.device).unsqueeze(1)
            cols = torch.arange(T, device=q.device).unsqueeze(0)
            sw = torch.where(
                (rows - cols) <= window_size[0],
                torch.zeros(1, device=q.device),
                torch.full((1,), float('-inf'), device=q.device),
            )
            attn_mask = attn_mask + sw.unsqueeze(0).unsqueeze(0)  # broadcast over (B, H)
        enable_gqa = self.n_head != self.n_kv_head
        # The memory-efficient SDPA backend does not implement backward for all
        # configurations (float additive mask + GQA).  Force flash/math only.
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            ctx = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH])
        except ImportError:
            ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False)
        with ctx:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class EBTBlock(Block):
    """Block subclass that replaces CausalSelfAttention with EBTSelfAttention."""

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        # Swap the standard attention module for the EBT-masked variant.
        # Preserve the layer_idx so weight shapes and ve_gate logic are unchanged.
        self.attn = EBTSelfAttention(config, layer_idx)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, attn_mask=None):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache, attn_mask=attn_mask)
        x = x + self.mlp(norm(x))
        return x


class NanoChatEBT(nn.Module):
    """
    EBT energy function built on GPT's architectural decisions:
      - Flash Attention with QK norm (no learnable params in norm)
      - relu² MLP (4x expansion, no bias)
      - RoPE (cos/sin buffers, bfloat16)
      - Per-layer residual scaling: resid_lambdas × x + x0_lambdas × x0
      - Functional RMSNorm throughout (no learnable scale/bias)
      - Sliding window support via GPTConfig.window_pattern (default "L" = full context)

    Inference flow is identical to EBTTimeConcat:
      Input:   embeddings  [B, 2*(S-1), D]  (real tokens first, predicted second)
      1. Prepend MCMC step time embedding   → [B, 2*(S-1)+1, D]
      2. norm → GPT-style blocks (standard causal attention) → norm
      3. Remove time embedding              → [B, 2*(S-1), D]
      4. Linear energy head                 → [B, 2*(S-1), 1]
      5. Return second half (predictions)   → [B, S-1,     1]

    Attention mask: uses EBT-specific masking via EBTSelfAttention + SDPA.
    pred_k (at position S+k-1) attends only to time+real₁..real_k and itself,
    preventing causality violations (seeing future real tokens) and independence
    violations (predicted tokens attending to each other).

    Value embeddings (ResFormer): alternating layers include a value_embeds lookup
    (same has_ve pattern as GPT). forward() accepts optional idx=[B, 2*(S-1)] token IDs;
    if idx is None, value_embeds are skipped (e.g. during inference).
    """

    def __init__(self, params: EBTModelArgs, max_mcmc_steps: int):
        super().__init__()

        n_kv_head = params.n_kv_heads if params.n_kv_heads is not None else params.n_heads
        kv_dim = n_kv_head * (params.dim // params.n_heads)

        # GPTConfig used to configure Block layers.
        # sequence_len covers the doubled EBT sequence (2*(S-1)+1) plus a small margin.
        # window_pattern is read from params (default "L" = full context on all layers).
        self.gpt_config = GPTConfig(
            sequence_len=params.max_seq_len * 2 + 4,
            vocab_size=params.vocab_size,       # placeholder; wte/lm_head not present in NanoChatEBT
            n_layer=params.n_layers,
            n_head=params.n_heads,
            n_kv_head=n_kv_head,
            n_embd=params.dim,
            window_pattern=params.window_pattern,
        )

        # EBT transformer blocks: same as GPT but use SDPA with explicit EBT mask.
        self.blocks = nn.ModuleList(
            [EBTBlock(self.gpt_config, i) for i in range(params.n_layers)]
        )

        # Per-layer scalars matching GPT:
        #   resid_lambdas[i] scales the residual stream before each block
        #   x0_lambdas[i]   blends the initial (post-norm) embedding back in
        self.resid_lambdas = nn.Parameter(torch.ones(params.n_layers))
        self.x0_lambdas = nn.Parameter(torch.zeros(params.n_layers))
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(params.vocab_size, kv_dim)
            for i in range(self.gpt_config.n_layer)
            if has_ve(i, self.gpt_config.n_layer)
        })

        # Window sizes per layer derived from window_pattern
        self.window_sizes = self._compute_window_sizes(self.gpt_config)

        # RoPE buffers: bfloat16, shape (1, seq_len, 1, head_dim/2), same format as GPT
        head_dim = params.dim // params.n_heads
        cos, sin = self._precompute_rotary_embeddings(self.gpt_config.sequence_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # MCMC step embedding (same as EBTTimeConcat.time_embeddings)
        self.time_embeddings = nn.Embedding(max_mcmc_steps, params.dim)

        # Scalar energy head (same as EBTTimeConcat.final_layer)
        self.final_layer = nn.Linear(params.dim, 1, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: int = 10000):
        """Same computation as GPT._precompute_rotary_embeddings."""
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().bfloat16()[None, :, None, :]  # (1, seq_len, 1, head_dim/2)
        sin = freqs.sin().bfloat16()[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config: GPTConfig):
        """Same logic as GPT._compute_window_sizes."""
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)   # final layer always full context
        return window_sizes

    def _init_weights(self):
        """
        Mirrors GPT.init_weights():
          attn c_q/c_k/c_v  : Uniform ±sqrt(3)/sqrt(n_embd)  (same std as Normal)
          attn c_proj        : zeros
          mlp  c_fc          : Uniform ±sqrt(3)/sqrt(n_embd)
          mlp  c_proj        : zeros
          value_embeds       : Uniform ±sqrt(3)/sqrt(n_embd)
          ve_gate            : zeros
          resid_lambdas      : 1.0
          x0_lambdas         : 0.1
          time_embeddings    : Normal(0, 1.0)
          final_layer        : Normal(0, 0.001)
        """
        n_embd = self.gpt_config.n_embd
        s = 3 ** 0.5 * n_embd ** -0.5          # sqrt(3) * 1/sqrt(n_embd)
        for block in self.blocks:
            nn.init.uniform_(block.attn.c_q.weight, -s, s)
            nn.init.uniform_(block.attn.c_k.weight, -s, s)
            nn.init.uniform_(block.attn.c_v.weight, -s, s)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            nn.init.zeros_(block.mlp.c_proj.weight)
            if block.attn.ve_gate is not None:
                nn.init.zeros_(block.attn.ve_gate.weight)
        for ve in self.value_embeds.values():
            nn.init.uniform_(ve.weight, -s, s)
        self.resid_lambdas.data.fill_(1.0)
        self.x0_lambdas.data.fill_(0.1)
        nn.init.normal_(self.time_embeddings.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.final_layer.weight, mean=0.0, std=0.001)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, embeddings: torch.Tensor, idx: torch.Tensor = None, start_pos: int = 0, mcmc_step: int = 0):
        """
        Args:
            embeddings : [B, 2*(S-1), D]  real tokens in the first half,
                         predicted (MCMC) embeddings in the second half
            idx        : [B, 2*(S-1)]  token IDs for real+predicted positions,
                         used for value_embeds (ResFormer) lookups.
                         Optional — if None, value_embeds are skipped.
            start_pos  : RoPE offset; always 0 for EBT (KV cache is not used)
            mcmc_step  : MCMC iteration index used to select the time embedding

        Returns:
            energies   : [B, S-1, 1]  scalar energy for each predicted token position
        """
        _bsz = embeddings.shape[0]

        # Prepend a zero time-placeholder to idx so full_idx aligns with the
        # [time, real₁..real_{S-1}, pred₁..pred_{S-1}] sequence layout.
        # Only computed when idx is provided (used for value_embeds ResFormer lookups).
        full_idx = F.pad(idx, (1, 0), value=0) if idx is not None else None  # [B, 2*(S-1)+1]

        # 1. Prepend MCMC step time embedding (identical to EBTTimeConcat)
        step_idx = torch.full(
            (_bsz,), fill_value=mcmc_step, device=embeddings.device, dtype=torch.long
        )
        time_embed = self.time_embeddings(step_idx).unsqueeze(1)   # [B, 1, D]
        x = torch.cat((time_embed, embeddings), dim=1)             # [B, 2*(S-1)+1, D]

        # 2. Prepare RoPE slice for this sequence length
        T = x.shape[1]   # 2*(S-1) + 1
        assert start_pos + T <= self.cos.size(1), (
            f"Sequence length {start_pos + T} exceeds RoPE cache {self.cos.size(1)}"
        )
        cos_sin = (
            self.cos[:, start_pos : start_pos + T],   # (1, T, 1, head_dim/2)
            self.sin[:, start_pos : start_pos + T],
        )

        # 2b. Build EBT attention mask: pred_k attends only to time+real₁..real_k and itself
        S = (T + 1) // 2   # number of real-side positions (time + real tokens)
        attn_mask = _build_ebt_mask(T, S, x.device).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        # 3. GPT-style trunk: input norm, then blocks with resid/x0 scalars, then output norm
        x = norm(x)   # functional RMSNorm, no learnable params
        x0 = x        # saved for x0_lambdas skip connection
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](full_idx) if (str(i) in self.value_embeds and full_idx is not None) else None
            x = block(x, ve, cos_sin, self.window_sizes[i], None, attn_mask=attn_mask)
        x = norm(x)

        # 4. Remove time embedding (identical to EBTTimeConcat)
        x = x[:, 1:]                              # [B, 2*(S-1), D]

        # 5. Project to scalar energies (identical to EBTTimeConcat)
        energies = self.final_layer(x)            # [B, 2*(S-1), 1]

        # 6. Return only the predicted-token half (identical to EBTTimeConcat)
        energies = energies[:, x.shape[1] // 2:]  # [B, S-1, 1]
        return energies
