"""TF (teacher-forced, Gemma-drafter-style) head modules for EBT.

Decoupled from the blockwise/MTP machinery in `nova/ebt/`. Each head takes
`(pred_hidden, prev_token_embed) -> logits` and is used in plain (K=1) NTP
on top of EBT's MCMC pred_hidden output. See `TF_HEAD_ARCHITECTURE.md` §9.

Three variants exposed via `build_tf_head(hparams)`:
  - "linear":         Linear(2D, D) -> Linear(D, V)
  - "direct_unembed": Linear(D, V) over trunk pred_hidden as an explicit test2 ablation
  - "transformer":    Linear(2D, D) -> L x causal AR block -> RMSNorm -> Linear(D, V)

The transformer variant matches Gemma-drafter's "concat-then-project + L tiny
attention blocks" pattern. L=1 is the empirical sweet spot in xxs experiments.

Init goes through `init_whole_model_weights(...)` (same as embeddings/trunk),
which respects `--weight_initialization_method` / `--weight_initialization_gain`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TFRMSNorm(nn.Module):
    """Standalone RMSNorm. Keeps an fp32 inner accumulation for bf16-mixed safety."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm.type_as(x)) * self.weight


class _TFHeadBlock(nn.Module):
    """Pre-norm causal SDPA + SwiGLU FFN. Minimal AR transformer block."""

    def __init__(self, dim: int, n_heads: int, ffn_mult: float = 4.0, norm_eps: float = 1e-5):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"tf_head: dim {dim} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.attn_norm = _TFRMSNorm(dim, eps=norm_eps)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        ffn_dim = ((int(dim * ffn_mult) + 63) // 64) * 64
        self.ffn_norm = _TFRMSNorm(dim, eps=norm_eps)
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        h = self.attn_norm(x)
        q = self.wq(h).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        if S > 1:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            attn = v
        x = x + self.wo(attn.transpose(1, 2).reshape(B, S, D))
        h = self.ffn_norm(x)
        x = x + self.w2(F.silu(self.w1(h)) * self.w3(h))
        return x


class TFLinearHead(nn.Module):
    """forward(pred_hidden[B,S,D], prev_embed[B,S,D]) -> logits[B,S,V]."""

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.down_proj = nn.Linear(2 * dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, pred_hidden: torch.Tensor, prev_token_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([pred_hidden, prev_token_embed], dim=-1)
        return self.out_proj(self.down_proj(x))


class TFDirectUnembedHead(nn.Module):
    """Test2 head: project the trunk last-layer candidate hidden to vocab logits.

    `prev_token_embed` is the first-layer embedding from self.embeddings(input_ids)
    and is intentionally ignored here. It stays in the signature only to match the
    shared TF-head interface used by the linear/transformer heads.
    """

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, pred_hidden: torch.Tensor, prev_token_embed: torch.Tensor) -> torch.Tensor:
        del prev_token_embed
        return self.proj(pred_hidden)


class TFTransformerHead(nn.Module):
    """forward(pred_hidden[B,S,D], prev_embed[B,S,D]) -> logits[B,S,V].

    `num_layers` causal transformer blocks between the concat-projection and the
    final V-projection. Gemma-drafter shape.
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        num_layers: int = 1,
        n_heads: int = 8,
        ffn_mult: float = 4.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.down_proj = nn.Linear(2 * dim, dim, bias=False)
        self.blocks = nn.ModuleList(
            [_TFHeadBlock(dim, n_heads, ffn_mult, norm_eps=norm_eps) for _ in range(num_layers)]
        )
        self.final_norm = _TFRMSNorm(dim, eps=norm_eps)
        self.proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, pred_hidden: torch.Tensor, prev_token_embed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([pred_hidden, prev_token_embed], dim=-1)
        x = self.down_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.proj(self.final_norm(x))


def build_tf_head(hparams) -> nn.Module:
    """Factory that consumes hparams namespace from train.py CLI flags.

    Reads:
      hparams.tf_head_type       in {"linear", "direct_unembed", "transformer"}
      hparams.tf_head_layers     int (transformer only)
      hparams.tf_head_n_heads    int, 0 -> inherit hparams.multiheaded_attention_heads
      hparams.tf_head_ffn_mult   float
      hparams.embedding_dim, hparams.vocab_size
    """
    dim = int(hparams.embedding_dim)
    vocab_size = int(hparams.vocab_size)
    head_type = getattr(hparams, "tf_head_type", "transformer")

    if head_type == "linear":
        return TFLinearHead(dim=dim, vocab_size=vocab_size)

    if head_type == "direct_unembed":
        return TFDirectUnembedHead(dim=dim, vocab_size=vocab_size)

    if head_type == "transformer":
        n_heads_override = int(getattr(hparams, "tf_head_n_heads", 0))
        n_heads_default = int(getattr(hparams, "multiheaded_attention_heads", 8))
        n_heads = n_heads_override if n_heads_override > 0 else n_heads_default
        return TFTransformerHead(
            dim=dim,
            vocab_size=vocab_size,
            num_layers=int(getattr(hparams, "tf_head_layers", 1)),
            n_heads=n_heads,
            ffn_mult=float(getattr(hparams, "tf_head_ffn_mult", 4.0)),
        )

    raise ValueError(f"Unknown tf_head_type: {head_type!r}")
