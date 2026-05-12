"""LLaMA-style autoregressive Transformer reused as the EBT backbone.

NOTE: Most of the low-level building blocks (RoPE, RMSNorm, SwiGLU FFN) are
adapted from the Meta LLaMA 2 reference implementation. Credit:
https://github.com/meta-llama/llama
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from openebm.elm.utils import *


@dataclass
class TransformerModelArgs:
    """Configuration dataclass for :class:`Transformer` and its sub-modules.

    :param dim: Model hidden dimension.
    :type dim: int
    :param n_layers: Number of stacked transformer blocks.
    :type n_layers: int
    :param n_heads: Number of query attention heads.
    :type n_heads: int
    :param n_kv_heads: Number of key/value heads. Falls back to ``n_heads``
        when ``None`` (i.e. standard MHA rather than GQA).
    :type n_kv_heads: Optional[int]
    :param ffn_dim_multiplier: Multiplier applied to ``dim`` when computing the
        FFN hidden size. ``None`` keeps the FFN hidden size equal to ``dim``.
    :type ffn_dim_multiplier: Optional[float]
    :param norm_eps: Epsilon for RMSNorm.
    :type norm_eps: float
    :param max_batch_size: Maximum batch size expected at inference.
    :type max_batch_size: int
    :param max_seq_len: Maximum sequence length supported by the RoPE cache.
    :type max_seq_len: int
    :param weight_initialization: Name of the weight initialization scheme.
    :type weight_initialization: str
    :param weight_initialization_gain: Gain factor forwarded to the init
        function (when applicable).
    :type weight_initialization_gain: float
    """

    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 64
    max_seq_len: int = 16
    weight_initialization: str = "xavier"
    weight_initialization_gain: float = 1.0



class RMSNorm(torch.nn.Module):
    """Root-mean-square layer normalization with a learnable scale."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize the RMSNorm layer.

        :param dim: Feature dimension being normalized.
        :type dim: int
        :param eps: Small constant added to the denominator for numerical
            stability.
        :type eps: float
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the RMS normalization step without the learnable scale.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Tensor with the same shape as ``x`` whose last dimension has
            unit RMS.
        :rtype: torch.Tensor
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full RMSNorm (normalize then rescale).

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Normalized and rescaled tensor.
        :rtype: torch.Tensor
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the complex-exponential frequency tensor used by RoPE.

    :param dim: Per-head dimension. Must be even.
    :type dim: int
    :param end: Number of positions to precompute.
    :type end: int
    :param theta: Base frequency scale. Matches the LLaMA default of 10000.
    :type theta: float
    :return: ``complex64`` tensor of shape ``(end, dim // 2)`` containing the
        rotation coefficients.
    :rtype: torch.Tensor
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape the RoPE frequency tensor for broadcasting against ``x``.

    :param freqs_cis: Frequency tensor to reshape.
    :type freqs_cis: torch.Tensor
    :param x: Target tensor whose layout dictates the broadcast shape.
    :type x: torch.Tensor
    :return: Reshaped frequency tensor compatible with ``x``.
    :rtype: torch.Tensor
    :raises AssertionError: If ``freqs_cis`` does not match the expected shape
        or if ``x`` does not have at least two dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to the query and key tensors.

    Reinterprets consecutive pairs along the head dimension as complex numbers,
    rotates them by ``freqs_cis``, and returns the result as real tensors with
    the original dtype.

    :param xq: Query tensor of shape ``(B, S, H, D)``.
    :type xq: torch.Tensor
    :param xk: Key tensor of shape ``(B, S, H_kv, D)``.
    :type xk: torch.Tensor
    :param freqs_cis: Precomputed RoPE frequencies.
    :type freqs_cis: torch.Tensor
    :return: A tuple ``(rotated_query, rotated_key)``.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Replicate key/value heads for grouped-query attention.

    Equivalent to ``torch.repeat_interleave(x, dim=2, repeats=n_rep)`` but
    implemented via views to avoid the extra allocation.

    :param x: Tensor of shape ``(B, S, H_kv, D)``.
    :type x: torch.Tensor
    :param n_rep: Number of times each head should be replicated.
    :type n_rep: int
    :return: Tensor of shape ``(B, S, H_kv * n_rep, D)``.
    :rtype: torch.Tensor
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Multi-head (optionally grouped-query) attention with RoPE."""

    def __init__(self, args: TransformerModelArgs) -> None:
        """Initialize the attention module from a configuration object.

        :param args: Model configuration.
        :type args: TransformerModelArgs
        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # NOTE: Model parallelism is hardcoded to 1 here because the project
        # relies on DDP rather than tensor-parallelism.
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        init_whole_model_weights(self.wq, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)

        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        init_whole_model_weights(self.wk, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)

        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        init_whole_model_weights(self.wv, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)

        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        init_whole_model_weights(self.wo, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute attention outputs for the input sequence.

        :param x: Input tensor of shape ``(B, S, D)``.
        :type x: torch.Tensor
        :param start_pos: Starting position (used by subclasses that implement
            KV caching; kept here for API compatibility).
        :type start_pos: int
        :param freqs_cis: Precomputed RoPE frequencies for the current window.
        :type freqs_cis: torch.Tensor
        :param mask: Optional additive attention mask broadcastable to
            ``(B, H, S, S)``.
        :type mask: Optional[torch.Tensor]
        :return: Attention output of shape ``(B, S, D)``.
        :rtype: torch.Tensor
        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # NOTE: The KV cache was removed because it broke the MCMC second
        # backward pass with "Trying to backward through the graph a second
        # time" during EBT refinement. The current implementation recomputes
        # keys/values on every step instead of caching them.
        keys = xk
        values = xv

        xq = xq.transpose(1, 2)      # (B, H, S, Dh)
        keys = keys.transpose(1, 2)  # (B, H, S, Dh)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """SwiGLU feed-forward block used in every transformer layer."""

    def __init__(
        self,
        dim: int,
        ffn_dim_multiplier: Optional[float],
        weight_initialization: str,
        weight_initialization_gain: float,
    ) -> None:
        """Initialize the SwiGLU feed-forward block.

        :param dim: Input and output dimension.
        :type dim: int
        :param ffn_dim_multiplier: Optional multiplier applied to ``dim`` to
            compute the hidden dimension. ``None`` keeps the hidden dimension
            equal to ``dim``.
        :type ffn_dim_multiplier: Optional[float]
        :param weight_initialization: Name of the weight initialization
            scheme forwarded to :func:`init_whole_model_weights`.
        :type weight_initialization: str
        :param weight_initialization_gain: Gain factor for the initialization.
        :type weight_initialization_gain: float
        """
        super().__init__()

        hidden_dim = dim if ffn_dim_multiplier is None else int(dim * ffn_dim_multiplier)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        init_whole_model_weights(self.w1, weight_initialization, weight_initialization_gain=weight_initialization_gain)

        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        init_whole_model_weights(self.w2, weight_initialization, weight_initialization_gain=weight_initialization_gain)

        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        init_whole_model_weights(self.w3, weight_initialization, weight_initialization_gain=weight_initialization_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SwiGLU transform ``W2(SiLU(W1 x) * W3 x)``.

        :param x: Input tensor of shape ``(..., dim)``.
        :type x: torch.Tensor
        :return: Output tensor of shape ``(..., dim)``.
        :rtype: torch.Tensor
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Single pre-norm transformer block (attention + SwiGLU FFN)."""

    def __init__(self, layer_id: int, args: TransformerModelArgs) -> None:
        """Initialize a transformer block.

        :param layer_id: Zero-based index identifying this block within the
            stack.
        :type layer_id: int
        :param args: Model configuration.
        :type args: TransformerModelArgs
        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            weight_initialization=args.weight_initialization,
            weight_initialization_gain=args.weight_initialization_gain
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run pre-norm attention followed by the SwiGLU FFN.

        :param x: Input tensor of shape ``(B, S, D)``.
        :type x: torch.Tensor
        :param start_pos: Starting position forwarded to the attention module.
        :type start_pos: int
        :param freqs_cis: Precomputed RoPE frequencies for the current window.
        :type freqs_cis: torch.Tensor
        :param mask: Optional additive attention mask.
        :type mask: Optional[torch.Tensor]
        :return: Output tensor of shape ``(B, S, D)``.
        :rtype: torch.Tensor
        """
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """Stack of :class:`TransformerBlock` plus a final RMSNorm."""

    def __init__(self, params: TransformerModelArgs) -> None:
        """Initialize the transformer stack.

        :param params: Model configuration.
        :type params: TransformerModelArgs
        """
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len
        )


    def forward(self, embeddings: torch.Tensor, start_pos: int, learning: bool = False) -> torch.Tensor:
        """Run the full transformer forward pass.

        :param embeddings: Input embeddings of shape ``(B, S, D)``.
        :type embeddings: torch.Tensor
        :param start_pos: Starting position for the RoPE cache window.
        :type start_pos: int
        :param learning: If ``False``, the forward pass is wrapped in
            :func:`torch.inference_mode` so no graph is retained; if ``True``
            gradients flow normally.
        :type learning: bool
        :return: Output hidden states of shape ``(B, S, D)`` (float32).
        :rtype: torch.Tensor
        """
        if not learning:
            with torch.inference_mode():
                _bsz, seqlen = embeddings.shape[:2]
                self.freqs_cis = self.freqs_cis.to(embeddings.device)
                freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

                mask = None
                if seqlen > 1:
                    mask = torch.full(
                        (seqlen, seqlen), float("-inf"), device=embeddings.device
                    )

                    mask = torch.triu(mask, diagonal=1)

                    # With KV caching the scores matrix has shape
                    # ``(seqlen, cache_len + seqlen)``; only entries ``(i, j)``
                    # with ``j > cache_len + i`` need to be masked out.
                    mask = torch.hstack([
                        torch.zeros((seqlen, start_pos), device=embeddings.device),
                        mask
                    ]).type_as(embeddings)

                for layer in self.layers:
                    embeddings = layer(embeddings, start_pos, freqs_cis, mask)
                embeddings = self.norm(embeddings).float()
                return embeddings
        else:
            _bsz, seqlen = embeddings.shape[:2]
            self.freqs_cis = self.freqs_cis.to(embeddings.device)
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

            mask = None
            if seqlen > 1:
                mask = torch.full(
                    (seqlen, seqlen), float("-inf"), device=embeddings.device
                )

                mask = torch.triu(mask, diagonal=1)

                # With KV caching the scores matrix has shape
                # ``(seqlen, cache_len + seqlen)``; only entries ``(i, j)``
                # with ``j > cache_len + i`` need to be masked out.
                mask = torch.hstack([
                    torch.zeros((seqlen, start_pos), device=embeddings.device),
                    mask
                ]).type_as(embeddings)

            for layer in self.layers:
                embeddings = layer(embeddings, start_pos, freqs_cis, mask)
            embeddings = self.norm(embeddings).float()
            return embeddings


