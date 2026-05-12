"""Autoregressive Energy-Based Transformer with per-step time embeddings.

Defines the EBT architecture used in the autoregressive setting, combining a
causal Transformer with a second attention stream dedicated to next-token
predictions and optional MCMC time embeddings. Most of the underlying
Transformer code is adapted from the Llama2 codebase.
"""
# NOTE: most code taken from the llama2 codebase; credit https://github.com/meta-llama/llama
import torch
from torch import nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Any

from openebm.elm.utils import init_whole_model_weights, EBTModelArgs
try:
    from openebm.elm.ve import build_layer_ve, build_value_embeds, has_ve
except ImportError:
    has_ve = None
    build_value_embeds = None
    build_layer_ve = None


class BackwardRMSNormFunction(torch.autograd.Function):
    """Identity in forward, RMSNorm applied to gradients in backward.

    The forward pass returns the input unchanged. The backward pass normalizes
    the incoming gradient using an RMSNorm transform parameterized by a
    learnable weight.
    """

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """Identity forward that stashes ``weight`` and ``eps`` for backward.

        :param ctx: Autograd context used to cache tensors for backward.
        :type ctx: Any
        :param input_: Input tensor, returned unchanged.
        :type input_: torch.Tensor
        :param weight: Learnable scale vector used during backward.
        :type weight: torch.Tensor
        :param eps: Numerical stability constant added inside the RMSNorm.
        :type eps: float
        :return: ``input_`` unchanged.
        :rtype: torch.Tensor
        """
        ctx.save_for_backward(weight)
        ctx.eps = eps
        return input_

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """Apply RMSNorm to ``grad_output`` using the cached ``weight``.

        :param ctx: Autograd context with cached tensors.
        :type ctx: Any
        :param grad_output: Gradient flowing from downstream ops.
        :type grad_output: torch.Tensor
        :return: Gradients for ``input_``, ``weight`` and ``eps`` (``None``).
        :rtype: Tuple[torch.Tensor, torch.Tensor, None]
        """
        (weight,) = ctx.saved_tensors
        eps = ctx.eps

        # Compute RMS of grad_output over the last dim
        rms = torch.rsqrt(grad_output.float().pow(2).mean(dim=-1, keepdim=True) + eps)

        # Normalized gradient wrt the input
        grad_input = grad_output * rms * weight  # shape matches grad_output

        # Gradient wrt 'weight': the transform is out_grad = grad_output * (rms * weight),
        # so the partial wrt weight is grad_output * rms, summed over all but the last dim.
        sum_dims = list(range(grad_output.dim() - 1))
        grad_weight = (grad_output * rms).sum(dim=sum_dims)

        return grad_input, grad_weight, None


class BackwardRMSNorm(nn.Module):
    """Identity in forward, RMSNorm in backward, with a learnable weight."""

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize the backward-only RMSNorm.

        :param dim: Size of the last dimension being normalized.
        :type dim: int
        :param eps: Numerical stability constant used inside the normalization.
        :type eps: float
        """
        super().__init__()
        self.weight = nn.Parameter(torch.full((dim,), 0.01))

        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` unchanged; normalization is only applied in backward.

        :param x: Input tensor of arbitrary shape ``(..., dim)``.
        :type x: torch.Tensor
        :return: The same tensor ``x``.
        :rtype: torch.Tensor
        """
        return BackwardRMSNormFunction.apply(x, self.weight, self.eps)

class BackwardLayerNormFunction(torch.autograd.Function):
    """Identity in forward, LayerNorm applied to gradients in backward."""

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
        """Identity forward that stashes LayerNorm parameters for backward.

        :param ctx: Autograd context used to cache tensors for backward.
        :type ctx: Any
        :param input_: Input tensor, returned unchanged.
        :type input_: torch.Tensor
        :param gamma: Learnable scale vector used during backward.
        :type gamma: torch.Tensor
        :param beta: Learnable bias vector used during backward.
        :type beta: torch.Tensor
        :param eps: Numerical stability constant added inside the LayerNorm.
        :type eps: float
        :return: ``input_`` unchanged.
        :rtype: torch.Tensor
        """
        ctx.save_for_backward(gamma, beta)
        ctx.eps = eps
        return input_

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Apply LayerNorm to ``grad_output`` using the cached ``gamma``/``beta``.

        :param ctx: Autograd context with cached tensors.
        :type ctx: Any
        :param grad_output: Gradient flowing from downstream ops.
        :type grad_output: torch.Tensor
        :return: Gradients for ``input_``, ``gamma``, ``beta`` and ``eps`` (``None``).
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]
        """
        gamma, beta = ctx.saved_tensors
        eps = ctx.eps

        # Compute mean and variance of grad_output over the last dimension
        mu = grad_output.mean(dim=-1, keepdim=True)
        var = grad_output.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + eps)

        # "LayerNorm" of the incoming gradient
        normed_grad = (grad_output - mu) / std

        # Output gradient for the input: out_grad = gamma * normed_grad + beta
        grad_input = gamma * normed_grad + beta

        # NOTE: partial(out_grad)/partial(gamma) = normed_grad and
        # partial(out_grad)/partial(beta) = 1. Matching the RMSNorm convention
        # above, we sum across all but the last dim using grad_output as the
        # final gradient coming from the next operation.
        sum_dims = list(range(grad_output.dim() - 1))
        grad_gamma = (grad_output * normed_grad).sum(dim=sum_dims)
        grad_beta = grad_output.sum(dim=sum_dims)

        return grad_input, grad_gamma, grad_beta, None


class BackwardLayerNorm(nn.Module):
    """Identity in forward, LayerNorm in backward with learnable gamma/beta.

    Normalizes over the last dimension of the input tensor.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        """Initialize the backward-only LayerNorm.

        :param dim: Size of the last dimension being normalized.
        :type dim: int
        :param eps: Numerical stability constant used inside the normalization.
        :type eps: float
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), 0.01))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` unchanged; normalization is only applied in backward.

        :param x: Input tensor of arbitrary shape ``(..., dim)``.
        :type x: torch.Tensor
        :return: The same tensor ``x``.
        :rtype: torch.Tensor
        """
        return BackwardLayerNormFunction.apply(x, self.gamma, self.beta, self.eps)


class EBMBackwardsRMSNorm(nn.Module):
    """Standard RMSNorm in forward plus a backward-only RMSNorm on the gradient.

    NOTE: none of this worked well in practice; it could lower the initial loss
    to ``-log(len(vocab))`` but then the run diverged or converged slower.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize the combined forward/backward RMSNorm.

        :param dim: Size of the last dimension being normalized.
        :type dim: int
        :param eps: Numerical stability constant for both normalizations.
        :type eps: float
        """
        super().__init__()
        # Forward RMSNorm has its own weight_fwd
        self.forward_rms = RMSNorm(dim, eps)
        # Backward RMSNorm has a separate weight_bwd
        self.backward_rms = BackwardRMSNorm(dim, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward RMSNorm then the backward-only RMSNorm (identity fwd).

        :param x: Input tensor of shape ``(..., dim)``.
        :type x: torch.Tensor
        :return: Tensor with the same shape as ``x``.
        :rtype: torch.Tensor
        """
        x = self.forward_rms(x)
        x = self.backward_rms(x)
        return x

class DyT(nn.Module):
    """Dynamic Tanh (DyT) normalization-free affine transform."""

    def __init__(self, num_features: int, alpha_init_value: float = 0.5, bias_learnable: bool = True):
        """Initialize the DyT module.

        :param num_features: Feature dimension of the input.
        :type num_features: int
        :param alpha_init_value: Initial value for the learnable scalar ``alpha``.
        :type alpha_init_value: float
        :param bias_learnable: If False, freezes the bias at zero.
        :type bias_learnable: bool
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=bias_learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``tanh(alpha * x) * weight + bias``.

        :param x: Input tensor of shape ``(..., num_features)``.
        :type x: torch.Tensor
        :return: Tensor with the same shape as ``x``.
        :rtype: torch.Tensor
        """
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias

class RMSNorm(torch.nn.Module):
    """Root-mean-square layer normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize the RMSNorm normalization layer.

        :param dim: Dimension of the input tensor.
        :type dim: int
        :param eps: Small value added to the denominator for numerical stability.
        :type eps: float
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the RMSNorm normalization to the input tensor.

        :param x: The input tensor.
        :type x: torch.Tensor
        :return: The normalized tensor.
        :rtype: torch.Tensor
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the RMSNorm layer.

        :param x: Input tensor of shape ``(..., dim)``.
        :type x: torch.Tensor
        :return: Tensor with the same shape as ``x`` after applying RMSNorm.
        :rtype: torch.Tensor
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the frequency tensor of complex exponentials for RoPE.

    Calculates a frequency tensor of complex exponentials using the given
    dimension ``dim`` and the end index ``end``. The ``theta`` parameter scales
    the frequencies. The returned tensor contains complex64 values.

    :param dim: Dimension of the frequency tensor.
    :type dim: int
    :param end: End index for precomputing frequencies.
    :type end: int
    :param theta: Scaling factor for frequency computation.
    :type theta: float
    :return: Precomputed frequency tensor with complex exponentials.
    :rtype: torch.Tensor
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape the frequency tensor for broadcasting with ``x``.

    Reshapes the frequency tensor so that it matches the shape of the target
    tensor ``x`` for elementwise operations during rotary embedding.

    :param freqs_cis: Frequency tensor to be reshaped.
    :type freqs_cis: torch.Tensor
    :param x: Target tensor providing the broadcasting shape.
    :type x: torch.Tensor
    :return: Reshaped frequency tensor.
    :rtype: torch.Tensor
    :raises AssertionError: If the frequency tensor does not match the expected shape or ``x`` has too few dimensions.
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
    """Apply rotary embeddings to the query and key tensors.

    Applies rotary embeddings to the given query ``xq`` and key ``xk`` tensors
    using the provided frequency tensor ``freqs_cis``. The inputs are viewed as
    complex numbers, the frequency tensor is reshaped for broadcasting, and the
    results are returned as real tensors with the original shapes/dtypes.

    :param xq: Query tensor to apply rotary embeddings to.
    :type xq: torch.Tensor
    :param xk: Key tensor to apply rotary embeddings to.
    :type xk: torch.Tensor
    :param freqs_cis: Precomputed frequency tensor of complex exponentials.
    :type freqs_cis: torch.Tensor
    :return: Tuple of the rotated query tensor and key tensor.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    if xq.dtype == torch.float32:
        xq_ = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
    else:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    if xk.dtype == torch.float32:
        xk_ = torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))
    else:
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads ``n_rep`` times along the head dimension.

    Equivalent to ``torch.repeat_interleave(x, dim=2, repeats=n_rep)``.

    :param x: Tensor of shape ``(bs, slen, n_kv_heads, head_dim)``.
    :type x: torch.Tensor
    :param n_rep: Number of times to repeat each head.
    :type n_rep: int
    :return: Tensor of shape ``(bs, slen, n_kv_heads * n_rep, head_dim)``.
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
    """Multi-head attention module for the EBT autoregressive model."""
    def __init__(self, layer_id: int, args: EBTModelArgs):
        """Initialize the Attention module.

        :param layer_id: Index of the layer this attention belongs to.
        :type layer_id: int
        :param args: Model configuration parameters.
        :type args: EBTModelArgs
        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1  # NOTE: hardcoded since we are using DDP
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.use_ve = args.use_ve and has_ve is not None and has_ve(layer_id, args.n_layers)
        if self.use_ve:
            self.ve_gate_channels = 12
            self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_local_kv_heads, bias=False)
            nn.init.zeros_(self.ve_gate.weight)
        else:
            self.ve_gate = None

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        init_whole_model_weights(self.wq, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)

        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        init_whole_model_weights(self.wk, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)

        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        init_whole_model_weights(self.wv, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)

        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        init_whole_model_weights(self.wo, args.weight_initialization, weight_initialization_gain=args.weight_initialization_gain)

        self.time_offset = 2 if args.use_mcmc_time_embed else 1
        self.register_buffer('superdiag_rows', torch.arange(args.max_seq_len - 1))
        self.register_buffer('superdiag_cols', torch.arange(self.time_offset, args.max_seq_len + self.time_offset - 1))

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        ve: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the attention module.

        :param x: Input tensor of shape ``(B, 2*(S-1)+1, D)`` combining the
            original sequence and predicted embeddings.
        :type x: torch.Tensor
        :param start_pos: Starting position for attention caching.
        :type start_pos: int
        :param freqs_cis: Precomputed frequency tensor for rotary embeddings.
        :type freqs_cis: torch.Tensor
        :param mask: Optional attention mask tensor.
        :type mask: Optional[torch.Tensor]
        :param ve: Optional value embeddings added to ``xv``.
        :type ve: Optional[torch.Tensor]
        :return: Output tensor after attention, same shape as ``x``.
        :rtype: torch.Tensor
        """
        # NOTE: the usage of S-1/S/S+1 is confusing here; see the paper for the canonical naming.
        bsz, full_seqlen, _ = x.shape  # full_seqlen includes real embeds and pred embeds
        original_seqlen = (full_seqlen + 1)//2  # condition plus all original tokens; +1 is for the condition
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, full_seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)
        if ve is not None and self.ve_gate is not None:
            ve = ve.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[:, :, :self.ve_gate_channels]))
            xv = xv + gate.unsqueeze(-1) * ve

        # _o is for the original attention path (B, S-1, N, H; N = num heads, H = head dim)
        xq_o = xq[:, :original_seqlen, :, :]
        xk_o = xk[:, :original_seqlen, :, :]
        xv_o = xv[:, :original_seqlen, :, :]

        # _p is for the predicted attention path (B, S-1, N, H)
        xq_p = xq[:, original_seqlen:, :, :]
        xk_p = xk[:, original_seqlen:, :, :]
        xv_p = xv[:, original_seqlen:, :, :]


        xq_o, xk_o = apply_rotary_emb(xq_o, xk_o, freqs_cis=freqs_cis[:original_seqlen])

        # Use time_offset since these are the next preds and may or may not carry time embeddings
        # (offset=2 with time embeddings, offset=1 without). Verified equivalent to prepending a row on S.
        xq_p, xk_p = apply_rotary_emb(xq_p, xk_p, freqs_cis=freqs_cis[self.time_offset:original_seqlen+1])

        # Original attention calculation (standard causal attention) ############################

        # seqlen here is S-1 which equals original_seqlen
        xq_o = xq_o.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys_o = xk_o.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        values_o = xv_o.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        scores_o = torch.matmul(xq_o, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim)  # B, N, S-1, S-1
        if mask is not None:
            # Slice the (S, S) mask down to (S-1, S-1) for the original sequence path.
            o_mask = mask[:-1, :-1]
            scores_o = scores_o + o_mask  # (bs, n_local_heads, seqlen, seqlen)
        scores_o = F.softmax(scores_o.float(), dim=-1).type_as(xq_o)
        output_o = torch.matmul(scores_o, values_o)  # (bs, n_local_heads, seqlen, head_dim)
        output_o = output_o.transpose(1, 2).contiguous().view(bsz, original_seqlen, -1)  # (B, S-1, D)

        # Predicted-sequence attention calculation (EBT-specific) ##############################

        # seqlen here is S-1 which equals original_seqlen
        xq_p = xq_p.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys_p = xk_p.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        values_p = xv_p.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        # B, N, S-1, S: uses xq_p and keys_o since each next pred measures similarity against all
        # prior words; the trailing S comes from the extra condition token.
        scores_p = torch.matmul(xq_p, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim)

        # B, N, S-1, 1: needed because context_length = original_length + 1 for the superdiag insertion
        temp_append = torch.zeros((scores_p.shape[0], scores_p.shape[1], scores_p.shape[2], 1), dtype=scores_p.dtype, device=scores_p.device)
        # B, N, S-1, S: each next pred attends to prior words (S-1) plus itself (+1)
        scores_p = torch.cat((scores_p, temp_append), dim = -1)

        insertion_superdiagonal = (xq_p * keys_p).sum(dim = 3) / math.sqrt(self.head_dim)
        insertion_superdiagonal = insertion_superdiagonal.to(scores_p.dtype)  # needed when using non-fp32 precision
        # bs, n, s-1: attention score of next preds with themselves (like extracting the matmul diagonal)

        seq_len_minus_1 = scores_p.shape[2]
        superdiag_rows = self.superdiag_rows[:seq_len_minus_1]
        superdiag_cols = self.superdiag_cols[:seq_len_minus_1]

        # First remove superdiagonal values so attention to future tokens is zeroed out — this
        # prevents leakage of probability mass. Done in a differentiable way via masking.
        zero_superdiag = torch.zeros_like(insertion_superdiagonal, dtype=scores_p.dtype, device=scores_p.device)
        diagonal_removal_mask = torch.ones_like(scores_p, dtype=scores_p.dtype, device=scores_p.device)
        diagonal_removal_mask[:, :, superdiag_rows, superdiag_cols] = zero_superdiag
        scores_p = scores_p * diagonal_removal_mask

        # Then set the diagonal to next-pred self attention scores in a differentiable way.
        diagonal_addition_mask = torch.zeros_like(scores_p, dtype=scores_p.dtype, device=scores_p.device)
        diagonal_addition_mask[:, :, superdiag_rows, superdiag_cols] = insertion_superdiagonal
        scores_p = scores_p + diagonal_addition_mask

        if mask is not None:
            # S-1, S+1: like 0 0 0 -inf -inf -inf; 0 0 0 0 -inf -inf; etc.
            p_mask = mask[self.time_offset:, :]
            scores_p = scores_p + p_mask
        if scores_p.dtype != torch.float32:
            scores_p = scores_p.float()
        scores_p = F.softmax(scores_p, dim=-1)
        if scores_p.dtype != xq_p.dtype:
            scores_p = scores_p.to(xq_p.dtype)

        # NOTE: we extract the superdiagonal instead of doing a plain matmul afterwards because that
        # would require the same subsequence in the value matrix. We only have the original subsequence
        # and, separately, all next preds; so self-attention of preds must be handled explicitly.
        scores_p_superdiagonal = scores_p.diagonal(offset=self.time_offset, dim1=2, dim2=3)  # B, N, S; how much each superdiag token attends to itself

        # Keep scores_p as-is except for the superdiagonal (next-preds self attention); these can't
        # be naively multiplied by values_p or values_o, so we handle them separately below.
        scores_p = scores_p * diagonal_removal_mask

        scores_p = scores_p[:, :, :, :-1]  # B, N, S-1, S: the extra col from temp_append is now dropped
        output_p = torch.matmul(scores_p, values_o)  # B, N, S-1, H: next preds attend to all original prior tokens

        # Weighted sum of each next pred to its final embed representation.
        next_pred_self_attention = values_p * scores_p_superdiagonal.unsqueeze(dim = -1)  # B, N, S-1, H

        output_p = output_p + next_pred_self_attention  # B, N, S-1, H: adds each next pred attending to itself
        n_pred = full_seqlen - original_seqlen
        output_p = output_p.transpose(1, 2).contiguous().view(bsz, n_pred, -1)  # B, S-1, D

        # Return linear projection of the concatenated outputs ################################

        output = torch.cat((output_o, output_p), dim = 1)  # B, 2(S-1)+1, D
        return self.wo(output)


class FeedForward(nn.Module):
    """SwiGLU-style feed-forward block used inside each TransformerBlock."""

    def __init__(
        self,
        dim: int,
        ffn_dim_multiplier: Optional[float],
        weight_initialization: str,
        ebt_act_func: str = "silu",
        weight_initialization_gain: float = 1.0
    ):
        """Initialize the FeedForward module.

        :param dim: Input dimension.
        :type dim: int
        :param ffn_dim_multiplier: Multiplier that sets the hidden dimension.
            When ``None``, the hidden dimension equals ``dim``.
        :type ffn_dim_multiplier: Optional[float]
        :param weight_initialization: Name of the weight initialization scheme.
        :type weight_initialization: str
        :param ebt_act_func: Activation function name; one of
            ``{"silu", "relu", "gelu", "elu"}``.
        :type ebt_act_func: str
        :param weight_initialization_gain: Gain passed to the initializer.
        :type weight_initialization_gain: float
        """
        super().__init__()

        hidden_dim = dim if ffn_dim_multiplier is None else int(dim*ffn_dim_multiplier)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        init_whole_model_weights(self.w1, weight_initialization, weight_initialization_gain=weight_initialization_gain)

        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        init_whole_model_weights(self.w2, weight_initialization, weight_initialization_gain=weight_initialization_gain)

        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        init_whole_model_weights(self.w3, weight_initialization, weight_initialization_gain=weight_initialization_gain)

        self.act_func = {
            "silu": F.silu,
            "relu": F.relu,
            "gelu": F.gelu,
            "elu": F.elu
        }[ebt_act_func]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ``w2(act(w1(x)) * w3(x))``.

        :param x: Input tensor of shape ``(..., dim)``.
        :type x: torch.Tensor
        :return: Output tensor of shape ``(..., dim)``.
        :rtype: torch.Tensor
        """
        return self.w2(self.act_func(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Single Transformer block with pre-normalization residual connections."""

    def __init__(self, layer_id: int, args: EBTModelArgs):
        """Initialize a TransformerBlock.

        :param layer_id: Identifier for the layer.
        :type layer_id: int
        :param args: Model configuration parameters.
        :type args: EBTModelArgs
        :raises ValueError: If ``args.ebt_norm`` is not a recognized value.
        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(layer_id, args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            weight_initialization=args.weight_initialization,
            ebt_act_func=args.ebt_act_func,
            weight_initialization_gain=args.weight_initialization_gain
        )
        self.layer_id = layer_id
        if args.ebt_norm == "rms":
            self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        elif args.ebt_norm == "layer":
            self.attention_norm = nn.LayerNorm(args.dim)
            self.ffn_norm = nn.LayerNorm(args.dim)
        elif args.ebt_norm == "none":
            self.attention_norm = nn.Identity()
            self.ffn_norm = nn.Identity()
        elif args.ebt_norm == "dyt":
            self.attention_norm = DyT(args.dim, alpha_init_value=args.dyt_alpha_init)
            self.ffn_norm = DyT(args.dim, alpha_init_value=args.dyt_alpha_init)
        elif args.ebt_norm == "ebm_backwards_norm":
            self.attention_norm = EBMBackwardsRMSNorm(args.dim, eps=args.norm_eps)
            self.ffn_norm = EBMBackwardsRMSNorm(args.dim, eps=args.norm_eps)
        else:
            raise ValueError(f"Invalid ebt_norm value: {args.ebt_norm}")

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        use_create_graph: bool = False,
        ve: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform a forward pass through the TransformerBlock.

        :param x: Input tensor of shape ``(B, 2*(S-1)+1, D)``.
        :type x: torch.Tensor
        :param start_pos: Starting position for attention caching.
        :type start_pos: int
        :param freqs_cis: Precomputed cosine and sine frequencies.
        :type freqs_cis: torch.Tensor
        :param mask: Masking tensor for attention.
        :type mask: Optional[torch.Tensor]
        :param use_create_graph: Whether we are in a ``create_graph=True`` MCMC
            step. When True, gradient checkpointing is disabled to avoid
            doubling activation memory — ``create_graph=True`` already requires
            keeping the recomputation graph's intermediate activations, so GC
            recomputation cannot save memory and instead stores a second copy.
        :type use_create_graph: bool
        :param ve: Optional value embeddings forwarded to the attention layer.
        :type ve: Optional[torch.Tensor]
        :return: Output tensor with the same shape as ``x``.
        :rtype: torch.Tensor
        """
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask, ve
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class EBTTimeConcat(nn.Module):
    """Energy-Based Transformer with optional per-MCMC-step time embeddings."""

    def __init__(self, params: EBTModelArgs, max_mcmc_steps: int, gradient_checkpointing: bool = False, use_mcmc_time_embed: bool = False):
        """Initialize a Transformer model.

        :param params: Model configuration parameters.
        :type params: EBTModelArgs
        :param max_mcmc_steps: Maximum number of MCMC steps; sets the size of
            the time-embedding table when ``use_mcmc_time_embed`` is True.
        :type max_mcmc_steps: int
        :param gradient_checkpointing: If True, enables gradient checkpointing
            inside the transformer blocks.
        :type gradient_checkpointing: bool
        :param use_mcmc_time_embed: If True, use per-step time embeddings
            (original behavior). If False, all MCMC steps share the same
            transition kernel, enabling arbitrary inference-time step counts.
        :type use_mcmc_time_embed: bool
        :raises ValueError: If ``params.ebt_norm`` is not a recognized value.
        """
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.use_mcmc_time_embed = use_mcmc_time_embed
        self.use_ve = params.use_ve and build_value_embeds is not None
        if self.use_ve:
            n_kv_heads = params.n_heads if params.n_kv_heads is None else params.n_kv_heads
            self.kv_dim = n_kv_heads * (params.dim // params.n_heads)
            self.value_embeds = build_value_embeds(params.n_layers, params.vocab_size, self.kv_dim)
            ve_init_bound = math.sqrt(3.0) * (params.dim ** -0.5)
            for ve in self.value_embeds.values():
                nn.init.uniform_(ve.weight, -ve_init_bound, ve_init_bound)
                ve.to(dtype=torch.bfloat16)
        else:
            self.value_embeds = None
            self.kv_dim = None

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        if params.ebt_norm == "rms":
            self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        elif params.ebt_norm == "layer":
            self.norm = nn.LayerNorm(params.dim)
        elif params.ebt_norm == "none":
            self.norm = nn.Identity()
        elif params.ebt_norm == "dyt":
            self.norm = DyT(params.dim, alpha_init_value=params.dyt_alpha_init, bias_learnable = False)  # no learnable bias since grad cannot be computed for a final bias term in EBT
        elif params.ebt_norm == "ebm_backwards_norm":
            self.norm = EBMBackwardsRMSNorm(params.dim, eps=params.norm_eps)
        else:
            raise ValueError(f"Invalid ebt_norm value: {params.ebt_norm}")

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len
        )

        if self.use_mcmc_time_embed:
            self.time_embeddings = nn.Embedding(max_mcmc_steps, params.dim)

        self.final_layer = nn.Linear(params.dim, 1, bias = False)
        init_whole_model_weights(self.final_layer, self.params.weight_initialization)

    def forward(
        self,
        embeddings: torch.Tensor,
        start_pos: int,
        mcmc_step: int = 0,
        use_create_graph: bool = False,
        real_token_ids: Optional[torch.Tensor] = None,
        predicted_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform a forward pass through the Transformer model.

        :param embeddings: Input embeddings of shape ``(B, 2*(S-1), D)``
            (used instead of token ids since this module is also used for
            vision inputs). When ``use_mcmc_time_embed`` is True, a time
            embedding token is prepended, giving shape ``(B, 2*(S-1)+1, D)``.
        :type embeddings: torch.Tensor
        :param start_pos: Starting position for attention caching.
        :type start_pos: int
        :param mcmc_step: Current MCMC step index, used for the time embeddings.
        :type mcmc_step: int
        :param use_create_graph: Whether we are in the final MCMC step that
            requires ``create_graph=True``. Forwarded to each TransformerBlock
            to decide whether gradient checkpointing should be enabled.

            * False (default): a non-final step or inference, GC can save memory.
            * True: final step with ``learning=True``; GC is disabled to avoid
              doubling activation memory (``create_graph=True`` already keeps
              the recomputation graph, so GC recomputation adds a second copy).
        :type use_create_graph: bool
        :param real_token_ids: Optional real token ids used to build value
            embeddings when value embedding is enabled.
        :type real_token_ids: Optional[torch.Tensor]
        :param predicted_tokens: Optional predicted tokens used to build value
            embeddings when value embedding is enabled.
        :type predicted_tokens: Optional[torch.Tensor]
        :return: Output energies tensor produced by the final projection.
        :rtype: torch.Tensor
        """
        _bsz = embeddings.shape[0]

        if self.use_mcmc_time_embed:
            mcmc_step_val = mcmc_step
            mcmc_step = torch.empty((_bsz,), device=embeddings.device, dtype=torch.long).fill_(mcmc_step_val)
            time_embeddings = self.time_embeddings(mcmc_step).unsqueeze(dim=1)  # expanded to B, 1, D
            embeddings = torch.cat((time_embeddings, embeddings), dim = 1)  # B, 2S - 1 + 1, D

        _bsz, seqlen = embeddings.shape[:2]
        if self.use_mcmc_time_embed:
            seqlen = (seqlen+3) // 2  # passed seqlen is 2(S-1)+1+1(time), so (+3)//2 = S+1
        else:
            seqlen = (seqlen+2) // 2  # passed seqlen is 2(S-1), so (+2)//2 = S
        self.freqs_cis = self.freqs_cis.to(embeddings.device)

        # Dynamically extend freqs_cis if the required length exceeds the precomputed length.
        required_length = start_pos + seqlen
        if required_length > self.freqs_cis.shape[0]:
            # Recompute a longer freqs_cis
            new_freqs_cis = precompute_freqs_cis(
                self.params.dim // self.params.n_heads,
                required_length
            ).to(embeddings.device)
            self.freqs_cis = new_freqs_cis

        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.empty((seqlen, seqlen), device=embeddings.device).fill_(float("-inf"))

            mask.triu_(diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=embeddings.device),
                mask
            ]).type_as(embeddings)
            # Causal mask default layout: 0, -inf, -inf / 0, 0, -inf / 0, 0, 0


            for i, layer in enumerate(self.layers):
                ve = None
                if self.use_ve and real_token_ids is not None and predicted_tokens is not None:
                    extra_prefix_tokens = 1 if self.use_mcmc_time_embed else 0
                    ve = build_layer_ve(
                        self.value_embeds,
                        i,
                        real_token_ids,
                        predicted_tokens,
                        extra_prefix_tokens=extra_prefix_tokens,
                    )
                embeddings = layer(embeddings, start_pos, freqs_cis, mask, ve=ve)
            embeddings = self.norm(embeddings)
            if self.use_mcmc_time_embed:
                embeddings = embeddings[:, 1:]  # remove temporal embed
            energies = self.final_layer(embeddings)

            energies = energies[:, embeddings.shape[1] // 2:]
            return energies
