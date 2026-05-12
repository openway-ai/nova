#NOTE most code gotten from llama2 codebase -- credit:https://github.com/meta-llama/llama
import torch
from torch import nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from openebm.elm.utils import init_whole_model_weights, EBTModelArgs
try:
    from openebm.elm.ve import build_layer_ve, build_value_embeds, has_ve
except ImportError:
    has_ve = None
    build_value_embeds = None
    build_layer_ve = None


class BackwardRMSNormFunction(torch.autograd.Function):
    """Identity in forward; RMSNorm applied to gradients in backward.

    The weight is learnable and enters only via the backward pass.
    """

    @staticmethod
    def forward(ctx: "torch.autograd.function.FunctionCtx", input_: torch.Tensor, weight: torch.Tensor, eps: float, use_fp32_norm: bool) -> torch.Tensor:
        """Identity forward that stashes ``weight``/``eps``/``use_fp32_norm`` for backward.

        :param ctx: Autograd context used to cache tensors for backward.
        :type ctx: torch.autograd.function.FunctionCtx
        :param input_: Input tensor; returned unchanged.
        :type input_: torch.Tensor
        :param weight: Learnable scale used only in backward.
        :type weight: torch.Tensor
        :param eps: Numerical-stability constant.
        :type eps: float
        :param use_fp32_norm: If ``True``, the backward RMS is computed in FP32
            then cast back; if ``False`` it stays in the input dtype.
        :type use_fp32_norm: bool
        :return: ``input_`` unchanged.
        :rtype: torch.Tensor
        """
        ctx.save_for_backward(weight)
        ctx.eps = eps
        ctx.use_fp32_norm = use_fp32_norm
        return input_  # Identity forward

    @staticmethod
    def backward(ctx: "torch.autograd.function.FunctionCtx", grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        """Apply RMSNorm to ``grad_output`` using the stashed ``weight``.

        :param ctx: Autograd context populated in :meth:`forward`.
        :type ctx: torch.autograd.function.FunctionCtx
        :param grad_output: Gradient tensor flowing back from the next op.
        :type grad_output: torch.Tensor
        :return: ``(grad_input, grad_weight, None, None)`` — one ``None`` per
            non-tensor forward argument.
        :rtype: Tuple[torch.Tensor, torch.Tensor, None, None]
        """
        # print("grad_output", grad_output)
        (weight,) = ctx.saved_tensors
        eps = ctx.eps

        # Compute RMS of grad_output over last dim
        if ctx.use_fp32_norm:
            rms = torch.rsqrt(grad_output.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        else:
            rms = torch.rsqrt(grad_output.pow(2).mean(dim=-1, keepdim=True) + eps)

        # Normalized gradient wrt the input
        grad_input = grad_output * rms * weight  # shape matches grad_output

        # Gradient wrt 'weight'
        # The transform is: out_grad = grad_output * (rms * weight).
        # So partial wrt weight is grad_output * rms.
        # We sum over all but the last dimension.
        sum_dims = list(range(grad_output.dim() - 1))
        grad_weight = (grad_output * rms).sum(dim=sum_dims)

        return grad_input, grad_weight, None, None


class BackwardRMSNorm(nn.Module):
    """Identity in forward, RMSNorm in backward, with a learnable weight."""

    def __init__(self, dim: int, eps: float = 1e-6, use_fp32_norm: bool = True) -> None:
        """Initialize the backward-only RMSNorm.

        :param dim: Size of the last dimension being normalized.
        :type dim: int
        :param eps: Numerical-stability constant.
        :type eps: float
        :param use_fp32_norm: Route the backward RMS through FP32 when ``True``.
        :type use_fp32_norm: bool
        """
        super().__init__()
        self.weight = nn.Parameter(torch.full((dim,), 0.01))
        # self.weight = nn.Parameter(torch.ones(dim))

        self.eps = eps
        self.use_fp32_norm = use_fp32_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` unchanged; normalization is only applied in backward.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: The same tensor ``x``.
        :rtype: torch.Tensor
        """
        return BackwardRMSNormFunction.apply(x, self.weight, self.eps, self.use_fp32_norm)
    
class BackwardLayerNormFunction(torch.autograd.Function):
    """Identity in forward; LayerNorm applied to gradients in backward."""

    @staticmethod
    def forward(ctx: "torch.autograd.function.FunctionCtx", input_: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
        """Identity forward that stashes ``gamma``/``beta``/``eps`` for backward.

        :param ctx: Autograd context used to cache tensors for backward.
        :type ctx: torch.autograd.function.FunctionCtx
        :param input_: Input tensor; returned unchanged.
        :type input_: torch.Tensor
        :param gamma: Learnable scale used only in backward.
        :type gamma: torch.Tensor
        :param beta: Learnable shift used only in backward.
        :type beta: torch.Tensor
        :param eps: Numerical-stability constant.
        :type eps: float
        :return: ``input_`` unchanged.
        :rtype: torch.Tensor
        """
        ctx.save_for_backward(gamma, beta)
        ctx.eps = eps
        # Identity forward pass
        return input_

    @staticmethod
    def backward(ctx: "torch.autograd.function.FunctionCtx", grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Apply LayerNorm to ``grad_output`` using the stashed ``gamma``/``beta``.

        :param ctx: Autograd context populated in :meth:`forward`.
        :type ctx: torch.autograd.function.FunctionCtx
        :param grad_output: Gradient tensor flowing back from the next op.
        :type grad_output: torch.Tensor
        :return: ``(grad_input, grad_gamma, grad_beta, None)`` — one ``None``
            for the non-tensor ``eps`` forward argument.
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

        # Output gradient for the input
        #    out_grad = gamma * normed_grad + beta
        grad_input = gamma * normed_grad + beta

        # For the chain rule:
        # partial(out_grad) / partial(gamma) = normed_grad
        # partial(out_grad) / partial(beta)  = 1
        # Then multiply each by grad_output for the chain rule, i.e. (grad_output_from_next * partial).
        # But in this scenario, the "local" gradient transform is the final. By convention (matching the RMSNorm example),
        # we multiply normed_grad by grad_output to find derivative wrt gamma, etc.
        # However, the code snippet below just sums across the relevant dims,
        # assuming grad_output is the final gradient from the next operation.

        sum_dims = list(range(grad_output.dim() - 1))
        grad_gamma = (grad_output * normed_grad).sum(dim=sum_dims)
        grad_beta = grad_output.sum(dim=sum_dims)

        return grad_input, grad_gamma, grad_beta, None


class BackwardLayerNorm(nn.Module):
    """Identity in forward, LayerNorm in backward, with learnable gamma and beta.

    Normalizes over the last dimension.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        """Initialize the backward-only LayerNorm.

        :param dim: Size of the last dimension being normalized.
        :type dim: int
        :param eps: Numerical-stability constant.
        :type eps: float
        """
        super().__init__()
        # self.gamma = nn.Parameter(torch.ones(dim))
        self.gamma = nn.Parameter(torch.full((dim,), 0.01))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` unchanged; normalization is only applied in backward.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: The same tensor ``x``.
        :rtype: torch.Tensor
        """
        return BackwardLayerNormFunction.apply(x, self.gamma, self.beta, self.eps)


class EBMBackwardsRMSNorm(nn.Module):
    """Forward RMSNorm followed by backward-only RMSNorm (identity forward).

    .. note::

        None of these variants worked better than plain RMSNorm. They could
        drive the initial loss close to ``-log(len(vocab))`` but then the
        run diverged or converged slower.
    """

    def __init__(self, dim: int, eps: float = 1e-6, use_fp32_norm: bool = True) -> None:
        """Initialize the combined forward/backward RMSNorm.

        :param dim: Size of the last dimension being normalized.
        :type dim: int
        :param eps: Numerical-stability constant.
        :type eps: float
        :param use_fp32_norm: Whether the inner RMSNorms use the FP32 path.
        :type use_fp32_norm: bool
        """
        super().__init__()
        # Forward RMSNorm has its own weight_fwd
        self.forward_rms = RMSNorm(dim, eps, use_fp32_norm=use_fp32_norm)
        # Backward RMSNorm has a separate weight_bwd
        self.backward_rms = BackwardRMSNorm(dim, eps, use_fp32_norm=use_fp32_norm)
        # self.backward_ln = BackwardLayerNorm(dim, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward RMSNorm then the backward-only RMSNorm (identity fwd).

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Tensor after forward RMSNorm (backward-only pass is identity
            on the forward path).
        :rtype: torch.Tensor
        """
        x = self.forward_rms(x)
        x = self.backward_rms(x)
        # x = self.backward_ln(x)
        return x


class DyT(nn.Module):
    """Dynamic Tanh (DyT) activation-norm hybrid.

    See https://jiachenzhu.github.io/DyT/ for the reference implementation.
    """

    def __init__(self, num_features: int, alpha_init_value: float = 0.5, bias_learnable: bool = True) -> None:
        """Initialize the DyT layer.

        :param num_features: Feature dimension ``D``.
        :type num_features: int
        :param alpha_init_value: Initial value for the scalar ``alpha``.
        :type alpha_init_value: float
        :param bias_learnable: If ``False``, ``bias`` is frozen at zero.
        :type bias_learnable: bool
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=bias_learnable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``tanh(alpha * x) * weight + bias``.

        :param x: Input tensor with trailing dim of size ``num_features``.
        :type x: torch.Tensor
        :return: Output tensor of the same shape.
        :rtype: torch.Tensor
        """
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias

class RMSNorm(torch.nn.Module):
    """Root-mean-square layer normalization."""

    def __init__(self, dim: int, eps: float = 1e-6, use_fp32_norm: bool = True) -> None:
        """Initialize the RMSNorm normalization layer.

        :param dim: Dimension of the input tensor.
        :type dim: int
        :param eps: Small value added to the denominator for numerical
            stability.
        :type eps: float
        :param use_fp32_norm: If ``True``, compute normalization in FP32 then
            cast back. Should be ``False`` for ``bf16-true`` to avoid FP32
            graph nodes.
        :type use_fp32_norm: bool
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_fp32_norm = use_fp32_norm

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization to ``x`` over the last dimension.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Normalized tensor with the same shape as ``x``.
        :rtype: torch.Tensor
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the RMSNorm layer.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Tensor with the same shape as ``x`` after applying RMSNorm
            and the learnable scale.
        :rtype: torch.Tensor
        """
        if self.use_fp32_norm:
            output = self._norm(x.float()).type_as(x)
        else:
            output = self._norm(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute the cosine and sine frequency tensors for rotary embeddings.

    Returns ``(cos, sin)`` tensors of shape ``(end, dim // 2)``, stored as
    float32 but applied via real-valued arithmetic to avoid dtype casting in
    the forward/backward graph.

    :param dim: Head dimension.
    :type dim: int
    :param end: Maximum sequence length.
    :type end: int
    :param theta: RoPE base.
    :type theta: float
    :return: ``(cos, sin)`` each of shape ``(end, dim // 2)``.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (end, dim//2)
    return freqs.cos(), freqs.sin()  # both float32, shape (end, dim//2)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings using real-valued cos/sin arithmetic.

    Avoids any dtype casting so the computation stays in the model's native
    dtype (for example bfloat16), which is important for keeping the MCMC
    autograd graph in low precision.

    :param xq: Query tensor of shape ``(..., seq, n_heads, head_dim)``.
    :type xq: torch.Tensor
    :param xk: Key tensor of shape ``(..., seq, n_kv_heads, head_dim)``.
    :type xk: torch.Tensor
    :param freqs_cis: ``(cos, sin)`` each of shape ``(seq, head_dim // 2)``,
        precomputed by :func:`precompute_freqs_cis`.
    :type freqs_cis: Tuple[torch.Tensor, torch.Tensor]
    :return: Rotated ``xq`` and ``xk`` in their original dtype.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    cos, sin = freqs_cis  # (seq, head_dim//2)
    # cast cos/sin to match input dtype (e.g. bfloat16) to avoid float32 intermediates
    cos = cos.to(dtype=xq.dtype)
    sin = sin.to(dtype=xq.dtype)

    # xq/xk: (bs, seq, n_heads, head_dim) → split last dim into pairs
    # cos/sin: (seq, head_dim//2) → broadcast over bs and n_heads
    # reshape: (bs, seq, n_heads, head_dim//2, 2)
    xq_r = xq.reshape(*xq.shape[:-1], -1, 2)   # (..., head_dim//2, 2)
    xk_r = xk.reshape(*xk.shape[:-1], -1, 2)

    xq0, xq1 = xq_r[..., 0], xq_r[..., 1]      # (..., head_dim//2)
    xk0, xk1 = xk_r[..., 0], xk_r[..., 1]

    # cos/sin need shape (1, seq, 1, head_dim//2) for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(2)          # (1, seq, 1, head_dim//2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    xq_out = torch.stack([xq0 * cos - xq1 * sin,
                          xq0 * sin + xq1 * cos], dim=-1).flatten(-2)
    xk_out = torch.stack([xk0 * cos - xk1 * sin,
                          xk0 * sin + xk1 * cos], dim=-1).flatten(-2)
    return xq_out, xk_out


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads ``n_rep`` times along the head dimension.

    Equivalent to ``torch.repeat_interleave(x, dim=2, repeats=n_rep)``.

    :param x: Tensor of shape ``(bs, slen, n_kv_heads, head_dim)``.
    :type x: torch.Tensor
    :param n_rep: Number of repetitions along the KV-head dimension.
    :type n_rep: int
    :return: Expanded tensor of shape
        ``(bs, slen, n_kv_heads * n_rep, head_dim)``.
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
    """Multi-head attention module with a dual original/predicted path.

    Implements the two-stream attention used by the time-embed EBT variant:
    one stream over the original tokens and a second stream over the
    predicted tokens that additionally attends to the original sequence.
    """

    def __init__(self, layer_id: int, args: EBTModelArgs) -> None:
        """Initialize the Attention module.

        :param layer_id: Layer index (used for VE routing).
        :type layer_id: int
        :param args: Model configuration.
        :type args: EBTModelArgs
        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1 #NOTE this is hardcoded since we are using DDP
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
        self.use_sdpa_attention = args.use_sdpa_attention
        self.register_buffer('superdiag_rows', torch.arange(args.max_seq_len - 1))
        self.register_buffer('superdiag_cols', torch.arange(self.time_offset, args.max_seq_len + self.time_offset - 1))
        # self.wq = ColumnParallelLinear(
        #     args.dim,
        #     args.n_heads * self.head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        # self.wk = ColumnParallelLinear(
        #     args.dim,
        #     self.n_kv_heads * self.head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        # self.wv = ColumnParallelLinear(
        #     args.dim,
        #     self.n_kv_heads * self.head_dim,
        #     bias=False,
        #     gather_output=False,
        #     init_method=lambda x: x,
        # )
        # self.wo = RowParallelLinear(
        #     args.n_heads * self.head_dim,
        #     args.dim,
        #     bias=False,
        #     input_is_parallel=True,
        #     init_method=lambda x: x,
        # )

        # self.cache_k = torch.zeros(
        #     (
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_local_kv_heads,
        #         self.head_dim,
        #     )
        # )
        # self.cache_v = torch.zeros(
        #     (
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_local_kv_heads,
        #         self.head_dim,
        #     )
        # )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        mask: Optional[torch.Tensor],
        ve: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the attention module.

        .. note::

            The ``S-1`` / ``S`` / ``S+1`` bookkeeping around predicted and
            original tokens is non-obvious; consult the EBT paper for
            definitions.

        :param x: Input tensor of shape ``(B, 2*(S-1) [+1], D)``; the
            condition prefix, original tokens, and predicted tokens are all
            concatenated along the sequence dim.
        :type x: torch.Tensor
        :param start_pos: Starting position for caching (reserved; KV cache
            is currently disabled in this module).
        :type start_pos: int
        :param freqs_cis: ``(cos, sin)`` tuple from
            :func:`precompute_freqs_cis` sliced to the current sequence
            range.
        :type freqs_cis: Tuple[torch.Tensor, torch.Tensor]
        :param mask: Optional attention mask of shape ``(seqlen, seqlen)``.
        :type mask: Optional[torch.Tensor]
        :param ve: Optional value-embedding tensor routed via :mod:`ve`;
            ``None`` when VE is disabled.
        :type ve: Optional[torch.Tensor]
        :return: Output tensor of shape ``(B, 2*(S-1) [+1], D)``.
        :rtype: torch.Tensor
        """
        # NOTE the usage of S-1/S/S+1 is messed up and confusing here, I recommend checking the paper
        bsz, full_seqlen, _ = x.shape # full_seqlen includes real embeds and pred embeds
        original_seqlen = (full_seqlen + 1)//2 # this is just the condition plus all original tokens, +1 is bc of condition
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, full_seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)
        if ve is not None and self.ve_gate is not None:
            ve = ve.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[:, :, :self.ve_gate_channels]))
            xv = xv + gate.unsqueeze(-1) * ve
        
        # _o is for original attention stuff
        xq_o = xq[:, :original_seqlen, :, :] #B, S-1, N, H (N and H are num head and head dim respectively)
        xk_o = xk[:, :original_seqlen, :, :]
        xv_o = xv[:, :original_seqlen, :, :]
        
        # _p is for predicted attention stuff
        xq_p = xq[:, original_seqlen:, :, :] #B, S-1, N, H (N and H are num head and head dim respectively)
        xk_p = xk[:, original_seqlen:, :, :]
        xv_p = xv[:, original_seqlen:, :, :]
        
        
        cos, sin = freqs_cis
        xq_o, xk_o = apply_rotary_emb(xq_o, xk_o, freqs_cis=(cos[:original_seqlen], sin[:original_seqlen]))

        xq_p, xk_p = apply_rotary_emb(xq_p, xk_p, freqs_cis=(cos[self.time_offset:original_seqlen+1], sin[self.time_offset:original_seqlen+1]))
        # I tested this compared to prepending row on S dimension and the tensors were the same

        # self.cache_k = self.cache_k.to(xq)
        # self.cache_v = self.cache_v.to(xq)

        # self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        # self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # keys = self.cache_k[:bsz, : start_pos + seqlen]
        # values = self.cache_v[:bsz, : start_pos + seqlen]

        # # repeat k/v heads if n_kv_heads < n_heads # this does nothing since self.n_rep = 1
        # keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        # values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        
        #original attn calc############################################

        # seqlen here is S-1 which = original_seqlen
        # Use F.scaled_dot_product_attention to avoid materialising the full [B, N, S, S]
        # attention score matrix, which OOMs at context=2048.
        xq_o = xq_o.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys_o = xk_o.transpose(1, 2) # (bs, n_local_heads, seqlen, head_dim)
        values_o = xv_o.transpose(1, 2) # (bs, n_local_heads, seqlen, head_dim)

        if self.use_sdpa_attention:
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False, enable_cudnn=False):
                output_o = F.scaled_dot_product_attention(xq_o, keys_o, values_o, is_causal=True)
            output_o = output_o.transpose(1, 2).contiguous().view(bsz, original_seqlen, -1)
        else:
            scores_o = torch.matmul(xq_o, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim) # B, N, S-1, S-1
            if mask is not None:
                #this mask needs to be seqlen, seqlen, was S, S
                o_mask = mask[:-1, :-1] #set to S-1, S-1 like 0 -inf -inf; 0 0 -inf, etc
                scores_o = scores_o + o_mask  # (bs, n_local_heads, seqlen, seqlen)
            scores_o = F.softmax(scores_o.float(), dim=-1).type_as(xq_o)
            output_o = torch.matmul(scores_o, values_o)  # (bs, n_local_heads, seqlen, head_dim)
            output_o = output_o.transpose(1, 2).contiguous().view(bsz, original_seqlen, -1) # has B, S-1, D after
        
        #pred sequence attn calc is for energy-based transformer ########################################################################################
        
        xq_p = xq_p.transpose(1, 2)  # (bs, n_local_heads, pred_seqlen, head_dim)
        keys_p = xk_p.transpose(1, 2) # (bs, n_local_heads, pred_seqlen, head_dim)
        values_p = xv_p.transpose(1, 2) # (bs, n_local_heads, pred_seqlen, head_dim)
        n_pred = full_seqlen - original_seqlen

        if self.use_sdpa_attention:
            K = original_seqlen           # number of orig tokens
            pred_seqlen = n_pred          # number of pred tokens

            keys_all   = torch.cat([keys_o,   keys_p],   dim=2)
            values_all = torch.cat([values_o, values_p], dim=2)
            orig_rows = torch.arange(pred_seqlen, device=x.device).unsqueeze(1)  # [pred_seqlen, 1]
            orig_cols = torch.arange(K,           device=x.device).unsqueeze(0)  # [1, K]
            orig_mask_part = torch.where(
                orig_cols <= orig_rows + (self.time_offset - 1),
                x.new_zeros(1),
                x.new_full((1,), float('-inf'))
            )  # [pred_seqlen, K]

            self_mask_part = x.new_full((pred_seqlen, pred_seqlen), float('-inf'))
            self_mask_part.fill_diagonal_(0.0)  # each pred token attends only to itself

            mask_p = torch.cat([orig_mask_part, self_mask_part], dim=1)  # [pred_seqlen, K + pred_seqlen]

            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False, enable_cudnn=False):
                output_p = F.scaled_dot_product_attention(xq_p, keys_all, values_all, attn_mask=mask_p)
            output_p = output_p.transpose(1, 2).contiguous().view(bsz, n_pred, -1)
        else:
            scores_p = torch.matmul(xq_p, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim) # B, N, S-1, S; this uses xq_p and keys_o since for every next pred calcs similarity to all prev words; right S is because have extra condition

            temp_append = torch.zeros((scores_p.shape[0], scores_p.shape[1], scores_p.shape[2], 1), dtype=scores_p.dtype, device=scores_p.device) # B, N, S-1, 1; is used since context_length = original_length +1, superdiag needs this
            scores_p = torch.cat((scores_p, temp_append), dim = -1)# is B, N, S-1, S; represents for each next pred (S-1 row) attending to all previous words (S-1) and then itself +1
            
            insertion_superdiagonal = (xq_p * keys_p).sum(dim = 3) / math.sqrt(self.head_dim)
            insertion_superdiagonal = insertion_superdiagonal.to(scores_p.dtype) # for if using non 32 precision
            # bs, n, s-1 ; this calcs attn score of next preds with themselves, is like grabbing diag of matmul
            
            seq_len_minus_1 = scores_p.shape[2]
            superdiag_rows = self.superdiag_rows[:seq_len_minus_1]
            superdiag_cols = self.superdiag_cols[:seq_len_minus_1]
      
            # first remove superdiagonal values so doesnt use attention to future tokens--prevents leakage of probability mass
            zero_superdiag = torch.zeros_like(insertion_superdiagonal, dtype=scores_p.dtype, device=scores_p.device) # for zeroing out superdiag since dont want to include in matmul, do this in differentiable way
            diagonal_removal_mask = torch.ones_like(scores_p, dtype=scores_p.dtype, device=scores_p.device)
            diagonal_removal_mask[:, :, superdiag_rows, superdiag_cols] = zero_superdiag
            scores_p = scores_p * diagonal_removal_mask
            
            # then set diagonal to next pred self attention scores in differentiable way
            diagonal_addition_mask = torch.zeros_like(scores_p, dtype=scores_p.dtype, device=scores_p.device)
            diagonal_addition_mask[:, :, superdiag_rows, superdiag_cols] = insertion_superdiagonal
            scores_p = scores_p + diagonal_addition_mask
            
            if mask is not None:
                p_mask = mask[self.time_offset:, :]  #S-1, S+1 like 0 0 0 -inf -inf -inf; 0 0 0 0 -inf -inf; etc
                scores_p = scores_p + p_mask
            if scores_p.dtype != torch.float32:
                scores_p = scores_p.float()
            scores_p = F.softmax(scores_p, dim=-1)
            if scores_p.dtype != xq_p.dtype:
                scores_p = scores_p.to(xq_p.dtype)
            
            #Q: why do I need to extract superdiagonal why cant i just do matmul after? A: its bc would need same subsequence in value matrix but dont have it, have original subsequence and then seperately all next preds
            scores_p_superdiagonal = scores_p.diagonal(offset=self.time_offset, dim1=2, dim2=3) # is B, N, S; basically how much each token on this superdiag should attent to itself; clone since dont want mask to change this
            
            scores_p = scores_p * diagonal_removal_mask # keeps scores_p as is except for superdiagonal which is next preds attention to selves, cant multiply these naively by values_p or values_o
            
            scores_p = scores_p[:, :, :, :-1] # B, N, S-1, S now; next preds/scores_p_superdiagonal was why needed extra col earlier (temp_append)
            output_p = torch.matmul(scores_p, values_o) # B, N, S-1, H; is how next preds attend to all original previous tokens;
            
            #next_pred_self_attention is to get self attention based on extracted superdiagonal and the values matrix (for predictions)
            next_pred_self_attention = values_p * scores_p_superdiagonal.unsqueeze(dim = -1) # B, N, S-1, H this is for weighted sum of each next pred to its final embed rep.
            
            output_p = output_p + next_pred_self_attention # B, N, S-1, H adding this is adding the aspect of each next pred embedding attending to itself
            output_p = output_p.transpose(1, 2).contiguous().view(bsz, n_pred, -1) # after this is B, S-1, D
        
        #return linear projection of concatted outputs ########################################################################################
        
        output = torch.cat((output_o, output_p), dim = 1) # B, 2(S-1)+1, D
        return self.wo(output)


class FeedForward(nn.Module):
    """Gated feed-forward network used inside each Transformer block."""

    def __init__(
        self,
        dim: int,
        ffn_dim_multiplier: Optional[float],
        weight_initialization: str,
        ebt_act_func: str = "silu",
        weight_initialization_gain: float = 1.0,
    ) -> None:
        """Initialize the FeedForward module.

        :param dim: Input/output feature dimension.
        :type dim: int
        :param ffn_dim_multiplier: Hidden-dim multiplier; when ``None`` the
            hidden dim equals ``dim``.
        :type ffn_dim_multiplier: Optional[float]
        :param weight_initialization: Name of the weight initializer.
        :type weight_initialization: str
        :param ebt_act_func: Activation function name; one of ``"silu"``,
            ``"relu"``, ``"gelu"``, ``"elu"``.
        :type ebt_act_func: str
        :param weight_initialization_gain: Gain forwarded to the initializer.
        :type weight_initialization_gain: float
        """
        super().__init__()
        # hidden_dim = int(2 * hidden_dim / 3)
        # # custom dim factor multiplier
        # if ffn_dim_multiplier is not None:
        #     hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        # hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

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
        
        # self.w1 = ColumnParallelLinear(
        #     dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        # )
        # self.w2 = RowParallelLinear(
        #     hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        # )
        # self.w3 = ColumnParallelLinear(
        #     dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated feed-forward: ``w2(act(w1(x)) * w3(x))``.

        :param x: Input tensor of shape ``(..., dim)``.
        :type x: torch.Tensor
        :return: Output tensor of shape ``(..., dim)``.
        :rtype: torch.Tensor
        """
        return self.w2(self.act_func(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """A single EBT transformer block: Attention + FeedForward with norms."""

    def __init__(self, layer_id: int, args: EBTModelArgs) -> None:
        """Initialize a TransformerBlock.

        :param layer_id: Identifier for the layer (forwarded to
            :class:`Attention`).
        :type layer_id: int
        :param args: Model configuration.
        :type args: EBTModelArgs
        :raises ValueError: If ``args.ebt_norm`` is not one of ``"rms"``,
            ``"layer"``, ``"none"``, ``"dyt"``, ``"ebm_backwards_norm"``.
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
        use_fp32 = getattr(args, 'float_precision', '') != "bf16-true"
        if args.ebt_norm == "rms":
            self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps, use_fp32_norm=use_fp32)
            self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, use_fp32_norm=use_fp32)
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
            self.attention_norm = EBMBackwardsRMSNorm(args.dim, eps=args.norm_eps, use_fp32_norm=use_fp32)
            self.ffn_norm = EBMBackwardsRMSNorm(args.dim, eps=args.norm_eps, use_fp32_norm=use_fp32)
        else:
            raise ValueError(f"Invalid ebt_norm value: {args.ebt_norm}")

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        mask: Optional[torch.Tensor],
        ve: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform a forward pass through the TransformerBlock.

        :param x: Input tensor of shape ``(B, 2*(S-1) [+1], D)``.
        :type x: torch.Tensor
        :param start_pos: Starting position for attention caching.
        :type start_pos: int
        :param freqs_cis: ``(cos, sin)`` frequency tuple forwarded to
            :class:`Attention`.
        :type freqs_cis: Tuple[torch.Tensor, torch.Tensor]
        :param mask: Optional attention mask.
        :type mask: Optional[torch.Tensor]
        :param ve: Optional value embeddings forwarded to :class:`Attention`.
        :type ve: Optional[torch.Tensor]
        :return: Output tensor of the same shape as ``x``.
        :rtype: torch.Tensor
        """
        # x has shape B, 2*(S-1), D?
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask, ve
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class EBTTimeConcat(nn.Module):
    """Energy-Based Transformer with time-embedding MCMC concatenation."""

    def __init__(
        self,
        params: EBTModelArgs,
        max_mcmc_steps: int,
        gradient_checkpointing: bool = False,
        use_mcmc_time_embed: bool = False,
    ) -> None:
        """Initialize the EBT transformer.

        :param params: Model configuration.
        :type params: EBTModelArgs
        :param max_mcmc_steps: Maximum MCMC step count used to size the time
            embedding table.
        :type max_mcmc_steps: int
        :param gradient_checkpointing: Enable gradient checkpointing when
            ``True``.
        :type gradient_checkpointing: bool
        :param use_mcmc_time_embed: When ``True``, use per-step time
            embeddings (original behavior). When ``False``, all MCMC steps
            share the same transition kernel, enabling arbitrary inference
            step counts.
        :type use_mcmc_time_embed: bool
        :raises ValueError: If ``params.ebt_norm`` is not one of the
            supported norm names.
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

        use_fp32 = getattr(params, 'float_precision', '') != "bf16-true"
        if params.ebt_norm == "rms":
            self.norm = RMSNorm(params.dim, eps=params.norm_eps, use_fp32_norm=use_fp32)
        elif params.ebt_norm == "layer":
            self.norm = nn.LayerNorm(params.dim)
        elif params.ebt_norm == "none":
            self.norm = nn.Identity()
        elif params.ebt_norm == "dyt":
            self.norm = DyT(params.dim, alpha_init_value=params.dyt_alpha_init, bias_learnable = False) # no learnable bias here since grad cant be computed for a final bias term in EBT
        elif params.ebt_norm == "ebm_backwards_norm":
            self.norm = EBMBackwardsRMSNorm(params.dim, eps=params.norm_eps, use_fp32_norm=use_fp32)
        else:
            raise ValueError(f"Invalid ebt_norm value: {params.ebt_norm}")

        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        if self.use_mcmc_time_embed:
            self.time_embeddings = nn.Embedding(max_mcmc_steps, params.dim)

        self.final_layer = nn.Linear(params.dim, 1, bias = False)
        init_whole_model_weights(self.final_layer, self.params.weight_initialization)

    def forward(
        self,
        embeddings: torch.Tensor,
        start_pos: int,
        mcmc_step: int = 0,
        real_token_ids: Optional[torch.Tensor] = None,
        predicted_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform a forward pass through the EBT transformer.

        :param embeddings: Token embeddings of shape
            ``(B, 2*(S-1), D)`` (plus one extra time-embedding row when
            ``use_mcmc_time_embed`` is enabled).
        :type embeddings: torch.Tensor
        :param start_pos: Starting position for attention caching.
        :type start_pos: int
        :param mcmc_step: Current MCMC step index used to index the time
            embedding table.
        :type mcmc_step: int
        :param real_token_ids: Optional tensor of real-token ids used by
            Value Embeddings.
        :type real_token_ids: Optional[torch.Tensor]
        :param predicted_tokens: Optional tensor of predicted token ids used
            by Value Embeddings.
        :type predicted_tokens: Optional[torch.Tensor]
        :return: Per-position energies for the predicted half of the
            sequence, shape ``(B, S-1, 1)``.
        :rtype: torch.Tensor
        """
        _bsz = embeddings.shape[0]

        if self.use_mcmc_time_embed:
            mcmc_step_val = mcmc_step
            mcmc_step = torch.empty((_bsz,), device=embeddings.device, dtype=torch.long).fill_(mcmc_step_val)
            time_embeddings = self.time_embeddings(mcmc_step).unsqueeze(dim=1) # needs to be expanded to B, 1, D
            embeddings = torch.cat((time_embeddings, embeddings), dim = 1) # B, 2S - 1 + 1, D

        _bsz, seqlen = embeddings.shape[:2]
        if self.use_mcmc_time_embed:
            seqlen = (seqlen+3) // 2 # passed in seqlen is 2(S-1)+1+1(time) so add 3 div 2 = S+1
        else:
            seqlen = (seqlen+2) // 2 # passed in seqlen is 2(S-1) so add 2 div 2 = S
        # 动态扩展 freqs_cos/freqs_sin 如果需要的长度超过预计算的长度
        required_length = start_pos + seqlen
        if required_length > self.freqs_cos.shape[0]:
            new_cos, new_sin = precompute_freqs_cis(
                self.params.dim // self.params.n_heads,
                required_length
            )
            self.freqs_cos = new_cos.to(embeddings.device)
            self.freqs_sin = new_sin.to(embeddings.device)

        freqs_cis = (
            self.freqs_cos[start_pos : start_pos + seqlen],
            self.freqs_sin[start_pos : start_pos + seqlen],
        )

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
            # causal mask is like this by default 0, -inf, -inf
            #                         0, 0,    -inf
            #                         0, 0,    0

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
                embeddings = embeddings[:, 1:] # remove temporal embed
            energies = self.final_layer(embeddings)

            energies = energies[:, embeddings.shape[1] // 2:]
            return energies