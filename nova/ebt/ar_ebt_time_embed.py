#NOTE most code gotten from llama2 codebase -- credit:https://github.com/meta-llama/llama
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from utils import (
    init_whole_model_weights,
    EBTModelArgs,
    BLOCK_MODE_CHOICES,
    EXPLICIT_BLOCK_LATENT_MODES,
    build_explicit_block_latent_mask,
    build_explicit_block_latent_freq_indices,
    build_explicit_block_latent_inference_mask,
    build_explicit_block_latent_inference_freq_indices,
)


def _resolve_block_mode(explicit: Optional[str], fallback: str) -> str:
    """Resolve a block_mode string, preferring the explicit argument.

    Any call site that owns a trained model must pass its block_mode
    explicitly. We only fall back to the model-configured default when
    no override is provided (e.g. from legacy call sites that will be
    updated progressively).
    """
    mode = explicit if explicit is not None else fallback
    if mode not in BLOCK_MODE_CHOICES:
        raise ValueError(
            f"Unknown block_mode={mode!r}; must be one of {BLOCK_MODE_CHOICES}"
        )
    return mode


def _mask_to_sdpa_attn_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Convert the repo's additive 0 / -inf masks into SDPA-friendly masks.

    Bool masks are preferred here because they are more broadly supported by
    optimized CUDA SDPA backends than dense additive float masks.
    """
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        return mask
    if torch.is_floating_point(mask):
        return torch.isfinite(mask)
    raise TypeError(f"Unsupported attention mask dtype for SDPA: {mask.dtype}")


def _run_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    prefer_memory_efficient: bool = False,
    allow_optimized_kernels: bool = True,
) -> torch.Tensor:
    """Run SDPA while preferring non-math CUDA backends when available."""
    if not query.is_cuda:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )

    backend_attempts = []
    if allow_optimized_kernels and prefer_memory_efficient:
        backend_attempts.extend(
            [
                dict(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=True,
                    enable_cudnn=False,
                ),
                dict(
                    enable_flash=False,
                    enable_math=False,
                    enable_mem_efficient=True,
                    enable_cudnn=False,
                ),
            ]
        )
    backend_attempts.append(
        dict(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
            enable_cudnn=False,
        )
    )

    last_error = None
    for backend_kwargs in backend_attempts:
        try:
            with torch.backends.cuda.sdp_kernel(**backend_kwargs):
                return F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                )
        except RuntimeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("No valid SDPA backend was available.")


class BackwardRMSNormFunction(torch.autograd.Function):
    """
    Forward: identity
    Backward: apply RMSNorm on grad_output, with its own weight parameter.
    """

    @staticmethod
    def forward(ctx, input_, weight, eps):
        # Save for backward
        ctx.save_for_backward(weight)
        ctx.eps = eps
        return input_  # Identity forward

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output -> RMSNorm(grad_output) using 'weight'.
        """
        # print("grad_output", grad_output)
        (weight,) = ctx.saved_tensors
        eps = ctx.eps

        # Compute RMS of grad_output over last dim
        rms = torch.rsqrt(grad_output.pow(2).mean(dim=-1, keepdim=True) + eps)

        # Normalized gradient wrt the input
        grad_input = grad_output * rms * weight  # shape matches grad_output

        # Gradient wrt 'weight'
        # The transform is: out_grad = grad_output * (rms * weight).
        # So partial wrt weight is grad_output * rms.
        # We sum over all but the last dimension.
        sum_dims = list(range(grad_output.dim() - 1))
        grad_weight = (grad_output * rms).sum(dim=sum_dims)

        return grad_input, grad_weight, None


class BackwardRMSNorm(nn.Module):
    """
    nn.Module that applies identity in forward, RMSNorm in backward,
    with its own learnable weight.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.full((dim,), 0.01))
        # self.weight = nn.Parameter(torch.ones(dim))

        self.eps = eps

    def forward(self, x: torch.Tensor):
        return BackwardRMSNormFunction.apply(x, self.weight, self.eps)
    
class BackwardLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, gamma, beta, eps):
        # Save for backward
        ctx.save_for_backward(gamma, beta)
        ctx.eps = eps
        # Identity forward pass
        return input_

    @staticmethod
    def backward(ctx, grad_output):
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
    """
    Identity on the forward pass, LayerNorm on the backward pass,
    with learnable gamma and beta. Normalizes over the last dimension.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        # self.gamma = nn.Parameter(torch.ones(dim))
        self.gamma = nn.Parameter(torch.full((dim,), 0.01))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return BackwardLayerNormFunction.apply(x, self.gamma, self.beta, self.eps)


class EBMBackwardsRMSNorm(nn.Module): # NOTE none of this worked well, could make loss lower initially (i.e. equal to -log(len(vocab))) but then it diverged or converged slower
    """
    Applies:
      1) A standard (forward) RMSNorm.
      2) A backward-only RMSNorm (identity forward) with its own parameter.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # Forward RMSNorm has its own weight_fwd
        self.forward_rms = RMSNorm(dim, eps)
        # Backward RMSNorm has a separate weight_bwd
        self.backward_rms = BackwardRMSNorm(dim, eps)
        # self.backward_ln = BackwardLayerNorm(dim, eps)

    def forward(self, x: torch.Tensor):
        """
        1) Apply standard RMSNorm in forward.
        2) Then apply backward-only RMSNorm (which is identity in forward).
        """
        x = self.forward_rms(x)
        x = self.backward_rms(x)
        # x = self.backward_ln(x)
        return x

class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5, bias_learnable = True):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=bias_learnable)
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the cosine and sine frequency tensors for rotary embeddings.

    Returns a tuple of (cos, sin) tensors with shape (end, dim//2), stored as
    float32 but applied via real-valued arithmetic to avoid dtype casting in
    the forward/backward graph.

    Args:
        dim (int): Head dimension.
        end (int): Maximum sequence length.
        theta (float, optional): RoPE base. Defaults to 10000.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (cos, sin) each of shape (end, dim//2).
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
    """
    Apply rotary embeddings using real-valued cos/sin arithmetic.

    Avoids any dtype casting so the computation stays in the model's native
    dtype (e.g. bfloat16) throughout, which is important for keeping the
    MCMC autograd graph in low precision.

    Args:
        xq (torch.Tensor): Query tensor, shape (..., seq, n_heads, head_dim).
        xk (torch.Tensor): Key tensor, shape (..., seq, n_kv_heads, head_dim).
        freqs_cis (Tuple[torch.Tensor, torch.Tensor]): (cos, sin) each of
            shape (seq, head_dim//2), precomputed by precompute_freqs_cis.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Rotated xq and xk in original dtype.
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
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args: EBTModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (EBTModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1 #NOTE this is hardcoded since we are using DDP
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
        context_len: int,
        pred_len: int,
        block_mode: str,
    ):
        """Forward pass of the attention module.

        block_mode dispatch:
          - ``dense_token`` / ``mtp_mcmc``: legacy symmetric EBT attention
            (matches ``main`` branch). Requires ``pred_len == context_len``
            as a shape invariant, not as a semantic selector. **The code
            path for these two modes is preserved verbatim from the
            previous version — do not refactor it for the new modes.**
          - ``future_latent_non_causal`` / ``blockwise``: dispatches to
            ``_forward_explicit_block_latent`` (a separate, plain SDPA
            implementation that uses the additive ``mask`` to encode
            visibility for ``S * K`` independent future latents).

        Args:
            x: Input tensor of shape ``(B, 1 + context_len + pred_len, D)``.
                For the explicit-block-latent modes the layout is instead
                ``(B, 1 + S + S * K, D)`` with ``pred_len = S * K``.
            start_pos: Starting position for rotary caching (unused here but
                kept for symmetry with other EBT variants).
            freqs_cis: Precomputed rotary frequency tensor. For the
                explicit-block-latent modes this must already be of length
                ``full_seqlen`` (the trunk gathers per-token positions).
            mask: Additive attention mask of shape
                ``(context_len + 2, context_len + 2)`` for the symmetric
                modes. For the explicit-block-latent modes this is the
                ``(full_seqlen, full_seqlen)`` mask produced by
                ``build_explicit_block_latent_mask``.
            context_len: Number of real context tokens (pre-time-embed).
            pred_len: Number of predicted-block tokens. For the explicit
                modes this equals ``context_len * block_size``.
            block_mode: Explicit attention semantic. See module docstring.
        """
        bsz, full_seqlen, _ = x.shape
        block_mode = _resolve_block_mode(block_mode, "dense_token")

        if block_mode in EXPLICIT_BLOCK_LATENT_MODES:
            # Plain full-sequence SDPA path with an explicit (S, K) mask.
            # Lives in its own method so that mtp_mcmc / dense_token logic
            # below is *byte-identical* to the previous version.
            return self._forward_explicit_block_latent(
                x=x,
                freqs_cis=freqs_cis,
                mask=mask,
            )
        if block_mode not in ("dense_token", "mtp_mcmc"):
            raise ValueError(f"Unsupported block_mode={block_mode!r}")


        # dense_token and mtp_mcmc share the legacy symmetric attention math.
        # pred_len == context_len is a shape invariant under these modes; it
        # is NOT used to pick between algorithms.
        if pred_len != context_len:
            raise ValueError(
                f"block_mode={block_mode!r} requires a symmetric layout with "
                f"pred_len == context_len; got context_len={context_len}, "
                f"pred_len={pred_len}. Non-symmetric layouts are reserved for "
                f"the 'blockwise' mode which is not implemented yet."
            )
        leading_time = full_seqlen == (1 + context_len + pred_len)
        plain_cat = full_seqlen == (context_len + pred_len)
        if not leading_time and not plain_cat:
            raise ValueError(
                f"Invalid symmetric layout: full_seqlen={full_seqlen}, "
                f"context_len={context_len}, pred_len={pred_len}"
            )

        cos, sin = freqs_cis
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, full_seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)

        if leading_time:
            context_with_time_len = context_len + 1
            xq_o = xq[:, :context_with_time_len, :, :]
            xk_o = xk[:, :context_with_time_len, :, :]
            xv_o = xv[:, :context_with_time_len, :, :]
            xq_p = xq[:, context_with_time_len:, :, :]
            xk_p = xk[:, context_with_time_len:, :, :]
            xv_p = xv[:, context_with_time_len:, :, :]
            n_ctx = context_with_time_len
            xq_o, xk_o = apply_rotary_emb(
                xq_o,
                xk_o,
                freqs_cis=(cos[:context_with_time_len], sin[:context_with_time_len]),
            )
            xq_p, xk_p = apply_rotary_emb(
                xq_p,
                xk_p,
                freqs_cis=(cos[2 : context_with_time_len + 1], sin[2 : context_with_time_len + 1]),
            )
            p_mask_start_row = 2
            pred_superdiag_offset = 2
        else:
            n_ctx = context_len
            xq_o = xq[:, :n_ctx, :, :]
            xk_o = xk[:, :n_ctx, :, :]
            xv_o = xv[:, :n_ctx, :, :]
            xq_p = xq[:, n_ctx:, :, :]
            xk_p = xk[:, n_ctx:, :, :]
            xv_p = xv[:, n_ctx:, :, :]
            xq_o, xk_o = apply_rotary_emb(
                xq_o, xk_o, freqs_cis=(cos[:n_ctx], sin[:n_ctx])
            )
            xq_p, xk_p = apply_rotary_emb(
                xq_p,
                xk_p,
                freqs_cis=(
                    cos[self.time_offset : n_ctx + 1],
                    sin[self.time_offset : n_ctx + 1],
                ),
            )
            p_mask_start_row = self.time_offset
            pred_superdiag_offset = self.time_offset

        o_mask = None
        if mask is not None:
            o_mask = mask[:-1, :-1]

        xq_ot = xq_o.transpose(1, 2)
        keys_o = xk_o.transpose(1, 2)
        values_o = xv_o.transpose(1, 2)

        if self.use_sdpa_attention:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
                enable_cudnn=False,
            ):
                output_o = F.scaled_dot_product_attention(
                    xq_ot, keys_o, values_o, is_causal=True
                )
            output_o = output_o.transpose(1, 2).contiguous().view(bsz, n_ctx, -1)
        else:
            scores_o = torch.matmul(xq_ot, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim)
            if o_mask is not None:
                scores_o = scores_o + o_mask
            scores_o = F.softmax(scores_o.float(), dim=-1).type_as(xq_ot)
            output_o = torch.matmul(scores_o, values_o)
            output_o = output_o.transpose(1, 2).contiguous().view(bsz, n_ctx, -1)

        xq_pt = xq_p.transpose(1, 2)
        keys_p = xk_p.transpose(1, 2)
        values_p = xv_p.transpose(1, 2)
        n_pred = pred_len
        Kkv = keys_o.shape[2]

        if self.use_sdpa_attention:
            keys_all = torch.cat([keys_o, keys_p], dim=2)
            values_all = torch.cat([values_o, values_p], dim=2)
            orig_rows = torch.arange(n_pred, device=x.device).unsqueeze(1)
            orig_cols = torch.arange(Kkv, device=x.device).unsqueeze(0)
            orig_mask_part = torch.where(
                orig_cols <= orig_rows + (self.time_offset - 1),
                x.new_zeros(1),
                x.new_full((1,), float("-inf")),
            )
            self_mask_part = x.new_full((n_pred, n_pred), float("-inf"))
            self_mask_part.fill_diagonal_(0.0)
            mask_p = torch.cat([orig_mask_part, self_mask_part], dim=1)
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
                enable_cudnn=False,
            ):
                output_p = F.scaled_dot_product_attention(
                    xq_pt, keys_all, values_all, attn_mask=mask_p
                )
            output_p = output_p.transpose(1, 2).contiguous().view(bsz, n_pred, -1)
        else:
            scores_p = torch.matmul(xq_pt, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim)
            temp_append = torch.zeros(
                (scores_p.shape[0], scores_p.shape[1], scores_p.shape[2], 1),
                dtype=scores_p.dtype,
                device=scores_p.device,
            )
            scores_p = torch.cat((scores_p, temp_append), dim=-1)

            insertion_superdiagonal = (xq_pt * keys_p).sum(dim=3) / math.sqrt(self.head_dim)
            insertion_superdiagonal = insertion_superdiagonal.to(scores_p.dtype)

            seq_len_minus_1 = scores_p.shape[2]
            superdiag_rows = self.superdiag_rows[:seq_len_minus_1]
            superdiag_cols = self.superdiag_cols[:seq_len_minus_1]

            zero_superdiag = torch.zeros_like(
                insertion_superdiagonal, dtype=scores_p.dtype, device=scores_p.device
            )
            diagonal_removal_mask = torch.ones_like(
                scores_p, dtype=scores_p.dtype, device=scores_p.device
            )
            diagonal_removal_mask[:, :, superdiag_rows, superdiag_cols] = zero_superdiag
            scores_p = scores_p * diagonal_removal_mask

            diagonal_addition_mask = torch.zeros_like(
                scores_p, dtype=scores_p.dtype, device=scores_p.device
            )
            diagonal_addition_mask[:, :, superdiag_rows, superdiag_cols] = insertion_superdiagonal
            scores_p = scores_p + diagonal_addition_mask

            if mask is not None:
                scores_p = scores_p + mask[p_mask_start_row:, :]
            scores_p = F.softmax(scores_p.float(), dim=-1).type_as(xq_pt)

            scores_p_superdiagonal = scores_p.diagonal(
                offset=pred_superdiag_offset, dim1=2, dim2=3
            ).clone()
            scores_p = scores_p * diagonal_removal_mask
            scores_p = scores_p[:, :, :, :-1]
            output_p = torch.matmul(scores_p, values_o)
            next_pred_self_attention = values_p * scores_p_superdiagonal.unsqueeze(dim=-1)
            output_p = output_p + next_pred_self_attention
            output_p = output_p.transpose(1, 2).contiguous().view(bsz, n_pred, -1)

        output = torch.cat((output_o, output_p), dim=1)
        return self.wo(output)

    def _forward_explicit_block_latent(
        self,
        x: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ):
        """Plain full-sequence SDPA for the future_latent / blockwise modes."""
        bsz, T, _ = x.shape
        if mask is None:
            raise ValueError(
                "explicit-block-latent attention requires a non-None mask "
                "(use utils.build_explicit_block_latent_mask)."
            )
        cos_f, sin_f = freqs_cis
        if cos_f.shape[0] != T or sin_f.shape[0] != T:
            raise ValueError(
                "freqs_cis length mismatch for explicit-block-latent attention: "
                f"got cos/sin length {cos_f.shape[0]}/{sin_f.shape[0]}, expected {T}."
            )

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, T, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, T, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, T, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=(cos_f, sin_f))

        xq = xq.transpose(1, 2)
        keys = xk.transpose(1, 2)
        values = xv.transpose(1, 2)

        # EBT training uses autograd.grad(create_graph=True) in the MCMC loop,
        # so this attention path must support double backward. The optimized
        # CUDA SDPA kernels selected by flash / mem-efficient backends do not
        # in this environment, so keep them for inference / no-grad only.
        allow_optimized_kernels = not (
            self.training
            and torch.is_grad_enabled()
            and (xq.requires_grad or keys.requires_grad or values.requires_grad)
        )

        output = _run_sdpa(
            xq,
            keys,
            values,
            attn_mask=_mask_to_sdpa_attn_mask(mask),
            prefer_memory_efficient=True,
            allow_optimized_kernels=allow_optimized_kernels,
        )
        output = output.transpose(1, 2).contiguous().view(bsz, T, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim_multiplier: Optional[float],
        weight_initialization: str,
        ebt_act_func: str = "silu",
        weight_initialization_gain: float = 1.0
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

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

    def forward(self, x):
        return self.w2(self.act_func(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: EBTModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (EBTModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

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
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        mask: Optional[torch.Tensor],
        context_len: int,
        pred_len: int,
        block_mode: str,
    ):
        """Perform a forward pass through the TransformerBlock.

        ``block_mode`` is forwarded to ``Attention.forward``.
        """
        h = x + self.attention(
            self.attention_norm(x),
            start_pos,
            freqs_cis,
            mask,
            context_len=context_len,
            pred_len=pred_len,
            block_mode=block_mode,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class EBTTimeConcat(nn.Module):
    def __init__(self, params: EBTModelArgs, max_mcmc_steps, gradient_checkpointing=False, use_mcmc_time_embed=False):
        """
        Initialize a Transformer model.

        Args:
            params (EBTModelArgs): Model configuration parameters.
            use_mcmc_time_embed (bool): If True, use per-step time embeddings (original behavior).
                If False, all MCMC steps share the same transition kernel, enabling arbitrary inference steps.

        Attributes:
            params (EBTModelArgs): Model configuration parameters.
            n_layers (int): Number of layers in the model.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.use_mcmc_time_embed = use_mcmc_time_embed

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
            self.norm = DyT(params.dim, alpha_init_value=params.dyt_alpha_init, bias_learnable = False) # no learnable bias here since grad cant be computed for a final bias term in EBT
        elif params.ebt_norm == "ebm_backwards_norm":
            self.norm = EBMBackwardsRMSNorm(params.dim, eps=params.norm_eps)
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

    def _run_layers(
        self,
        embeddings: torch.Tensor,
        start_pos: int,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        mask: Optional[torch.Tensor],
        *,
        context_len: int,
        pred_len: int,
        block_mode: str,
    ) -> torch.Tensor:
        """Apply transformer layers, optionally checkpointing each block."""
        for layer in self.layers:
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
                and embeddings.requires_grad
            ):
                def custom_forward(
                    hidden_states: torch.Tensor,
                    layer: TransformerBlock = layer,
                ) -> torch.Tensor:
                    return layer(
                        hidden_states,
                        start_pos,
                        freqs_cis,
                        mask,
                        context_len=context_len,
                        pred_len=pred_len,
                        block_mode=block_mode,
                    )

                embeddings = checkpoint(
                    custom_forward,
                    embeddings,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                embeddings = layer(
                    embeddings,
                    start_pos,
                    freqs_cis,
                    mask,
                    context_len=context_len,
                    pred_len=pred_len,
                    block_mode=block_mode,
                )
        return embeddings

    def forward(
        self,
        embeddings: torch.Tensor,
        start_pos: int,
        mcmc_step = 0,
        context_len = None,
        pred_len = None,
        return_pred_hidden: bool = False,
        return_context_hidden: bool = False,
        block_mode: Optional[str] = None,
        block_size: Optional[int] = None,
    ):
        """Perform a forward pass through the Transformer model.

        Attention / mask semantics are selected explicitly via ``block_mode``.
        Shape validation (``pred_len == context_len``) is treated as a shape
        invariant for the symmetric modes, not as a semantic branch.

        For the explicit-block-latent modes (``future_latent_non_causal`` /
        ``blockwise``), this dispatches to a separate forward path
        (``_forward_explicit_block_latent``) that uses a different sequence
        layout (``[time, c_1..c_S, z_{1,1}..z_{S,K}]``) and a per-(t, j)
        attention mask. The dense_token / mtp_mcmc code below is kept
        verbatim from the previous version.
        """
        block_mode = _resolve_block_mode(block_mode, self.params.block_mode)
        if block_mode in EXPLICIT_BLOCK_LATENT_MODES:
            return self._forward_explicit_block_latent(
                embeddings=embeddings,
                start_pos=start_pos,
                mcmc_step=mcmc_step,
                context_len=context_len,
                pred_len=pred_len,
                block_size=block_size,
                block_mode=block_mode,
                return_pred_hidden=return_pred_hidden,
                return_context_hidden=return_context_hidden,
            )
        if block_mode not in ("dense_token", "mtp_mcmc"):
            raise ValueError(f"Unsupported block_mode={block_mode!r}")

        if context_len is None or pred_len is None:
            full_len = embeddings.shape[1]
            if full_len % 2 != 0:
                raise ValueError(
                    f"Unable to infer context_len/pred_len from odd embeddings length {full_len}; pass them explicitly."
                )
            context_len = full_len // 2
            pred_len = full_len - context_len
        if context_len + pred_len != embeddings.shape[1]:
            raise ValueError(
                f"context_len + pred_len must equal embeddings length, got {context_len}+{pred_len}!={embeddings.shape[1]}"
            )
        if pred_len != context_len:
            raise ValueError(
                f"block_mode={block_mode!r} requires symmetric context/pred layout, "
                f"got context_len={context_len}, pred_len={pred_len}. "
                f"Non-symmetric block prediction will be enabled by block_mode='blockwise' "
                f"once implemented."
            )

        _bsz = embeddings.shape[0]

        if self.use_mcmc_time_embed:
            mcmc_step_t = torch.full(
                size=(_bsz,),
                fill_value=mcmc_step,
                device=embeddings.device,
                dtype=torch.long,
            )
            time_embeddings = self.time_embeddings(mcmc_step_t).unsqueeze(dim=1)
            embeddings = torch.cat((time_embeddings, embeddings), dim=1)
            legacy_seqlen = context_len + 2
        else:
            _, raw_len = embeddings.shape[:2]
            legacy_seqlen = (raw_len + 2) // 2

        required_length = start_pos + legacy_seqlen
        if required_length > self.freqs_cos.shape[0]:
            new_cos, new_sin = precompute_freqs_cis(
                self.params.dim // self.params.n_heads,
                required_length,
            )
            self.freqs_cos = new_cos.to(embeddings.device)
            self.freqs_sin = new_sin.to(embeddings.device)

        freqs_cis = (
            self.freqs_cos[start_pos : start_pos + legacy_seqlen],
            self.freqs_sin[start_pos : start_pos + legacy_seqlen],
        )

        mask = None
        if legacy_seqlen > 1:
            mask = torch.full((legacy_seqlen, legacy_seqlen), float("-inf"), device=embeddings.device)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.type_as(embeddings)

        embeddings = self._run_layers(
            embeddings,
            start_pos,
            freqs_cis,
            mask,
            context_len=context_len,
            pred_len=pred_len,
            block_mode=block_mode,
        )
        embeddings = self.norm(embeddings)
        if self.use_mcmc_time_embed:
            embeddings = embeddings[:, 1:]  # remove time embedding
        context_hidden = embeddings[:, :context_len]
        pred_hidden = embeddings[:, context_len:]
        energies = self.final_layer(embeddings)
        energies = energies[:, context_len:]
        if return_context_hidden and return_pred_hidden:
            return energies, context_hidden, pred_hidden
        if return_context_hidden:
            return energies, context_hidden
        if return_pred_hidden:
            return energies, pred_hidden
        return energies

    def _forward_explicit_block_latent(
        self,
        embeddings: torch.Tensor,
        start_pos: int,
        mcmc_step,
        context_len: Optional[int],
        pred_len: Optional[int],
        block_size: Optional[int],
        block_mode: str,
        return_pred_hidden: bool,
        return_context_hidden: bool,
    ):
        """Dispatch between training-layout and inference-layout for the
        explicit-block-latent modes (``future_latent_non_causal`` /
        ``blockwise``).

        The two layouts are distinguished by ``pred_len``:

          * ``pred_len == context_len * block_size`` → **training layout**:
            S source positions, each with K future latents (S*K pred tokens
            in position-major order).
          * ``pred_len == block_size`` → **inference layout**:
            C context tokens followed by exactly K pred latents anchored at
            the end of context.

        Both layouts use the same attention semantics, just at different
        scales. Inference layout is mathematically equivalent to extracting
        the last source position's K latents from the training layout, but
        avoids paying O(S^2) attention cost.
        """
        if context_len is None or pred_len is None or block_size is None:
            raise ValueError(
                "EBTTimeConcat explicit-block-latent path requires "
                "context_len, pred_len and block_size to be set explicitly."
            )
        S_or_C = int(context_len)
        K = int(block_size)
        if S_or_C <= 0 or K <= 0:
            raise ValueError(
                f"context_len/block_size must be > 0; got context_len={S_or_C}, K={K}"
            )

        if int(pred_len) == S_or_C * K:
            return self._forward_explicit_block_latent_training(
                embeddings=embeddings,
                start_pos=start_pos,
                mcmc_step=mcmc_step,
                context_len=S_or_C,
                block_size=K,
                block_mode=block_mode,
                return_pred_hidden=return_pred_hidden,
                return_context_hidden=return_context_hidden,
            )
        if int(pred_len) == K:
            return self._forward_explicit_block_latent_inference(
                embeddings=embeddings,
                start_pos=start_pos,
                mcmc_step=mcmc_step,
                context_len=S_or_C,
                block_size=K,
                block_mode=block_mode,
                return_pred_hidden=return_pred_hidden,
                return_context_hidden=return_context_hidden,
            )
        raise ValueError(
            "explicit-block-latent path expects pred_len in "
            f"{{context_len * block_size, block_size}}; got pred_len={pred_len}, "
            f"context_len={S_or_C}, block_size={K}."
        )

    def _forward_explicit_block_latent_training(
        self,
        embeddings: torch.Tensor,
        start_pos: int,
        mcmc_step,
        context_len: int,
        block_size: int,
        block_mode: str,
        return_pred_hidden: bool,
        return_context_hidden: bool,
    ):
        """Training-layout forward.

        Sequence layout (1-indexed semantic, 0-indexed positions in the mask):
            ``[time, c_1, ..., c_S, z_{1,1}, ..., z_{1,K}, z_{2,1}, ...,``
            ``z_{2,K}, ..., z_{S,1}, ..., z_{S,K}]``

        Pred latents are stored in **position-major** order (outer ``t``,
        inner ``j``), matching ``build_explicit_block_latent_mask``.
        Returns energies of shape ``(B, S * K, 1)`` in the same order.
        """
        S = int(context_len)
        K = int(block_size)
        full_len = embeddings.shape[1]
        if full_len != S + S * K:
            raise ValueError(
                "explicit-block-latent (training) expects embeddings of length "
                f"S + S * K = {S + S * K}, got {full_len}."
            )

        # ------ optional time embedding (prepended like the legacy path) ------
        bsz = embeddings.shape[0]
        if self.use_mcmc_time_embed:
            mcmc_step_t = torch.full(
                size=(bsz,),
                fill_value=mcmc_step,
                device=embeddings.device,
                dtype=torch.long,
            )
            time_embeddings = self.time_embeddings(mcmc_step_t).unsqueeze(dim=1)  # (B, 1, D)
            embeddings = torch.cat((time_embeddings, embeddings), dim=1)
            T_time = 1
        else:
            T_time = 0
        T = T_time + S + S * K
        assert embeddings.shape[1] == T

        # ------ rotary frequencies (gather per-token positions) ------
        device = embeddings.device
        self.freqs_cos = self.freqs_cos.to(device)
        self.freqs_sin = self.freqs_sin.to(device)
        max_pos_needed = start_pos + S + K
        required_length = max_pos_needed + 1
        if required_length > self.freqs_cos.shape[0]:
            new_cos, new_sin = precompute_freqs_cis(
                self.params.dim // self.params.n_heads,
                required_length,
            )
            self.freqs_cos = new_cos.to(device)
            self.freqs_sin = new_sin.to(device)

        pos_indices = build_explicit_block_latent_freq_indices(
            context_len=S,
            block_size=K,
            time_len=T_time,
            start_pos=start_pos,
        )
        pos_index_tensor = torch.tensor(
            pos_indices, dtype=torch.long, device=device
        )
        freqs_cis = (
            self.freqs_cos.index_select(0, pos_index_tensor),
            self.freqs_sin.index_select(0, pos_index_tensor),
        )

        mask = build_explicit_block_latent_mask(
            context_len=S,
            block_size=K,
            time_len=T_time,
            block_mode=block_mode,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )

        embeddings = self._run_layers(
            embeddings,
            start_pos,
            freqs_cis,
            mask,
            context_len=S,
            pred_len=S * K,
            block_mode=block_mode,
        )
        embeddings = self.norm(embeddings)
        embeddings = embeddings[:, T_time:]  # drop the time token
        context_hidden = embeddings[:, :S]
        pred_hidden = embeddings[:, S:]      # [B, S*K, D] in position-major order
        energies = self.final_layer(embeddings)
        energies = energies[:, S:]           # [B, S*K, 1]

        if return_context_hidden and return_pred_hidden:
            return energies, context_hidden, pred_hidden
        if return_context_hidden:
            return energies, context_hidden
        if return_pred_hidden:
            return energies, pred_hidden
        return energies

    def _forward_explicit_block_latent_inference(
        self,
        embeddings: torch.Tensor,
        start_pos: int,
        mcmc_step,
        context_len: int,
        block_size: int,
        block_mode: str,
        return_pred_hidden: bool,
        return_context_hidden: bool,
    ):
        """Inference-layout forward: C context + K pred latents at the end.

        Sequence layout: ``[time, c_1, ..., c_C, z_1, ..., z_K]``
        with ``z_j`` semantically equal to ``z_{C, j}`` from training. The
        attention mask (built by ``build_explicit_block_latent_inference_mask``)
        encodes the same per-mode visibility rules as training.

        Returns energies of shape ``(B, K, 1)`` for the K pred latents.
        """
        C = int(context_len)
        K = int(block_size)
        full_len = embeddings.shape[1]
        if full_len != C + K:
            raise ValueError(
                "explicit-block-latent (inference) expects embeddings of length "
                f"C + K = {C + K}, got {full_len}."
            )

        bsz = embeddings.shape[0]
        if self.use_mcmc_time_embed:
            mcmc_step_t = torch.full(
                size=(bsz,),
                fill_value=mcmc_step,
                device=embeddings.device,
                dtype=torch.long,
            )
            time_embeddings = self.time_embeddings(mcmc_step_t).unsqueeze(dim=1)
            embeddings = torch.cat((time_embeddings, embeddings), dim=1)
            T_time = 1
        else:
            T_time = 0
        T = T_time + C + K
        assert embeddings.shape[1] == T

        device = embeddings.device
        self.freqs_cos = self.freqs_cos.to(device)
        self.freqs_sin = self.freqs_sin.to(device)
        max_pos_needed = start_pos + C + K
        required_length = max_pos_needed + 1
        if required_length > self.freqs_cos.shape[0]:
            new_cos, new_sin = precompute_freqs_cis(
                self.params.dim // self.params.n_heads,
                required_length,
            )
            self.freqs_cos = new_cos.to(device)
            self.freqs_sin = new_sin.to(device)

        pos_indices = build_explicit_block_latent_inference_freq_indices(
            context_len=C,
            block_size=K,
            time_len=T_time,
            start_pos=start_pos,
        )
        pos_index_tensor = torch.tensor(
            pos_indices, dtype=torch.long, device=device
        )
        freqs_cis = (
            self.freqs_cos.index_select(0, pos_index_tensor),
            self.freqs_sin.index_select(0, pos_index_tensor),
        )

        mask = build_explicit_block_latent_inference_mask(
            context_len=C,
            block_size=K,
            time_len=T_time,
            block_mode=block_mode,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )

        embeddings = self._run_layers(
            embeddings,
            start_pos,
            freqs_cis,
            mask,
            context_len=C,
            pred_len=K,
            block_mode=block_mode,
        )
        embeddings = self.norm(embeddings)
        embeddings = embeddings[:, T_time:]  # drop the time token
        context_hidden = embeddings[:, :C]
        pred_hidden = embeddings[:, C:]      # [B, K, D]
        energies = self.final_layer(embeddings)
        energies = energies[:, C:]           # [B, K, 1]

        if return_context_hidden and return_pred_hidden:
            return energies, context_hidden, pred_hidden
        if return_context_hidden:
            return energies, context_hidden
        if return_pred_hidden:
            return energies, pred_hidden
        return energies
