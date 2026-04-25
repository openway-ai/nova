#NOTE most code gotten from llama2 codebase -- credit:https://github.com/meta-llama/llama
import torch
from torch import nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from utils import init_whole_model_weights, EBTModelArgs, BLOCK_MODE_CHOICES


def _resolve_block_mode(explicit: Optional[str], fallback: str) -> str:
    mode = explicit if explicit is not None else fallback
    if mode not in BLOCK_MODE_CHOICES:
        raise ValueError(
            f"Unknown block_mode={mode!r}; must be one of {BLOCK_MODE_CHOICES}"
        )
    return mode

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
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    
        

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
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
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


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
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        context_len: int,
        pred_len: int,
        block_mode: str,
    ):
        """Forward pass of the ar_ebt_default Attention module.

        block_mode dispatch:
          - ``dense_token``: original EBT superdiagonal attention (matches
            ``main`` branch).
          - ``mtp_mcmc``: the current dev-blockwise ``cat(ctx, self)``
            attention (preserves current dev-blockwise behavior for this
            trunk). A separate code path from dense_token.
          - ``future_latent_non_causal`` / ``blockwise``: not yet implemented.

        Both symmetric modes require ``pred_len == context_len`` as a shape
        invariant.
        """
        bsz, full_seqlen, _ = x.shape
        block_mode = _resolve_block_mode(block_mode, "dense_token")

        if block_mode in ("future_latent_non_causal", "blockwise"):
            raise NotImplementedError(
                f"ar_ebt_default Attention does not implement block_mode={block_mode!r} yet; "
                f"only 'dense_token' and 'mtp_mcmc' are supported."
            )
        if block_mode not in ("dense_token", "mtp_mcmc"):
            raise ValueError(f"Unsupported block_mode={block_mode!r}")

        if pred_len != context_len:
            raise ValueError(
                f"block_mode={block_mode!r} requires pred_len == context_len; "
                f"got context_len={context_len}, pred_len={pred_len}. "
                f"Non-symmetric layouts are reserved for the 'blockwise' mode."
            )
        if context_len + pred_len != full_seqlen:
            raise ValueError(
                f"context_len + pred_len must equal full_seqlen, got "
                f"{context_len}+{pred_len}!={full_seqlen}"
            )

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, full_seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)

        xq_o = xq[:, :context_len, :, :]
        xk_o = xk[:, :context_len, :, :]
        xv_o = xv[:, :context_len, :, :]

        xq_p = xq[:, context_len:, :, :]
        xk_p = xk[:, context_len:, :, :]
        xv_p = xv[:, context_len:, :, :]

        xq_o, xk_o = apply_rotary_emb(xq_o, xk_o, freqs_cis=freqs_cis[:context_len])
        xq_p, xk_p = apply_rotary_emb(xq_p, xk_p, freqs_cis=freqs_cis[1:1 + pred_len])

        xq_o = xq_o.transpose(1, 2)
        keys_o = xk_o.transpose(1, 2)
        values_o = xv_o.transpose(1, 2)
        xq_p = xq_p.transpose(1, 2)
        keys_p = xk_p.transpose(1, 2)
        values_p = xv_p.transpose(1, 2)

        if block_mode == "dense_token":
            # Main-branch EBT Attention: context uses causal mask,
            # predicted tokens use the superdiagonal self-attention trick.
            scores_o = torch.matmul(xq_o, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores_o = scores_o + mask[:-1, :-1]
            scores_o = F.softmax(scores_o.float(), dim=-1).type_as(xq_o)
            output_o = torch.matmul(scores_o, values_o)
            output_o = output_o.transpose(1, 2).contiguous().view(bsz, context_len, -1)

            # scores_p: B, N, K, S ; append one zero column for the superdiag slot
            scores_p = torch.matmul(xq_p, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim)
            temp_append = torch.zeros(
                (scores_p.shape[0], scores_p.shape[1], scores_p.shape[2], 1),
                dtype=scores_p.dtype,
                device=scores_p.device,
            )
            scores_p = torch.cat((scores_p, temp_append), dim=-1)

            insertion_superdiagonal = (xq_p * keys_p).sum(dim=3) / math.sqrt(self.head_dim)
            insertion_superdiagonal = insertion_superdiagonal.to(scores_p.dtype)

            superdiag_rows = torch.arange(scores_p.shape[2], device=scores_p.device)
            superdiag_cols = torch.arange(1, scores_p.shape[3], device=scores_p.device)

            zero_superdiag = torch.zeros_like(insertion_superdiagonal, dtype=scores_p.dtype, device=scores_p.device)
            diagonal_removal_mask = torch.ones_like(scores_p, dtype=scores_p.dtype, device=scores_p.device)
            diagonal_removal_mask[:, :, superdiag_rows, superdiag_cols] = zero_superdiag
            scores_p = scores_p * diagonal_removal_mask

            diagonal_addition_mask = torch.zeros_like(scores_p, dtype=scores_p.dtype, device=scores_p.device)
            diagonal_addition_mask[:, :, superdiag_rows, superdiag_cols] = insertion_superdiagonal
            scores_p = scores_p + diagonal_addition_mask

            if mask is not None:
                scores_p = scores_p + mask[1:, :]
            scores_p = F.softmax(scores_p.float(), dim=-1).type_as(xq_p)

            scores_p_superdiagonal = scores_p.diagonal(offset=1, dim1=2, dim2=3).clone()
            scores_p = scores_p * diagonal_removal_mask
            scores_p = scores_p[:, :, :, :-1]
            output_p = torch.matmul(scores_p, values_o)
            next_pred_self_attention = values_p * scores_p_superdiagonal.unsqueeze(dim=-1)
            output_p = output_p + next_pred_self_attention
            output_p = output_p.transpose(1, 2).contiguous().view(bsz, pred_len, -1)
        else:
            # block_mode == "mtp_mcmc": preserve current dev-blockwise
            # cat(ctx, self) attention for ar_ebt_default.
            scores_o = torch.matmul(xq_o, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores_o = scores_o + mask
            scores_o = F.softmax(scores_o.float(), dim=-1).type_as(xq_o)
            output_o = torch.matmul(scores_o, values_o)
            output_o = output_o.transpose(1, 2).contiguous().view(bsz, context_len, -1)

            scores_ctx = torch.matmul(xq_p, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores_self = (xq_p * keys_p).sum(dim=3, keepdim=True) / math.sqrt(self.head_dim)
            scores_p = torch.cat((scores_ctx, scores_self), dim=-1)
            scores_p = F.softmax(scores_p.float(), dim=-1).type_as(xq_p)

            scores_ctx = scores_p[:, :, :, :context_len]
            scores_self = scores_p[:, :, :, context_len]
            output_ctx = torch.matmul(scores_ctx, values_o)
            output_self = values_p * scores_self.unsqueeze(dim=-1)
            output_p = output_ctx + output_self
            output_p = output_p.transpose(1, 2).contiguous().view(bsz, pred_len, -1)

        output = torch.cat((output_o, output_p), dim=1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim_multiplier: Optional[float],
        weight_initialization: str,
        weight_initialization_gain: float
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
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


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


class EBTDefault(nn.Module):
    def __init__(self, params: EBTModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (EBTModelArgs): Model configuration parameters.

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

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len
        )

        self.final_layer = nn.Linear(params.dim, 1, bias = False)
        init_whole_model_weights(self.final_layer, self.params.weight_initialization)


    def forward(
        self,
        embeddings: torch.Tensor,
        start_pos: int,
        mcmc_step = None,
        context_len: Optional[int] = None,
        pred_len: Optional[int] = None,
        return_pred_hidden: bool = False,
        return_context_hidden: bool = False,
        block_mode: Optional[str] = None,
    ):  # NOTE mcmc_step not used here
        """Perform a forward pass through the EBTDefault trunk.

        Attention / mask semantics are chosen via ``block_mode``. See
        ``Attention.forward`` for the exact dispatch.
        """
        block_mode = _resolve_block_mode(block_mode, self.params.block_mode)
        if block_mode in ("future_latent_non_causal", "blockwise"):
            raise NotImplementedError(
                f"EBTDefault.forward does not implement block_mode={block_mode!r} yet; "
                f"only 'dense_token' and 'mtp_mcmc' are supported."
            )
        if block_mode not in ("dense_token", "mtp_mcmc"):
            raise ValueError(f"Unsupported block_mode={block_mode!r}")

        _bsz, full_len = embeddings.shape[:2]
        if context_len is None or pred_len is None:
            if full_len % 2 != 0:
                raise ValueError(
                    f"Unable to infer context/pred lengths from odd full length {full_len}; pass context_len and pred_len."
                )
            context_len = full_len // 2
            pred_len = full_len - context_len
        if context_len + pred_len != full_len:
            raise ValueError(
                f"context_len + pred_len must equal full length, got {context_len}+{pred_len}!={full_len}"
            )
        # Shape invariant for both symmetric modes.
        if pred_len != context_len:
            raise ValueError(
                f"block_mode={block_mode!r} requires pred_len == context_len for ar_ebt_default; "
                f"got context_len={context_len}, pred_len={pred_len}."
            )
        self.freqs_cis = self.freqs_cis.to(embeddings.device)

        # dense_token needs rotary positions up to context_len + 1 (superdiag shift).
        # mtp_mcmc needs positions up to max(context_len, pred_len + 1); with
        # pred_len == context_len this equals context_len + 1 as well.
        rotary_len = context_len + 1
        required_length = start_pos + rotary_len
        if required_length > self.freqs_cis.shape[0]:
            new_freqs_cis = precompute_freqs_cis(
                self.params.dim // self.params.n_heads,
                required_length,
            ).to(embeddings.device)
            self.freqs_cis = new_freqs_cis

        freqs_cis = self.freqs_cis[start_pos : start_pos + rotary_len]

        mask = None
        if block_mode == "dense_token":
            # Full mask of shape (context_len + 1, context_len + 1) for the
            # superdiagonal layout; Attention slices as mask[:-1,:-1] / mask[1:,:].
            if context_len + 1 > 1:
                mask = torch.full(
                    (context_len + 1, context_len + 1), float("-inf"), device=embeddings.device
                )
                mask = torch.triu(mask, diagonal=1)
                mask = mask.type_as(embeddings)
        else:  # mtp_mcmc
            if context_len > 1:
                mask = torch.full(
                    (context_len, context_len), float("-inf"), device=embeddings.device
                )
                mask = torch.triu(mask, diagonal=1)
                mask = mask.type_as(embeddings)

        for layer in self.layers:
            embeddings = layer(
                embeddings,
                start_pos,
                freqs_cis,
                mask,
                context_len=context_len,
                pred_len=pred_len,
                block_mode=block_mode,
            )
        embeddings = self.norm(embeddings)
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
