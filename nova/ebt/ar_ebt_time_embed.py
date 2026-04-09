#NOTE most code gotten from llama2 codebase -- credit:https://github.com/meta-llama/llama
import torch
from torch import nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from utils import init_whole_model_weights, EBTModelArgs

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


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
        rms = torch.rsqrt(grad_output.float().pow(2).mean(dim=-1, keepdim=True) + eps)

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
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        # """
        # NOTE the usage of S-1/S/S+1 is messed up and confusing here, I recommend checking the paper
        bsz, full_seqlen, _ = x.shape # full_seqlen includes real embeds and pred embeds
        original_seqlen = (full_seqlen + 1)//2 # this is just the condition plus all original tokens, +1 is bc of condition
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, full_seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)
        
        # _o is for original attention stuff
        xq_o = xq[:, :original_seqlen, :, :] #B, S-1, N, H (N and H are num head and head dim respectively)
        xk_o = xk[:, :original_seqlen, :, :]
        xv_o = xv[:, :original_seqlen, :, :]
        
        # _p is for predicted attention stuff
        xq_p = xq[:, original_seqlen:, :, :] #B, S-1, N, H (N and H are num head and head dim respectively)
        xk_p = xk[:, original_seqlen:, :, :]
        xv_p = xv[:, original_seqlen:, :, :]
        
        
        xq_o, xk_o = apply_rotary_emb(xq_o, xk_o, freqs_cis=freqs_cis[:original_seqlen])

        xq_p, xk_p = apply_rotary_emb(xq_p, xk_p, freqs_cis=freqs_cis[2:original_seqlen+1]) # use 2 since are the next preds and also have time embeddings and thus need to condition on two tokens
        
        #original attn calc is more normal############################################

        # seqlen here is original_seqlen (includes time_embed token)
        xq_o = xq_o.transpose(1, 2)  # (bs, n_local_heads, original_seqlen, head_dim)
        keys_o = xk_o.transpose(1, 2)
        values_o = xv_o.transpose(1, 2)

        # Path A: standard causal self-attention on original tokens via SDPA.
        # o_mask is [original_seqlen, original_seqlen]; SDPA fuses softmax+matmul
        # and avoids materialising the score matrix in HBM (FlashAttention backend).
        o_mask = mask[:-1, :-1] if mask is not None else None
        output_o = F.scaled_dot_product_attention(xq_o, keys_o, values_o, attn_mask=o_mask)
        output_o = output_o.transpose(1, 2).contiguous().view(bsz, original_seqlen, -1)

        #pred sequence attn calc is for energy-based transformer ########################################################################################

        # Path B: each pred token i must attend to
        #   (1) original tokens 0..i+1  (causal cross-attention; +2 offset due to time_embed)
        #   (2) itself (pred_i self-attention)
        #   (3) nothing else (no future orig tokens, no other pred tokens)
        #
        # Implementation: concatenate orig and pred K/V into a single sequence of
        # length K + (K-1) = 2K-1, then pass an explicit additive mask to SDPA.
        # SDPA (FlashAttention backend) avoids materialising the [B,N,K-1,2K-1]
        # score matrix in HBM → O(B·K·D) peak memory vs O(B·N·K²) in the original.
        # FLOPs are ~2x higher for Path B, but memory savings dominate at large K.

        xq_p = xq_p.transpose(1, 2)  # (bs, n_local_heads, pred_seqlen, head_dim)
        keys_p = xk_p.transpose(1, 2)
        values_p = xv_p.transpose(1, 2)

        K = original_seqlen           # number of orig tokens (time_embed + real tokens)
        pred_seqlen = K - 1           # number of pred tokens

        # Combined K/V: [B, N, K + pred_seqlen, H] = [B, N, 2K-1, H]
        keys_all   = torch.cat([keys_o,   keys_p],   dim=2)
        values_all = torch.cat([values_o, values_p], dim=2)

        # Build additive mask [pred_seqlen, 2K-1].
        # Row i allows: orig cols 0..i+1 (causal, time_embed at col 0 gives +1 offset)
        #               self  col  K+i   (pred_i attends only to itself)
        # All other entries are -inf.
        orig_rows = torch.arange(pred_seqlen, device=x.device).unsqueeze(1)  # [pred_seqlen, 1]
        orig_cols = torch.arange(K,           device=x.device).unsqueeze(0)  # [1, K]
        orig_mask_part = torch.where(
            orig_cols <= orig_rows + 1,
            x.new_zeros(1),
            x.new_full((1,), float('-inf'))
        )  # [pred_seqlen, K]

        self_mask_part = x.new_full((pred_seqlen, pred_seqlen), float('-inf'))
        self_mask_part.fill_diagonal_(0.0)  # each pred token attends only to itself

        mask_p = torch.cat([orig_mask_part, self_mask_part], dim=1)  # [pred_seqlen, 2K-1]

        output_p = F.scaled_dot_product_attention(xq_p, keys_all, values_all, attn_mask=mask_p)
        output_p = output_p.transpose(1, 2).contiguous().view(bsz, original_seqlen-1, -1)
        
        #return linear projection of concatted outputs ########################################################################################
        
        output = torch.cat((output_o, output_p), dim = 1) # B, 2(S-1)+1, D
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
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        # x has shape B, 2*(S-1), D?
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class EBTTimeConcat(nn.Module):
    def __init__(self, params: EBTModelArgs, max_mcmc_steps):
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

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len
        )

        self.time_embeddings = nn.Embedding(max_mcmc_steps, params.dim)

        self.final_layer = nn.Linear(params.dim, 1, bias = False)
        init_whole_model_weights(self.final_layer, self.params.weight_initialization)

    def forward(self, embeddings: torch.Tensor, start_pos: int, mcmc_step = 0):
        """
        Perform a forward pass through the Transformer model.

        Args:
            embeds (torch.Tensor): Embeddings (instead of tokens since is for vision).
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output energies after applying the Transformer model.

        """
        _bsz = embeddings.shape[0]
        mcmc_step = torch.full(size=(_bsz,), fill_value=mcmc_step, device = embeddings.device, dtype=torch.long)
        time_embeddings = self.time_embeddings(mcmc_step).unsqueeze(dim=1) # needs to be expanded to B, 1, D
        embeddings = torch.cat((time_embeddings, embeddings), dim = 1) # B, 2S - 1, D

        _bsz, seqlen = embeddings.shape[:2]
        seqlen = (seqlen+3) // 2 # do this since passed in seqlen is 2(S-1)+1 so add 3 div 2 = S+1 which corresponds to concatting time embed
        self.freqs_cis = self.freqs_cis.to(embeddings.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=embeddings.device
            )

            mask = torch.triu(mask, diagonal=1)

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
                embeddings = layer(embeddings, start_pos, freqs_cis, mask)
            embeddings = self.norm(embeddings)
            embeddings = embeddings[:, 1:] # remove temporal embed
            energies = self.final_layer(embeddings)

            energies = energies[:, embeddings.shape[1] // 2:]
            return energies
