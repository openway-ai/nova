"""Autoregressive Energy-Based Transformer (EBT) with Adaptive Layer Normalization.

This module defines the building blocks of the AR-EBT model, including
:class:`RMSNorm`, rotary positional embeddings, a specialized multi-head
:class:`Attention` that jointly processes real and predicted token embeddings,
the :class:`FeedForward` network, the AdaLN-conditioned
:class:`AdaLNTransformerBlock`, a :class:`FinalLayer` that produces scalar
energies, and the top-level :class:`EBTAdaLN` model.

The transformer architecture follows Llama2
(https://github.com/meta-llama/llama) and the AdaLN conditioning follows DiT
(https://github.com/facebookresearch/DiT, https://arxiv.org/pdf/2212.09748).
"""
# NOTE most code gotten from llama2 codebase -- credit:https://github.com/meta-llama/llama
# adaln code also gotten from DiT codebase -- credit:https://github.com/facebookresearch/DiT, paper https://arxiv.org/pdf/2212.09748
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


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization module."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize the RMSNorm normalization layer.

        :param dim: The dimension of the input tensor.
        :type dim: int
        :param eps: A small value added to the denominator for numerical
            stability.
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

        :param x: The input tensor.
        :type x: torch.Tensor
        :return: The output tensor after applying RMSNorm.
        :rtype: torch.Tensor
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the frequency tensor for complex exponentials (cis).

    Calculates a frequency tensor with complex exponentials using the given
    dimension ``dim`` and the end index ``end``. The ``theta`` parameter scales
    the frequencies. The returned tensor contains complex values in complex64
    data type.

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
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape frequency tensor for broadcasting with another tensor.

    Reshapes the frequency tensor to have the same shape as the target tensor
    ``x`` for the purpose of broadcasting the frequency tensor during
    element-wise operations.

    :param freqs_cis: Frequency tensor to be reshaped.
    :type freqs_cis: torch.Tensor
    :param x: Target tensor for broadcasting compatibility.
    :type x: torch.Tensor
    :return: Reshaped frequency tensor.
    :rtype: torch.Tensor
    :raises AssertionError: If the frequency tensor doesn't match the expected
        shape, or if the target tensor ``x`` doesn't have the expected number
        of dimensions.
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
    """Apply rotary embeddings to input tensors using the given frequency tensor.

    Applies rotary embeddings to the given query ``xq`` and key ``xk`` tensors
    using the provided frequency tensor ``freqs_cis``. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for
    broadcasting compatibility. The resulting tensors contain rotary
    embeddings and are returned as real tensors.

    :param xq: Query tensor to apply rotary embeddings.
    :type xq: torch.Tensor
    :param xk: Key tensor to apply rotary embeddings.
    :type xk: torch.Tensor
    :param freqs_cis: Precomputed frequency tensor for complex exponentials.
    :type freqs_cis: torch.Tensor
    :return: Tuple of modified query tensor and key tensor with rotary
        embeddings.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Equivalent to ``torch.repeat_interleave(x, dim=2, repeats=n_rep)``.

    :param x: Input tensor of shape ``(bs, slen, n_kv_heads, head_dim)``.
    :type x: torch.Tensor
    :param n_rep: Number of times to repeat each KV head.
    :type n_rep: int
    :return: Tensor with KV heads repeated along dim 2.
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

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN-style affine modulation to ``x``.

    :param x: Input tensor of shape ``(B, S, D)``.
    :type x: torch.Tensor
    :param shift: Shift tensor of shape ``(B, D)``.
    :type shift: torch.Tensor
    :param scale: Scale tensor of shape ``(B, D)``.
    :type scale: torch.Tensor
    :return: Modulated tensor ``x * (1 + scale) + shift`` broadcast across the
        sequence dimension.
    :rtype: torch.Tensor
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, layer_id: int, args: EBTModelArgs) -> None:
        """Initialize the Attention module.

        :param layer_id: Identifier for the layer.
        :type layer_id: int
        :param args: Model configuration parameters.
        :type args: EBTModelArgs
        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1  # NOTE this is hardcoded since we are using DDP
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

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        ve: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the attention module.

        :param x: Input tensor with concatenated real and predicted embeddings
            of shape ``(B, 2*(S-1), D)``.
        :type x: torch.Tensor
        :param start_pos: Starting position for caching.
        :type start_pos: int
        :param freqs_cis: Precomputed frequency tensor.
        :type freqs_cis: torch.Tensor
        :param mask: Attention mask tensor.
        :type mask: Optional[torch.Tensor]
        :param ve: Optional value-embeddings tensor gated into ``xv``.
        :type ve: Optional[torch.Tensor]
        :return: Output tensor after attention of shape ``(B, 2*(S-1), D)``.
        :rtype: torch.Tensor
        """
        # NOTE the usage of S-1/S/S+1 is messed up and confusing here, I recommend checking the paper
        bsz, full_seqlen, _ = x.shape  # full_seqlen includes real embeds and pred embeds
        original_seqlen = full_seqlen//2  # length of original sequence without next pred
        context_length = original_seqlen + 1  # actual context length of model
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, full_seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)
        if ve is not None and self.ve_gate is not None:
            ve = ve.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[:, :, :self.ve_gate_channels]))
            xv = xv + gate.unsqueeze(-1) * ve

        # _o is for original attention stuff
        xq_o = xq[:, :original_seqlen, :, :]  # B, S-1, N, H (N and H are num head and head dim respectively)
        xk_o = xk[:, :original_seqlen, :, :]
        xv_o = xv[:, :original_seqlen, :, :]

        # _p is for predicted attention stuff
        xq_p = xq[:, original_seqlen:, :, :]  # B, S-1, N, H (N and H are num head and head dim respectively)
        xk_p = xk[:, original_seqlen:, :, :]
        xv_p = xv[:, original_seqlen:, :, :]



        xq_o, xk_o = apply_rotary_emb(xq_o, xk_o, freqs_cis=freqs_cis[:original_seqlen])

        xq_p, xk_p = apply_rotary_emb(xq_p, xk_p, freqs_cis=freqs_cis[1:context_length])  # use 1 since are the next preds and thus need to condition on a frame
        # I tested this compared to prepending row on S dimension and the tensors were the same

        # original attn calc is more normal ############################################

        # seqlen here is S-1 which = original_seqlen
        xq_o = xq_o.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys_o = xk_o.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        values_o = xv_o.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        scores_o = torch.matmul(xq_o, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim)  # B, N, S-1, S-1
        if mask is not None:
            # this mask needs to be seqlen, seqlen, was S, S
            o_mask = mask[:-1, :-1]  # set to S-1, S-1 like 0 -inf -inf; 0 0 -inf, etc
            scores_o = scores_o + o_mask  # (bs, n_local_heads, seqlen, seqlen)
        scores_o = F.softmax(scores_o.float(), dim=-1).type_as(xq_o)
        output_o = torch.matmul(scores_o, values_o)  # (bs, n_local_heads, seqlen, head_dim)
        output_o = output_o.transpose(1, 2).contiguous().view(bsz, original_seqlen, -1)  # has B, S-1, D after

        # pred sequence attn calc is for energy-based transformer ########################################################################################

        # seqlen here is S-1 which = original_seqlen
        xq_p = xq_p.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys_p = xk_p.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        values_p = xv_p.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        scores_p = torch.matmul(xq_p, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim)  # B, N, S-1, S-1; this uses xq_p and keys_o since for every next pred calcs similarity to all prev words
        temp_append = torch.zeros((scores_p.shape[0], scores_p.shape[1], scores_p.shape[2], 1), dtype=scores_p.dtype, device=scores_p.device)  # B, N, S-1, 1; is used since context_length = original_length +1, superdiag needs this
        scores_p = torch.cat((scores_p, temp_append), dim = -1)  # is B, N, S-1, S; represents for each next pred (S-1 row) attending to all previous words (S-1) and then itself +1

        insertion_superdiagonal = (xq_p * keys_p).sum(dim = 3) / math.sqrt(self.head_dim)
        insertion_superdiagonal = insertion_superdiagonal.to(scores_p.dtype)  # for if using non 32 precision
        # bs, n, s-1 ; this calcs attn score of next preds with themselves, is like grabbing diag of matmul

        superdiag_rows = torch.arange(scores_p.shape[2])  # [0, ..., S-2] (len 15)
        superdiag_cols = torch.arange(1, scores_p.shape[3])  # [1, ..., S-1] (len 15)
        # use [3] last line since is [2]+1 and scores_p is wider than is tall as has B, N, S-1, S

        # first remove superdiagonal values so doesnt use attention to future tokens--prevents leakage of probability mass
        zero_superdiag = torch.zeros_like(insertion_superdiagonal, dtype=scores_p.dtype, device=scores_p.device)  # for zeroing out superdiag since dont want to include in matmul, do this in differentiable way
        diagonal_removal_mask = torch.ones_like(scores_p, dtype=scores_p.dtype, device=scores_p.device)
        diagonal_removal_mask[:, :, superdiag_rows, superdiag_cols] = zero_superdiag
        scores_p = scores_p * diagonal_removal_mask

        # then set diagonal to next pred self attention scores in differentiable way
        diagonal_addition_mask = torch.zeros_like(scores_p, dtype=scores_p.dtype, device=scores_p.device)
        diagonal_addition_mask[:, :, superdiag_rows, superdiag_cols] = insertion_superdiagonal
        scores_p = scores_p + diagonal_addition_mask

        if mask is not None:
            p_mask = mask[1:, :]  # S-1, S like 0 0 -inf -inf; 0 0 0, -inf, etc
            scores_p = scores_p + p_mask
        scores_p = F.softmax(scores_p.float(), dim=-1).type_as(xq_p)

        # Q: why do I need to extract superdiagonal why cant i just do matmul after? A: its bc would need same subsequence in value matrix but dont have it, have original subsequence and then seperately all next preds
        scores_p_superdiagonal = scores_p.diagonal(offset=1, dim1=2, dim2=3).clone()  # is B, N, S-1; basically how much each token on this superdiag should attent to itself; clone since dont want mask to change this

        scores_p = scores_p * diagonal_removal_mask  # keeps scores_p as is except for superdiagonal which is next preds attention to selves, cant multiply these naively by values_p or values_o

        scores_p = scores_p[:, :, :, :-1]  # B, N, S-1, S-1 now; next preds/scores_p_superdiagonal was why needed extra col earlier (temp_append)
        output_p = torch.matmul(scores_p, values_o)  # B, N, S-1, H; is how next preds attend to all original previous tokens;

        # next_pred_self_attention is to get self attention based on extracted superdiagonal and the values matrix (for predictions)
        next_pred_self_attention = values_p * scores_p_superdiagonal.unsqueeze(dim = -1)  # B, N, S-1, H this is for weighted sum of each next pred to its final embed rep.

        output_p = output_p + next_pred_self_attention  # B, N, S-1, H adding this is adding the aspect of each next pred embedding attending to itself
        output_p = output_p.transpose(1, 2).contiguous().view(bsz, original_seqlen, -1)  # after this is B, S-1, D

        # return linear projection of concatted outputs ########################################################################################

        output = torch.cat((output_o, output_p), dim = 1)  # B, 2(S-1), D
        return self.wo(output)


class FeedForward(nn.Module):
    """SwiGLU feed-forward network used inside each transformer block."""

    def __init__(
        self,
        dim: int,
        ffn_dim_multiplier: Optional[float],
        weight_initialization: str,
        weight_initialization_gain: float
    ) -> None:
        """Initialize the FeedForward module.

        :param dim: Input (and output) dimension.
        :type dim: int
        :param ffn_dim_multiplier: Custom multiplier for hidden dimension. If
            ``None``, the hidden dimension equals ``dim``.
        :type ffn_dim_multiplier: Optional[float]
        :param weight_initialization: Name of the weight initialization scheme
            forwarded to :func:`init_whole_model_weights`.
        :type weight_initialization: str
        :param weight_initialization_gain: Gain value forwarded to
            :func:`init_whole_model_weights`.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SwiGLU feed-forward network.

        :param x: Input tensor of shape ``(..., dim)``.
        :type x: torch.Tensor
        :return: Output tensor of shape ``(..., dim)``.
        :rtype: torch.Tensor
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class AdaLNTransformerBlock(nn.Module):
    """Transformer block conditioned on a time embedding via AdaLN-Zero."""

    def __init__(self, layer_id: int, args: EBTModelArgs) -> None:
        """Initialize an AdaLNTransformerBlock.

        :param layer_id: Identifier for the layer.
        :type layer_id: int
        :param args: Model configuration parameters.
        :type args: EBTModelArgs
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
            weight_initialization_gain=args.weight_initialization_gain
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, 6 * self.dim, bias=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        time_embeddings: torch.Tensor,
        ve: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform a forward pass through the transformer block.

        :param x: Input tensor.
        :type x: torch.Tensor
        :param start_pos: Starting position for attention caching.
        :type start_pos: int
        :param freqs_cis: Precomputed cosine and sine frequencies.
        :type freqs_cis: torch.Tensor
        :param mask: Masking tensor for attention.
        :type mask: Optional[torch.Tensor]
        :param time_embeddings: MCMC step time embedding used as the AdaLN
            conditioning signal.
        :type time_embeddings: torch.Tensor
        :param ve: Optional value embeddings passed through to attention.
        :type ve: Optional[torch.Tensor]
        :return: Output tensor after applying attention and feed-forward
            layers.
        :rtype: torch.Tensor
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(time_embeddings).chunk(6, dim=1)

        h = x + gate_msa.unsqueeze(1) * self.attention(
            modulate(self.attention_norm(x), shift_msa, scale_msa), start_pos, freqs_cis, mask, ve,
        )
        out = h + gate_mlp.unsqueeze(1) * self.feed_forward(modulate(self.ffn_norm(h), shift_mlp, scale_mlp))
        return out

class FinalLayer(nn.Module):
    """The final layer of EBT when using AdaLN.

    Applies an AdaLN shift/scale conditioned on the time embedding and projects
    the hidden state to a scalar energy per position.
    """

    def __init__(self, hidden_size: int) -> None:
        """Initialize the final AdaLN output head.

        :param hidden_size: Dimensionality of the model's hidden state.
        :type hidden_size: int
        """
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1, bias = False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Project hidden states to scalar energies with AdaLN conditioning.

        :param x: Hidden state tensor of shape ``(B, S, D)``.
        :type x: torch.Tensor
        :param c: Conditioning tensor of shape ``(B, D)`` (time embedding).
        :type c: torch.Tensor
        :return: Scalar energy tensor of shape ``(B, S, 1)``.
        :rtype: torch.Tensor
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(x, shift, scale)
        x = self.linear(x)
        return x

class EBTAdaLN(nn.Module):
    """Energy-Based Transformer using AdaLN conditioning on MCMC step embeddings."""

    def __init__(self, params: EBTModelArgs, max_mcmc_steps: int) -> None:
        """Initialize a Transformer model.

        :param params: Model configuration parameters.
        :type params: EBTModelArgs
        :param max_mcmc_steps: Maximum number of MCMC steps, used to size the
            time-step embedding table.
        :type max_mcmc_steps: int
        """
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers
        self.use_ve = params.use_ve and build_value_embeds is not None
        if self.use_ve:
            n_kv_heads = params.n_heads if params.n_kv_heads is None else params.n_kv_heads
            self.kv_dim = n_kv_heads * (params.dim // params.n_heads)
            self.value_embeds = build_value_embeds(params.n_layers, params.vocab_size, self.kv_dim)
            ve_init_bound = math.sqrt(3.0) * (params.dim ** -0.5)
            for ve in self.value_embeds.values():
                nn.init.uniform_(ve.weight, -ve_init_bound, ve_init_bound)
        else:
            self.value_embeds = None
            self.kv_dim = None

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            block = AdaLNTransformerBlock(layer_id, params)
            if params.adaln_zero_init:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            self.layers.append(block)  # confirmed all layers and final layer are initialized to 0

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len
        )

        self.time_embeddings = nn.Embedding(max_mcmc_steps, params.dim)

        self.final_layer = FinalLayer(params.dim)
        if params.adaln_zero_init:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_layer.linear.weight, 0)
        else:
            init_whole_model_weights(self.final_layer.linear, self.params.weight_initialization)

    def forward(
        self,
        embeddings: torch.Tensor,
        start_pos: int,
        mcmc_step: int = 0,
        real_token_ids: Optional[torch.Tensor] = None,
        predicted_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform a forward pass through the Transformer model.

        :param embeddings: Concatenated real-and-predicted embeddings of shape
            ``(B, 2*(S-1), D)``. Embeddings are used directly (no token lookup).
        :type embeddings: torch.Tensor
        :param start_pos: Starting position for attention caching.
        :type start_pos: int
        :param mcmc_step: Current MCMC step index used to select the time
            embedding.
        :type mcmc_step: int
        :param real_token_ids: Optional token ids used to build value
            embeddings when ``use_ve`` is enabled.
        :type real_token_ids: Optional[torch.Tensor]
        :param predicted_tokens: Optional predicted token ids used to build
            value embeddings when ``use_ve`` is enabled.
        :type predicted_tokens: Optional[torch.Tensor]
        :return: Output energies after applying the Transformer model, shape
            ``(B, S-1, 1)``.
        :rtype: torch.Tensor
        """
        _bsz, seqlen = embeddings.shape[:2]
        seqlen = (seqlen+2) // 2  # do this since passed in seqlen is 2(S-1) so add 2 div 2 = S
        self.freqs_cis = self.freqs_cis.to(embeddings.device)

        # Dynamically extend freqs_cis if the required length exceeds the precomputed length.
        required_length = start_pos + seqlen
        if required_length > self.freqs_cis.shape[0]:
            # Recompute a longer freqs_cis.
            new_freqs_cis = precompute_freqs_cis(
                self.params.dim // self.params.n_heads,
                required_length
            ).to(embeddings.device)
            self.freqs_cis = new_freqs_cis

        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        mcmc_step = torch.full(size=(_bsz,), fill_value=mcmc_step, device = embeddings.device, dtype=torch.long)
        time_embeddings = self.time_embeddings(mcmc_step)

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
                ve = None
                if self.use_ve and real_token_ids is not None and predicted_tokens is not None:
                    ve = build_layer_ve(self.value_embeds, i, real_token_ids, predicted_tokens)
                embeddings = layer(embeddings, start_pos, freqs_cis, mask, time_embeddings, ve)
            embeddings = self.norm(embeddings)
            energies = self.final_layer(embeddings, time_embeddings)

            energies = energies[:, embeddings.shape[1] // 2:]
            return energies
