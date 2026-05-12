"""Default autoregressive Energy-Based Transformer (EBT) model components.

Contains the Llama2-style building blocks (RMSNorm, rotary embeddings,
attention, feed-forward, transformer block) together with the
energy-based head that scores predicted-token embeddings.
"""
# NOTE: most code adapted from the llama2 codebase. Credit: https://github.com/meta-llama/llama
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
    """Root-mean-square layer normalization.

    :ivar eps: Small constant added to the denominator for numerical stability.
    :vartype eps: float
    :ivar weight: Learnable per-feature scale parameter.
    :vartype weight: torch.nn.Parameter
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize the RMSNorm normalization layer.

        :param dim: Feature dimension to normalize over (last axis).
        :type dim: int
        :param eps: Small value added to the denominator for numerical stability.
        :type eps: float
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the RMS normalization formula to the input tensor.

        :param x: Input tensor whose last dimension is normalized.
        :type x: torch.Tensor
        :return: RMS-normalized tensor with the same shape as ``x``.
        :rtype: torch.Tensor
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the RMSNorm layer.

        Computes the normalization in float32 for numerical stability and
        casts the result back to the input dtype before multiplying by the
        learnable scale.

        :param x: Input tensor of shape ``(..., dim)``.
        :type x: torch.Tensor
        :return: Scaled, RMS-normalized tensor with the same shape and dtype
            as ``x``.
        :rtype: torch.Tensor
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the rotary-embedding complex frequency tensor.

    Computes the complex exponentials used by rotary position embeddings for
    positions ``[0, end)`` and head dimension ``dim``. The frequencies are
    scaled by ``theta`` in the standard RoPE formulation. The returned tensor
    holds ``complex64`` values.

    :param dim: Per-head feature dimension.
    :type dim: int
    :param end: Exclusive upper bound on the position index.
    :type end: int
    :param theta: Base for the geometric frequency progression.
    :type theta: float
    :return: Complex frequency tensor of shape ``(end, dim // 2)``.
    :rtype: torch.Tensor
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # NOTE: complex64 representation of the rotation angles.
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape the frequency tensor so it broadcasts against ``x``.

    Inserts singleton dimensions so that the sequence axis (``dim 1``) and
    the head-dim axis (``dim -1``) of ``freqs_cis`` align with ``x``.

    :param freqs_cis: Precomputed complex frequency tensor of shape
        ``(x.shape[1], x.shape[-1])``.
    :type freqs_cis: torch.Tensor
    :param x: Target tensor whose shape dictates the broadcast layout.
    :type x: torch.Tensor
    :return: View of ``freqs_cis`` reshaped for broadcasting against ``x``.
    :rtype: torch.Tensor
    :raises AssertionError: If ``x.ndim`` or ``freqs_cis.shape`` are not
        compatible with the expected layout.
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

    Interprets the last dimension of ``xq`` and ``xk`` as complex pairs,
    multiplies by ``freqs_cis``, and converts back to real tensors with the
    original dtype.

    :param xq: Query tensor of shape ``(B, S, H, D)`` where ``D`` is even.
    :type xq: torch.Tensor
    :param xk: Key tensor of shape ``(B, S, H, D)`` where ``D`` is even.
    :type xk: torch.Tensor
    :param freqs_cis: Precomputed complex frequency tensor.
    :type freqs_cis: torch.Tensor
    :return: Tuple ``(xq_out, xk_out)`` of rotated query and key tensors,
        each with the same shape and dtype as the inputs.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads along the head dimension.

    Equivalent to ``torch.repeat_interleave(x, dim=2, repeats=n_rep)`` but
    implemented via ``expand`` + ``reshape`` to avoid materializing the
    intermediate copy when possible.

    :param x: Tensor of shape ``(B, S, n_kv_heads, head_dim)``.
    :type x: torch.Tensor
    :param n_rep: Number of times to repeat each KV head.
    :type n_rep: int
    :return: Tensor of shape ``(B, S, n_kv_heads * n_rep, head_dim)``.
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
    """Multi-head attention module with EBT-specific prediction branch.

    Implements causal self-attention for the ``original`` token stream and a
    parallel attention path for the ``predicted`` tokens, whose self-scores
    are spliced on the super-diagonal of the attention matrix.
    """

    def __init__(self, layer_id: int, args: EBTModelArgs):
        """Initialize the Attention module.

        :param layer_id: Identifier of this layer within the stack; used for
            deciding whether value embeddings (VE) should be enabled.
        :type layer_id: int
        :param args: Model configuration parameters.
        :type args: EBTModelArgs
        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1  # NOTE: hardcoded to 1 since we use DDP rather than tensor parallelism.
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
        """Forward pass of the EBT attention module.

        The input ``x`` concatenates the original token embeddings and the
        predicted token embeddings along the sequence dimension. The module
        produces attended outputs for both halves and concatenates them on
        return.

        :param x: Input tensor of shape ``(B, 2*(S-1), D)``.
        :type x: torch.Tensor
        :param start_pos: Starting position (currently unused, retained for
            API compatibility with the KV-cache variant).
        :type start_pos: int
        :param freqs_cis: Precomputed rotary frequency tensor.
        :type freqs_cis: torch.Tensor
        :param mask: Optional additive attention mask (``-inf`` for masked
            positions).
        :type mask: Optional[torch.Tensor]
        :param ve: Optional value-embedding tensor of shape
            ``(B, 2*(S-1), n_local_kv_heads * head_dim)`` mixed into ``xv``
            through a learned gate.
        :type ve: Optional[torch.Tensor]
        :return: Output tensor of shape ``(B, 2*(S-1), D)``.
        :rtype: torch.Tensor
        """
        # NOTE: the usage of S-1/S/S+1 below is confusing; see the paper for details.
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

        # _o suffix: tensors for the original sequence.
        xq_o = xq[:, :original_seqlen, :, :]  # B, S-1, N, H (N=num heads, H=head dim)
        xk_o = xk[:, :original_seqlen, :, :]
        xv_o = xv[:, :original_seqlen, :, :]

        # _p suffix: tensors for the predicted (next-token) sequence.
        xq_p = xq[:, original_seqlen:, :, :]  # B, S-1, N, H
        xk_p = xk[:, original_seqlen:, :, :]
        xv_p = xv[:, original_seqlen:, :, :]


        xq_o, xk_o = apply_rotary_emb(xq_o, xk_o, freqs_cis=freqs_cis[:original_seqlen])

        # NOTE: use freqs_cis[1:context_length] for predictions because each next pred must
        # condition on a shifted frame.
        xq_p, xk_p = apply_rotary_emb(xq_p, xk_p, freqs_cis=freqs_cis[1:context_length])
        # NOTE: verified to be equivalent to prepending a row along the S dimension.

        # Original (causal) attention branch.

        # seqlen here is S-1 which = original_seqlen
        xq_o = xq_o.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys_o = xk_o.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        values_o = xv_o.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        scores_o = torch.matmul(xq_o, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim)  # B, N, S-1, S-1
        if mask is not None:
            # Mask needs to be (seqlen, seqlen); the incoming full mask is (S, S) so we slice to (S-1, S-1).
            o_mask = mask[:-1, :-1]  # (S-1, S-1) causal mask: row 0 = [0, -inf, ...], row 1 = [0, 0, -inf, ...].
            scores_o = scores_o + o_mask  # (bs, n_local_heads, seqlen, seqlen)
        scores_o = F.softmax(scores_o.float(), dim=-1).type_as(xq_o)
        output_o = torch.matmul(scores_o, values_o)  # (bs, n_local_heads, seqlen, head_dim)
        output_o = output_o.transpose(1, 2).contiguous().view(bsz, original_seqlen, -1)  # B, S-1, D

        # Predicted-sequence attention branch (EBT specific).

        # seqlen here is S-1 which = original_seqlen
        xq_p = xq_p.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys_p = xk_p.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        values_p = xv_p.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        # xq_p vs keys_o: each next-pred attends to all previous original tokens.
        scores_p = torch.matmul(xq_p, keys_o.transpose(2, 3)) / math.sqrt(self.head_dim)  # B, N, S-1, S-1
        # context_length = original_length + 1, so the super-diagonal needs an extra column.
        temp_append = torch.zeros((scores_p.shape[0], scores_p.shape[1], scores_p.shape[2], 1), dtype=scores_p.dtype, device=scores_p.device)  # B, N, S-1, 1
        scores_p = torch.cat((scores_p, temp_append), dim=-1)  # B, N, S-1, S: each next pred attends to all previous words (S-1) plus itself (+1).

        insertion_superdiagonal = (xq_p * keys_p).sum(dim=3) / math.sqrt(self.head_dim)
        insertion_superdiagonal = insertion_superdiagonal.to(scores_p.dtype)  # NOTE: cast needed when running in non-fp32 precision.
        # (bs, n, s-1); self-attention score of each next pred, i.e. the diagonal of the matmul.

        superdiag_rows = torch.arange(scores_p.shape[2])  # [0, ..., S-2] (length S-1)
        superdiag_cols = torch.arange(1, scores_p.shape[3])  # [1, ..., S-1] (length S-1)
        # Indexing uses shape[3] = shape[2] + 1 since scores_p has shape B, N, S-1, S (wider than tall).

        # First, zero the super-diagonal so we do not leak attention probability mass onto future tokens.
        zero_superdiag = torch.zeros_like(insertion_superdiagonal, dtype=scores_p.dtype, device=scores_p.device)  # NOTE: built this way to keep the operation differentiable.
        diagonal_removal_mask = torch.ones_like(scores_p, dtype=scores_p.dtype, device=scores_p.device)
        diagonal_removal_mask[:, :, superdiag_rows, superdiag_cols] = zero_superdiag
        scores_p = scores_p * diagonal_removal_mask

        # Then insert the next-pred self-attention scores onto the super-diagonal in a differentiable way.
        diagonal_addition_mask = torch.zeros_like(scores_p, dtype=scores_p.dtype, device=scores_p.device)
        diagonal_addition_mask[:, :, superdiag_rows, superdiag_cols] = insertion_superdiagonal
        scores_p = scores_p + diagonal_addition_mask

        if mask is not None:
            p_mask = mask[1:, :]  # (S-1, S) pred mask: row 0 = [0, 0, -inf, ...], row 1 = [0, 0, 0, -inf, ...].
            scores_p = scores_p + p_mask
        scores_p = F.softmax(scores_p.float(), dim=-1).type_as(xq_p)

        # NOTE: we extract the super-diagonal rather than doing a single matmul because the predicted
        # tokens live in a separate value matrix (values_p), not inside values_o.
        scores_p_superdiagonal = scores_p.diagonal(offset=1, dim1=2, dim2=3).clone()  # B, N, S-1; cloned so later mutation of scores_p does not affect it.

        # Re-apply the removal mask so scores_p carries only the original-token contributions.
        scores_p = scores_p * diagonal_removal_mask

        scores_p = scores_p[:, :, :, :-1]  # B, N, S-1, S-1; temp_append is no longer needed after the super-diagonal was split off.
        output_p = torch.matmul(scores_p, values_o)  # B, N, S-1, H: each next pred attending to all previous original tokens.

        # Add the contribution of each next-pred embedding attending to itself via the extracted super-diagonal.
        next_pred_self_attention = values_p * scores_p_superdiagonal.unsqueeze(dim=-1)  # B, N, S-1, H

        output_p = output_p + next_pred_self_attention  # B, N, S-1, H
        output_p = output_p.transpose(1, 2).contiguous().view(bsz, original_seqlen, -1)  # B, S-1, D

        # Concatenate both branches and project back to the model dimension.

        output = torch.cat((output_o, output_p), dim=1)  # B, 2(S-1), D
        return self.wo(output)


class FeedForward(nn.Module):
    """SwiGLU feed-forward block used inside each transformer layer."""

    def __init__(
        self,
        dim: int,
        ffn_dim_multiplier: Optional[float],
        weight_initialization: str,
        weight_initialization_gain: float
    ):
        """Initialize the FeedForward module.

        :param dim: Input and output feature dimension.
        :type dim: int
        :param ffn_dim_multiplier: Multiplier applied to ``dim`` to obtain
            the hidden dimension. If ``None`` the hidden dimension equals
            ``dim``.
        :type ffn_dim_multiplier: Optional[float]
        :param weight_initialization: Name of the weight-initialization
            scheme passed to :func:`init_whole_model_weights`.
        :type weight_initialization: str
        :param weight_initialization_gain: Gain factor for the
            initialization scheme.
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
        """Apply the SwiGLU feed-forward transformation.

        :param x: Input tensor of shape ``(..., dim)``.
        :type x: torch.Tensor
        :return: Output tensor of shape ``(..., dim)``.
        :rtype: torch.Tensor
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """One transformer layer: RMSNorm -> attention -> RMSNorm -> feed-forward, with residuals."""

    def __init__(self, layer_id: int, args: EBTModelArgs):
        """Initialize a TransformerBlock.

        :param layer_id: Zero-based index of this layer within the stack.
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

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        ve: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one transformer block on the concatenated (original + predicted) sequence.

        :param x: Input tensor of shape ``(B, 2*(S-1), D)``.
        :type x: torch.Tensor
        :param start_pos: Starting position for attention caching.
        :type start_pos: int
        :param freqs_cis: Precomputed rotary frequency tensor.
        :type freqs_cis: torch.Tensor
        :param mask: Optional additive attention mask.
        :type mask: Optional[torch.Tensor]
        :param ve: Optional value-embedding tensor forwarded to the attention
            module.
        :type ve: Optional[torch.Tensor]
        :return: Output tensor of shape ``(B, 2*(S-1), D)``.
        :rtype: torch.Tensor
        """
        # x has shape B, 2*(S-1), D
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask, ve
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class EBTDefault(nn.Module):
    """Default autoregressive Energy-Based Transformer.

    Stacks :class:`TransformerBlock` layers on top of input embeddings and
    produces a scalar energy per predicted token via a final linear head.
    """

    def __init__(self, params: EBTModelArgs):
        """Initialize the EBT model.

        :param params: Model configuration parameters.
        :type params: EBTModelArgs
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
        mcmc_step=None,
        real_token_ids: Optional[torch.Tensor] = None,
        predicted_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the EBT model.

        :param embeddings: Concatenated embeddings of shape
            ``(B, 2*(S-1), D)`` containing both real and predicted-token
            embeddings.
        :type embeddings: torch.Tensor
        :param start_pos: Starting position for attention caching.
        :type start_pos: int
        :param mcmc_step: MCMC step index. Accepted for API compatibility but
            not used in this implementation.
        :type mcmc_step: Optional[int]
        :param real_token_ids: Optional token IDs for the real sequence,
            consumed by value-embedding layers when ``use_ve`` is enabled.
        :type real_token_ids: Optional[torch.Tensor]
        :param predicted_tokens: Optional token IDs for the predicted
            sequence, consumed by value-embedding layers when ``use_ve`` is
            enabled.
        :type predicted_tokens: Optional[torch.Tensor]
        :return: Energy tensor of shape ``(B, S-1, 1)`` scoring the
            predicted-token half of the sequence.
        :rtype: torch.Tensor
        """
        # NOTE: mcmc_step is unused in this variant; kept for signature compatibility.
        _bsz, seqlen = embeddings.shape[:2]
        seqlen = (seqlen+2) // 2  # incoming seqlen is 2*(S-1); (2*(S-1) + 2) // 2 = S
        self.freqs_cis = self.freqs_cis.to(embeddings.device)

        # Dynamically extend freqs_cis if the required length exceeds the precomputed one.
        required_length = start_pos + seqlen
        if required_length > self.freqs_cis.shape[0]:
            # Recompute a longer freqs_cis tensor.
            new_freqs_cis = precompute_freqs_cis(
                self.params.dim // self.params.n_heads,
                required_length
            ).to(embeddings.device)
            self.freqs_cis = new_freqs_cis

        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=embeddings.device
            )

            mask = torch.triu(mask, diagonal=1)

            # NOTE: when performing key-value caching, attention is computed only for the new sequence.
            # The resulting score matrix is (seqlen, cache_len + seqlen); only entries (i, j) with
            # j > cache_len + i are masked since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=embeddings.device),
                mask
            ]).type_as(embeddings)
            # Default causal mask layout:
            #   0,    -inf, -inf
            #   0,    0,    -inf
            #   0,    0,    0


            for i, layer in enumerate(self.layers):
                ve = None
                if self.use_ve and real_token_ids is not None and predicted_tokens is not None:
                    ve = build_layer_ve(self.value_embeds, i, real_token_ids, predicted_tokens)
                embeddings = layer(embeddings, start_pos, freqs_cis, mask, ve)
            embeddings = self.norm(embeddings)
            energies = self.final_layer(embeddings)

            energies = energies[:, embeddings.shape[1] // 2:]
            return energies
