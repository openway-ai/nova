"""Value Embedding (VE) module.

Implements GPT-style Value Embeddings by maintaining an
``nn.Embedding(vocab_size, kv_dim)`` table for every layer that opts in:

- ``real`` branch: indexes the table with the discrete ground-truth token ids.
- ``predicted`` branch: discretizes the predicted logits via ``argmax`` before
  indexing the table, which avoids materializing a ``softmax + matmul`` over
  ``(B, S, vocab_size)`` and therefore saves a large amount of activation memory.

Enabled through the ``--use_ve`` command-line flag.
"""

from typing import Optional

import torch
import torch.nn as nn


def has_ve(layer_idx: int, n_layers: int) -> bool:
    """Check whether a given layer has a Value Embedding table.

    Layers that opt-in alternate with a stride of two, starting from the last
    layer (e.g. for ``n_layers=12`` the VE-enabled layers are
    ``1, 3, 5, 7, 9, 11``).

    :param layer_idx: Zero-based layer index.
    :type layer_idx: int
    :param n_layers: Total number of transformer layers.
    :type n_layers: int
    :return: ``True`` if the layer uses a VE table, ``False`` otherwise.
    :rtype: bool
    """
    return layer_idx % 2 == (n_layers - 1) % 2


def build_value_embeds(n_layers: int, vocab_size: int, kv_dim: int) -> nn.ModuleDict:
    """Build VE tables for every layer that opts in.

    :param n_layers: Total number of transformer layers.
    :type n_layers: int
    :param vocab_size: Vocabulary size of the tokenizer.
    :type vocab_size: int
    :param kv_dim: Embedding dimension used by the KV branch.
    :type kv_dim: int
    :return: A module dict keyed by the stringified layer index whose values
        are ``nn.Embedding(vocab_size, kv_dim)`` tables.
    :rtype: nn.ModuleDict
    """
    return nn.ModuleDict(
        {str(i): nn.Embedding(vocab_size, kv_dim) for i in range(n_layers) if has_ve(i, n_layers)}
    )


def build_layer_ve(
    value_embeds: nn.ModuleDict,
    layer_id: int,
    real_token_ids: torch.Tensor,
    predicted_tokens: torch.Tensor,
    extra_prefix_tokens: int = 0,
) -> Optional[torch.Tensor]:
    """Build the VE tensor for a single layer.

    Both branches look up the same embedding table with discrete token ids.
    The ``predicted`` branch first reduces ``predicted_tokens`` with ``argmax``
    rather than multiplying a softmax probability distribution with the
    embedding weight, which avoids creating a ``(B, S, vocab_size)`` activation.

    :param value_embeds: The ``ModuleDict`` produced by :func:`build_value_embeds`.
    :type value_embeds: nn.ModuleDict
    :param layer_id: Target layer index.
    :type layer_id: int
    :param real_token_ids: Tensor of shape ``(B, S)`` with ground-truth ids.
    :type real_token_ids: torch.Tensor
    :param predicted_tokens: Tensor of shape ``(B, S, vocab_size)`` holding
        predicted logits or probabilities.
    :type predicted_tokens: torch.Tensor
    :param extra_prefix_tokens: Number of zero-filled prefix positions to
        prepend to the resulting tensor (used when the layer processes extra
        special-token positions).
    :type extra_prefix_tokens: int
    :return: A tensor of shape
        ``(B, extra_prefix_tokens + 2 * S, kv_dim)`` concatenating the real and
        predicted branches (plus the optional prefix), or ``None`` if the given
        layer has no VE table.
    :rtype: Optional[torch.Tensor]
    """
    layer_key = str(layer_id)
    if layer_key not in value_embeds:
        return None

    value_table = value_embeds[layer_key]

    ve_real = value_table(real_token_ids)
    pred_ids = predicted_tokens.argmax(dim=-1)
    ve_pred = value_table(pred_ids)

    if extra_prefix_tokens > 0:
        batch_size = real_token_ids.shape[0]
        kv_dim = value_table.weight.shape[1]
        prefix = torch.zeros(
            batch_size,
            extra_prefix_tokens,
            kv_dim,
            device=ve_real.device,
            dtype=ve_real.dtype,
        )
        return torch.cat((prefix, ve_real, ve_pred), dim=1)

    return torch.cat((ve_real, ve_pred), dim=1)
