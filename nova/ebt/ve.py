"""
VE (Value Embedding) 核心模块

实现 GPT 风格的 Value Embedding，为每个启用层建立 nn.Embedding(vocab_size, kv_dim) 表：
- real 分支：用离散 token id 查表
- predicted 分支：用 soft 分布加权

通过 --use_ve 参数控制是否启用该功能。
"""

import torch
import torch.nn as nn


def has_ve(layer_idx, n_layers):
    """
    判断某一层是否使用 VE。

    策略：每隔一层使用 VE，从最后一层开始交替。
    例如：n_layers=12 时，层 1, 3, 5, 7, 9, 11 使用 VE。

    Args:
        layer_idx: 当前层索引（从 0 开始）
        n_layers: 总层数

    Returns:
        bool: 该层是否使用 VE
    """
    return layer_idx % 2 == (n_layers - 1) % 2


def build_value_embeds(n_layers, vocab_size, kv_dim):
    """
    为需要 VE 的层构建 Embedding 表。

    Args:
        n_layers: Transformer 层数
        vocab_size: 词汇表大小
        kv_dim: KV 维度（n_kv_heads * head_dim）

    Returns:
        nn.ModuleDict: 键为层索引字符串，值为对应的 nn.Embedding
    """
    return nn.ModuleDict(
        {str(i): nn.Embedding(vocab_size, kv_dim) for i in range(n_layers) if has_ve(i, n_layers)}
    )


def build_layer_ve(value_embeds, layer_id, real_token_ids, predicted_tokens, extra_prefix_tokens=0):
    """
    为指定层构建 VE 张量。

    对于 real 分支，使用离散 token ID 直接查表；
    对于 predicted 分支，使用 softmax 后的概率分布加权嵌入表。

    Args:
        value_embeds: ModuleDict，包含各层的 Embedding 表
        layer_id: 当前层索引
        real_token_ids: 真实 token IDs，shape (B, S)
        predicted_tokens: 预测的 logits/概率分布，shape (B, S, V)
        extra_prefix_tokens: 额外前缀 token 数（time_embed 变体需要为 1）

    Returns:
        VE 张量，shape (B, 2*S + extra_prefix_tokens, kv_dim) 或 None（如果该层不使用 VE）
    """
    layer_key = str(layer_id)
    if layer_key not in value_embeds:
        return None

    value_table = value_embeds[layer_key]

    # Real 分支：离散 token id 查表
    ve_real = value_table(real_token_ids)  # (B, S, kv_dim)

    # Predicted 分支：soft 分布加权
    ve_pred = torch.matmul(
        torch.softmax(predicted_tokens, dim=-1),  # (B, S, V)
        value_table.weight  # (V, kv_dim)
    )  # (B, S, kv_dim)

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
