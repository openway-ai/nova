"""
VE (Value Embedding) 核心模块

实现 GPT 风格的 Value Embedding：
- real 分支：用离散 token id 查表
- 默认表结构：nn.Embedding(vocab_size, kv_dim)
- 低秩表结构：nn.Embedding(vocab_size, ve_rank) + nn.Linear(ve_rank, kv_dim)

通过 --use_ve 参数控制是否启用该功能。
"""

import math
import torch
import torch.nn as nn


def has_ve(layer_idx, n_layers, use_sparse_ve=False):
    """
    判断某一层是否使用 VE。

    默认策略：每隔一层使用 VE，从最后一层开始交替。
    稀疏策略：仅在最后一层启用 VE。
    """
    if not use_sparse_ve:
        return layer_idx % 2 == (n_layers - 1) % 2

    first_third = n_layers // 3
    second_third = (2 * n_layers) // 3

    if layer_idx < first_third:
        return False
    return layer_idx == n_layers - 1


def build_value_embeds(n_layers, vocab_size, kv_dim, use_low_rank_ve=False, ve_rank=256, use_sparse_ve=False):
    """
    为需要 VE 的层构建 Embedding 表。
    """
    if use_low_rank_ve:
        return nn.ModuleDict(
            {
                str(i): nn.ModuleDict(
                    {
                        "embed": nn.Embedding(vocab_size, ve_rank),
                        "proj": nn.Linear(ve_rank, kv_dim, bias=False),
                    }
                )
                for i in range(n_layers)
                if has_ve(i, n_layers, use_sparse_ve)
            }
        )

    return nn.ModuleDict(
        {str(i): nn.Embedding(vocab_size, kv_dim) for i in range(n_layers) if has_ve(i, n_layers, use_sparse_ve)}
    )


def init_value_embeds(value_embeds, base_dim):
    """
    初始化 VE 参数。

    普通 VE 初始化 Embedding；低秩 VE 分别初始化低秩 Embedding 和投影层。
    """
    ve_init_bound = math.sqrt(3.0) * (base_dim ** -0.5)
    for ve in value_embeds.values():
        if isinstance(ve, nn.ModuleDict):
            nn.init.uniform_(ve["embed"].weight, -ve_init_bound, ve_init_bound)
            proj_init_bound = math.sqrt(3.0) * (ve["embed"].embedding_dim ** -0.5)
            nn.init.uniform_(ve["proj"].weight, -proj_init_bound, proj_init_bound)
        else:
            nn.init.uniform_(ve.weight, -ve_init_bound, ve_init_bound)


def build_layer_ve(value_embeds, layer_id, real_token_ids, predicted_tokens, extra_prefix_tokens=0):
    """
    为指定层构建 VE 张量。

    普通 VE：real 分支和 predicted 分支均使用离散 token ID 直接查表。
    低秩 VE：predicted 分支使用 softmax 后的概率分布加权低秩嵌入表，再投影到 kv_dim。
    """
    layer_key = str(layer_id)
    if layer_key not in value_embeds:
        return None

    value_module = value_embeds[layer_key]
    is_low_rank = isinstance(value_module, nn.ModuleDict)
    value_table = value_module["embed"] if is_low_rank else value_module

    ve_real = value_table(real_token_ids)
    if is_low_rank:
        ve_pred = torch.matmul(
            torch.softmax(predicted_tokens, dim=-1),
            value_table.weight,
        )
    else:
        pred_ids = predicted_tokens.argmax(dim=-1)  # (B, S) 离散化
        ve_pred = value_table(pred_ids)              # 直接查表

    if extra_prefix_tokens > 0:
        batch_size = real_token_ids.shape[0]
        ve_dim = value_table.weight.shape[1]
        prefix = torch.zeros(
            batch_size,
            extra_prefix_tokens,
            ve_dim,
            device=ve_real.device,
            dtype=ve_real.dtype,
        )
        ve = torch.cat((prefix, ve_real, ve_pred), dim=1)
    else:
        ve = torch.cat((ve_real, ve_pred), dim=1)

    return value_module["proj"](ve) if is_low_rank else ve
