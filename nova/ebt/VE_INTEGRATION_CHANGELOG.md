# VE (Value Embedding) 集成改动文档

## 概述

本文档记录了将 VE (Value Embedding) 功能从 `ebt_veexp` 项目集成到 `nova-dev-ebt/nova/ebt` 项目的所有代码改动。

VE 功能实现 GPT 风格的 Value Embedding：
- 为交替层建立 `nn.Embedding(vocab_size, kv_dim)` 嵌入表
- Real 分支：使用离散 token ID 查表
- Predicted 分支：使用 soft 概率分布加权嵌入表
- 通过 `--use_ve` 命令行参数控制启用/禁用

---

## 1. 新建文件

### `/home/liqinuo/nova-dev-ebt/nova/ebt/ve.py`

**用途**: VE 核心模块，包含三个主要函数

```python
def has_ve(layer_idx, n_layers):
    """判断某一层是否使用 VE（交替层策略）"""
    return layer_idx % 2 == (n_layers - 1) % 2

def build_value_embeds(n_layers, vocab_size, kv_dim):
    """为需要 VE 的层构建 Embedding 表"""
    return nn.ModuleDict(
        {str(i): nn.Embedding(vocab_size, kv_dim) for i in range(n_layers) if has_ve(i, n_layers)}
    )

def build_layer_ve(value_embeds, layer_id, real_token_ids, predicted_tokens, extra_prefix_tokens=0):
    """为指定层构建 VE 张量"""
    # Real 分支: ve_real = value_table(real_token_ids)
    # Predicted 分支: ve_pred = torch.matmul(softmax(predicted_tokens), value_table.weight)
    # 返回 cat((ve_real, ve_pred), dim=1) 或带前缀的版本
```

---

## 2. 修改文件详情

### 2.1 `utils.py`

**改动位置 1**: `EBTModelArgs` dataclass (第 34-35 行)

```python
# 添加的字段:
vocab_size: int = 32768  # VE 需要用到
use_ve: bool = False     # 是否启用 Value Embedding
```

**改动位置 2**: `setup_ebt` 函数 (第 452-455 行)

```python
# 添加参数获取和传递:
vocab_size = getattr(hparams, 'vocab_size', 32768)
use_ve = getattr(hparams, 'use_ve', False)

# 传递给 EBTModelArgs:
EBTModelArgs(
    ...,
    vocab_size=vocab_size,
    use_ve=use_ve,
)
```

---

### 2.2 `ar_ebt_default.py`

**改动位置 1**: 文件顶部 - VE 模块导入 (第 13-17 行)

```python
try:
    from ve import build_layer_ve, build_value_embeds, has_ve
except ImportError:
    has_ve = None
    build_value_embeds = None
    build_layer_ve = None
```

**改动位置 2**: `Attention.__init__` 方法 - 添加 layer_id 参数和 VE gate (第 153 行起)

```python
def __init__(self, layer_id: int, args: EBTModelArgs):
    # ... 原有代码 ...

    # 新增 VE 相关:
    self.layer_id = layer_id
    self.use_ve = args.use_ve and has_ve is not None and has_ve(layer_id, args.n_layers)
    if self.use_ve:
        self.ve_gate_channels = min(32, args.dim)
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_local_kv_heads, bias=False)
        nn.init.zeros_(self.ve_gate.weight)  # 零初始化
    else:
        self.ve_gate = None
```

**改动位置 3**: `Attention.forward` 方法 - 添加 ve 参数和注入逻辑 (第 242 行起)

```python
def forward(
    self,
    x: torch.Tensor,
    start_pos: int,
    freqs_cis: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    ve: Optional[torch.Tensor] = None,  # 新增参数
):
    # ... 原有代码直到 xv view 之后 ...

    # 新增 VE 注入:
    if ve is not None and self.ve_gate is not None:
        ve = ve.view(bsz, full_seqlen, self.n_local_kv_heads, self.head_dim)
        gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
        xv = xv + gate.unsqueeze(-1) * ve
```

**改动位置 4**: `TransformerBlock.__init__` - 传递 layer_id (第 426 行起)

```python
def __init__(self, layer_id: int, args: EBTModelArgs):
    # 修改:
    self.attention = Attention(layer_id, args)  # 原来是 Attention(args)
```

**改动位置 5**: `TransformerBlock.forward` - 添加 ve 参数 (第 461 行起)

```python
def forward(
    self,
    x: torch.Tensor,
    start_pos: int,
    freqs_cis: torch.Tensor,
    mask: Optional[torch.Tensor],
    ve: Optional[torch.Tensor] = None,  # 新增参数
):
    # 修改 attention 调用:
    h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask, ve=ve)
```

**改动位置 6**: `EBTDefault.__init__` - 构建 value_embeds (第 538-548 行)

```python
# 新增 VE 相关:
self.kv_dim = params.n_kv_heads * (params.dim // params.n_heads)
self.use_ve = params.use_ve and build_value_embeds is not None
if self.use_ve:
    self.value_embeds = build_value_embeds(params.n_layers, params.vocab_size, self.kv_dim)
    # Xavier 初始化
    for ve in self.value_embeds.values():
        nn.init.xavier_uniform_(ve.weight)
else:
    self.value_embeds = None
```

**改动位置 7**: `EBTDefault.forward` - 添加参数和 VE 构建 (第 524 行起)

```python
def forward(
    self,
    tokens: torch.Tensor,
    start_pos: int = 0,
    mcmc_step: int = 0,
    real_token_ids: torch.Tensor = None,      # 新增
    predicted_tokens: torch.Tensor = None,     # 新增
):
    # 在 transformer 层循环中:
    for i, layer in enumerate(self.layers):
        ve = None
        if self.use_ve and self.value_embeds is not None:
            if real_token_ids is not None and predicted_tokens is not None:
                ve = build_layer_ve(self.value_embeds, i, real_token_ids, predicted_tokens)
        h = layer(h, start_pos, freqs_cis, mask, ve=ve)
```

---

### 2.3 `ar_ebt_time_embed.py`

与 `ar_ebt_default.py` 相同的修改，但 `build_layer_ve` 调用使用 `extra_prefix_tokens=1`:

```python
ve = build_layer_ve(self.value_embeds, i, real_token_ids, predicted_tokens, extra_prefix_tokens=1)
```

---

### 2.4 `ar_ebt_adaln.py`

与 `ar_ebt_default.py` 相同的修改:
- 添加 VE 模块导入
- `Attention.__init__` 添加 layer_id 和 ve_gate
- `Attention.forward` 添加 ve 参数和注入逻辑
- `AdaLNTransformerBlock.__init__` 传递 layer_id
- `AdaLNTransformerBlock.forward` 添加 ve 参数
- `EBTAdaLN.__init__` 构建 value_embeds
- `EBTAdaLN.forward` 添加参数和 VE 构建

---

### 2.5 `modeling_ebt.py`

**改动位置 1**: `_mcmc_step_excluded` 方法签名 (第 71-73 行)

```python
def _mcmc_step_excluded(self, predicted_tokens, real_embeddings_input, mcmc_step, i, num_mcmc_steps,
                  langevin_dynamics_noise_std, alpha, learning,
                  real_token_ids=None):  # 新增参数
```

**改动位置 2**: `_mcmc_step_excluded` 内 transformer 调用 (第 104-107 行)

```python
energy_preds = self.transformer(
    combined_embeddings,
    start_pos=start_pos,
    mcmc_step=mcmc_step,
    real_token_ids=real_token_ids,        # 新增
    predicted_tokens=predicted_tokens,     # 新增
)
```

**改动位置 3**: `forward` 方法中 `_mcmc_step_excluded` 调用 (第 192-195 行)

```python
predicted_tokens, energy_preds, predicted_tokens_for_loss = self._mcmc_step_excluded(
    predicted_tokens, real_embeddings_input, mcmc_step, i, len(mcmc_steps),
    langevin_dynamics_noise_std, alpha, learning,
    real_token_ids=x  # 新增: 传递真实 token IDs
)
```

**改动位置 4**: `calculate_contrastive_loss` 中 transformer 调用 (第 342-345 行)

```python
neg_energies = self.transformer(
    neg_embeddings,
    start_pos=start_pos,
    mcmc_step=mcmc_step,
    real_token_ids=input_ids,             # 新增
    predicted_tokens=true_pred_tokens,     # 新增
)
```

**改动位置 5**: `_run_ebt_inference_steps` 方法 (第 489 行起)

```python
def _run_ebt_inference_steps(
    self, embeddings, start_pos,
    ...
    real_token_ids=None,  # 新增参数
):
    # 内部函数 do_mcmc_step 和 get_energy 也相应修改
```

---

### 2.6 `train.py`

**改动位置**: EBT 参数部分 (第 481 行)

```python
parser.add_argument("--use_ve", help="启用 Value Embedding (VE)，为交替层添加可学习的值嵌入", action="store_true", default=False)
```

---

## 3. 参数传递链路

```
--use_ve (命令行)
    ↓
train.py: args.use_ve
    ↓
ModelTrainer: hparams.use_ve, hparams.vocab_size
    ↓
setup_ebt(): EBTModelArgs(use_ve=..., vocab_size=...)
    ↓
EBTDefault/EBTTimeEmbed/EBTAdaLN: self.value_embeds (如果 use_ve=True)
    ↓
Attention: self.ve_gate (如果 use_ve 且 has_ve(layer_id))
    ↓
forward: ve = build_layer_ve(...) → xv = xv + gate * ve
```

---

## 4. 兼容性保证

1. `use_ve` 默认 `False`，不启用时不创建任何 VE 模块
2. 使用 `getattr(hparams, 'use_ve', False)` 确保旧配置兼容
3. 使用 `try-except` 导入 VE 模块，import 失败时优雅降级
4. 所有 VE 相关逻辑都有 `ve is not None` 检查
5. forward 方法新增参数都有默认值 `None`

---

## 5. 使用方法

### 不启用 VE (默认行为)
```bash
python train.py --run_name test_no_ve --model_size "4xs" --max_steps 100
```

### 启用 VE
```bash
python train.py --run_name test_with_ve --model_size "4xs" --max_steps 100 --use_ve
```

---

## 6. 改动文件清单

| 文件 | 操作 | 主要改动 |
|------|------|----------|
| `ve.py` | 新建 | VE 核心函数 (has_ve, build_value_embeds, build_layer_ve) |
| `utils.py` | 修改 | EBTModelArgs 添加 vocab_size, use_ve 字段 |
| `ar_ebt_default.py` | 修改 | Attention/TransformerBlock/EBTDefault 添加 VE 支持 |
| `ar_ebt_time_embed.py` | 修改 | 同上，extra_prefix_tokens=1 |
| `ar_ebt_adaln.py` | 修改 | 同上 |
| `modeling_ebt.py` | 修改 | transformer 调用传递 real_token_ids, predicted_tokens |
| `train.py` | 修改 | 添加 --use_ve 命令行参数 |

---

*文档生成时间: 2026-04-03*
