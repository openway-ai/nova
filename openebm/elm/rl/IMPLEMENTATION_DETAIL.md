# EBT Sudoku RL 详细实现文档

> 简洁版见 [README.md](./README.md)

## 1. 算法原理

### 1.1 GRPO（Group Relative Policy Optimization）

对每个 prompt x，生成 N 个 completion {y₁, ..., yₙ}，计算 reward r(yᵢ)，然后用组内相对优势进行策略更新：

```
advantage_i = (r(yᵢ) - mean(r)) / std(r)
```

### 1.2 EBT 的 Energy-Based Policy Gradient

EBT 不像标准 LLM 那样直接输出 token 概率，而是通过能量函数定义分布：
```
p_θ(y|x) ∝ exp(-E_θ(x, y))
```

因此 log-likelihood ratio 有 closed-form：
```
log(p_θ / p_θ_old) = -(E_θ - E_θ_old)   （partition function在 group 内抵消）
```

### 1.3 三种 Loss 变体

#### Energy-GSPO（默认）
```python
ratio = exp(-(E_θ(x,y) - E_θ_old(x,y)) / |y|)
clipped_ratio = clamp(ratio, 1-ε, 1+ε)
loss = -min(ratio * advantage, clipped_ratio * advantage).mean()
```

**问题**：on-policy 时 E_θ == E_θ_old → ratio=1.0 → loss≈0，梯度极弱。

#### Energy-REINFORCE（推荐）
```python
loss = (advantages * E_θ(x,y)).mean()
```

直接优化：高 advantage 的 completion 应有低能量。无 ratio，梯度信号强。

#### Token Logprobs（研究用）
```python
logps = get_per_token_logps(model, ids, prompt_len, learning=True)
ratio = exp(current_logps - old_logps)
# 标准 PPO clip
```

需要 `create_graph=True` 通过 MCMC 链反传，Hessian 数值不稳定，已知产生 NaN。

## 2. compute_sequence_energy 实现

核心函数，计算已知 completion 的 sequence-level 能量（一阶，无 MCMC 迭代）：

```python
def compute_sequence_energy(model, input_ids, prompt_length):
    # input_ids: (B, S) = prompt + completion
    model_input = input_ids[:, :-1]       # (B, S-1)
    targets = input_ids[:, 1:]            # (B, S-1)

    real_embeddings = model.embeddings(model_input).detach()
    one_hot_targets = F.one_hot(targets, vocab_size).float()
    predicted_embeddings = model.vocab_to_embed(one_hot_targets)

    all_embeddings = cat([real_embeddings, predicted_embeddings], dim=1)  # (B, 2*(S-1), D)
    energy_preds = transformer(all_embeddings, ...)  # (B, S-1, 1) — 只返回 predicted half

    # 切到 completion 位置
    comp_energy = energy_preds[:, prompt_length-1:]
    return comp_energy.mean(dim=1)  # (B,)
```

关键设计：
- `real_embeddings.detach()` — 梯度只通过 predicted_embeddings 和 transformer 参数流动
- Transformer energy head 只输出 predicted half 的能量（不是完整 2*(S-1)）
- 一阶可微，无 autograd.grad，无 Hessian

## 3. Reward 设计

### 3.1 组件分解

| 组件 | 分值范围 | 计算方式 |
|---|---|---|
| **Format** | 0 或 0.5 | `parse_board_from_text(text)` 能否返回 9×9 grid |
| **Clue Preservation** | 0.0 - 0.5 | `preserved_clues / total_clues × 0.5` |
| **Blank Accuracy** | 0.0 - 1.5 | `correct_blanks / total_blanks × 1.5` |
| **Full Solve** | 0 或 0.5 | 81 格全部与 ground truth 一致 |

总分范围：[0.0, 3.0]

### 3.2 设计决策

**为什么没有 "Validity Bonus"（行列宫约束检查）？**
- 如果 clue_preservation=1.0 且 blank_accuracy=1.0 → full_solve=1.0 → validity 自动满足
- 单独的 validity bonus 会奖励"解了另一道题"的情况（改了 clue 但满足约束）

**为什么 Blank Accuracy 权重最高（1.5）？**
- 这是核心学习信号：模型需要学会填空
- Format 和 Clue Preservation 是前置条件，SFT 模型通常已经满足

### 3.3 Board 解析

支持两种格式（`sudoku_evaluator.py::parse_board_from_text`）：
1. 空格分隔的 9 行（每行 9 个数字）
2. 连续 81 个数字

## 4. 数据管道

### 4.1 Prompt 编码

使用 `nanochat.tokenizer.render_for_completion(conv)` 生成 prompt token IDs：
```
[BOS, <|user_start|>, ...puzzle_text..., <|user_end|>, <|assistant_start|>]
```

**关键**：必须用 `encode_special()` 编码 special tokens（单 token ID 32759-32763），不能用 `encode()`（会拆成多个字符 token）。

### 4.2 Stop Token 检测

Rollout 使用 `encode_special('<|assistant_end|>')` = 32763 和 `encode_special('<|user_start|>')` = 32760 作为停止信号。

### 4.3 数据增强

训练集使用 `augment_board()`：digit relabel + band/stack permutation + row/col-within-block permutation + 50% transpose。

## 5. 已知问题与修复历史

| 版本 | 问题 | 根因 | 修复 |
|---|---|---|---|
| v4-v8 | NaN loss, DDP hang | MCMC Hessian 数值不稳定 | 切换到 Energy-GSPO（一阶） |
| v9 | reward=0, loss=NaN | `tokenizer.encode()` 把 special token 当文本编码 | 改用 `render_for_completion` |
| v10 | 模型输出乱码英文 | `_orig_mod.` 前缀未 strip，权重未加载 | strip 两层前缀 |
| v3(0521) | energy=NaN | transformer 输出 (B,S-1,1) 被当作 (B,2*(S-1)) 切片 | 去掉"取后半"逻辑 |
| v4(0521) | loss≈0, ratio=1.0 | Energy-GSPO on-policy ratio 恒为 1 | 建议切换 energy_reinforce |

## 6. 调参指南

### 6.1 推荐配置

```bash
# 基础
--rl_loss_type energy_reinforce
--learning_rate 1e-4
--max_grad_per_param 0.5
--gradient_clip_val 1.0

# 生成
--num_generations 8
--max_completion_length 180
--temperature 0.9
--top_p 0.9

# 训练
--max_steps 1000
--warmup_steps 20
--val_check_interval 50
--float_precision "32-true"
```

### 6.2 关键调参维度

| 参数 | 影响 | 建议范围 |
|---|---|---|
| `learning_rate` | 学习速度 | 1e-5 ~ 1e-3（energy 变体需要高 lr） |
| `temperature` | 生成多样性 | 0.7-1.0（太低→退化，太高→乱码） |
| `num_generations` | advantage 估计质量 | 4-16（越多越稳但越慢） |
| `max_grad_per_param` | 梯度稳定性 | 0.1-1.0（0=禁用） |
| `epsilon` | PPO clip 范围（仅 gspo） | 0.1-0.3 |

### 6.3 监控健康指标

| 指标 | 健康范围 | 异常信号 |
|---|---|---|
| `reward_mean` | 随 step 上升 | 持续为 0 → prompt/生成问题 |
| `grad_norm` | 1e-4 ~ 1.0 | 0 → loss 无梯度；>10 → 爆炸 |
| `nan_params` | 0 | >0 → 数值不稳定 |
| `degenerate_rate` | <0.5 | 1.0 → 所有 completion reward 相同 |
| `comp_len` | 80-180 | 224(max) → stop token 未触发 |
| `ratio_mean` | 0.8-1.2 | 恒=1.0 → gspo 退化 |

## 7. Checkpoint 加载

SFT checkpoint 可能包含 `torch.compile` 产生的 `_orig_mod.` 前缀：
```python
# train_rl_sudoku.py::load_sft_model_and_tokenizer
for key, val in state_dict.items():
    clean_key = key
    if clean_key.startswith("model."):       # Lightning prefix
        clean_key = clean_key[6:]
    if clean_key.startswith("_orig_mod."):   # torch.compile prefix
        clean_key = clean_key[10:]
    model_state[clean_key] = val
```

加载后检查 `missing_keys` 和 `unexpected_keys` 必须为空。

## 8. 冻结参数

RL 阶段冻结以下参数（防止 MCMC 步长被破坏）：
- `model.alpha`（MCMC step size，SFT 校准值 ~500）
- `model.langevin_dynamics_noise_std`
