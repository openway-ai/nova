# EBT Sudoku RL (GRPO) 训练管道

基于 Energy-Based Transformer 的 Group Relative Policy Optimization，用于 9×9 数独后训练。

> 详细实现原理、Reward 设计、调参指南见 [IMPLEMENTATION_DETAIL.md](./IMPLEMENTATION_DETAIL.md)

## 快速开始

```bash
# 1. 确保 SFT checkpoint 和数据就绪
export SFT_CKPT="/path/to/sft_checkpoint.ckpt"
export SUDOKU_DATA_DIR_V2="/path/to/sudoku_cache_v2"

# 2. 启动训练（推荐 energy_reinforce）
bash openebm/elm/runs/run_ebt_sudoku_rl.sh
```

默认配置：6-8 GPU，`energy_gspo` loss，lr=1e-6，8 generations/prompt。

## 核心架构

```
训练循环 (ebm_grpo_trainer.py)
├── Phase 1: Rollout（无梯度）
│   ├── 每个 prompt 生成 8 个 completion (rollout.py)
│   ├── 计算多组件 reward (rewards.py)
│   └── 计算 group-relative advantages
└── Phase 2: Policy Update（有梯度）
    ├── 计算 sequence energy (logprobs.py)
    └── PPO-clipped loss + optimizer step
```

## 三种 Loss 变体

| 变体 | `--rl_loss_type` | 原理 | 推荐场景 |
|---|---|---|---|
| **Energy-GSPO** | `energy_gspo` | ratio = exp(-(E_θ - E_old)/\|y\|)，PPO clip | 默认，但梯度信号弱 |
| **Energy-REINFORCE** | `energy_reinforce` | loss = advantages × E_θ，无 ratio | **推荐**，梯度强 100x |
| Token Logprobs | `token_logprobs` | 传统 MCMC logprobs ratio | 研究用，Hessian 不稳定 |

## 文件结构

| 文件 | 职责 |
|---|---|
| `train_rl_sudoku.py` | 入口：加载 SFT checkpoint，构建 Trainer |
| `ebm_grpo_trainer.py` | Lightning Module：训练循环、loss、optimizer |
| `ebm_grpo_config.py` | 所有超参数的 dataclass |
| `logprobs.py` | `compute_sequence_energy()`：一阶能量计算 |
| `rollout.py` | 自回归生成 + stop token 检测 |
| `rewards.py` | 多组件 Sudoku reward（总分 0-3） |
| `sudoku_dataset_rl.py` | RL prompt 数据集 + collate |
| `../runs/run_ebt_sudoku_rl.sh` | 多 GPU 启动脚本 |

## Reward 组件（总分 0.0 - 3.0）

| 组件 | 分值 | 含义 |
|---|---|---|
| Format | 0.5 | 输出能否 parse 为 9×9 网格 |
| Clue Preservation | 0.5 | 给定数字是否保持不变 |
| Blank Accuracy | 1.5 | 空格填对比例 × 1.5 |
| Full Solve | 0.5 | 完全正确（81 格全对） |

## 关键超参数

```bash
--rl_loss_type energy_reinforce  # 推荐
--learning_rate 1e-4             # energy 变体需要较高 lr
--max_grad_per_param 0.5         # 放宽 per-param clip
--num_generations 8              # completions per prompt
--max_completion_length 180      # 9x9 grid ≈ 160 tokens
--temperature 0.9
--epsilon 0.2                    # PPO clip range (仅 gspo)
```

## 监控指标

训练日志每 `log_interval` 步输出：
```
[GRPO] step=10 | loss=0.0312 | reward=1.85±0.42 | fmt=0.50 clue=0.48 blank_acc=0.87 solve=0.00 | comp_len=165 | clip=0.15 degen=0.12
[GRPO] step=10 grad_norm=3.21e-03 max_param_grad=1.87e-04 nan_params=0
```

健康训练的标志：
- `reward` 随 step 上升
- `grad_norm > 0`，`nan_params = 0`
- `degen < 0.5`（退化组比例低）
- `comp_len` 在 80-180 之间（非顶到 max）
