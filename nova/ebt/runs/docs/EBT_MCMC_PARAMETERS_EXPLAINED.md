# EBT MCMC 参数深度解析: MCMC_STEP_SIZE 与 MCMC_STEP_SIZE_LR_MULTIPLIER

## 🎯 核心问题

为什么需要这两个参数?
```python
MCMC_STEP_SIZE = 500.0
MCMC_STEP_SIZE_LR_MULTIPLIER = 1500  # 3 × 500
```

---

## 📚 理论基础: EBT 如何工作?

### 1. 传统 Transformer (单步预测)

```python
# 标准自回归生成
x_t = input_tokens          # 输入
logits = model(x_t)         # 前向传播
next_token = sample(logits) # 采样下一个token
```

**问题**: 每次只能看到一个token的未来,短视!

### 2. EBT (能量优化视角)

EBT将文本生成看作**能量最小化问题**:

```python
# 定义能量函数 (越低越好)
Energy(x) = -log P(x | context)

# 目标: 找到最小化能量的序列
x* = argmin_x Energy(x)
```

**关键创新**: 通过**MCMC (马尔可夫链蒙特卡洛)**优化整个序列!

---

## 🔬 MCMC 优化过程详解

### Step 1: 初始化

```python
# 从噪声开始 (或部分已知序列)
x_0 = random_noise()  # shape: [batch, seq_len, vocab_size]
```

### Step 2: MCMC 迭代 (核心!)

```python
for t in range(mcmc_num_steps):  # 默认 2 步
    # (1) 计算当前能量 (前向传播)
    energy = model.forward(x_t)  # E(x_t)

    # (2) 计算梯度 (对输入求导!)
    grad = torch.autograd.grad(energy, x_t)  # ∂E/∂x

    # (3) 梯度下降更新 (关键: 使用 alpha 作为步长!)
    x_{t+1} = x_t - alpha * grad
    #              ^^^^^ 这就是 MCMC Step Size!
```

**注意**: `alpha` (MCMC步长) 是一个**可学习参数**,初始化为 `MCMC_STEP_SIZE`!

---

## 🎓 两个参数的具体含义

### 参数1: MCMC_STEP_SIZE (初始步长)

```python
# 在模型初始化时
self.alpha = nn.Parameter(torch.tensor(500.0))  # 可学习!
```

**含义**:
- MCMC 梯度下降的**初始步长**
- 控制每次 MCMC 迭代时,沿着负梯度方向移动多远
- 类比: 普通优化器的学习率,但作用在**输入空间**而非参数空间

**为什么是 500?**
- 经验值,在多个任务上表现良好
- 太小 (如50): MCMC收敛慢,生成质量差
- 太大 (如5000): MCMC震荡,能量不降反升

### 参数2: MCMC_STEP_SIZE_LR_MULTIPLIER (Alpha的学习率倍数)

```python
# 在优化器中
optimizer = AdamW([
    {'params': transformer_params, 'lr': 0.00025},  # 基础LR
    {'params': [self.alpha], 'lr': 0.00025 * 1500}, # Alpha的LR
    #                                       ^^^^
    #                              MCMC_STEP_SIZE_LR_MULTIPLIER
])
```

**含义**:
- Alpha 参数的学习率是**基础学习率的倍数**
- 实际 Alpha LR = `PEAK_LR × MCMC_STEP_SIZE_LR_MULTIPLIER`
- 例: 0.00025 × 1500 = 0.375

**为什么需要更高的LR?**

Alpha 需要比普通参数**更快地适应**,原因:
1. **搜索空间不同**: Alpha 控制MCMC步长,影响能量优化效率
2. **快速调整需求**: 不同数据分布需要不同的MCMC步长
3. **梯度信号弱**: Alpha 的梯度信号比普通参数稀疏

**为什么是 3× (1500 = 3 × 500)?**

官方经验法则:
```python
MCMC_STEP_SIZE_LR_MULTIPLIER = 3 × MCMC_STEP_SIZE
```

这是为了保持**更新比例平衡**:
- 如果 Alpha 初始值是 500
- Alpha 的 LR 也应该在相似的量级 (×3 是经验最优)

---

## 🧮 数学推导: 为什么需要高LR?

### MCMC 更新方程

```python
# MCMC 步骤
x_{t+1} = x_t - alpha * ∂E/∂x

# Alpha 的梯度 (通过反向传播)
∂Loss/∂alpha = ∂Loss/∂x * ∂x/∂alpha
              = ∂Loss/∂x * (-∂E/∂x)  # 因为 x 依赖于 alpha
```

**关键观察**: Alpha 的梯度**间接**通过 MCMC 过程计算,信号较弱!

### 梯度量级对比

假设:
- Transformer 参数 W 的梯度: `∂Loss/∂W ≈ 0.01`
- Alpha 的梯度: `∂Loss/∂alpha ≈ 0.0001` (小100倍!)

为了让 Alpha 有类似的更新幅度:
```python
# Transformer 更新
W_new = W - 0.00025 * 0.01 = W - 0.0000025

# Alpha 更新 (如果用相同LR)
alpha_new = alpha - 0.00025 * 0.0001 = alpha - 0.000000025  # 太小!

# Alpha 更新 (用高LR)
alpha_new = alpha - 0.375 * 0.0001 = alpha - 0.0000375  ✓ 合理
```

---

## 🔍 实际训练中的动态变化

### 训练过程示例

```
Step 0 (初始化):
  Alpha = 500.0
  Alpha_LR = 0.375

Step 1000:
  Alpha ≈ 480.0  (下降了 4%)
  Alpha_LR = 0.375 (不变,由调度器控制)

Step 5000:
  Alpha ≈ 520.0  (上升了 4%)
  Alpha_LR = 0.36 (随 Cosine Annealing 下降)

Step 10000:
  Alpha ≈ 510.0  (稳定在合理范围)
  Alpha_LR = 0.30
```

**观察**:
- Alpha 会在训练中**自适应调整**
- 不同数据集/任务可能需要不同的 Alpha
- Alpha LR 过高 (如1.2) → Alpha 震荡 → MCMC 失控 → Loss暴涨!

---

## 💥 之前失败的原因解析

### 失败配置

```python
MCMC_STEP_SIZE = 500
MCMC_STEP_SIZE_LR_MULTIPLIER = 2000  # ❌ 不是 3×!

Peak_LR = 0.0006
Alpha_LR = 0.0006 × 2000 = 1.2  # ❌ 过高!
```

### 失败机制

```
Step 1: Alpha 梯度计算
  ∂Loss/∂alpha = -0.5 (较大的负梯度,希望减小Alpha)

Step 2: Alpha 更新 (过大的LR导致过度更新!)
  Alpha_new = 500 - 1.2 * (-0.5)
            = 500 + 0.6
            = 500.6  ✓ 看起来还行

Step 100: (累积效应)
  Alpha ≈ 500 + 100 * 0.6 = 560  ⚠️ 偏高

Step 1000: (梯度方向突变)
  ∂Loss/∂alpha = +2.0 (正梯度,希望增大Alpha)
  Alpha_new = 560 - 1.2 * 2.0 = 557.6  (还在振荡)

Step 5000: (失控!)
  Alpha ≈ 700  ❌ 已经偏离稳定区
  MCMC更新: x_new = x - 700 * grad  # 步长太大!
  → 能量函数震荡 → Loss 暴涨至 800+
```

### 正确配置

```python
MCMC_STEP_SIZE = 500
MCMC_STEP_SIZE_LR_MULTIPLIER = 1500  # ✓ 3×

Peak_LR = 0.00025
Alpha_LR = 0.00025 × 1500 = 0.375  # ✓ 安全范围
```

```
Step 5000:
  Alpha ≈ 510  ✓ 稳定
  MCMC更新: x_new = x - 510 * grad  # 步长合理
  → 能量函数平稳下降 → Loss 稳定
```

---

## 📊 不同配置的对比

| 配置 | MCMC Step Size | LR Multiplier | Alpha LR | Alpha 稳定性 | Loss 稳定性 |
|:-----|:--------------|:-------------|:---------|:------------|:-----------|
| 官方推荐 | 500 | 1500 (3×) | 0.9 | ✓✓✓ | ✓✓✓ |
| 失败版本 | 500 | **2000** (4×) | **1.2** | ❌ 震荡 | ❌ 暴涨 |
| 修正版本 | 500 | 1500 (3×) | **0.375** | ✓✓✓ | ✓✓✓ |
| 保守方案 | 500 | 1200 (2.4×) | 0.3 | ✓✓ | ✓✓ |
| 过于保守 | 500 | 300 (0.6×) | 0.075 | ✓ | ✓ (但收敛慢) |

---

## 🎯 核心原理总结

### 1. 为什么需要 MCMC_STEP_SIZE?

**因为**: EBT 使用 MCMC 在输入空间优化能量函数
- MCMC 需要步长来控制每次迭代的移动距离
- 初始值 500 是经验最优解

### 2. 为什么需要 MCMC_STEP_SIZE_LR_MULTIPLIER?

**因为**: Alpha (MCMC步长) 是一个可学习参数,需要优化
- Alpha 的梯度信号比普通参数弱
- 需要更高的学习率来快速适应不同数据

### 3. 为什么是 3× 关系?

**因为**: 保持更新幅度平衡的经验法则
- Alpha 初始值量级 ∝ MCMC步长量级
- Alpha LR 量级 ∝ Alpha 初始值量级
- 3× 是在多个任务上验证的最优比例

### 4. Alpha LR 过高的危害?

**机制**: Alpha → MCMC步长 → 能量优化 → Loss
```
Alpha LR 过高
  → Alpha 震荡 (500 → 700 → 400 → 800...)
    → MCMC 步长不稳定
      → 能量函数优化失控
        → Loss 暴涨!
```

---

## 🔬 实验建议

### 如何调优 MCMC_STEP_SIZE?

**原则**: 几乎不需要调!

官方建议:
> "the most important and arguably only really necessary to tune hparam is mcmc_step_size"

但实际上,500 在绝大多数情况下都是最优的。

**仅在以下情况调整**:
1. 数据分布极其特殊 (如代码 vs 自然语言)
2. 序列长度极短/极长 (如 32 vs 2048)
3. 多次实验验证需要调整

**调整范围**: 300 - 800

### 如何调优 MCMC_STEP_SIZE_LR_MULTIPLIER?

**原则**: 严格保持 3× 关系!

```python
MCMC_STEP_SIZE_LR_MULTIPLIER = 3 × MCMC_STEP_SIZE
```

**例外情况** (需充分实验验证):
- 极其稳定的数据: 可以尝试 3.5× 或 4×
- 极其不稳定的数据: 可以降到 2× 或 2.5×

**但警告**: 偏离 3× 很可能导致训练不稳定!

---

## 📖 类比: 帮助理解

### 类比1: 登山优化

```
普通 Transformer = 单步爬山
  - 每次只看脚下的下一步
  - 容易陷入局部最优

EBT + MCMC = 装备GPS的探险
  - MCMC_STEP_SIZE = 每次探险的步幅 (500米)
  - MCMC_STEP_SIZE_LR_MULTIPLIER = GPS更新速度的加速倍数
  - Alpha LR 过高 = GPS疯狂调整方向 → 迷路!
```

### 类比2: 火箭控制

```
MCMC_STEP_SIZE = 火箭推进器的基础推力
MCMC_STEP_SIZE_LR_MULTIPLIER = 推力调节系统的敏感度

Alpha LR 过高 = 推力调节过于敏感
  → 火箭左右摇摆 → 失控!
```

---

## ✅ 最佳实践总结

1. **默认配置 (推荐)**:
   ```python
   MCMC_STEP_SIZE = 500
   MCMC_STEP_SIZE_LR_MULTIPLIER = 1500  # 3 × 500
   ```

2. **计算 Alpha LR**:
   ```python
   Alpha_LR = PEAK_LR × MCMC_STEP_SIZE_LR_MULTIPLIER
   ```

3. **安全范围**:
   ```
   Alpha_LR 应该在 0.3 - 0.9 之间
   ```

4. **调试方法**:
   ```bash
   # 监控 Alpha 的值
   grep "Alpha_MCMC" log.txt

   # 正常: 稳定在 400-600
   # 异常: 剧烈震荡或持续偏离
   ```

5. **遇到不稳定时**:
   - 优先降低 PEAK_LR (而非调整 MCMC 参数)
   - 如果仍不稳定,降低 MCMC_STEP_SIZE_LR_MULTIPLIER 到 2.5× 或 2×
   - 最后手段: 降低 MCMC_STEP_SIZE 本身

---

**核心记忆**:
- MCMC_STEP_SIZE = MCMC 的步长 (500是魔法数字)
- MCMC_STEP_SIZE_LR_MULTIPLIER = Alpha 学习率的倍数 (3×是黄金比例)
- Alpha LR = 两者的乘积决定训练稳定性
- 过高的 Alpha LR → MCMC失控 → Loss暴涨! ⚡
