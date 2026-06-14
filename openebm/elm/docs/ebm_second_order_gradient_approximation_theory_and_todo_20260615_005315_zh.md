# EBM 二阶梯度近似实现的理论分层与实验 TODO

时间戳：2026-06-15 00:53:15 Asia/Hong_Kong  
修订：自包含版本，作为当前实现理解与后续实验判断依据。

除非另有说明，代码路径均以 `/mnt/shared-storage-user/puyuan/code/OpenEBM/` 为根。

## 0. 结论先行

如果下游采样器实际使用

$$
s_\theta(x)=-\nabla_x E_\theta(x)
$$

那么训练目标最应该对齐的是采样轨迹上这个向量场本身，而不只是能量值高低。把采样器实际访问的 relaxed token 状态分布记作 \(\mu\)，理想误差可写成：

$$
\mathcal D(\theta)
=
\mathbb E_{x\sim\mu}
\left[
\|\nabla_xE_\theta(x)-\nabla_xE_\*(x)\|^2
\right].
$$

按“对 \(\nabla_xE_\theta\) 的约束有多直接”排序，当前判断是：

$$
\text{second\_order}
\succ
\text{first\_order\_cd + trajectory-local descent CD}
\succ
\text{first\_order\_cd}
\approx
\text{proposal\_aware\_nce + relaxed\_cd}
\succ
\text{first\_order\_nce / pure proposal NCE}.
$$

最关键的代码理解：

- 当前 `second_order` 是唯一已有的 pathwise-field 训练路径：loss 反传穿过每一步 MCMC 的 \(\nabla_xE_\theta\)。
- 当前 `first_order_cd` 是 graph-safe 的 endpoint relaxed ranking：跑 MCMC 到终点、detach、重算正负 energy，优化 \(\operatorname{softplus}(E^+-E^-)\)。
- 当前 `first_order_cd` 不是“小步 raw CD / score-matching leading term”实现，因为它只使用最终 endpoint，且不是原始相邻小步能量差。
- NCE / proposal-aware NCE 主要改善负样本覆盖和能量校准，不会自动恢复 \(\nabla_xE_\theta\) 的 field 监督阶数。
- 本次新增的最高效可行选项是 **trajectory-local descent CD**：不回传 sampler 图，复用已保存的 detached MCMC 轨迹点，重算相邻状态能量并最小化 \(E(z_{k+s})-E(z_k)\)。它提升的是“轨迹覆盖 + 采样方向能量斜率”近似，不等同于 data-started raw CD 的严格 score-matching Taylor 项。

## 1. 本文的理论标尺

### 1.1 二阶 pathwise-field 目标

OpenEBM 的 MCMC 更新形式可抽象为：

$$
z_{k+1}
=
z_k-\alpha_k\nabla_z E_\theta(c,z_k)+\epsilon_k,
$$

其中 \(z_k\) 是 relaxed token state，\(\epsilon_k\) 是可选 Langevin noise。若训练 loss 反传穿过 \(z_{k+1}\)，则有：

$$
\frac{\partial z_{k+1}}{\partial\theta}
=
\frac{\partial z_k}{\partial\theta}
-\alpha_k
\left(
\nabla_\theta\nabla_zE_\theta(c,z_k)
+
\nabla_z^2E_\theta(c,z_k)
\frac{\partial z_k}{\partial\theta}
\right).
$$

这里出现了 \(\nabla_\theta\nabla_zE_\theta\)，也就是“模型参数如何改变采样器实际使用的 score field”。这类目标最接近下游采样算子，代价是二阶图和长激活图。

### 1.2 短步 raw CD 为什么有 field 信息

考虑从某个状态 \(x\) 出发的一步小步长 Langevin：

$$
\tilde x
=
x-\frac{\varepsilon^2}{2}\nabla_xE_\theta(x)+\varepsilon\xi,
\quad
\xi\sim\mathcal N(0,I).
$$

令 \(\Delta=\tilde x-x\)，\(H=\nabla_x^2E_\theta(x)\)。Taylor 展开：

$$
E_\theta(\tilde x)-E_\theta(x)
=
\nabla_xE_\theta(x)^\top\Delta
+
\frac12\Delta^\top H\Delta
+
O(\|\Delta\|^3).
$$

代入 \(\Delta\) 并对噪声取期望：

$$
\mathbb E_\xi
\left[
E_\theta(x)-E_\theta(\tilde x)
\right]
=
\frac{\varepsilon^2}{2}
\left(
\|\nabla_xE_\theta(x)\|^2
-
\mathrm{tr}\nabla_x^2E_\theta(x)
\right)
+
O(\varepsilon^3).
$$

所以，**原始相邻小步能量差** 虽然训练时只需要一阶参数梯度，但在小步长条件下会隐式约束 score norm 和 Hessian trace。这是“一阶训练里最接近 field shaping”的通道。

该结论成立需要两个前提：

- 负样本是相邻小步状态，\(\tilde x-x=O(\varepsilon)\)。
- loss 保留 raw energy difference 的局部信号；如果只看远距离 endpoint 或强饱和 ranking，这个 Taylor 通道会显著变弱。

### 1.3 Endpoint ranking 的局限

如果 loss 只比较数据正样本 \(x^+\) 和 MCMC 终点 \(z_K\)：

$$
L=\operatorname{softplus}(E_\theta(x^+)-E_\theta(z_K)),
$$

它主要约束“正样本所在 basin 的能量低于终点所在 basin”。这个约束有价值，但它是 **level-set / endpoint ranking**，不直接告诉模型在 \(z_0,\dots,z_K\) 中间各点的 \(\nabla_xE_\theta\) 应该指向哪里、大小应是多少。

因此判断一阶近似优劣时，我采用三个标准：

| 标准 | 问题 | 越好意味着 |
|---|---|---|
| 阶数 | loss 是否显式或隐式含 \(\nabla_xE_\theta\) 信息 | 更像 field objective |
| 覆盖 | 约束覆盖 data 邻域、endpoint，还是整条轨迹 | 更贴采样器实际路径 |
| 饱和 | 能量差拉开后训练信号是否关闭 | 更稳定地继续塑形 |

## 2. 当前实现对照

| 实现 | 关键代码 | 当前目标/路径 | 几何对象 | 对二阶 field 的近似 |
|---|---|---|---|---|
| `second_order` | `openebm/elm/modeling_ebt.py:140-150` 中 `create_graph_for_mcmc=True`；CE 聚合在 `:318-342`、`:396-401` | loss 反传穿过每步 \(\nabla_zE_\theta\)，包含 \(\nabla_\theta\nabla_zE_\theta\) | pathwise-field，覆盖整条 MCMC 轨迹 | 最高；小模型 teacher / gold baseline |
| `first_order_debug` | mode 说明在 `openebm/elm/train.py:567-579`；FSDP2 warning 在 `openebm/elm/train_engines/fsdp2.py:50-55` | 只关闭 sampler 的 `create_graph`；CE 对模型主参数信号不完整 | 诊断路径 | 不作为训练近似方案 |
| `first_order_cd` | graph-safe detach 在 `openebm/elm/modeling_ebt.py:239-244`；combined 重算在 `:536-552`；CE ranking 在 `:583-604` | \(L=\mathrm{CE}([-E^+,-E^-],0)=\mathrm{softplus}(E^+-E^-)\) | endpoint relaxed level-set/ranking | 中；当前最实用一阶 baseline，但不是 local raw CD |
| `first_order_cd + first_order_local_cd_coeff>0` | loss 接入在 `openebm/elm/modeling_ebt.py:386-395`；局部能量重算在 `:624-694`；参数在 `openebm/elm/train.py:610-630` | \(L=L_{\mathrm{endpoint}}+\lambda\,\mathbb E[E(z_{k+s})-E(z_k)]\) | endpoint ranking + 轨迹局部能量斜率 | 中偏高；覆盖 MCMC 中间状态，但仍不含 \(\nabla_\theta\nabla_zE\) |
| `first_order_nce` | 同 `first_order_cd` 重算路径；NCE loss 在 `openebm/elm/modeling_ebt.py:587-594` | \(L=\mathrm{softplus}(E^+)+\mathrm{softplus}(-E^-)\) | endpoint 绝对能量校准 | 中偏低；校准能量零点，不增加 field 阶数 |
| `proposal_aware_nce + uniform` | 无 MCMC 快路径在 `openebm/elm/modeling_ebt.py:257-297`；uniform proposal 在 `:670-703` | 均匀离散负样本 + log-q / offset 修正 | density-ratio / 分类边界 | 低；覆盖广但离 MCMC 轨迹远 |
| `proposal_aware_nce + mcmc_final` | endpoint proposal 在 `openebm/elm/modeling_ebt.py:705-737`；NCE logit 在 `:822-831` | 从 final logits 采离散负样本，用近似 logq 修正 | endpoint proposal-aware classification | 中偏低；改善覆盖/校准，不改变零阶本质 |
| `proposal_aware_nce + mcmc_final + relaxed_cd` | relaxed endpoint 在 `openebm/elm/modeling_ebt.py:774-820`；loss 在 `:849-859`；参数在 `openebm/elm/train.py:641-652` | NCE + \(\mathrm{softplus}(E^+-E_{\mathrm{relaxed}}+m)\) | endpoint level-set + density-ratio 校准 | 中；比纯 NCE 更贴 MCMC endpoint，但仍不是 field loss |

## 3. 关键代码理解

### 3.1 `second_order` 是唯一已有 pathwise-field 约束

`openebm/elm/modeling_ebt.py:142-150`：

```python
mcmc_gradient_mode = getattr(self.hparams, "mcmc_gradient_mode", "second_order")
create_graph_for_mcmc = learning and mcmc_gradient_mode == "second_order" ...
predicted_tokens_grad = torch.autograd.grad(
    [energy_f32.sum()],
    [predicted_tokens],
    create_graph=create_graph_for_mcmc,
)[0]
```

判断：

- `second_order` 会让后续 loss 的 backward 穿过 `predicted_tokens_grad`。
- 这条路径包含 \(\nabla_\theta\nabla_zE_\theta\)，理论上最接近采样器实际使用的 score field。
- FSDP2 对该路径有明确高风险 warning，见 `openebm/elm/train_engines/fsdp2.py:44-49`。
- ZeRO-3 当前只允许一阶 surrogate，见 `openebm/elm/train_engines/deepspeed_zero.py:465-477`。

### 3.2 `first_order_cd` 是 endpoint relaxed ranking

`openebm/elm/modeling_ebt.py:239-244`：

```python
if getattr(self.hparams, "mcmc_gradient_mode", "second_order") in {"first_order_cd", "first_order_nce", "proposal_aware_nce"}:
    predicted_energies.append(energy_preds.detach())
    predicted_distributions.append(predicted_tokens_for_loss.detach())
```

`openebm/elm/modeling_ebt.py:370-378`：

```python
contrastive_loss, surrogate_metrics = self.calculate_contrastive_loss(
    predicted_energies,
    input_ids,
    next_token_indices,
    fake_pred_tokens=predicted_distributions[-1].detach(),
    recompute_fake_energy=True,
    combine_recomputed_energies=mcmc_gradient_mode in {"first_order_cd", "first_order_nce"},
)
```

`openebm/elm/modeling_ebt.py:582-603`：

```python
energy_stack = torch.cat([real_energies, fake_energies], dim=1)
contrastive_loss = F.cross_entropy(-1 * energy_stack, energy_targets, ignore_index=-100)
```

等价：

$$
L_{\mathrm{fo\_cdv2}}
=
\operatorname{softplus}(E^+-E^-).
$$

判断：

- 它保留了“fake 来自当前 EBM MCMC 终点”这一点，因此比完全外部 proposal 更贴近当前 landscape。
- 它不让训练梯度穿过 MCMC 更新，因此截断了二阶 field credit assignment。
- 它只比较正样本和终点 fake 的能量，不重算相邻小步能量差，因此不是 local raw CD。
- softplus 是软饱和：当 \(E^+\ll E^-\) 后，梯度权重 \(\sigma(E^+-E^-)\) 会变小。

### 3.3 `proposal_aware_nce` 是覆盖/校准项，不是 field 项

`openebm/elm/modeling_ebt.py:822-831`：

```python
r_pos = -real_energies.float() - logq_pos.float() - logz_offset - log_k
r_neg = -fake_energies.float() - logq_neg.float() - logz_offset - log_k
pos_loss = F.softplus(-r_pos)...
neg_loss = F.softplus(r_neg)...
```

判断：

- 这显式加入 \(\log q\)、\(\log K\)、offset，比简化 `first_order_nce` 更接近 density-ratio classification。
- 但 loss 仍然只依赖能量值 \(E\)，不包含 \(\nabla_xE\) 或 \(\nabla_\theta\nabla_xE\)。
- 因此它适合做 calibration / coverage auxiliary，不应被当作二阶 field loss 的直接替代。

## 4. Trajectory-local raw CD 的合理性判断

### 4.1 严格 data-started raw CD 与当前 OpenEBM 轨迹不同

第 1.2 节的 Taylor 结论严格对应的是 data-started CD：从正样本或数据邻域状态 \(x^+\) 出发，取一个小步负样本 \(\tilde x=x^++O(\varepsilon)\)，然后最小化 \(E(x^+)-E(\tilde x)\)。这时 raw energy difference 的期望里会出现 \(\|\nabla_xE\|^2-\mathrm{tr}H\)。

当前 OpenEBM 的 MCMC 轨迹不是这种 data-started CD。代码事实：

- 初始 relaxed token 来自 `corrupt_embeddings`，`random_noise`/`zeros` 分支在 `openebm/elm/modeling_ebt.py:464-469`，不是从 one-hot 正样本附近初始化。
- `first_order_cd`/`first_order_nce`/`proposal_aware_nce` 在 `openebm/elm/modeling_ebt.py:239-244` 保存的是每步更新后的 detached `predicted_distributions`。
- 当前 endpoint `first_order_cd` 只使用最后一个 detached 状态，loss 接入在 `openebm/elm/modeling_ebt.py:364-386`。

因此，如果把 data-started 公式 \(E(z_k)-E(z_{k+1})\) 直接套到当前 noise-started sampler 轨迹上，会出现方向问题：当前 MCMC 更新本身沿能量下降方向前进，通常希望 \(E(z_{k+1})<E(z_k)\)。若最小化 \(E(z_k)-E(z_{k+1})\)，训练会倾向于把早期状态能量压低、把后续状态能量抬高，和 sampler descent 方向相反，也会和 endpoint CD 希望“data 能量低于 fake endpoint”的排序产生冲突。

结论：**当前实现条件下，不应把 trajectory-local raw CD 理解为严格 score-matching Taylor 项；更合理的是把它改造成 sampler-direction 的局部能量斜率约束。**

### 4.2 与二阶梯度的真实近似关系

二阶 pathwise 梯度包含：

$$
\nabla_\theta L
\supset
\sum_k
\frac{\partial L}{\partial z_K}
\left(\prod_{j>k}\frac{\partial z_{j+1}}{\partial z_j}\right)
\left[-\alpha_k\nabla_\theta\nabla_zE_\theta(c,z_k)\right].
$$

本次实现的一阶局部项是：

$$
L_{\mathrm{local}}
=
\mathbb E_{k,t}
\left[
E_\theta(c,z_{k+s,t})-E_\theta(c,z_{k,t})
\right],
$$

其中 \(z\) 已 detach，\(s\) 是 stride，\(t\) 是有效 token 位置。其参数梯度为：

$$
\nabla_\theta L_{\mathrm{local}}
=
\mathbb E_{k,t}
\left[
\nabla_\theta E_\theta(c,z_{k+s,t})
-
\nabla_\theta E_\theta(c,z_{k,t})
\right].
$$

它比 endpoint ranking 更接近二阶目标的原因：

- 覆盖从 endpoint 扩展到 MCMC 中间状态，监督区域更接近采样器实际访问的 \(\mathrm{supp}(\mu)\)。
- raw 版本不饱和，能持续塑造局部能量斜率。
- 最小化 \(E(z_{k+s})-E(z_k)\) 会鼓励 sampler 前进方向能量下降，和当前 MCMC update 的使用方式一致。

它仍然不是二阶 field loss，主要误差来源：

- detach 轨迹，完全没有 \(\nabla_\theta\nabla_zE\) 和跨步 credit assignment。
- 只约束 scalar energy slope，不约束 \(\nabla_zE\) 的具体方向和范数。
- 当前实现复用最后一步 `mcmc_step=self.hparams.mcmc_num_steps - 1` 的 energy landscape 重算所有 pair，见 `openebm/elm/modeling_ebt.py:663-669`；如果 time embedding 让不同步能量面差异很大，这会引入近似误差。
- 现有 `predicted_distributions` 保存的是更新后的状态；没有保存 MCMC 初始 \(z_0\) 到第一步后的 pair。若要覆盖真正第一步，需要额外保存初始 relaxed token，当前为兼容性暂不改 forward 返回结构。

### 4.3 最有效可行设计

最稳妥的方案不是替换 `first_order_cd`，而是在 endpoint CD 上加小权重局部项：

$$
L
=
c_{\mathrm{endpoint}}L_{\mathrm{endpoint}}
+
\lambda_{\mathrm{local}}L_{\mathrm{local}}.
$$

其中：

$$
L_{\mathrm{endpoint}}
=
\operatorname{softplus}(E(x^+)-E(z_K)),
\quad
L_{\mathrm{local}}
=
\frac{1}{|\mathcal K|}
\sum_{k\in\mathcal K}
\left[
E(z_{k+s})-E(z_k)
\right].
$$

优先级判断：

- `num_pairs=1, stride=1` 最值得先做：只多覆盖一对相邻小步，额外成本最低，最不容易偏离局部假设。
- `num_pairs=2` 是第二优先级：检查覆盖增加是否提升二阶 teacher cosine。
- 不建议一开始用全轨迹 pair：中后段状态距离可能较大，局部假设弱，且额外 transformer 前向会增加显存/时间。
- `raw` 是最直接的局部斜率项；若能量 delta 过大、loss 变负过快或梯度不稳，再切到 `softplus`。

## 5. 已实现的兼容 option

### 5.1 代码路径与公式

本次实现把局部项挂在 `mcmc_gradient_mode=first_order_cd` 下，默认关闭。

代码事实：

- 总 loss 接入：`openebm/elm/modeling_ebt.py:386-395`。
- 局部 pair 构造：`openebm/elm/modeling_ebt.py:624-650`，从 `predicted_distributions[pair_idx]` 和 `predicted_distributions[pair_idx + stride]` 取 detached relaxed logits。
- 合并一次 transformer 重算局部能量：`openebm/elm/modeling_ebt.py:652-669`。
- raw / softplus 局部 loss：`openebm/elm/modeling_ebt.py:671-684`。
- 参数定义：`openebm/elm/train.py:610-630`。
- 参数校验：`openebm/elm/train.py:79-94`。
- c2048/c1024 运行脚本环境变量透传：`openebm/elm/runs/run_ebt_muon_adamw_c2048.sh:273-286`、`openebm/elm/runs/run_ebt_muon_adamw_c1024.sh:189-202`。

实际优化目标：

$$
L_{\mathrm{train}}
=
\texttt{first\_order\_cd\_loss\_coeff}\cdot L_{\mathrm{endpoint}}
+
\texttt{first\_order\_local\_cd\_coeff}\cdot L_{\mathrm{local}}.
$$

`raw`：

$$
L_{\mathrm{local}}
=
\mathbb E_{\mathrm{valid}}
\left[E(z_{k+s})-E(z_k)\right].
$$

`softplus`：

$$
L_{\mathrm{local}}
=
\mathbb E_{\mathrm{valid}}
\left[
\operatorname{softplus}(E(z_{k+s})-E(z_k)+m)
\right].
$$

### 5.2 新增参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--first_order_local_cd_coeff` | `0.0` | 默认关闭，保证旧 `first_order_cd` 行为不变 |
| `--first_order_local_cd_num_pairs` | `1` | 使用多少对相邻 stored MCMC states |
| `--first_order_local_cd_pair_stride` | `1` | pair 间隔 \(s\) |
| `--first_order_local_cd_loss_type` | `raw` | `raw` 或 `softplus` |
| `--first_order_local_cd_margin` | `0.0` | `softplus` 内部 margin |

### 5.3 推荐实验配置

第一条建议：

```bash
MCMC_GRADIENT_MODE=first_order_cd \
FIRST_ORDER_LOCAL_CD_COEFF=0.1 \
FIRST_ORDER_LOCAL_CD_NUM_PAIRS=1 \
FIRST_ORDER_LOCAL_CD_PAIR_STRIDE=1 \
FIRST_ORDER_LOCAL_CD_LOSS_TYPE=raw
```

如果 `first_order_local_cd_energy_delta` 快速变成很大的负数、gradient norm 明显异常，改跑：

```bash
MCMC_GRADIENT_MODE=first_order_cd \
FIRST_ORDER_LOCAL_CD_COEFF=0.1 \
FIRST_ORDER_LOCAL_CD_NUM_PAIRS=1 \
FIRST_ORDER_LOCAL_CD_PAIR_STRIDE=1 \
FIRST_ORDER_LOCAL_CD_LOSS_TYPE=softplus \
FIRST_ORDER_LOCAL_CD_MARGIN=0.0
```

推荐 sweep：

- `FIRST_ORDER_LOCAL_CD_COEFF=0.03/0.1/0.3`
- `FIRST_ORDER_LOCAL_CD_NUM_PAIRS=1/2`
- 只有小模型 teacher cosine 提升后，再尝试 `0.5` 或更多 pairs。

### 5.4 NCE 作为辅助校准线

现有代码已经能跑一条无需先改代码的 hybrid：

```bash
MCMC_GRADIENT_MODE=proposal_aware_nce \
PROPOSAL_AWARE_NCE_PROPOSAL=mcmc_final \
PROPOSAL_AWARE_NCE_RELAXED_CD_COEFF=1.0 \
PROPOSAL_AWARE_NCE_BASE_COEFF=0.05 \
PROPOSAL_AWARE_NCE_K=1
```

这条线的定位：

- 保留 MCMC endpoint relaxed CD。
- 额外加入少量 proposal-aware NCE 校准。
- 如果任务指标提升但二阶 field cosine 不提升，应解释为“校准/覆盖收益”，不是“二阶近似更准”。

## 6. 实验判断标准

### 6.1 P0：二阶 teacher 对齐评估

先用小模型和小 batch 建立 teacher：

1. 固定同一 batch、同一 seed、同一初始权重。
2. 跑 `second_order` 得到 teacher gradient \(g_2\)。
3. 跑候选一阶方案得到 \(g_m\)。
4. 比较：

$$
\cos(g_m,g_2)
=
\frac{g_m^\top g_2}{\|g_m\|\|g_2\|},
\quad
\mathrm{relerr}
=
\frac{\|g_m-g_2\|}{\|g_2\|}.
$$

必须分层统计：

- embeddings / vocab_to_embed
- transformer 前层
- transformer 中层
- transformer 后层或 energy/output head

主要候选：

| 候选 | 目的 |
|---|---|
| `first_order_cd` | 当前 baseline |
| `first_order_cd + local_cd(raw,num_pairs=1,coeff=0.1)` | 验证局部轨迹斜率是否更接近二阶梯度 |
| `first_order_cd + local_cd(raw,num_pairs=2,coeff=0.1)` | 验证增加轨迹覆盖是否继续有益 |
| `first_order_cd + local_cd(softplus,num_pairs=1,coeff=0.1)` | 验证稳定化是否牺牲 field 对齐 |
| `proposal_aware_nce + mcmc_final + relaxed_cd` | 现有 hybrid |

### 6.2 P1：验证已实现的 trajectory-local descent CD

已实现参数：

- `--first_order_local_cd_coeff`
- `--first_order_local_cd_num_pairs`
- `--first_order_local_cd_pair_stride`
- `--first_order_local_cd_loss_type {raw,softplus}`
- `--first_order_local_cd_margin`

优先消融：

| 实验 | 判断 |
|---|---|
| raw, `coeff=0.03/0.1/0.3`, `num_pairs=1` | 是否比 `first_order_cd` 更接近二阶 gradient |
| raw, `num_pairs=2` | 增加局部覆盖是否继续有益 |
| raw vs softplus | 稳定化是否损害 field 对齐 |
| `stride=1` vs `stride=2` | 更长局部间隔是否破坏近似 |

必须同时记录：

- `first_order_energy_gap`：endpoint fake 与 real 的能量差。
- `first_order_local_cd_energy_delta`：局部 \(E(z_{k+s})-E(z_k)\)。
- `first_order_local_cd_loss`：raw 或 softplus 后的局部项。
- 梯度方向 cosine / relerr：最终判断依据，不能只看 loss 下降。

### 6.3 P1：现有 hybrid 辅助线

无需代码改动，先跑：

```bash
MCMC_GRADIENT_MODE=proposal_aware_nce \
PROPOSAL_AWARE_NCE_PROPOSAL=mcmc_final \
PROPOSAL_AWARE_NCE_RELAXED_CD_COEFF=1.0 \
PROPOSAL_AWARE_NCE_BASE_COEFF=0.05 \
PROPOSAL_AWARE_NCE_K=1
```

判据：

- 若 gradient cosine 低于 `first_order_cd`，说明 NCE 校准偏离二阶 field。
- 若 valid/task metric 提升但 cosine 不提升，应把收益归因到 calibration / coverage。
- 若 cosine 和 task metric 都提升，再考虑提高 `K` 或 `base_coeff`。

### 6.4 P2：NCE 工程增强

这些放在 local CD 验证之后：

- `PROPOSAL_AWARE_NCE_K=2/4`：增加离散负样本覆盖，但显存随 K 增。
- `PROPOSAL_AWARE_NCE_RANK_COEFF=0.02/0.05`：hard negative ranking。
- 可学习 logZ / offset：改善 NCE calibration，不直接提升 field 阶数。

### 6.5 P3：大模型稳定性回归

小模型确认后再上 FSDP2/ZeRO：

- 比较 peak memory / tokens/sec。
- 保持 `no_mcmc_detach=False`。
- 先禁用随机 step count，降低评估噪声。
- 监控 energy gap、gradient norm、`pct_gradient_clipped`。

## 7. 最终执行顺序

1. 用小模型建立 `second_order` teacher 的 gradient / field 对齐评估。
2. 对比当前 `first_order_cd` 和现有 `proposal_aware_nce + mcmc_final + relaxed_cd`。
3. 跑已实现的 `first_order_cd + first_order_local_cd_coeff=0.1,num_pairs=1,raw`。
4. 若 local descent CD 的二阶 cosine 更高，再做 `num_pairs=2`、`coeff=0.03/0.3` 和 `softplus` 稳定化消融。
5. 只有在 field 对齐不退化时，再加入 NCE 的 `K>1`、hard negative、logZ offset。

这份文档的判断依据是：**训练目标是否在采样器实际轨迹上直接约束 \(\nabla_xE_\theta\)**。能量 gap、valid loss、NCE loss 都有参考价值，但不能单独证明某个一阶目标更接近二阶梯度图。
