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
\mathcal{D}(\theta) = \mathbb{E}_{x\sim\mu} \left[ \|\nabla_x E_\theta(x) - \nabla_x E_*(x)\|^2 \right]
$$

按“对 \(\nabla_xE_\theta\) 的约束有多直接”排序，当前判断是：

$$
\text{second\_order}
\succ
\text{待实现：data-anchored local CD / denoising-style 一阶 surrogate}
\succ
\text{first\_order\_cd}
\approx
\text{proposal\_aware\_nce + relaxed\_cd}
\approx
\text{first\_order\_cd + trajectory-local descent CD}
\succ
\text{first\_order\_nce / pure proposal NCE}.
$$

最关键的代码理解：

- 当前 `second_order` 是唯一已有的 pathwise-field 训练路径：loss 反传穿过每一步 MCMC 的 \(\nabla_xE_\theta\)。
- 当前 `first_order_cd` 是 graph-safe 的 endpoint relaxed ranking：跑 MCMC 到终点、detach、重算正负 energy，优化 \(\operatorname{softplus}(E^+-E^-)\)。
- 当前 `first_order_cd` 不是“小步 raw CD / score-matching leading term”实现，因为它只使用最终 endpoint，且不是原始相邻小步能量差。
- NCE / proposal-aware NCE 主要改善负样本覆盖和能量校准，不会自动恢复 \(\nabla_xE_\theta\) 的 field 监督阶数。
- 已实现的 **trajectory-local descent CD** 只应视作默认关闭的 sampler-consistency regularizer：因为 \(z_{k+1}\) 本来就是当前能量梯度下降产生的，\(E(z_{k+1})<E(z_k)\) 通常已由 sampler 自身满足；该项不应被当作强二阶近似主线。
- 更值得新增的主线是 **data-anchored local CD / denoising-style 一阶 surrogate**：在真实 token 邻域构造局部负样本或 near/far denoising pair，用一阶能量差逼近“朝数据方向能量下降”的局部 directional derivative。

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

更关键的是，它和原始二阶目标的语义不同。原始 `second_order` 的训练信号来自最终预测分布 \(z_K\) 与真实 token 的 CE / NLL，希望的是：

$$
z_K \rightarrow x^+.
$$

而 `first_order_cd` 的 endpoint ranking 希望的是：

$$
E_\theta(c,x^+) < E_\theta(c,z_K).
$$

当 \(z_K\) 明显错误、离 \(x^+\) 很远时，这个排序是合理的：真实 token 应该处在更低能量 basin，采样器才有动力朝数据走。但当 \(z_K\) 已经接近 \(x^+\)，继续把 \(z_K\) 当作 negative 并强行推高能量就不再符合“让最终预测等于真实 token”的原始二阶目标。理想行为应接近：

$$
E_\theta(c,z_K)-E_\theta(c,x^+) \approx m\cdot d(z_K,x^+),
$$

其中 \(d(z_K,x^+)\) 是预测错误程度，例如 CE、KL、\(1-p_{z_K}(x^+)\) 或 embedding/probability 距离。于是：

- \(z_K\) 很错：\(E(x^+)\) 应明显低于 \(E(z_K)\)。
- \(z_K\) 接近正确：energy gap 应变小，避免把 near-correct 状态继续推高。
- \(z_K=x^+\)：两者能量应相等或至少不再继续拉开。

因此，`first_order_cd` 的方向不是完全错，但它缺少 **distance-aware / correctness-aware gap control**。后续一阶 surrogate 不应只追求更大的 \(E(z_K)-E(x^+)\)，而应让 energy gap 与预测错误程度一致。

因此判断一阶近似优劣时，我采用三个标准：

| 标准 | 问题 | 越好意味着 |
|---|---|---|
| 阶数 | loss 是否显式或隐式含 \(\nabla_xE_\theta\) 信息 | 更像 field objective |
| 覆盖 | 约束覆盖 data 邻域、endpoint，还是整条轨迹 | 更贴采样器实际路径 |
| 饱和 | 能量差拉开后训练信号是否关闭 | 更稳定地继续塑形 |
| gap 校准 | energy gap 是否随预测错误程度变小/变大 | 更贴近 \(z_K\rightarrow x^+\) 的原始二阶语义 |

## 2. 当前实现对照

| 实现 | 关键代码 | 当前目标/路径 | 几何对象 | 对二阶 field 的近似 |
|---|---|---|---|---|
| `second_order` | `openebm/elm/modeling_ebt.py:140-150` 中 `create_graph_for_mcmc=True`；CE 聚合在 `:318-342`、`:396-401` | loss 反传穿过每步 \(\nabla_zE_\theta\)，包含 \(\nabla_\theta\nabla_zE_\theta\) | pathwise-field，覆盖整条 MCMC 轨迹 | 最高；小模型 teacher / gold baseline |
| `first_order_debug` | mode 说明在 `openebm/elm/train.py:567-579`；FSDP2 warning 在 `openebm/elm/train_engines/fsdp2.py:50-55` | 只关闭 sampler 的 `create_graph`；CE 对模型主参数信号不完整 | 诊断路径 | 不作为训练近似方案 |
| `first_order_cd` | graph-safe detach 在 `openebm/elm/modeling_ebt.py:239-244`；combined 重算在 `:536-552`；CE ranking 在 `:583-604` | \(L=\mathrm{CE}([-E^+,-E^-],0)=\mathrm{softplus}(E^+-E^-)\) | endpoint relaxed level-set/ranking | 中；当前最实用一阶 baseline，但不是 local raw CD |
| 待实现：`distance-aware first_order_cd` | 建议在 `calculate_contrastive_loss()` 的 endpoint delta 上加 stop-gradient distance 权重或 margin | \(d(z_K,x^+)\operatorname{softplus}(E^+-E_K)\) 或 \(\operatorname{softplus}(E^+-E_K+m d)\) | correctness-aware endpoint ranking | 中偏高；修复 near-correct fake 被继续推高的问题，但仍不是 field loss |
| `first_order_cd + first_order_local_cd_coeff>0` | loss 接入在 `openebm/elm/modeling_ebt.py:386-395`；局部能量重算在 `:624-694`；参数在 `openebm/elm/train.py:610-630` | \(L=L_{\mathrm{endpoint}}+\lambda\,\mathbb E[E(z_{k+s})-E(z_k)]\) | endpoint ranking + 轨迹局部能量斜率 | 中/不确定；能量下降通常已由 sampler 构造出来，更多是弱 consistency regularizer |
| 待实现：`data-anchored local CD / denoise-rank` | 建议复用 `calculate_contrastive_loss` 的 one-hot 构造与 combined transformer 重算路径；新增辅助函数见第 5 节 | data 邻域 \(x^+\)、\(\tilde x\) 或 near/far pair 的局部能量差 | data-anchored local directional derivative | 中偏高；仍是一阶参数梯度，但比 trajectory-local descent 更有监督锚点 |
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

## 4. Trajectory-local descent CD 的合理性修正

### 4.1 你的质疑是正确的：当前 MCMC 轨迹天然倾向能量下降

当前 OpenEBM 的 MCMC 轨迹从初始 relaxed/noise token 出发，按当前模型能量做梯度下降：

$$
z_{k+1}=z_k-\alpha\nabla_zE_\theta(c,z_k)+\epsilon_k.
$$

忽略噪声并做小步 Taylor 展开，记 \(g_k=\nabla_zE_\theta(c,z_k)\)、\(H_k=\nabla_z^2E_\theta(c,z_k)\)，则

$$
E(z_{k+1})-E(z_k)
=
-\alpha\|g_k\|^2
+
\frac{\alpha^2}{2}g_k^\top H_kg_k
+
O(\alpha^3).
$$

只要步长不太大、噪声不主导，第一项会使 \(E(z_{k+1})<E(z_k)\)。因此，已实现的 trajectory-local descent CD：

$$
L_{\mathrm{local}}
=
\mathbb E[E(z_{k+s})-E(z_k)]
$$

很大程度上是在强化 sampler 自己已经构造出来的能量下降关系。它不是 data-started raw CD，也不自动恢复第 1.2 节的 score-matching Taylor 通道。

### 4.2 该项剩余的意义：弱 consistency regularizer，而不是强二阶近似

代码事实：

- 初始 relaxed token 来自 `corrupt_embeddings`，`random_noise`/`zeros` 分支在 `openebm/elm/modeling_ebt.py:464-469`，不是从 one-hot 正样本附近初始化。
- `first_order_cd`/`first_order_nce`/`proposal_aware_nce` 在 `openebm/elm/modeling_ebt.py:239-244` 保存的是每步更新后的 detached `predicted_distributions`。
- 已实现局部项从这些 detached states 取相邻 pair，见 `openebm/elm/modeling_ebt.py:624-650`，并优化 \(E(z_{k+s})-E(z_k)\)，见 `:671-684`。

它仍可能有少量工程价值：

- 作为 consistency regularizer，显式要求重算 energy 后的排序与 sampler 生成轨迹一致。
- 若 detached sampler 状态与重算 energy 的数值路径存在偏差，它能暴露这种不一致。
- `softplus` 小系数版本可以作为稳定化探针，观察 `first_order_local_cd_energy_delta` 是否与预期一致。

但它对“更接近二阶梯度图”的贡献是高度不确定的：

- 没有 data anchor，只比较模型自己生成的两个状态，不告诉模型这条下降路是否朝向真实 token。
- detach 轨迹，完全没有 \(\nabla_\theta\nabla_zE\) 和跨步 credit assignment。
- 只约束 scalar energy slope，不约束 \(\nabla_zE\) 的方向和范数。
- raw 版本会持续把 \(E(z_{k+s})-E(z_k)\) 推得更负，可能只是放大自生成坡度，而不是让 field 更正确。

结论：**trajectory-local descent CD 应保留为默认关闭的低优先级实验项，不应作为最高优先级二阶近似方案。** 若要跑，建议只用小系数并以 teacher gradient cosine 判断，而不是看它自己的 loss 是否下降。

## 5. 更优先的待实现方案：data-anchored local CD / denoising-style 一阶 surrogate

### 5.1 为什么它比 trajectory-local descent CD 更有意义

trajectory-local descent CD 的问题是没有真实 token 锚点；而 data-anchored 方案从 \(x^+\) 或 \(x^+\) 的局部扰动出发，直接告诉模型“哪个方向更靠近数据”。这更接近下游想要的 denoising field：

$$
s_\theta(c,z)=-\nabla_zE_\theta(c,z)
\quad\text{应指向}\quad
x^+-z.
$$

如果只用能量值做一阶参数训练，可以通过局部 near/far pair 逼近这个方向约束。给定数据邻域扰动点 \(z_\sigma\)，定义

$$
v=x^+-z_\sigma,
\quad
z_{\mathrm{near}}=z_\sigma+\eta v,
\quad
z_{\mathrm{far}}=z_\sigma-\eta v.
$$

对能量差做 Taylor：

$$
E(z_{\mathrm{near}})-E(z_{\mathrm{far}})
\approx
2\eta\nabla_zE_\theta(c,z_\sigma)^\top v.
$$

最小化该差值会推动

$$
\nabla_zE_\theta(c,z_\sigma)^\top (x^+-z_\sigma)<0,
$$

等价于让 score \(-\nabla_zE_\theta\) 与回到数据的方向 \(x^+-z_\sigma\) 对齐。它仍然只需要对参数的一阶梯度，因为 loss 本身只含 energy values，不显式反传 \(\nabla_zE\)。

### 5.2 三个可落地变体

**变体 A：data-anchored local CD**

构造 data 邻域负样本：

$$
z_{\mathrm{neg}}
=
\operatorname{perturb}(x^+;\sigma),
\quad
L_{\mathrm{data\_local\_cd}}
=
E_\theta(c,x^+)-E_\theta(c,z_{\mathrm{neg}}).
$$

如果 \(z_{\mathrm{neg}}\) 是 \(x^+\) 的小步 Langevin 或小噪声扰动，这比 endpoint CD 更接近第 1.2 节的 local raw CD / score-matching leading term。它的覆盖主要是 data 邻域，不覆盖整条 MCMC trajectory，但监督方向明确。

**变体 B：denoising-style near/far ranking**

先采样局部 corrupted state：

$$
z_\sigma=(1-\sigma)x^+ + \sigma u,
$$

其中 \(u\) 可取 uniform token distribution、softmax Gaussian logits，或从当前模型 proposal 中采样的 detached distribution。再构造 near/far：

$$
z_{\mathrm{near}}=(1-\eta)z_\sigma+\eta x^+,
\quad
z_{\mathrm{far}}=(1+\eta)z_\sigma-\eta x^+,
$$

并投影/裁剪回有效 relaxed-token 表示。目标：

$$
L_{\mathrm{denoise\_rank}}
=
E_\theta(c,z_{\mathrm{near}})
-
E_\theta(c,z_{\mathrm{far}}),
$$

或稳定版本：

$$
L_{\mathrm{denoise\_rank}}
=
\operatorname{softplus}
\left(
E_\theta(c,z_{\mathrm{near}})
-
E_\theta(c,z_{\mathrm{far}})
+m
\right).
$$

这个目标比 trajectory-local descent CD 更值得优先验证，因为它约束的是“朝真实 token 方向能量下降”，不是“沿模型自己刚走过的方向继续下降”。

**变体 C：distance-aware endpoint CD**

这不是 data-neighborhood field 约束，但它直接修复当前 endpoint CD 的一个语义偏差：当 \(z_K\) 已接近真实 token 时，不应继续把它作为强 negative 推高。定义 stop-gradient 距离：

$$
d_K=\operatorname{sg}\left[d(z_K,x^+)\right],
$$

例如：

$$
d_K=1-p_{z_K}(x^+),
\quad\text{或}\quad
d_K=\operatorname{CE}(z_K,x^+).
$$

可以使用 distance-weighted CD：

$$
L_{\mathrm{dw\_cd}}
=
d_K\cdot \operatorname{softplus}(E(x^+)-E(z_K)).
$$

也可以使用 distance-aware margin：

$$
L_{\mathrm{dam\_cd}}
=
\operatorname{softplus}(E(x^+)-E(z_K)+m\cdot d_K).
$$

直觉：

- \(z_K\) 很错时，\(d_K\) 大，仍要求 \(E(x^+)\ll E(z_K)\)。
- \(z_K\) 接近正确时，\(d_K\) 小，降低继续推高 \(z_K\) 的力度。
- \(z_K=x^+\) 时，理想上不再制造额外 energy gap。

这个方案的优点是改动小，直接复用当前 `first_order_cd` 的 endpoint energy 重算路径；缺点是仍然是 endpoint scalar ranking，不提供显式 field 方向监督。因此它适合作为 `first_order_cd` 的低成本修正，而不是替代 data-anchored denoising surrogate。

### 5.3 最小实现设计

建议先把 data-anchored denoise-rank 作为 `first_order_cd` 的另一个默认关闭 auxiliary，而不是新增主模式：

- 新增参数：
  - `--first_order_data_local_cd_coeff`，默认 `0.0`。
  - `--first_order_data_local_cd_noise`，推荐先试 `0.05/0.1/0.2`。
  - `--first_order_data_local_cd_eta`，推荐先试 `0.25/0.5`。
  - `--first_order_data_local_cd_loss_type {raw,softplus}`，默认先用 `softplus` 更稳。
  - `--first_order_data_local_cd_num_samples`，默认 `1`。
  - `--first_order_data_local_cd_noise_mode {uniform,gaussian_logits}`，默认 `uniform`。
- 复用 `calculate_contrastive_loss()` 中 one-hot target 构造逻辑，或抽出公共 helper。
- 在 `forward_loss_wrapper()` 的 `first_order_cd` 分支中，和 endpoint CD 一起组合：

```python
total_loss = first_order_cd_loss_coeff * endpoint_cd_loss
if first_order_data_local_cd_coeff > 0:
    data_local_loss, data_local_metrics = self.calculate_data_anchored_local_cd_loss(
        input_ids,
        next_token_indices,
    )
    total_loss = total_loss + first_order_data_local_cd_coeff * data_local_loss
```

`calculate_data_anchored_local_cd_loss()` 最小伪代码：

```python
true_probs = one_hot(next_token_indices).masked_fill(~valid_positions, 0)
noise_probs = uniform_or_gaussian_probs_like(true_probs)
z_sigma = (1 - sigma) * true_probs + sigma * noise_probs

near = normalize_or_clamp((1 - eta) * z_sigma + eta * true_probs)
far = normalize_or_clamp((1 + eta) * z_sigma - eta * true_probs)

near_energy, far_energy = recompute_energies_in_one_transformer_call(near, far)
delta = near_energy - far_energy
loss = delta.mean()              # raw
# or loss = F.softplus(delta + margin).mean()
```

工程注意点：

- `far` 可能出现负概率；最简单方案是先在 logits 空间做 near/far，或对 probability-space 结果 `clamp_min(0)` 后重新归一化。第一版应在文档和日志里标注该投影带来的近似误差。
- valid mask 必须和 `calculate_contrastive_loss()` 一样处理 padding token，避免无效位置参与 energy ranking。
- 该项可与 endpoint `first_order_cd` 混合；不建议一开始替代 endpoint CD。
- 若需要更接近 data-started CD，可增加一版 one-step detached Langevin negative：从 \(x^+\) 或 \(z_\sigma\) 出发计算一次 \(\nabla_zE\)，`create_graph=False`，得到 `z_neg.detach()` 后再做 \(E(x^+)-E(z_neg)\)。这比 near/far ranking 更贵，但仍避免二阶 sampler 图。

distance-aware endpoint CD 的最小实现可放在 `calculate_contrastive_loss()` 或其外层：

```python
fake_probs = softmax(fake_pred_tokens.detach())
target_prob = fake_probs.gather(-1, safe_next_token_indices.unsqueeze(-1)).squeeze(-1)
d = (1.0 - target_prob).detach()

base_delta = real_energies.squeeze(-1) - fake_energies.squeeze(-1)
if loss_type == "distance_weighted_ce":
    per_token_loss = d.reshape(-1) * F.softplus(base_delta)
elif loss_type == "distance_margin":
    margin = first_order_cd_distance_margin * d.reshape(-1)
    per_token_loss = F.softplus(base_delta + margin)
```

建议新增参数：

- `--first_order_cd_loss_type {ce,margin,distance_weighted_ce,distance_margin}`
- `--first_order_cd_distance_metric {one_minus_true_prob,ce}`
- `--first_order_cd_distance_margin`，默认 `1.0`
- `--first_order_cd_distance_weight_floor`，默认 `0.0`，用于避免 near-correct 样本完全无梯度。

### 5.4 风险与验证

风险：

- 只覆盖 data 邻域，不能直接监督远离数据但采样器会经过的中段区域。
- 噪声太小会导致信号弱；噪声太大会破坏 local Taylor 假设。
- raw loss 可能无界，第一版建议优先跑 `softplus`。
- near/far projection 若处理不当，可能学到 simplex 边界伪影。

验证：

- 首要指标仍是与 `second_order` teacher 的 gradient cosine / relerr。
- 额外记录 \(E(z_{\mathrm{near}})-E(z_{\mathrm{far}})\)、endpoint energy gap、valid/task metric。
- 若 teacher cosine 提升但 valid/task 不提升，说明 local field 方向更接近但全局 basin 排序不足；可增加 endpoint CD 权重。
- 若 valid/task 提升但 teacher cosine 不提升，应把收益解释为 data-neighborhood regularization，而不是更准确二阶近似。

## 6. 已实现的兼容 option

### 6.1 代码路径与公式

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

### 6.2 新增参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--first_order_local_cd_coeff` | `0.0` | 默认关闭，保证旧 `first_order_cd` 行为不变 |
| `--first_order_local_cd_num_pairs` | `1` | 使用多少对相邻 stored MCMC states |
| `--first_order_local_cd_pair_stride` | `1` | pair 间隔 \(s\) |
| `--first_order_local_cd_loss_type` | `raw` | `raw` 或 `softplus` |
| `--first_order_local_cd_margin` | `0.0` | `softplus` 内部 margin |

### 6.3 低优先级 sanity-check 配置

该项不再作为主要二阶近似方向。若需要验证它是否还有 consistency regularization 收益，建议先用小系数和 `softplus`：

```bash
MCMC_GRADIENT_MODE=first_order_cd \
FIRST_ORDER_LOCAL_CD_COEFF=0.03 \
FIRST_ORDER_LOCAL_CD_NUM_PAIRS=1 \
FIRST_ORDER_LOCAL_CD_PAIR_STRIDE=1 \
FIRST_ORDER_LOCAL_CD_LOSS_TYPE=softplus \
FIRST_ORDER_LOCAL_CD_MARGIN=0.0
```

只在 teacher cosine 明确提升时，再短跑 raw 对照：

```bash
MCMC_GRADIENT_MODE=first_order_cd \
FIRST_ORDER_LOCAL_CD_COEFF=0.03 \
FIRST_ORDER_LOCAL_CD_NUM_PAIRS=1 \
FIRST_ORDER_LOCAL_CD_PAIR_STRIDE=1 \
FIRST_ORDER_LOCAL_CD_LOSS_TYPE=raw
```

不建议一开始 sweep 大范围参数。若确实有效，再试：

- `FIRST_ORDER_LOCAL_CD_COEFF=0.03/0.1`
- `FIRST_ORDER_LOCAL_CD_NUM_PAIRS=1/2`
- 只有小模型 teacher cosine 提升后，再考虑更多 pairs；否则停止。

### 6.4 NCE 作为辅助校准线

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

## 7. 实验判断标准

### 7.1 P0：二阶 teacher 对齐评估

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
| 待实现：`distance-aware first_order_cd` | 验证 correctness-aware gap 是否更贴近 \(z_K\rightarrow x^+\) |
| 待实现：`first_order_cd + data_anchored_denoise_rank` | 验证 data-neighborhood directional derivative 是否更接近二阶 field |
| 待实现：`first_order_cd + data_local_cd` | 验证 data-started 小扰动 raw CD 是否恢复 score-matching 通道 |
| `first_order_cd + trajectory_local_descent_cd` | 低优先级 sanity check；验证 sampler-consistency 是否有额外收益 |
| `proposal_aware_nce + mcmc_final + relaxed_cd` | 现有 hybrid |

### 7.2 P1：实现并验证 data-anchored local CD / denoise-rank

建议先实现第 5.3 节的参数：

- `--first_order_data_local_cd_coeff`
- `--first_order_data_local_cd_noise`
- `--first_order_data_local_cd_eta`
- `--first_order_data_local_cd_loss_type {raw,softplus}`
- `--first_order_data_local_cd_num_samples`
- `--first_order_data_local_cd_noise_mode {uniform,gaussian_logits}`

优先消融：

| 实验 | 判断 |
|---|---|
| distance-weighted endpoint CD, \(d=1-p(x^+)\) | near-correct \(z_K\) 是否不再被过度推高 |
| distance-margin endpoint CD, \(m d\) | energy gap 是否能随错误程度缩放 |
| denoise-rank softplus, `noise=0.1`, `eta=0.25`, `coeff=0.1` | 第一条最稳配置；验证 data-anchored 方向约束 |
| denoise-rank raw vs softplus | raw 是否提供更强 field 对齐，或是否不稳定 |
| `noise=0.05/0.1/0.2` | 找到 local Taylor 仍成立且信号足够强的噪声尺度 |
| `eta=0.25/0.5` | near/far 间隔是否过大导致非局部偏差 |
| data-local CD: \(E(x^+)-E(z_{\mathrm{neg}})\) | 验证更接近 data-started CD 的能量差形式 |
| one-step detached Langevin negative | 验证更接近 CD-1 的负样本是否值得额外一次 token-gradient 开销 |

必须同时记录：

- `first_order_energy_gap`：endpoint fake 与 real 的能量差。
- endpoint distance：\(1-p_{z_K}(x^+)\) 或 CE，用来检查 energy gap 是否随错误程度缩放。
- data-local energy delta：\(E(z_{\mathrm{near}})-E(z_{\mathrm{far}})\) 或 \(E(x^+)-E(z_{\mathrm{neg}})\)。
- denoise pair 的 noise/eta、valid mask token 数、梯度 norm。
- 梯度方向 cosine / relerr：最终判断依据，不能只看 loss 下降。

### 7.3 P2：低优先级验证已实现的 trajectory-local descent CD

该项已经实现，但由于 \(z_{k+1}\) 本来由当前能量下降产生，预期只可能提供弱 consistency regularization。若要保留 sanity check，使用小系数：

```bash
MCMC_GRADIENT_MODE=first_order_cd \
FIRST_ORDER_LOCAL_CD_COEFF=0.03 \
FIRST_ORDER_LOCAL_CD_NUM_PAIRS=1 \
FIRST_ORDER_LOCAL_CD_PAIR_STRIDE=1 \
FIRST_ORDER_LOCAL_CD_LOSS_TYPE=softplus
```

判据：

- 若 teacher cosine 不高于 `first_order_cd`，该项不应继续投入。
- 若 `first_order_local_cd_energy_delta` 本来已经为负且继续变得更负，但 teacher cosine 不提升，说明它只是在放大自生成坡度。
- 不建议将 raw trajectory-local loss 作为主目标；raw 只适合短 smoke。

### 7.4 P2：现有 hybrid 辅助线

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

### 7.5 P3：NCE 工程增强

这些放在 data-anchored local CD 验证之后：

- `PROPOSAL_AWARE_NCE_K=2/4`：增加离散负样本覆盖，但显存随 K 增。
- `PROPOSAL_AWARE_NCE_RANK_COEFF=0.02/0.05`：hard negative ranking。
- 可学习 logZ / offset：改善 NCE calibration，不直接提升 field 阶数。

### 7.6 P3：大模型稳定性回归

小模型确认后再上 FSDP2/ZeRO：

- 比较 peak memory / tokens/sec。
- 保持 `no_mcmc_detach=False`。
- 先禁用随机 step count，降低评估噪声。
- 监控 energy gap、gradient norm、`pct_gradient_clipped`。

## 8. 最终执行顺序

1. 用小模型建立 `second_order` teacher 的 gradient / field 对齐评估。
2. 对比当前 `first_order_cd` 和现有 `proposal_aware_nce + mcmc_final + relaxed_cd`。
3. 先实现 `distance-aware first_order_cd`，验证 near-correct \(z_K\) 不再被继续强推高。
4. 再实现 `data-anchored denoise-rank`，先跑 `softplus, noise=0.1, eta=0.25, coeff=0.1`。
5. 若 data-anchored 方案的 teacher cosine 提升，再做 raw/softplus、noise、eta 和 data-local CD 变体消融。
6. 已实现的 trajectory-local descent CD 只作为低优先级 sanity check；若 teacher cosine 不提升，不继续投入。
7. 只有在 field 对齐不退化时，再加入 NCE 的 `K>1`、hard negative、logZ offset。

这份文档的判断依据是：**训练目标是否以有监督锚点约束 \(\nabla_xE_\theta\) 的方向/局部 directional derivative，而不只是制造能量 gap**。能量 gap、valid loss、NCE loss 都有参考价值，但不能单独证明某个一阶目标更接近二阶梯度图。
