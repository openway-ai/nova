# Test2: last-layer embedding 直接 unembedding 的 free embedding 方案

## 要测试的新问题

Test1 当前代码中，EBT trunk 输出的 candidate hidden 会进入 TF-head 内部的小 transformer，得到 head 内部的 `t1 hidden` 后再做 unembedding / vocab projection。

Test2 想测试另一种更直接的路径：

```text
last-layer candidate hidden -> unembedding -> t1 pred -> CE with t1 label
```

这里需要区分两类 D 维表示：

```text
first-layer embedding:
    context token -> token embedding layer

last-layer candidate hidden:
    post-update free embedding state -> EBT trunk -> post-norm candidate hidden
```

Test2 要直接过 unembedding 的是第二类，也就是 EBT trunk 输出的 last-layer candidate hidden；不是 context token 刚过 `self.embeddings` 得到的 first-layer embedding。后面的监督方式不变，仍然用同一个 `t1 label` 做 CE。

## Test2 方案简图

```text
gold context token
    |
    v
token embedding layer / self.embeddings
    |
    +--------------------------+
    |                          |
    v                          v
real_embed [B,S,D]        prev_embed [B,S,D]
first-layer context       first-layer anchor
    |
    v
cat(real_embed, predicted_embed state)
    |
    v
EBT trunk / energy transformer
    |
    +--> energy -> grad -> update free D-dim MCMC state
    |
    +--> post-update trunk forward
                |
                v
        pred_hidden [B,S,D]
        post-norm last-layer candidate hidden
                |
                v
        unembedding / vocab projection
                |
                v
        t1 pred logits [B,S,V]
                |
                v
        CE(t1 pred, t1 label)
```

和 test1 的差异集中在监督头内部：

```text
Test1:
    pred_hidden
        + first-layer prev_embed
        -> down_proj
        -> TF-head transformer
        -> t1 hidden
        -> unembedding
        -> t1 pred
        -> CE with t1 label

Test2:
    pred_hidden
        -> unembedding
        -> t1 pred
        -> CE with t1 label
```

## 我对该方案的理解

Test2 不是要改 free embedding MCMC 的主体流程。MCMC 仍然在 D 维 embedding state 上做：

```text
predicted_embedding = predicted_embedding - alpha * grad_energy
```

也不是要改 label 或 loss。监督仍然是：

```text
CE(t1 pred logits, t1 label)
```

真正变化的是 `tf_head` 的解码路径：去掉 `pred_hidden + prev_embed -> down_proj -> transformer -> t1 hidden` 这段额外变换，让 trunk 产出的 last-layer candidate hidden 直接接受 vocab-level 监督。这样可以测试一个更强的假设：

> free embedding MCMC + EBT trunk 产出的 last-layer candidate hidden 是否已经足够可解码，是否不需要额外的小 transformer head 才能映射回 vocab。

如果 test2 效果接近或优于 test1，说明 last-layer candidate hidden 的语义可解码性较强，额外 TF-head transformer 可能不是必要组件。如果 test2 明显变差，则说明当前 EBT trunk/free embedding MCMC 产出的 last-layer 表示还需要一个额外的 teacher-forced transformer head 做 token-level translation。

## 已进行的代码改动

结论：当前新的 xxs 训练流程符合 Test2 方案。代码中已明确区分 first-layer embedding 和 last-layer candidate hidden：`prev_embed` 是 first-layer anchor，不会在 `direct_unembed` 中被 unembedding；真正直接过 vocab projection 的是 post-update trunk forward 返回的 `pred_hidden_step`。

### 1. CLI option

`openebm/elm/train.py` 已新增显式 option：

```text
--tf_head_type {linear,transformer,direct_unembed}
```

其中 `direct_unembed` 是 Test2 的独立 ablation 名称，用来避免和已有 `linear` 路径混在一起。默认值仍是 `transformer`，所以未显式传参的旧训练路径不受影响。

### 2. Direct unembedding head

`openebm/elm/tf_head.py` 已新增 `TFDirectUnembedHead`，实际路径为：

```text
pred_hidden [B,S,D]
    # trunk post-norm last-layer candidate hidden
    -> vocab projection [B,S,V]
    -> t1 pred logits
```

`prev_token_embed` 仍保留在 `forward()` 签名里，但只是为了兼容统一 TF-head 调用接口；在 `direct_unembed` 中会被显式忽略。该 head 不包含 TF-head transformer block，也不包含额外 `down_proj/final_norm`，因此与 Test1 的差异集中在：

```text
Test1: down_proj -> TF-head transformer -> final_norm -> vocab projection
Test2: pred_hidden -> vocab projection
```

### 3. Head factory 接入

`build_tf_head(hparams)` 已支持：

```text
tf_head_type == "direct_unembed" -> TFDirectUnembedHead
```

`modeling_ebt.py` 不需要为 Test2 增加分支，因为 TF-head 的统一接口仍是：

```text
self.tf_head(pred_hidden_step, prev_embed) -> logits [B,S,V]
```

其中 `prev_embed = self.embeddings(input_ids)` 是 first-layer embedding。`direct_unembed` 只保留该参数以匹配接口，实际不会使用它。

### 4. 主训练路径保持不变

当前代码检查后确认以下路径仍沿用 Test1/free-embedding 逻辑：

```text
free embedding MCMC:
    predicted_state [B,S,D] -> energy grad -> predicted_state - alpha * grad

post-update hidden:
    updated D-dim state -> second trunk forward -> pred_hidden_step
    pred_hidden_step is post-norm last-layer candidate hidden

supervision:
    tf_head logits -> CE with next_token_indices
```

因此 Test2 的变量是干净的：只去掉 TF-head transformer，并确保 unembedding 的对象是 last-layer `pred_hidden_step`，不改 MCMC、label 或 CE 聚合逻辑。

### 5. xxs 启动脚本

已新增并启动使用：

```text
openebm/elm/runs/launch_xxs_local_free_embed_direct_unembed.sh
```

核心配置为：

```text
RUN_NAME=ebt-xxs-directunembed-freeembed-4gpu-tokmatch-bs_256_s1_lr_0.0002
TF_HEAD_TYPE=direct_unembed

--use_tf_head
--tf_head_type direct_unembed
--free_embedding_mcmc
--free_embed_noise_scale "${FREE_EMBED_NOISE_SCALE}"
```

当前脚本按 4 卡 H200 设置，并将 `MAX_STEP` 调整为 `25000`，使总 token 数与原 2 卡 50k step xxs 对照一致：

```text
2 GPU * 2 batch/device * 2 grad_accum * 256 ctx * 50000 = 102.4M tokens
4 GPU * 2 batch/device * 2 grad_accum * 256 ctx * 25000 = 102.4M tokens
```

peak LR 仍保持 `0.0002`，避免在 Test2 中额外引入 batch-size LR scaling 变量；`WARMUP_STEP` 调整为 `500`，使 warmup token 数也与原 2 卡配置一致：

```text
old warmup: 1000 steps * 2048 tokens/step = 2.048M tokens
new warmup:  500 steps * 4096 tokens/step = 2.048M tokens
```

### 6. 已做轻量验证

已完成以下检查：

```text
python -m py_compile openebm/elm/train.py openebm/elm/tf_head.py openebm/elm/modeling_ebt.py
bash -n openebm/elm/runs/launch_xxs_local_free_embed_direct_unembed.sh
```

并确认 `direct_unembed` head 的输出 shape 为 `[B,S,V]`，可被现有 CE 路径直接消费。
