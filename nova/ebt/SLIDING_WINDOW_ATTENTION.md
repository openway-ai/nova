# Sliding Window Attention for NanoChatEBT

## Overview

This document describes the code changes that transplant Sliding Window Attention (SWA) support
into `nova_forked/nova`. The feature is exposed as a new EBT type `nanochat_time_embed` with an
optional `--window_pattern` flag. All existing EBT types (`default`, `time_embed`, `adaln`,
`adaln_zero`, `nanochat_d26`) are unchanged.

### Background

The base GPT model in `nanochat/gpt.py` already supports SWA via Flash Attention 3's native
`window_size=(left, right)` parameter, controlled by `GPTConfig.window_pattern` (a string like
`"SSSL"` tiled across layers). However, EBT training requires a **non-standard attention mask**
that Flash Attention 3 cannot express: predicted tokens must attend only to a restricted set of
real tokens, not the standard lower-triangular causal pattern. Because of this, EBT must fall back
to PyTorch's `scaled_dot_product_attention` (SDPA) with an explicit float additive mask. SWA is
then composed with the EBT mask by summing the two additive float masks.

---

## Changed Files

### 1. `nova/ebt/nanochat_ebt.py` — New implementation (replaces 83-line stub)

Previously a non-functional stub. Now a complete 307-line module containing four components.

#### `_build_ebt_mask(T, S, device)`

Builds the EBT-specific additive float attention mask for a sequence of length `T`.

**EBT sequence layout:**
```
position:  0      1         S-1    S        T-1
token:     time   real₁ … real_{S-1}  pred₁ … pred_{S-1}
           T = 2*(S-1) + 1,   S = (T+1)//2
```

**Mask rules:**
- Real positions `0..S-1`: standard lower-triangular causal (each token sees itself and all prior tokens).
- Predicted position `pred_k` (at row `S+k-1`, 1-indexed `k`):
  - Allowed: columns `0..k` (time token + real₁..real_k)
  - Allowed: self (column `S+k-1`)
  - Blocked: everything else (future real tokens; all other predicted tokens)

This prevents two classes of violations:
1. **Causality violation**: a predicted token seeing real tokens that come after it.
2. **Independence violation**: predicted tokens attending to each other (they must be conditionally independent given the real context).

Returns a `(T, T)` float tensor with values `0.0` (allowed) or `-inf` (blocked).

#### `EBTSelfAttention(CausalSelfAttention)`

Subclass of `CausalSelfAttention` from `nanochat/gpt.py`. Inherits all weight definitions
(`c_q`, `c_k`, `c_v`, `c_proj`, `ve_gate`) and the QK-norm + RoPE logic unchanged. The only
difference is in `forward()`.

**Changes in `forward(self, x, ve, cos_sin, window_size, kv_cache, attn_mask=None)`:**

1. **Drops `flash_attn_func`**: The parent class uses FA3/SDPA via `flash_attn.flash_attn_func(causal=True)`. This subclass replaces that with `F.scaled_dot_product_attention` and an explicit mask, because FA3's `causal=True` only supports standard lower-triangular masks.

2. **Tensor layout transpose**: FA3 uses `(B, T, H, D)` natively; SDPA requires `(B, H, T, D)`. Adds `q/k/v = x.transpose(1, 2)` before the attention call and `y.transpose(1, 2)` after.

3. **Sliding window mask composition**: When `window_size[0] > 0` (i.e. not full context), builds a `(T, T)` additive float mask that blocks any position where `row - col > window_size[0]` (attending too far back), then adds it to the incoming EBT mask:
   ```python
   sw = torch.where(
       (rows - cols) <= window_size[0],
       torch.zeros(1, device=q.device),
       torch.full((1,), float('-inf'), device=q.device),
   )
   attn_mask = attn_mask + sw.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
   ```
   Summing two `{0, -inf}` masks ANDs them: a position is blocked if either constraint fires.

4. **SDPA backend selection**: PyTorch's auto-selected backend for float additive mask + GQA is the memory-efficient backend, which has no backward implementation for that configuration, causing `RuntimeError: derivative for aten::_scaled_dot_product_efficient_attention_backward is not implemented`. Fixed by explicitly selecting only flash and math backends:
   ```python
   try:
       from torch.nn.attention import sdpa_kernel, SDPBackend
       ctx = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH])
   except ImportError:  # older PyTorch
       ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False)
   with ctx:
       y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)
   ```

#### `EBTBlock(Block)`

Thin subclass of `Block` from `nanochat/gpt.py`. In `__init__`, after calling `super().__init__()`,
replaces `self.attn` with an `EBTSelfAttention` instance (same `layer_idx`, so weight shapes and
`ve_gate` presence are identical). Overrides `forward` to accept and pass through the `attn_mask`
keyword argument.

#### `NanoChatEBT(nn.Module)`

Standalone energy model. Does **not** subclass `GPT` — it builds its own set of `EBTBlock` layers,
RoPE buffers, and scalars, all configured via a `GPTConfig` derived from `EBTModelArgs`.

**Constructor (`__init__(params: EBTModelArgs, max_mcmc_steps: int)`):**

| Component | Detail |
|---|---|
| `self.gpt_config` | `GPTConfig` with `sequence_len = params.max_seq_len * 2 + 4` (covers doubled EBT sequence), `window_pattern = params.window_pattern` |
| `self.blocks` | `nn.ModuleList` of `EBTBlock` (one per layer) |
| `self.resid_lambdas` | `nn.Parameter(ones(n_layers))` — scales residual stream before each block |
| `self.x0_lambdas` | `nn.Parameter(zeros(n_layers))` — blends initial embedding back in at each block |
| `self.value_embeds` | `nn.ModuleDict` of `nn.Embedding(vocab_size, kv_dim)` for alternating layers (same `has_ve` pattern as GPT) |
| `self.window_sizes` | List of `(left, right)` tuples computed by `_compute_window_sizes` |
| `self.cos`, `self.sin` | RoPE buffers, bfloat16, shape `(1, seq_len, 1, head_dim/2)`, non-persistent |
| `self.time_embeddings` | `nn.Embedding(max_mcmc_steps, dim)` — maps MCMC step index to a `[B,1,D]` prefix |
| `self.final_layer` | `nn.Linear(dim, 1, bias=False)` — scalar energy projection |

**`_compute_window_sizes(config)`**: Mirrors `GPT._compute_window_sizes` exactly. Tiles the
`window_pattern` string across layers; maps `"L"` → `(sequence_len, 0)` (full context),
`"S"` → `(sequence_len // 2, 0)` (half context). The final layer is always forced to full context
regardless of the pattern string.

**`_init_weights()`**: Initialises all parameters:
- `c_q/c_k/c_v`: Uniform `±sqrt(3)/sqrt(n_embd)` (same std as Normal, avoids outliers)
- `c_proj`, `mlp.c_proj`: zeros (residual projections start neutral)
- `mlp.c_fc`: Uniform `±sqrt(3)/sqrt(n_embd)`
- `ve_gate`: zeros (gate starts at `sigmoid(0)*2 = 1.0`, neutral blend)
- `value_embeds`: Uniform `±sqrt(3)/sqrt(n_embd)`
- `resid_lambdas`: `1.0` (standard residual connection)
- `x0_lambdas`: `0.1` (small initial skip-connection weight)
- `time_embeddings`: Normal(0, 1.0)
- `final_layer`: Normal(0, 0.001)

Uses `.data.fill_()` for `resid_lambdas` and `x0_lambdas` to avoid in-place op on leaf tensors
with `requires_grad=True`.

**`forward(embeddings, idx=None, start_pos=0, mcmc_step=0)`**:

```
Input:  embeddings [B, 2*(S-1), D]   — real embeddings (first half) + predicted embeddings (second half)
        idx        [B, 2*(S-1)]       — token IDs (optional; enables ResFormer value_embeds)
        start_pos  int                — RoPE offset (always 0; KV cache unused in EBT)
        mcmc_step  int                — selects the time embedding row

Output: energies   [B, S-1, 1]       — scalar energy per predicted position
```

Forward steps:
1. If `idx` is provided, prepend a zero time-placeholder: `full_idx = F.pad(idx, (1,0), value=0)` → `[B, 2*(S-1)+1]`. If `idx=None`, `full_idx=None` and value_embeds are skipped.
2. Look up MCMC step time embedding `[B,1,D]` and prepend to `embeddings` → `x [B, T, D]` where `T = 2*(S-1)+1`.
3. Slice RoPE buffers to length `T`: `cos_sin = (cos[:,T0:T0+T], sin[:,T0:T0+T])`.
4. Build EBT attention mask via `_build_ebt_mask(T, S, device)`, broadcast to `(1, 1, T, T)`.
5. Apply functional RMSNorm to `x`; save `x0 = x`.
6. For each block `i`:
   - `x = resid_lambdas[i] * x + x0_lambdas[i] * x0`
   - Look up value embed `ve = value_embeds[i](full_idx)` if layer `i` has a value embed and `full_idx` is not None; else `ve = None`.
   - `x = block(x, ve, cos_sin, window_sizes[i], kv_cache=None, attn_mask=attn_mask)`
7. Apply final RMSNorm.
8. Remove time-prefix: `x = x[:, 1:]` → `[B, 2*(S-1), D]`.
9. Project: `energies = final_layer(x)` → `[B, 2*(S-1), 1]`.
10. Return second half: `energies[:, x.shape[1]//2:]` → `[B, S-1, 1]`.

---

### 2. `nova/ebt/utils.py` — Two additions

#### `EBTModelArgs` dataclass — two new fields

```python
# Before (line 43):
    weight_initialization_gain: float = 1.0

# After:
    weight_initialization_gain: float = 1.0
    vocab_size: int = None       # required for NanoChatEBT value_embeds
    window_pattern: str = "L"    # "L"=full context (default), "S"=half context; e.g. "SSSL"
```

`vocab_size` is needed by `NanoChatEBT.__init__` to size the `value_embeds` embedding tables.
`window_pattern` is propagated into `GPTConfig` to configure per-layer window sizes.
Default `"L"` means full context — identical to no sliding window, preserving existing behaviour.

#### `setup_ebt()` — new `nanochat_time_embed` branch

Added between the `time_embed` branch and the `else` (adaln) fallback:

```python
elif hparams.ebt_type == "nanochat_time_embed":
    from nanochat_ebt import NanoChatEBT
    transformer_args.window_pattern = hparams.window_pattern
    transformer_args.vocab_size = hparams.vocab_size
    ebt = NanoChatEBT(params=transformer_args, max_mcmc_steps=hparams.mcmc_num_steps)
```

`window_pattern` and `vocab_size` are written into `transformer_args` after construction because
`EBTModelArgs` is built from the generic `hparams` fields shared with all EBT types; these two
fields are `nanochat_time_embed`-specific and not part of that generic construction.

---

### 3. `nova/ebt/train.py` — Two additions

#### `--ebt_type` choices extended

```python
# Before:
choices=["default", "time_embed", "adaln", "adaln_zero", "nanochat_d26"]

# After:
choices=["default", "time_embed", "adaln", "adaln_zero", "nanochat_d26", "nanochat_time_embed"]
```

#### New `--window_pattern` argument

Added immediately after `--ebt_type`:

```python
parser.add_argument(
    "--window_pattern",
    help="Per-layer attention window pattern for NanoChatEBT. L=full context, S=half context. E.g. 'SSSL'.",
    type=str,
    default="L",
)
```

Default `"L"` means no sliding window. Only used when `--ebt_type nanochat_time_embed` is set;
ignored for all other EBT types.

---

### 4. `nova/ebt/modeling_ebt.py` — One addition in the training MCMC loop

In `EBT_NLP._mcmc_step_excluded()`, the call to `self.transformer(...)` now conditionally passes
`idx` when using `nanochat_time_embed`:

```python
# Before (single line):
energy_preds = self.transformer(all_embeddings, start_pos=start_pos, mcmc_step=mcmc_step)

# After:
if self.hparams.ebt_type == "nanochat_time_embed":
    energy_preds = self.transformer(all_embeddings, idx=torch.cat((x, x), dim=1), start_pos=start_pos, mcmc_step=mcmc_step)
else:
    energy_preds = self.transformer(all_embeddings, start_pos=start_pos, mcmc_step=mcmc_step)
```

`x` is the real token ID tensor `[B, S]` available in the enclosing `forward(self, x, ...)` scope.
`torch.cat((x, x), dim=1)` creates a `[B, 2*S]` tensor that covers both the real and predicted
positions in the EBT sequence layout (predicted positions share the same token IDs as their real
counterparts for the purposes of the ResFormer value embedding lookup).

The inference path (`_run_ebt_inference_steps`) is intentionally unchanged: it calls
`self.transformer(combined_embeddings, start_pos=start_pos, mcmc_step=step_idx)` without `idx`,
which maps to `NanoChatEBT.forward(idx=None)` and simply skips the value_embeds lookup. SWA
remains fully active during inference; only the ResFormer contribution is absent.

---

## What Is Unchanged

| File | Status |
|---|---|
| `nanochat/gpt.py` | Unchanged — already has `window_pattern`, `_compute_window_sizes`, SWA via FA3 |
| `nanochat/flash_attention.py` | Unchanged |
| `nanochat/checkpoint_manager.py` | Unchanged — already patches `window_pattern="L"` for old checkpoints |
| `nova/ebt/ar_ebt_default.py` | Unchanged |
| `nova/ebt/ar_ebt_time_embed.py` | Unchanged |
| `nova/ebt/ar_ebt_adaln.py` | Unchanged |
| `nova/ebt/modeling_ebt.py` `_run_ebt_inference_steps` | Unchanged (no function signature change) |

---

## Usage

```bash
# Enable SWA with SSSL pattern (layers 0..N-2 use half-context window, final layer full context)
python train.py \
    --ebt_type nanochat_time_embed \
    --window_pattern SSSL \
    ...

# Full context (equivalent to old behaviour, no SWA)
python train.py \
    --ebt_type nanochat_time_embed \
    --window_pattern L \
    ...
```

### Window pattern semantics

| Character | Window size | Description |
|---|---|---|
| `L` | `sequence_len` | Full causal context (no restriction) |
| `S` | `sequence_len // 2` | Short window (half context) |

The pattern string is tiled across layers. The **final layer always uses `L`** regardless of the
pattern, to ensure global context integration before the energy head.

Example — 6-layer model with `"SSSL"`:
```
Layer:    0    1    2    3    4    5
Pattern:  S    S    S    L    S    S
Override: S    S    S    L    S    L   ← layer 5 forced to L
```

---

## Design Notes

### Why not use FA3 for EBT?

FA3's `causal=True` supports only standard lower-triangular masks. EBT's predicted positions
require a mask where `pred_k` attends only to `{time, real₁..real_k, self}` — this cannot be
expressed as a simple causal mask. SDPA with an explicit float additive mask is the correct
fallback.

### Why summing masks works

Both the EBT mask and the sliding window mask are additive float tensors with values in `{0.0, -inf}`.
Adding them is equivalent to a logical AND on the allowed-position sets:
- `0.0 + 0.0 = 0.0` (both allow → allow)
- `0.0 + (-inf) = -inf` (one blocks → block)
- `(-inf) + 0.0 = -inf` (one blocks → block)
- `(-inf) + (-inf) = -inf` (both block → block)

### Why the memory-efficient SDPA backend must be disabled

PyTorch auto-selects SDPA backends based on input properties. For float additive masks combined
with GQA (`n_head != n_kv_head`), it chooses the memory-efficient backend, which does not
implement `backward` for that configuration. This would silently fail during gradient computation.
Explicitly selecting only `FLASH_ATTENTION` and `MATH` backends avoids this.
