"""EBT log-probability and energy estimation for GRPO.

Provides three gradient mechanisms for RL:
  1. compute_sequence_energy: First-order energy for Energy-GSPO/REINFORCE
  2. get_per_token_logps: Token-level logprobs via MCMC (legacy, Hessian-unstable)
"""

import torch
import torch.nn.functional as F


def compute_sequence_energy(model, input_ids, prompt_length, completion_mask=None):
    """Compute per-sequence energy for known completions (no MCMC iteration).

    Sets predicted_tokens to one-hot of actual targets, runs transformer once.
    Fully first-order differentiable — no autograd.grad, no Hessian.

    Args:
        model: EBT_NLP model instance
        input_ids: (B, S) full sequence (prompt + completion)
        prompt_length: int, number of prompt tokens
        completion_mask: optional (B, completion_length) binary mask. When
            provided, only real generated completion tokens contribute to the
            sequence mean. This prevents semantic-stop/padding tail tokens from
            polluting GSPO ratios and the reference energy anchor.

    Returns:
        energies: (B,) mean energy per sequence (completion positions only)
    """
    B, S = input_ids.shape
    model_input = input_ids[:, :-1]  # (B, S-1)
    targets = input_ids[:, 1:]       # (B, S-1)

    real_embeddings = model.embeddings(model_input)

    if getattr(model, "use_free_embedding_mcmc", False):
        predicted_tokens = model.embeddings(targets)
        predicted_embeddings = predicted_tokens
    else:
        one_hot_targets = F.one_hot(targets, model.vocab_size).float()
        predicted_tokens = one_hot_targets
        if getattr(model.hparams, 'vocab_to_embed_uses_prob_dist', False):
            predicted_embeddings = torch.matmul(one_hot_targets, model.embeddings.weight)
        else:
            predicted_embeddings = model.vocab_to_embed(one_hot_targets)

    all_embeddings = torch.cat([real_embeddings.detach(), predicted_embeddings], dim=1)

    transformer = getattr(model, 'transformer_eager', model.transformer)
    energy_preds = transformer(
        all_embeddings, start_pos=0, mcmc_step=0,
        real_token_ids=model_input, predicted_tokens=predicted_tokens,
    )

    # ── Debug: log transformer output shape and stats (first call only) ──
    if not getattr(model, '_dbg_energy_transformer_logged', False):
        model._dbg_energy_transformer_logged = True
        import os as _os
        _rank = _os.environ.get('LOCAL_RANK', '0')
        if _rank == '0':
            with torch.no_grad():
                print(
                    f"[DBG-ENERGY] transformer raw output: shape={tuple(energy_preds.shape)} "
                    f"dtype={energy_preds.dtype} "
                    f"min={energy_preds.min().item():.4f} max={energy_preds.max().item():.4f} "
                    f"any_nan={torch.isnan(energy_preds).any().item()} "
                    f"any_inf={torch.isinf(energy_preds).any().item()} "
                    f"requires_grad={energy_preds.requires_grad}",
                    flush=True,
                )
                print(
                    f"[DBG-ENERGY] input shapes: B={B} S={S} prompt_len={prompt_length} "
                    f"all_embeddings={tuple(all_embeddings.shape)} "
                    f"predicted_embeddings.requires_grad={predicted_embeddings.requires_grad}",
                    flush=True,
                )

    energy_preds = energy_preds.float().reshape(B, -1)  # (B, S-1) — transformer returns predicted-half only

    # ── Debug: check reshape result ──
    if not getattr(model, '_dbg_energy_reshape_logged', False):
        model._dbg_energy_reshape_logged = True
        import os as _os
        _rank = _os.environ.get('LOCAL_RANK', '0')
        if _rank == '0':
            seq_len = S - 1
            actual_cols = energy_preds.shape[1]
            print(
                f"[DBG-ENERGY] after reshape: shape={tuple(energy_preds.shape)} "
                f"seq_len(S-1)={seq_len} actual_cols={actual_cols} "
                f"match={seq_len == actual_cols}",
                flush=True,
            )

    # The transformer energy head outputs one scalar per position in the
    # predicted half only (not the real half). So energy_preds is (B, S-1),
    # aligned with targets[0..S-2]. Slice to completion-only positions.
    comp_start = prompt_length - 1
    comp_energy = energy_preds[:, comp_start:]  # (B, comp_len)

    if completion_mask is None:
        return comp_energy.mean(dim=1)  # (B,)

    mask = completion_mask.to(device=comp_energy.device, dtype=comp_energy.dtype)
    if mask.shape[0] != comp_energy.shape[0]:
        raise ValueError(
            f"completion_mask batch mismatch: {tuple(mask.shape)} vs energy {tuple(comp_energy.shape)}"
        )
    if mask.shape[1] != comp_energy.shape[1]:
        common_len = min(mask.shape[1], comp_energy.shape[1])
        mask = mask[:, :common_len]
        comp_energy = comp_energy[:, :common_len]

    denom = mask.sum(dim=1).clamp(min=1.0)
    return (comp_energy * mask).sum(dim=1) / denom  # (B,)


def get_per_token_logps(model, input_ids, prompt_length, learning=False):
    """Compute per-token log-probabilities via MCMC chain (legacy path).

    WARNING: With learning=True, create_graph=True produces NaN gradients
    for EBT's architecture. Use compute_sequence_energy for RL instead.

    Returns:
        (completion_logps, None): tuple for API compatibility
    """
    B, S = input_ids.shape
    model_input = input_ids[:, :-1]
    targets = input_ids[:, 1:]

    with torch.set_grad_enabled(learning), torch.amp.autocast('cuda', enabled=False):
        predicted_distributions, _ = model.forward(
            model_input, start_pos=0, learning=learning,
            return_raw_logits=True, no_randomness=True,
        )

    final_logits = predicted_distributions[-1]
    flat_logits = final_logits.reshape(-1, final_logits.shape[-1]).clamp(-100.0, 100.0)
    flat_targets = targets.reshape(-1)
    per_token_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')
    per_token_logps = -per_token_loss.view(B, S - 1)

    completion_start = prompt_length - 1
    return per_token_logps[:, completion_start:], None
