"""EBT log-probability estimation for GRPO.

Computes per-token log-probabilities by running the EBT forward pass (MCMC chain)
and taking cross-entropy of the final predicted distribution against actual tokens.

This is analogous to d1's mask-then-predict approach for masked diffusion models,
but uses EBT's natural next-token prediction through iterative energy minimization.
"""

import torch
import torch.nn.functional as F


def get_per_token_logps(model, input_ids, prompt_length, learning=True):
    """Compute per-token log-probabilities for completion tokens.

    Algorithm:
      1. Split input_ids into model_input (all but last token) and targets (all but first)
      2. Run EBT forward pass (MCMC chain) on model_input
      3. Take final MCMC step logits
      4. Compute per-token log-probs via cross-entropy against targets
      5. Return only completion positions

    Args:
        model: EBT_NLP model instance
        input_ids: (B, S) full sequence (prompt + completion)
        prompt_length: int, number of prompt tokens
        learning: if True, gradients flow through the MCMC chain

    Returns:
        per_token_logps: (B, completion_length) log-probabilities for each completion token
    """
    B, S = input_ids.shape

    # Standard next-token prediction framing
    model_input = input_ids[:, :-1]   # (B, S-1)
    targets = input_ids[:, 1:]        # (B, S-1)

    # Run EBT forward — returns (list_of_logits_per_mcmc_step, list_of_energies)
    # With return_raw_logits=True, each element in list_of_logits is (B, S-1, V)
    # Always enable the outer grad-enabled context: this keeps the LAST MCMC
    # step's transformer params in the autograd graph, so cross_entropy below
    # can backprop into them. The `learning` flag is now ONLY used to control
    # `create_graph` inside `_mcmc_step_excluded` (i.e., whether MCMC builds a
    # 2nd-order graph). Callers wrap with @torch.no_grad() if they want detach
    # (e.g. old_logps / ref_logps in _generate_and_score).
    with torch.set_grad_enabled(True):
        predicted_distributions, _ = model.forward(
            model_input,
            start_pos=0,
            learning=learning,
            return_raw_logits=True,
            no_randomness=True,
        )

    # Take the final MCMC step's logits
    final_logits = predicted_distributions[-1]  # (B, S-1, V)

    # Compute per-token log-probs: -cross_entropy(logits, targets)
    # Reshape for cross_entropy: (B*(S-1), V) vs (B*(S-1),)
    flat_logits = final_logits.reshape(-1, final_logits.shape[-1])
    flat_targets = targets.reshape(-1)
    per_token_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')
    per_token_logps = -per_token_loss.view(B, S - 1)

    # Return only completion positions
    # prompt_length tokens in input → prompt_length-1 positions in the shifted targets
    # Completion starts at position (prompt_length - 1) in the targets tensor
    completion_start = prompt_length - 1
    completion_logps = per_token_logps[:, completion_start:]

    return completion_logps


def get_per_token_logps_batch(model, input_ids, prompt_lengths, learning=True):
    """Batch version that handles variable prompt lengths via padding.

    For simplicity in GRPO where all prompts in a group have the same length,
    this just calls get_per_token_logps with a single prompt_length.

    Args:
        model: EBT_NLP model instance
        input_ids: (B, S) padded sequences
        prompt_lengths: int or (B,) tensor of prompt lengths
        learning: gradient mode

    Returns:
        per_token_logps: (B, max_completion_length) log-probs, padded with 0
    """
    if isinstance(prompt_lengths, int):
        return get_per_token_logps(model, input_ids, prompt_lengths, learning=learning)

    # Variable prompt lengths — process each sample individually
    # (This path is less efficient but handles edge cases)
    B, S = input_ids.shape
    max_comp_len = S - prompt_lengths.min().item()
    all_logps = torch.zeros(B, max_comp_len, device=input_ids.device, dtype=torch.float32)

    for i in range(B):
        pl = prompt_lengths[i].item()
        seq = input_ids[i:i+1, :pl + (S - pl)]  # just the non-padded part
        logps = get_per_token_logps(model, seq, pl, learning=learning)
        comp_len = logps.shape[1]
        all_logps[i, :comp_len] = logps[0]

    return all_logps
