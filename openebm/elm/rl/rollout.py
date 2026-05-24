"""Generation/rollout module for EBM-GRPO.

Generates multiple completions per prompt using the EBT model's autoregressive
generation (reusing logic from generate.py). Returns both token IDs (for log-prob
computation) and decoded text (for reward computation).
"""

from typing import List, Optional

import torch
import torch.nn.functional as F

from openebm.elm.generate import sample_top_p, call_model_forward_decode


def generate_completions(
    model,
    prompt_ids,
    tokenizer,
    hparams,
    num_generations: int = 8,
    max_completion_length: int = 512,
    temperature: float = 0.9,
    top_p: float = 0.95,
    generation_batch_size: int = 4,
    extra_stop_strings: Optional[List[str]] = None,
):
    """Generate multiple completions per prompt.

    Args:
        model: EBT_NLP model (in eval mode, no grad)
        prompt_ids: (num_prompts, prompt_len) tokenized prompts (left-padded)
        tokenizer: nanochat tokenizer for decoding
        hparams: model hparams namespace (needs model_name, context_length, etc.)
        num_generations: completions per prompt
        max_completion_length: max tokens to generate
        temperature: sampling temperature
        top_p: nucleus sampling threshold
        generation_batch_size: sub-batch size for VRAM management

    Returns:
        completion_ids: (num_prompts * num_generations, max_completion_length) token IDs
        completion_texts: list of str, decoded completions
        completion_masks: (num_prompts * num_generations, max_completion_length) binary mask
    """
    num_prompts, prompt_len = prompt_ids.shape
    device = prompt_ids.device

    # Determine pad token: prefer bos (nanochat convention), then eos, then 0.
    # In nanochat, bos == eos token id, but using bos for pad is the documented
    # path (see chat_ebt.py:300-309).
    if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        pad_id = tokenizer.bos_token_id
    elif hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        pad_id = tokenizer.eos_token_id
    else:
        pad_id = 0

    # Determine stop tokens — only semantic stoppers, NOT bos/eos.
    # In nanochat bos == eos == pad, so adding eos_token_id here would stop
    # generation on the very first token. See chat_ebt.py:335-343.
    # CRITICAL: use encode_special (single-token ID) not encode (multi-char text).
    stop_token_ids = set()
    inner_tok = getattr(tokenizer, 'tokenizer', tokenizer)
    encode_special_fn = getattr(inner_tok, 'encode_special', None)
    if encode_special_fn is not None:
        for special in ['<|user_start|>', '<|assistant_end|>']:
            try:
                tid = encode_special_fn(special)
                if tid is not None and isinstance(tid, int):
                    stop_token_ids.add(tid)
            except Exception:
                pass

    # Expand prompts: each prompt repeated num_generations times
    # Shape: (num_prompts * num_generations, prompt_len)
    expanded_prompts = prompt_ids.repeat_interleave(num_generations, dim=0)
    total_seqs = expanded_prompts.shape[0]

    # Allocate output buffers
    total_len = min(hparams.context_length, prompt_len + max_completion_length)
    actual_comp_len = total_len - prompt_len

    all_completion_ids = torch.full(
        (total_seqs, actual_comp_len), pad_id, dtype=torch.long, device=device
    )
    all_completion_masks = torch.zeros(
        (total_seqs, actual_comp_len), dtype=torch.long, device=device
    )

    # Generate in sub-batches
    with torch.no_grad():
        for batch_start in range(0, total_seqs, generation_batch_size):
            batch_end = min(batch_start + generation_batch_size, total_seqs)
            batch_prompts = expanded_prompts[batch_start:batch_end]
            bsz = batch_prompts.shape[0]

            # Build token buffer
            tokens = torch.full(
                (bsz, total_len), pad_id, dtype=torch.long, device=device
            )
            tokens[:, :prompt_len] = batch_prompts

            # Track which positions are real (not pad) in the prompt.
            # Use a POSITION mask, NOT a token-value mask: in nanochat bos==pad,
            # so a legitimate bos token inside the prompt would otherwise be
            # misclassified as "pad" and overwritten by generation.
            input_text_mask = torch.zeros(bsz, total_len, dtype=torch.bool, device=device)
            input_text_mask[:, :prompt_len] = True
            eos_reached = torch.zeros(bsz, dtype=torch.bool, device=device)

            # Autoregressive generation
            for cur_pos in range(prompt_len, total_len):
                input_tokens = tokens[:, :cur_pos]
                logits = call_model_forward_decode(
                    hparams, model, input_tokens, 0, bsz
                )

                if temperature > 0:
                    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                    next_token = sample_top_p(probs, top_p).reshape(-1)
                else:
                    next_token = torch.argmax(logits[:, -1], dim=-1)

                # Only replace if we're past the prompt
                next_token = torch.where(
                    input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
                )
                tokens[:, cur_pos] = next_token

                # Check for stop tokens
                for stop_id in stop_token_ids:
                    eos_reached |= (~input_text_mask[:, cur_pos]) & (
                        next_token == stop_id
                    )

                if eos_reached.all():
                    break

            # Extract completions
            completions = tokens[:, prompt_len:total_len]

            # Build masks (1 for real tokens, 0 after a semantic stop token).
            # Do NOT zero on tok == pad_id: in nanochat bos==pad, and the model
            # may legitimately emit bos tokens mid-completion.
            comp_masks = torch.ones_like(completions, dtype=torch.long)
            for i in range(bsz):
                for pos in range(completions.shape[1]):
                    tok = completions[i, pos].item()
                    if tok in stop_token_ids:
                        comp_masks[i, pos:] = 0
                        break

            all_completion_ids[batch_start:batch_end, :completions.shape[1]] = completions
            all_completion_masks[batch_start:batch_end, :completions.shape[1]] = comp_masks

            # NOTE: do NOT call torch.cuda.empty_cache() here. It forces a
            # full GPU sync + allocator reset on every sub-batch (very
            # expensive on long-completion runs). PyTorch's caching allocator
            # already reuses buffers efficiently across sub-batches.
            del tokens, logits

    # Decode completions to text
    completion_texts = []
    for i in range(total_seqs):
        # Get only the masked (real) tokens
        mask = all_completion_masks[i]
        real_len = mask.sum().item()
        token_list = all_completion_ids[i, :real_len].tolist()
        text = tokenizer.decode(token_list)
        completion_texts.append(text)

    return all_completion_ids, completion_texts, all_completion_masks
