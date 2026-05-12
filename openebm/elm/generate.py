"""Text generation and perplexity utilities for EBT / baseline transformers.

Most of the decoding code is adapted from
``https://github.com/meta-llama/llama/blob/main/llama/generation.py#L129``.

.. note::

    The energy-landscape plotting code currently works for fixed-step EBT
    variants; per-landscape plotting for time-embed EBT variants is still
    pending.
"""

import io
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from openebm.elm.tokenizer import AutoTokenizer


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Top-p (nucleus) sampling over a probability distribution.

    Selects the smallest set of tokens whose cumulative probability mass
    exceeds ``p``, then renormalizes over that set and samples.

    :param probs: Probability distribution.
    :type probs: torch.Tensor
    :param p: Nucleus threshold in ``[0, 1]``.
    :type p: float
    :return: Sampled token indices.
    :rtype: torch.Tensor
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def call_model_forward_decode(hparams: Any, model: Any, input_tokens: torch.Tensor, start_pos: int, bsz: int) -> torch.Tensor:
    """Run a forward pass appropriate for decoding.

    KV caching is not yet supported, so ``start_pos`` is forced to ``0``
    inside the call.

    :param hparams: Training/inference hparams (needs ``model_name`` and,
        when ``ebt``, ``infer_ebt_advanced``).
    :type hparams: Any
    :param model: Model to call.
    :type model: Any
    :param input_tokens: Input token ids.
    :type input_tokens: torch.Tensor
    :param start_pos: KV-cache start position (currently unused).
    :type start_pos: int
    :param bsz: Batch size.
    :type bsz: int
    :return: Logits at the final position.
    :rtype: torch.Tensor
    """
    if hparams.model_name == "ebt":
        if hparams.infer_ebt_advanced:
            ebt_outputs = model.ebt_advanced_inference(input_tokens, start_pos = 0, learning = False)
            logits = ebt_outputs[0]
        else:
            ebt_outputs = model.forward(input_tokens, start_pos = 0, learning = False, return_raw_logits = True)
            # EBT returns (logits_list, energy_list) per MCMC step; take the
            # last step's logits. ``learning=False`` avoids gradient tracking.
            logits = ebt_outputs[0][-1]
        energies = ebt_outputs[1]
        # energies[step] has shape (bsz, ...); mean over the trailing dims.
        energies = [energy_tensor.reshape(bsz, -1).mean(dim=1) for energy_tensor in energies]
    else:
        logits = model.forward(input_tokens, start_pos = 0, learning = False, return_raw_logits = True)
    return logits


def call_model_forward_ppl(hparams: Any, model: Any, input_tokens: torch.Tensor, start_pos: int, bsz: int) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
    """Run a forward pass appropriate for perplexity scoring.

    Same KV-cache caveat as :func:`call_model_forward_decode`.

    :param hparams: See :func:`call_model_forward_decode`.
    :type hparams: Any
    :param model: Model to call.
    :type model: Any
    :param input_tokens: Input token ids.
    :type input_tokens: torch.Tensor
    :param start_pos: KV-cache start position (currently unused).
    :type start_pos: int
    :param bsz: Batch size.
    :type bsz: int
    :return: ``(logits, energies)`` — ``energies`` is ``None`` for non-EBT
        models.
    :rtype: Tuple[torch.Tensor, Optional[List[torch.Tensor]]]
    """
    if hparams.model_name == "ebt":
        if hparams.infer_ebt_advanced:
            ebt_outputs = model.ebt_advanced_inference(input_tokens, start_pos = 0, learning = False)
            logits = ebt_outputs[0]
        else:
            ebt_outputs = model.forward(input_tokens, start_pos = 0, learning = False, return_raw_logits = True)
            # Same indexing convention as ``call_model_forward_decode``.
            logits = ebt_outputs[0][-1]
        energies = ebt_outputs[1]
        energies = [energy_tensor.reshape(bsz, -1).mean(dim=1) for energy_tensor in energies]
    else:
        logits = model.forward(input_tokens, start_pos = 0, learning = False, return_raw_logits = True)
        energies = None
    return logits, energies


def _get_tokenizer(hparams: Any) -> Any:
    """Return a cached tokenizer, creating and wrapping it on first use.

    Avoids re-wrapping a :class:`NanoChatTokenizerWrapper` on every batch.

    :param hparams: Hparams object; the tokenizer is cached on it as
        ``_cached_gen_tokenizer``.
    :type hparams: Any
    :return: A HuggingFace-like tokenizer.
    :rtype: Any
    """
    if hasattr(hparams, '_cached_gen_tokenizer'):
        return hparams._cached_gen_tokenizer

    from openebm.elm.nanochat_tokenizer_adapter import NanoChatTokenizerWrapper

    if hasattr(hparams, 'tokenizer_obj') and hparams.tokenizer_obj is not None:
        tokenizer = hparams.tokenizer_obj
        if hasattr(tokenizer, 'enc') and hasattr(tokenizer.enc, 'encode'):
            tokenizer = NanoChatTokenizerWrapper(tokenizer_obj=tokenizer)
    elif hasattr(hparams, 'tokenizer_path'):
        tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_path, clean_up_tokenization_spaces=False)
    else:
        if isinstance(hparams.tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer, clean_up_tokenization_spaces=False)
        else:
            tokenizer = hparams.tokenizer
            if hasattr(tokenizer, 'enc') and hasattr(tokenizer.enc, 'encode'):
                tokenizer = NanoChatTokenizerWrapper(tokenizer_obj=tokenizer)

    hparams._cached_gen_tokenizer = tokenizer
    return tokenizer


def generate_text(model: Any, batch: Tuple[Any, Any], hparams: Any) -> List[Dict[str, Any]]:
    """Generate text for ``(questions, answers)`` and return decoded outputs.

    :param model: Model to decode from (must expose ``transformer.params``).
    :type model: Any
    :param batch: ``(questions, answers)`` pair as emitted by the eval
        dataloader. Both halves may be dict-like or raw tensors.
    :type batch: Tuple[Any, Any]
    :param hparams: Inference hparams (``infer_max_gen_len``, ``infer_temp``,
        ``infer_topp``, ``infer_logprobs``, ``infer_echo``,
        ``context_length``).
    :type hparams: Any
    :return: One dict per sample with keys ``generation``, ``target``,
        ``prompt`` and optionally ``tokens`` / ``logprobs``.
    :rtype: List[Dict[str, Any]]
    """
    tokenizer = _get_tokenizer(hparams)

    # Tokenizers can expose eos / pad / bos in any combination; fall back
    # through them so this works across HF and NanoChat tokenizers.
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        tokenizer_pad_token_id = tokenizer.eos_token_id
    elif hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        tokenizer_pad_token_id = tokenizer.pad_token_id
    elif hasattr(tokenizer, 'bos_token_id'):
        tokenizer_pad_token_id = tokenizer.bos_token_id
    else:
        tokenizer_pad_token_id = 0

    questions, answers = batch

    # Normalize the batch: accept either dict-like inputs (from a
    # DataCollator) or raw tensors. After this block both ``questions`` and
    # ``answers`` are dicts with ``input_ids`` (and ``attention_mask`` for
    # questions).
    if not isinstance(questions, dict) and not hasattr(questions, '__getitem__'):
        ids = questions if isinstance(questions, torch.Tensor) else torch.tensor(questions, dtype=torch.long)
        attn_mask = (ids != tokenizer_pad_token_id).long()
        questions = {'input_ids': ids, 'attention_mask': attn_mask}

    if not isinstance(answers, dict) and not hasattr(answers, '__getitem__'):
        ans_ids = answers if isinstance(answers, torch.Tensor) else torch.tensor(answers, dtype=torch.long)
        answers = {'input_ids': ans_ids}

    ids = questions['input_ids']
    attn_mask = questions["attention_mask"]
    max_gen_len = hparams.infer_max_gen_len
    temperature = hparams.infer_temp
    top_p = hparams.infer_topp
    logprobs = hparams.infer_logprobs
    echo = hparams.infer_echo

    # Use the attention mask to find the true prompt length per row. This
    # avoids a subtle bug with bs > 1 when pad_token_id == eos_token_id and
    # min_prompt_len would otherwise be miscomputed.
    prompt_tokens = []
    for row_ids, row_mask in zip(ids, attn_mask):
        seq_len = row_mask.sum().item()
        prompt_tokens.append(row_ids[:seq_len].tolist())

    params = model.transformer.params
    bsz = len(prompt_tokens)

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= hparams.context_length
    total_len = min(hparams.context_length, max_gen_len + max_prompt_len)
    pad_id = tokenizer_pad_token_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    if logprobs:
        token_logprobs = torch.zeros_like(tokens, dtype=torch.float)
    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz, device="cuda")
    input_text_mask = tokens != pad_id
    with torch.no_grad():
        if min_prompt_len == total_len:
            logits = call_model_forward_decode(hparams, model, tokens, prev_pos, bsz)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )
        for cur_pos in range(min_prompt_len, total_len):
            input_tokens = tokens[:, :cur_pos]
            logits = call_model_forward_decode(hparams, model, input_tokens, prev_pos, bsz)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # Only replace the token when the prompt has already been consumed.
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            # Same fallback sequence as tokenizer_pad_token_id above.
            eos_token_id = getattr(tokenizer, 'eos_token_id', getattr(tokenizer, 'bos_token_id', 0))
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == eos_token_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

    if logprobs:
        token_logprobs = token_logprobs.tolist()
    out_tokens, out_logprobs = [], []
    for i, toks in enumerate(tokens.tolist()):
        start = 0 if echo else len(prompt_tokens[i])
        toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
        probs = None
        if logprobs:
            probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
        eos_token_id = getattr(tokenizer, 'eos_token_id', getattr(tokenizer, 'bos_token_id', 0))
        if eos_token_id in toks:
            eos_idx = toks.index(eos_token_id)
            toks = toks[:eos_idx]
            probs = probs[:eos_idx] if logprobs else None
        out_tokens.append(toks)
        out_logprobs.append(probs)

    if logprobs:
        return [
            {
                "generation": tokenizer.decode(t, skip_special_tokens=True),
                "tokens": [tokenizer.decode(x) for x in t],
                "logprobs": logprobs_i,
                "target": tokenizer.decode(gt_ans, skip_special_tokens=True),
                "prompt": tokenizer.decode(question, skip_special_tokens=True),
            }
            for t, logprobs_i, gt_ans, question in zip(out_tokens, out_logprobs, answers['input_ids'], questions['input_ids'])
        ]
    return [
        {
            "generation": tokenizer.decode(t, skip_special_tokens=True),
            "target": tokenizer.decode(gt_ans, skip_special_tokens=True),
            "prompt": tokenizer.decode(question, skip_special_tokens=True),
        } for t, gt_ans, question in zip(out_tokens, answers['input_ids'], questions['input_ids'])
    ]


def get_ppl(model: Any, batch: Dict[str, torch.Tensor], hparams: Any) -> Dict[str, Any]:
    """Compute perplexity and optionally plot energy-landscape diagnostics.

    Similar to ``model.forward_loss_wrapper`` but skips the list-of-logits
    path and avoids token-by-token ``inference_mode`` for sequence-level PPL.

    :param model: Model to score with.
    :type model: Any
    :param batch: Dict with ``input_ids`` of shape ``[B, 1, S]``.
    :type batch: Dict[str, torch.Tensor]
    :param hparams: Hparams (``model_name``, ``infer_plot_energy_landscape``).
    :type hparams: Any
    :return: Dict with ``loss``, ``perplexity`` and, for EBT models,
        per-MCMC-step mean energies.
    :rtype: Dict[str, Any]
    :raises AssertionError: If ``infer_plot_energy_landscape`` is set for
        non-EBT models.
    """
    batch_size = batch['input_ids'].shape[0]
    # EBT models may enable grad internally even inside this ``no_grad`` block.
    with torch.no_grad():
        input_ids = batch['input_ids'].squeeze(dim=1)[:, :-1]
        if hparams.model_name == "ebt":
            logits, energies = call_model_forward_ppl(hparams, model, input_ids, 0, batch_size)
        else:
            logits, _ = call_model_forward_ppl(hparams, model, input_ids, 0, batch_size)

    # Targets must be 1-D for cross entropy.
    next_token_indices = batch['input_ids'].squeeze(dim=1)[:, 1:].reshape(-1)

    # NanoChat has no pad token; fall back to -100 (standard ignore index).
    pad_token_id = model.tokenizer_pad_token_id
    if pad_token_id is None:
        pad_token_id = -100

    cce_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), next_token_indices, ignore_index=pad_token_id)
    perplexity = torch.exp(cce_loss).detach()

    outputs = {
        'loss': cce_loss,
        'perplexity': perplexity
    }

    if hparams.model_name == "ebt":
        energy_tensors = []
        for step_energies in zip(*[energies]):
            step_tensor = torch.stack(step_energies)
            avg_step_energy = torch.mean(step_tensor, dim=0)
            energy_tensors.append(avg_step_energy)

        for step_idx, energy in enumerate(energy_tensors):
            if energy.numel() > 1:
                energy_scalar = energy.mean().item()
            else:
                energy_scalar = energy.item()
            outputs[f"mcmc_step_{step_idx}_energy"] = energy_scalar

    if hparams.infer_plot_energy_landscape:
        assert hparams.model_name == "ebt", "Energy landscape plotting only works with EBT models"

        _, energies_list, predicted_tokens_list = model.ebt_advanced_inference(input_ids, start_pos=0, learning=False)

        image_tensors = {}

        # Plot all MCMC steps (first, intermediate, last).
        num_steps = len(predicted_tokens_list)
        steps_to_plot = list(range(num_steps))

        for step_idx in steps_to_plot:
            token_losses = []
            step_energies = []

            target_tokens = batch['input_ids'].squeeze(dim=1)[:, 1:]

            for batch_idx in range(batch_size):
                for pos_idx in range(input_ids.shape[1]):
                    if pos_idx >= target_tokens.shape[1] or target_tokens[batch_idx, pos_idx] == model.tokenizer_pad_token_id:
                        continue

                    target_token = target_tokens[batch_idx, pos_idx]

                    token_logit = predicted_tokens_list[step_idx][batch_idx, pos_idx]

                    token_loss = F.cross_entropy(
                        token_logit.unsqueeze(0),
                        target_token.unsqueeze(0),
                        reduction='none'
                    ).item()

                    token_energy = energies_list[step_idx][batch_idx, pos_idx].item()

                    token_losses.append(token_loss)
                    step_energies.append(token_energy)

            plt.figure(figsize=(10, 6))
            plt.scatter(step_energies, token_losses, alpha=0.5)
            plt.xlabel('Predicted Energy')
            plt.ylabel('Ground Truth Cross-Entropy Loss')
            plt.title(f'Energy Landscape vs Ground Truth Loss (MCMC Step {step_idx})')

            if len(step_energies) > 5:
                z = np.polyfit(step_energies, token_losses, 1)
                p = np.poly1d(z)
                plt.plot(sorted(step_energies), p(sorted(step_energies)), "r--", alpha=0.8)

                from scipy.stats import pearsonr
                corr, _ = pearsonr(step_energies, token_losses)
                plt.annotate(f"Correlation: {corr:.3f}", xy=(0.05, 0.95), xycoords='axes fraction')

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()

        # Combined plot showing the evolution across MCMC steps.
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']

        for i, step_idx in enumerate(steps_to_plot):
            token_losses = []
            step_energies = []

            for batch_idx in range(batch_size):
                for pos_idx in range(input_ids.shape[1]):
                    if pos_idx >= target_tokens.shape[1] or target_tokens[batch_idx, pos_idx] == model.tokenizer_pad_token_id:
                        continue

                    target_token = target_tokens[batch_idx, pos_idx]
                    token_logit = predicted_tokens_list[step_idx][batch_idx, pos_idx]

                    token_loss = F.cross_entropy(
                        token_logit.unsqueeze(0),
                        target_token.unsqueeze(0),
                        reduction='none'
                    ).item()

                    token_energy = energies_list[step_idx][batch_idx, pos_idx].item()

                    token_losses.append(token_loss)
                    step_energies.append(token_energy)

            color_idx = i % len(colors)
            plt.scatter(step_energies, token_losses, alpha=0.5, color=colors[color_idx],
                        label=f'Step {step_idx}')

        plt.xlabel('Predicted Energy')
        plt.ylabel('Ground Truth Cross-Entropy Loss')
        plt.title('Energy Landscape Evolution During MCMC Steps')
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

    return outputs
