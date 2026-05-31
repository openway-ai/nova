"""GSM8K reward functions for EBM-GRPO RL.

Reward components:
  1. format_reward  (0.0 or 0.2): completion contains "####" marker
  2. partial_credit (0.0 or 0.1): any numerical answer can be parsed
  3. exact_match    (0.0 or 0.7): extracted number == ground truth
  4. length_penalty (0.0 to -0.1): penalise very short (<50 chars) gibberish

Total range: [-0.1, 1.0]

Design notes:
  - Keeping the reward spread narrow ([0, 1.0]) avoids the step-size mismatch
    that caused the Sudoku clue-corruption bug (where reward ±0.5 gradient
    overwhelmed the KL anchor). The format partial credit prevents reward_std
    from collapsing to 0 when the model is just learning the output format.
  - No scaling by answer magnitude (e.g. percentage error) — GSM8K answers
    span 1–1e6; normalising by magnitude would create inconsistent gradients.
"""

import re
from typing import List, Optional

_HASH_RE = re.compile(r"####\s*(-?[\d,\.]+)")

# Flexible digit-answer patterns for lenient parsing when #### is missing.
_LOOSE_RE = re.compile(
    r"(?:answer\s+is|=|:\s*)\s*(?:\$\s*)?(-?[\d,\.]+)",
    re.IGNORECASE,
)


def _normalise(s: str) -> str:
    """Strip commas and trailing zeros: '1,024.00' -> '1024'."""
    s = s.replace(",", "").strip()
    try:
        f = float(s)
        # If integer-valued, drop the decimal part
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return s


def extract_answer(completion: str) -> Optional[str]:
    """Extract numerical answer from model completion.

    1. Look for '#### <number>' (official format).
    2. Fall back to loose patterns ('the answer is X', '= X').
    Returns None if nothing found.
    """
    m = _HASH_RE.search(completion)
    if m:
        return _normalise(m.group(1))
    m = _LOOSE_RE.search(completion)
    if m:
        return _normalise(m.group(1))
    return None


def compute_gsm8k_reward(
    completion: str,
    ground_truth: str,
) -> float:
    """Score a single model completion against the ground-truth answer string.

    Returns a float in [-0.1, 1.0].
    """
    has_marker = "####" in completion
    parsed = extract_answer(completion)
    gt_norm = _normalise(ground_truth)

    correct = (parsed is not None) and (parsed == gt_norm)

    format_r = 0.2 if has_marker else 0.0
    partial_r = 0.1 if parsed is not None else 0.0
    exact_r = 0.7 if correct else 0.0

    # Length penalty: punish extremely short completions (< 50 chars)
    # that are unlikely to contain real reasoning.
    length_r = -0.1 if len(completion.strip()) < 50 else 0.0

    return format_r + partial_r + exact_r + length_r


def compute_gsm8k_rewards(
    completions: List[str],
    ground_truths: List[str],
) -> List[float]:
    """Batch wrapper for compute_gsm8k_reward."""
    return [
        compute_gsm8k_reward(c, g)
        for c, g in zip(completions, ground_truths)
    ]


def compute_gsm8k_rewards_detailed(
    completions: List[str],
    ground_truths: List[str],
) -> List[dict]:
    """Like compute_gsm8k_rewards but returns per-component breakdown.

    Returns list of dicts with keys:
      total, format, exact_match, length_penalty, parsed_answer, is_correct.
    """
    results = []
    for completion, gt in zip(completions, ground_truths):
        has_marker = "####" in completion
        parsed = extract_answer(completion)
        gt_norm = _normalise(gt)
        correct = (parsed is not None) and (parsed == gt_norm)
        length_r = -0.1 if len(completion.strip()) < 50 else 0.0

        format_r = 0.2 if has_marker else 0.0
        partial_r = 0.1 if parsed is not None else 0.0
        exact_r = 0.7 if correct else 0.0
        d = {
            "total": format_r + partial_r + exact_r + length_r,
            "format": format_r,
            "partial_credit": partial_r,
            "exact_match": exact_r,
            "length_penalty": length_r,
            "parsed_answer": parsed,
            "is_correct": correct,
        }
        results.append(d)
    return results
