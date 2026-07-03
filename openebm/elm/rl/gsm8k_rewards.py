"""GSM8K reward functions for EBM-GRPO RL.

Reward components:
  1. format_reward     (0.0 or 0.2): completion contains "####" marker
  2. partial_credit    (0.0 or 0.05): any numerical answer can be parsed
  3. answer_proximity  (0.0 to 0.25): wrong parsed answer is numerically close
  4. exact_match       (0.0 or 0.75): extracted number == ground truth
  5. length_penalty    (0.0 to -0.1): penalise very short (<50 chars) gibberish

Total range: [-0.1, 1.0]

Design notes:
  - Keeping the reward spread narrow ([0, 1.0]) avoids the step-size mismatch
    that caused the Sudoku clue-corruption bug (where reward ±0.5 gradient
    overwhelmed the KL anchor).
  - Exact-match is sparse in early GSM8K RL. A fixed parsed-answer bonus alone
    makes whole GRPO groups tie at 0.1 and yields zero advantages. The bounded
    proximity term provides a small, scale-invariant ranking signal for wrong
    numeric answers while keeping exact-match dominant.
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


def _to_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(s.replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def answer_proximity_score(parsed: Optional[str], ground_truth: str) -> float:
    """Return a bounded, scale-invariant closeness score for wrong answers.

    The score is 1.0 for equal numeric values and decays linearly with symmetric
    relative error. It is only used as a small shaping term; exact-match remains
    the main reward.
    """
    pred = _to_float(parsed)
    gt = _to_float(_normalise(ground_truth))
    if pred is None or gt is None:
        return 0.0
    denom = max(abs(pred), abs(gt), 1.0)
    rel_err = abs(pred - gt) / denom
    return max(0.0, 1.0 - min(rel_err, 1.0))


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
    partial_r = 0.05 if parsed is not None else 0.0
    proximity_r = 0.0 if correct else 0.25 * answer_proximity_score(parsed, gt_norm)
    exact_r = 0.75 if correct else 0.0

    # Length penalty: punish extremely short completions (< 50 chars)
    # that are unlikely to contain real reasoning.
    length_r = -0.1 if len(completion.strip()) < 50 else 0.0

    return format_r + partial_r + proximity_r + exact_r + length_r


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
      total, format, partial_credit, answer_proximity, exact_match,
      length_penalty, parsed_answer, is_correct.
    """
    results = []
    for completion, gt in zip(completions, ground_truths):
        has_marker = "####" in completion
        parsed = extract_answer(completion)
        gt_norm = _normalise(gt)
        correct = (parsed is not None) and (parsed == gt_norm)
        length_r = -0.1 if len(completion.strip()) < 50 else 0.0

        format_r = 0.2 if has_marker else 0.0
        partial_r = 0.05 if parsed is not None else 0.0
        proximity_r = 0.0 if correct else 0.25 * answer_proximity_score(parsed, gt_norm)
        exact_r = 0.75 if correct else 0.0
        d = {
            "total": format_r + partial_r + proximity_r + exact_r + length_r,
            "format": format_r,
            "partial_credit": partial_r,
            "answer_proximity": proximity_r,
            "exact_match": exact_r,
            "length_penalty": length_r,
            "parsed_answer": parsed,
            "is_correct": correct,
        }
        results.append(d)
    return results
