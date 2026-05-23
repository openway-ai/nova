"""Multi-component reward functions for 9x9 Sudoku RL training.

Reuses the existing sudoku_evaluator.py for board parsing and validation.

Reward components:
  1. Format reward (0.0-0.5): parseable 9x9 board?
     - If a full 9x9 board parses cleanly:      0.5
     - If completion contains >=40 ASCII digits: up to 0.1 (partial credit,
       prevents reward variance from collapsing to 0 when the model is
       still emitting reasonable token streams but mis-formatted output).
  2. Clue preservation (0.0-0.5): all given clues unchanged?
  3. Blank accuracy reward (0.0-1.5): blank fill quality. Even when no
     blanks are correct, a tiny "effort floor" (0.02) is granted so reward
     std doesn't degenerate to 0 on a bad batch.
  4. Full solve bonus (0.0-0.5): perfect solution (implies validity)

Total reward range: [0.0, 3.0] (effort floors keep it just above 0 when the
completion at least looks like sudoku output).
"""

from typing import List, Optional
import re

from openebm.elm.data.sudoku_evaluator import (
    parse_board_from_text,
    validate_sudoku_board,
)


_DIGIT_RE = re.compile(r"[0-9]")


def compute_sudoku_rewards(
    completions: List[str],
    puzzles: List[str],
    solutions: List[List[List[int]]],
) -> List[float]:
    """Compute multi-component rewards for a batch of sudoku completions.

    Args:
        completions: list of model-generated text strings
        puzzles: list of 81-char puzzle strings (0 = blank)
        solutions: list of 9x9 ground-truth solution grids

    Returns:
        list of float rewards in [0.0, 3.0]
    """
    rewards = []
    for completion, puzzle, solution in zip(completions, puzzles, solutions):
        reward = _score_single(completion, puzzle, solution)
        rewards.append(reward)
    return rewards


def compute_sudoku_rewards_detailed(
    completions: List[str],
    puzzles: List[str],
    solutions: List[List[List[int]]],
) -> List[dict]:
    """Like compute_sudoku_rewards but returns per-component breakdown.

    Returns:
        list of dicts with keys: total, format, clue_preservation, blank_accuracy, full_solve
    """
    results = []
    for completion, puzzle, solution in zip(completions, puzzles, solutions):
        detail = _score_single_detailed(completion, puzzle, solution)
        results.append(detail)
    return results


def _score_single(completion: str, puzzle: str, solution: List[List[int]]) -> float:
    """Score a single completion."""
    return _score_single_detailed(completion, puzzle, solution)["total"]


def _score_single_detailed(
    completion: str, puzzle: str, solution: List[List[int]]
) -> dict:
    """Score a single completion with per-component breakdown."""
    result = {
        "total": 0.0,
        "format": 0.0,
        "clue_preservation": 0.0,
        "blank_accuracy": 0.0,
        "full_solve": 0.0,
    }

    # 1. Format reward
    pred_board = parse_board_from_text(completion)
    if pred_board is None:
        # Partial credit: completion at least contains lots of digits (shape
        # the model is trying to produce sudoku output). Capped at 0.1 to
        # stay well below the 0.5 awarded for clean parse.
        n_digits = len(_DIGIT_RE.findall(completion or ""))
        if n_digits >= 40:
            result["format"] = min(0.1, 0.1 * n_digits / 81.0)
            result["total"] = result["format"]
        return result

    result["format"] = 0.5

    # 2. Clue preservation: did the model keep all given clues unchanged?
    clue_count = 0
    clue_preserved_count = 0

    for r in range(9):
        for c in range(9):
            puzzle_idx = r * 9 + c
            if puzzle_idx < len(puzzle) and puzzle[puzzle_idx] != '0':
                clue_count += 1
                clue_value = int(puzzle[puzzle_idx])
                if pred_board[r][c] == clue_value:
                    clue_preserved_count += 1

    if clue_count > 0:
        clue_preservation_frac = clue_preserved_count / clue_count
        result["clue_preservation"] = clue_preservation_frac * 0.5
    else:
        # No clues (empty board) - give full credit
        result["clue_preservation"] = 0.5

    # 3. Blank accuracy reward: fraction of blank cells filled correctly
    # Only count cells that were blank in the original puzzle. A small
    # "effort floor" is added so that even all-wrong fills still produce
    # >0 reward — prevents reward_std → 0 when the entire batch is bad.
    blank_count = 0
    correct_blank_count = 0

    for r in range(9):
        for c in range(9):
            puzzle_idx = r * 9 + c
            if puzzle_idx < len(puzzle) and puzzle[puzzle_idx] == '0':
                blank_count += 1
                if pred_board[r][c] == solution[r][c]:
                    correct_blank_count += 1

    if blank_count > 0:
        blank_accuracy_frac = correct_blank_count / blank_count
    else:
        blank_accuracy_frac = 1.0  # no blanks = trivially correct

    # Effort floor: parseable board with any digit fill gets 0.02 baseline,
    # then scaled accuracy on top. This keeps the gradient signal alive
    # when a batch has uniformly poor accuracy.
    result["blank_accuracy"] = 0.02 + blank_accuracy_frac * 1.48

    # 4. Full solve bonus: perfect match with ground truth
    # This implicitly checks validity (correct solution must satisfy constraints)
    fully_correct = all(
        pred_board[r][c] == solution[r][c]
        for r in range(9)
        for c in range(9)
    )
    if fully_correct:
        result["full_solve"] = 0.5

    result["total"] = (
        result["format"]
        + result["clue_preservation"]
        + result["blank_accuracy"]
        + result["full_solve"]
    )
    return result
