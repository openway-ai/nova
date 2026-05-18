"""Multi-component reward functions for 9x9 Sudoku RL training.

Reuses the existing sudoku_evaluator.py for board parsing and validation.

Reward components:
  1. Format reward (0.0-0.5): parseable 9x9 board?
  2. Clue preservation (0.0-0.5): all given clues unchanged?
  3. Blank accuracy reward (0.0-1.5): fraction of blanks filled correctly × 1.5
  4. Full solve bonus (0.0-0.5): perfect solution (implies validity)

Total reward range: [0.0, 3.0]

Note: We removed the separate 'validity bonus' because:
  - If clues are preserved AND blanks are all correct → full_solve=true → validity is guaranteed
  - If clues are NOT preserved → the solution is invalid regardless of constraints
  - Validity without correctness is not useful (could be a different puzzle's solution)
"""

from typing import List, Optional

from openebm.elm.data.sudoku_evaluator import (
    parse_board_from_text,
    validate_sudoku_board,
)


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

    # 1. Format reward: can we parse a 9x9 board?
    pred_board = parse_board_from_text(completion)
    if pred_board is None:
        return result  # all zeros

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
    # Only count cells that were blank in the original puzzle
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

    result["blank_accuracy"] = blank_accuracy_frac * 1.5

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
