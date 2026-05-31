"""Multi-component reward functions for 9x9 Sudoku RL training.

Reuses the existing sudoku_evaluator.py for board parsing and validation.

Reward components:
  1. Format reward (0.0-0.5): parseable 9x9 board?
     - If a full 9x9 board parses cleanly:      0.5
     - If completion contains >=40 ASCII digits: up to 0.1 (partial credit,
       prevents reward variance from collapsing to 0 when the model is
       still emitting reasonable token streams but mis-formatted output).
  2. Clue preservation (-0.5..+0.5): SYMMETRIC penalty/reward.
     full preservation → +0.5; full corruption → −0.5.
     v3: changed from one-sided (0..0.5) after v2 showed clue dropping
     0.34 → 0.07 because corruption had no explicit cost.
  3. Blank accuracy reward (0.0-1.0): blank fill quality. This is gated on
     preserving every given clue; otherwise the model can get positive reward
     by corrupting clues and locally matching blank cells.
  4. Constraint validity reward (0.0-0.4): row/column/box consistency, multiplied
     by blank fill ratio to avoid rewarding conservative all-zero boards.
  5. Full solve bonus (0.0-0.6): perfect solution (implies validity)

Total reward range: [-0.5, 3.0]. If any clue is corrupted, total reward is
capped at format + clue_preservation and blank/full-solve credit is disabled.
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
        list of dicts with reward components plus diagnostic metrics such as
        parse_ok, clue_accuracy, blank_accuracy_frac, valid_sudoku, and solved.
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
        "constraint_validity": 0.0,
        "full_solve": 0.0,
        "parse_ok": False,
        "valid_sudoku": False,
        "solved": False,
        "clue_count": 0,
        "clue_correct": 0,
        "clue_accuracy": 0.0,
        "blank_count": 0,
        "blank_correct": 0,
        "blank_accuracy_frac": 0.0,
        "filled_blank_count": 0,
        "blank_fill_ratio": 0.0,
        "constraint_validity_frac": 0.0,
        "all_clues_preserved": False,
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
        result["digit_count"] = n_digits
        return result

    result["parse_ok"] = True
    result["format"] = 0.5
    result["valid_sudoku"] = bool(validate_sudoku_board(pred_board))

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

    result["clue_count"] = clue_count
    result["clue_correct"] = clue_preserved_count

    if clue_count > 0:
        clue_preservation_frac = clue_preserved_count / clue_count
        result["clue_accuracy"] = clue_preservation_frac
        # SYMMETRIC: full preserve → +0.5, full corrupt → −0.5.
        # v3 fix: v2 had (frac * 0.5) which lets the model corrupt clues for
        # free if blank gain exceeds preservation cost. v3 makes corruption
        # strictly punished, breaking the cheat path observed in 20260524 log.
        result["clue_preservation"] = (2.0 * clue_preservation_frac - 1.0) * 0.5
        all_clues_preserved = (clue_preserved_count == clue_count)
    else:
        # No clues (empty board) - give full credit
        result["clue_preservation"] = 0.5
        result["clue_accuracy"] = 1.0
        all_clues_preserved = True
    result["all_clues_preserved"] = all_clues_preserved

    # Stability guard: Sudoku givens are hard constraints, not soft hints.
    # The 20260529 GSPO run showed a reward-hacking path where the model kept
    # parseable format, corrupted clues, and still earned positive blank-cell
    # reward. Do not award blank/full-solve credit unless every given clue is
    # preserved.
    if not all_clues_preserved:
        result["total"] = result["format"] + result["clue_preservation"]
        return result

    # 3. Blank accuracy reward: fraction of blank cells filled correctly
    # Only count cells that were blank in the original puzzle.
    blank_count = 0
    correct_blank_count = 0
    filled_blank_count = 0

    for r in range(9):
        for c in range(9):
            puzzle_idx = r * 9 + c
            if puzzle_idx < len(puzzle) and puzzle[puzzle_idx] == '0':
                blank_count += 1
                if pred_board[r][c] != 0:
                    filled_blank_count += 1
                if pred_board[r][c] == solution[r][c]:
                    correct_blank_count += 1

    if blank_count > 0:
        blank_accuracy_frac = correct_blank_count / blank_count
        blank_fill_ratio = filled_blank_count / blank_count
    else:
        blank_accuracy_frac = 1.0  # no blanks = trivially correct
        blank_fill_ratio = 1.0
    result["blank_count"] = blank_count
    result["blank_correct"] = correct_blank_count
    result["filled_blank_count"] = filled_blank_count
    result["blank_accuracy_frac"] = blank_accuracy_frac
    result["blank_fill_ratio"] = blank_fill_ratio

    result["blank_accuracy"] = blank_accuracy_frac * 1.0

    # 4. Partial structural validity over 27 Sudoku constraints. Scale it by
    # fill ratio so all-zero/under-filled boards cannot receive high reward.
    constraint_validity_frac = _constraint_validity_fraction(pred_board)
    result["constraint_validity_frac"] = constraint_validity_frac
    result["constraint_validity"] = constraint_validity_frac * blank_fill_ratio * 0.4

    # 5. Full solve bonus: perfect match with ground truth
    # This implicitly checks validity (correct solution must satisfy constraints)
    fully_correct = all(
        pred_board[r][c] == solution[r][c]
        for r in range(9)
        for c in range(9)
    )
    if fully_correct:
        result["full_solve"] = 0.6
    result["solved"] = bool(fully_correct)

    result["total"] = (
        result["format"]
        + result["clue_preservation"]
        + result["blank_accuracy"]
        + result["constraint_validity"]
        + result["full_solve"]
    )
    return result


def _constraint_validity_fraction(board: List[List[int]]) -> float:
    """Fraction of row/column/box constraints with no duplicate non-zero digits."""
    valid = 0
    total = 0

    def no_duplicate_nonzero(values: List[int]) -> bool:
        digits = [v for v in values if v != 0]
        return len(digits) == len(set(digits))

    for r in range(9):
        total += 1
        valid += int(no_duplicate_nonzero(board[r]))

    for c in range(9):
        total += 1
        valid += int(no_duplicate_nonzero([board[r][c] for r in range(9)]))

    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            total += 1
            box = [
                board[r][c]
                for r in range(br, br + 3)
                for c in range(bc, bc + 3)
            ]
            valid += int(no_duplicate_nonzero(box))

    return valid / total if total else 0.0
