"""Sudoku SFT evaluator.

Reports two metrics:

- **board accuracy** — fraction of puzzles where every cell matches the
  ground truth.
- **cell accuracy** — fraction of individual cells that match the ground
  truth, averaged across all puzzles.
"""

import re
from typing import Any, Dict, List, Optional, Sequence


def parse_board_from_text(text: Optional[str]) -> Optional[List[List[int]]]:
    """Extract a 9x9 integer grid from ``text`` if possible.

    Two layouts are supported:

    - a 9-row grid where each row contains at least nine digits, or
    - any text from which 81 consecutive digits can be extracted.

    :param text: Model-generated text.
    :type text: Optional[str]
    :return: A 9x9 ``list[list[int]]``, or ``None`` when parsing fails.
    :rtype: Optional[List[List[int]]]
    """
    if text is None:
        return None

    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if len(lines) >= 9:
        board: List[List[int]] = []
        for line in lines[:9]:
            digits = re.findall(r"\d", line)
            if len(digits) >= 9:
                board.append([int(d) for d in digits[:9]])
        if len(board) == 9:
            return board

    all_digits = re.findall(r"\d", text)
    if len(all_digits) >= 81:
        board = []
        for r in range(9):
            board.append([int(all_digits[r * 9 + c]) for c in range(9)])
        return board

    return None


def evaluate_sudoku(
    predictions: Sequence[str],
    references: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate sudoku predictions against ground truth.

    :param predictions: Model-generated text for each puzzle.
    :type predictions: Sequence[str]
    :param references: Reference dicts; each must contain at least a
        ``solution`` key holding the 9x9 ground-truth grid.
    :type references: Sequence[Dict[str, Any]]
    :return: Dict with the following fields:

        - ``board_accuracy``: fraction of puzzles solved completely correctly.
        - ``cell_accuracy``: fraction of individual cells correct.
        - ``n_samples``: number of evaluated samples.
        - ``n_parsed``: number of predictions successfully parsed as 9x9 grids.
    :rtype: Dict[str, Any]
    :raises AssertionError: If ``predictions`` and ``references`` have
        different lengths.
    """
    assert len(predictions) == len(references), (
        f"Mismatch: {len(predictions)} predictions vs {len(references)} references"
    )

    n_samples = len(predictions)
    n_parsed = 0
    n_board_correct = 0
    total_cells = 0
    correct_cells = 0

    for pred_text, ref in zip(predictions, references):
        solution = ref["solution"]
        pred_board = parse_board_from_text(pred_text)

        if pred_board is None:
            # Parsing failed; count the whole board as incorrect.
            total_cells += 81
            continue

        n_parsed += 1
        board_correct = True
        for r in range(9):
            for c in range(9):
                total_cells += 1
                if pred_board[r][c] == solution[r][c]:
                    correct_cells += 1
                else:
                    board_correct = False

        if board_correct:
            n_board_correct += 1

    return {
        "board_accuracy": n_board_correct / n_samples if n_samples > 0 else 0.0,
        "cell_accuracy": correct_cells / total_cells if total_cells > 0 else 0.0,
        "n_samples": n_samples,
        "n_parsed": n_parsed,
    }


def validate_sudoku_board(board: Optional[List[List[int]]]) -> bool:
    """Check whether ``board`` is a valid sudoku solution.

    A board is valid when every row, column, and 3x3 box contains the digits
    1-9 exactly once.

    :param board: Candidate 9x9 grid, or ``None``.
    :type board: Optional[List[List[int]]]
    :return: ``True`` if the board is a valid completed sudoku,
        ``False`` otherwise.
    :rtype: bool
    """
    if board is None or len(board) != 9:
        return False

    target = set(range(1, 10))

    for row in board:
        if len(row) != 9 or set(row) != target:
            return False

    for c in range(9):
        col = {board[r][c] for r in range(9)}
        if col != target:
            return False

    for box_r in range(3):
        for box_c in range(3):
            box = set()
            for r in range(box_r * 3, box_r * 3 + 3):
                for c in range(box_c * 3, box_c * 3 + 3):
                    box.add(board[r][c])
            if box != target:
                return False

    return True
