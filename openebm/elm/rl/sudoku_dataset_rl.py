"""RL-specific Sudoku dataset that provides prompts (not full sequences).

Reuses puzzle loading and augmentation from sudoku_dataset_v2.py.
Each item yields a prompt (up to the assistant turn) plus metadata for reward computation.
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset

from nanochat.common import get_dist_info

from openebm.elm.data.sudoku_dataset_v2 import (
    DEFAULT_DATA_DIR,
    PROMPT_TEMPLATES,
    _format_board,
    _pick_format,
    _load_sudoku_split_v2,
    augment_board,
)
from openebm.elm.rl.data_sharding import build_rank_worker_indices


class SudokuRLPromptDataset(IterableDataset):
    """Dataset that yields sudoku prompts for RL rollout generation.

    Each item is a dict with:
      - prompt_text: str, the full conversation up to assistant turn
      - puzzle: str, 81-char puzzle string (for reward computation)
      - solution: list[list[int]], 9x9 ground truth
      - difficulty: str, difficulty bucket name
      - num_givens: int, number of given digits
    """

    def __init__(
        self,
        tokenizer,
        max_prompt_length: int = 256,
        split: str = "train",
        data_dir: str = None,
        augment: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.split = split
        self.data_dir = data_dir or os.environ.get("SUDOKU_DATA_DIR_V2", DEFAULT_DATA_DIR)
        self.augment = augment and (split == "train")
        self._seed = seed

        # Cache the inner RustBPETokenizer (may be wrapped by NanoChatTokenizerWrapper).
        # We need the inner one to access render_for_completion / encode_special, which
        # produce single-token IDs for chat special tokens (matching SFT training).
        self._inner_tok = getattr(tokenizer, 'tokenizer', tokenizer)

        # Sanity check: encode_special must return single int IDs in the special range
        # (>32000 in nanochat). Guards against the v9 regression where string-concat
        # chat prompts silently produced reward=0 for an entire training run because
        # encode_ordinary tokenized "<|user_start|>" as 7 character tokens instead of
        # the single special-token ID 32760.
        asst_start_id = self._inner_tok.encode_special('<|assistant_start|>')
        asst_end_id = self._inner_tok.encode_special('<|assistant_end|>')
        assert isinstance(asst_start_id, int) and asst_start_id > 32000, (
            f"Tokenizer special-token encoding broken: <|assistant_start|>={asst_start_id}"
        )
        assert isinstance(asst_end_id, int) and asst_end_id > 32000, (
            f"Tokenizer special-token encoding broken: <|assistant_end|>={asst_end_id}"
        )

        # Load puzzle data
        effective_split = "val" if split == "val" else "train"
        self.samples = _load_sudoku_split_v2(effective_split, self.data_dir)

    def __iter__(self):
        # DDP-aware sharding
        is_ddp, rank, local_rank, world_size = get_dist_info()
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        # Unique seed per rank + worker
        seed = self._seed + rank * 1000 + worker_id
        rng = np.random.default_rng(seed)
        py_rng = random.Random(seed)

        epoch = 0
        while True:
            indices = build_rank_worker_indices(
                len(self.samples),
                rank=rank,
                world_size=world_size,
                worker_id=worker_id,
                num_workers=num_workers,
                seed=self._seed + epoch,
            )
            for idx in indices:
                sample = self.samples[idx]
                # Get puzzle and solution from sample (both are 9x9 nested lists)
                puzzle_grid = sample["puzzle"]  # 9x9 nested list
                solution = sample["solution"]  # 9x9 nested list

                # Convert puzzle grid to flat string for later use
                puzzle_flat = ''.join(str(puzzle_grid[r][c]) for r in range(9) for c in range(9))

                # Apply augmentation
                if self.augment:
                    puzzle_grid, solution = augment_board(puzzle_grid, solution, rng)
                    # Update flat puzzle string after augmentation
                    puzzle_flat = "".join(
                        str(puzzle_grid[r][c]) for r in range(9) for c in range(9)
                    )

                # Pick random prompt template and format
                fmt = _pick_format(py_rng)
                puzzle_str = _format_board(puzzle_grid, fmt)
                prompt_idx = py_rng.randrange(len(PROMPT_TEMPLATES))
                prompt_tpl = PROMPT_TEMPLATES[prompt_idx]
                user_content = prompt_tpl.format(puzzle=puzzle_str)

                # Build prompt using nanochat's render_for_completion: this is the
                # canonical RL entry point that produces single-token IDs for
                # <|user_start|>, <|user_end|>, <|assistant_start|>, matching the
                # exact distribution the model saw during SFT.
                conv = {
                    "messages": [
                        {"role": "user", "content": user_content},
                        # placeholder assistant message — popped by render_for_completion
                        {"role": "assistant", "content": ""},
                    ]
                }
                prompt_ids = self._inner_tok.render_for_completion(conv)
                # Left-truncate to preserve the trailing <|assistant_start|>.
                if len(prompt_ids) > self.max_prompt_length:
                    prompt_ids = prompt_ids[-self.max_prompt_length:]

                # Compute difficulty
                num_givens = sum(1 for c in puzzle_flat if c != '0')
                if num_givens <= 22:
                    difficulty = "hard"
                elif num_givens <= 28:
                    difficulty = "medium"
                else:
                    difficulty = "easy"

                yield {
                    "prompt_ids": prompt_ids,
                    "puzzle": puzzle_flat,
                    "solution": solution,
                    "difficulty": difficulty,
                    "num_givens": num_givens,
                }
            epoch += 1

    def _format_chat_prompt(self, user_content: str) -> str:
        """Deprecated: kept only for diagnostic comparisons. Do NOT use for RL prompts —
        the resulting string, when passed to tokenizer.encode(), tokenizes special
        tokens as multi-char text instead of single special IDs. See `__iter__` for
        the correct render_for_completion path.
        """
        return (
            f"<|user_start|>{user_content}<|user_end|>"
            f"<|assistant_start|>"
        )


def collate_rl_prompts(batch, tokenizer, max_prompt_length):
    """Collate a batch of RL prompt dicts into tensors.

    Args:
        batch: list of dicts from SudokuRLPromptDataset, each with pre-encoded `prompt_ids`
        tokenizer: nanochat tokenizer (only used for pad_id)
        max_prompt_length: max token length for prompts

    Returns:
        dict with:
          - prompt_ids: (B, max_len) padded token IDs (left-padded)
          - prompt_lengths: (B,) actual lengths
          - puzzles: list of 81-char strings
          - solutions: list of 9x9 grids
          - difficulties: list of str
    """
    all_ids = [item["prompt_ids"] for item in batch]
    puzzles = [item["puzzle"] for item in batch]
    solutions = [item["solution"] for item in batch]
    difficulties = [item["difficulty"] for item in batch]

    # Defensive truncate (should already be capped by dataset)
    all_ids = [ids[-max_prompt_length:] if len(ids) > max_prompt_length else ids for ids in all_ids]

    # Pad to max length in batch (left-pad for generation)
    max_len = max(len(ids) for ids in all_ids)
    # Use bos_token_id as pad (nanochat convention)
    pad_id = tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else 0

    prompt_ids = torch.full((len(all_ids), max_len), pad_id, dtype=torch.long)
    prompt_lengths = torch.zeros(len(all_ids), dtype=torch.long)

    for i, ids in enumerate(all_ids):
        length = len(ids)
        prompt_lengths[i] = length
        # Left-pad: place tokens at the end
        prompt_ids[i, max_len - length:] = torch.tensor(ids, dtype=torch.long)

    return {
        "prompt_ids": prompt_ids,
        "prompt_lengths": prompt_lengths,
        "puzzles": puzzles,
        "solutions": solutions,
        "difficulties": difficulties,
    }
