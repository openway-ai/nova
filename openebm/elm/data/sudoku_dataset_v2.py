"""
Sudoku SFT Dataset V2 — 数据多样化 + 对称增强
================================================

相对于 v1 的改进：
  1. **自然语言 prompt 多样性**：5 种 prompt 模板 + 3 种 response 模板随机组合
  2. **On-the-fly 对称增强**：digit relabel / band perm / stack perm / row-within-band /
     col-within-stack / transpose —— 所有变换保持数独合法性
  3. **版本隔离**：默认缓存目录 sudoku_cache_v2/，不干扰 v1

输出格式与 SFTIterableDataset 完全一致: (inputs, targets) shape (B, T)。
"""

import copy
import os
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset as _IterableDataset, DataLoader

from nanochat.common import get_dist_info


DEFAULT_DATA_DIR = os.environ.get(
    "SUDOKU_DATA_DIR_V2",
    os.path.join(os.path.dirname(__file__), "sudoku_cache_v2"),
)


# ── Prompt / Response 多样化 ────────────────────────────────────────────────

PROMPT_TEMPLATES = [
    "Solve this sudoku puzzle. Replace each 0 with the correct digit (1-9):\n{puzzle}",
    "Complete the following sudoku grid. Empty cells are marked as 0:\n{puzzle}",
    "Fill in the missing numbers in this 9x9 sudoku:\n{puzzle}",
    "Here is a sudoku puzzle where 0 represents an empty cell. Find the solution:\n{puzzle}",
    "I need help solving this sudoku. Each row, column, and 3x3 box must contain digits 1-9 exactly once:\n{puzzle}",
]

RESPONSE_TEMPLATES = [
    "{solution}",
    "Here is the completed sudoku:\n{solution}",
    "The solution is:\n{solution}",
]


def _format_board(board):
    """Format a 9x9 int grid as a string (rows separated by \n, cols by spaces)."""
    return "\n".join(" ".join(str(c) for c in row) for row in board)


def board_to_conversation_v2(puzzle, solution, rng=None):
    """Convert 9x9 puzzle/solution to nanochat chat conversation with random templates."""
    if rng is None:
        rng = random
    puzzle_str = _format_board(puzzle)
    solution_str = _format_board(solution)
    prompt_tpl = rng.choice(PROMPT_TEMPLATES)
    response_tpl = rng.choice(RESPONSE_TEMPLATES)
    return {
        "messages": [
            {"role": "user", "content": prompt_tpl.format(puzzle=puzzle_str)},
            {"role": "assistant", "content": response_tpl.format(solution=solution_str)},
        ]
    }


# ── 对称增强 ────────────────────────────────────────────────────────────────

def augment_board(puzzle, solution, rng):
    """Apply random validity-preserving symmetry transform.

    Six transforms stacked:
      1. Digit relabel (1-9 permutation)
      2. Band permutation (3 horizontal band blocks)
      3. Row-within-band permutation
      4. Stack permutation (3 vertical stack blocks)
      5. Col-within-stack permutation
      6. Transpose (50% probability)

    Args:
        puzzle:  9x9 nested list, 0 = empty
        solution: 9x9 nested list, 1-9
        rng:     numpy.random.Generator (preferred) or np.random

    Returns:
        (puzzle_aug, solution_aug) as 9x9 nested lists
    """
    puzzle = np.array(puzzle, dtype=np.int64)
    solution = np.array(solution, dtype=np.int64)

    # 1. digit relabel (keeps 0 fixed)
    perm = rng.permutation(9) + 1  # random permutation of [1..9]
    mapping = np.zeros(10, dtype=np.int64)
    mapping[1:] = perm
    puzzle = mapping[puzzle]
    solution = mapping[solution]

    # 2. band permutation (3 horizontal band blocks of 3 rows each)
    band_order = rng.permutation(3)
    puzzle = np.vstack([puzzle[b * 3:(b + 1) * 3] for b in band_order])
    solution = np.vstack([solution[b * 3:(b + 1) * 3] for b in band_order])

    # 3. row-within-band permutation
    for b in range(3):
        row_order = rng.permutation(3)
        puzzle[b * 3:(b + 1) * 3] = puzzle[b * 3:(b + 1) * 3][row_order]
        solution[b * 3:(b + 1) * 3] = solution[b * 3:(b + 1) * 3][row_order]

    # 4. stack permutation (3 vertical stack blocks of 3 cols each)
    stack_order = rng.permutation(3)
    puzzle = np.hstack([puzzle[:, s * 3:(s + 1) * 3] for s in stack_order])
    solution = np.hstack([solution[:, s * 3:(s + 1) * 3] for s in stack_order])

    # 5. col-within-stack permutation
    for s in range(3):
        col_order = rng.permutation(3)
        puzzle[:, s * 3:(s + 1) * 3] = puzzle[:, s * 3:(s + 1) * 3][:, col_order]
        solution[:, s * 3:(s + 1) * 3] = solution[:, s * 3:(s + 1) * 3][:, col_order]

    # 6. transpose (50%)
    if rng.random() < 0.5:
        puzzle = puzzle.T
        solution = solution.T

    return puzzle.tolist(), solution.tolist()


# ── Dataset loader ──────────────────────────────────────────────────────────

def _load_sudoku_split_v2(split, data_dir):
    """Load cached sudoku v2 data for a given split."""
    split_files = {
        "train": "rrn_train.pt",
        "val": "rrn_val.pt",
        "test": "satnet_test.pt",
    }
    if split not in split_files:
        raise ValueError(f"Unknown split '{split}', expected one of {list(split_files)}")
    path = os.path.join(data_dir, split_files[split])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sudoku v2 cache not found: {path}\n"
            f"Run: python openebm/elm/data/prepare_sudoku_data_v2.py --data_dir {data_dir}"
        )
    samples = torch.load(path, weights_only=False)
    print(f"[Sudoku v2] Loaded {len(samples)} samples from {path}")
    return samples


class SudokuSFTV2IterableDataset(_IterableDataset):
    """Sudoku SFT dataset v2 with prompt diversity + on-the-fly symmetry augmentation."""

    STATE_VERSION = "sudoku_sft_v2"

    def __init__(
        self,
        tokenizer,
        batch_size,
        max_len,
        split,
        max_iter,
        device="cuda",
        data_dir=None,
        resume_state_dict=None,
        augment=True,
        seed=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.B = batch_size
        self.T = max_len
        self.split = split
        self.max_iter = max_iter
        self.device = device
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.resume_state_dict = resume_state_dict
        self._resume_state_locked = self._can_restore(resume_state_dict)
        self.last_state_dict = None
        self._runtime_initialized = False
        # v2 新增：训练时开启增强，验证时关闭（确保 valid_loss 可比较）
        self.augment = augment and (split == "train")
        self._seed = seed

        self.row_capacity = self.T + 1
        self.buffer_size = 500  # v2 RRN 数据更大
        self.conv_buffer = []
        self.cursor = None
        self.consumed = None
        self.epoch = 1
        self.it = 0
        self.ddp_rank = None
        self.ddp_world_size = None
        self._np_rng = None
        self._py_rng = None

        # v2 训练集用 'train'，验证集用 'val'
        effective_split = "val" if split == "val" else "train"
        self.samples = _load_sudoku_split_v2(effective_split, self.data_dir)

    def _can_restore(self, state):
        return isinstance(state, dict) and (
            state.get("state_version") == self.STATE_VERSION or "conv_buffer" in state
        )

    def _init_rngs(self, ddp_rank):
        """Init rng with a rank-dependent seed for reproducibility + cross-rank diversity."""
        base_seed = self._seed if self._seed is not None else 20260508
        seed = base_seed + (ddp_rank or 0) * 10007 + self.it
        self._np_rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

    def _initialize_fresh_state(self, ddp_rank, ddp_world_size):
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.row_capacity = self.T + 1
        self.conv_buffer = []
        self.cursor = ddp_rank
        self.consumed = ddp_rank
        self.epoch = 1
        self.it = 0
        self._init_rngs(ddp_rank)

    def _restore_state(self, state, ddp_rank, ddp_world_size):
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.row_capacity = int(state.get("row_capacity", self.T + 1))
        self.conv_buffer = copy.deepcopy(state.get("conv_buffer", []))
        self.cursor = int(state.get("cursor", ddp_rank))
        self.consumed = int(state.get("consumed", ddp_rank))
        self.epoch = int(state.get("epoch", 1))
        self.it = int(state.get("it", 0))
        self._init_rngs(ddp_rank)
        print(
            f"[Sudoku v2 Resume] Rank {ddp_rank}: "
            f"cursor={self.cursor}, consumed={self.consumed}, epoch={self.epoch}, "
            f"it={self.it}, conv_buffer={len(self.conv_buffer)}"
        )

    def _build_state_dict(self, copy_buffer):
        return {
            "state_version": self.STATE_VERSION,
            "split": self.split,
            "cursor": self.cursor,
            "consumed": self.consumed,
            "epoch": self.epoch,
            "it": self.it,
            "buffer_size": self.buffer_size,
            "row_capacity": self.row_capacity,
            "rank": self.ddp_rank,
            "world_size": self.ddp_world_size,
            "conv_buffer": copy.deepcopy(self.conv_buffer) if copy_buffer else self.conv_buffer,
        }

    def __iter__(self):
        """BOS-aligned bestfit packing, mirroring SFTIterableDataset.__iter__"""
        ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
        dataset_size = len(self.samples)
        assert dataset_size > 0, f"Empty sudoku dataset for split={self.split}"

        bos_token = self.tokenizer.get_bos_token_id()
        use_cuda = "cuda" in str(self.device)

        if self.split == "train":
            if not self._runtime_initialized:
                if self._can_restore(self.resume_state_dict):
                    self._restore_state(self.resume_state_dict, ddp_rank, ddp_world_size)
                else:
                    self._initialize_fresh_state(ddp_rank, ddp_world_size)
                self._runtime_initialized = True
        else:
            self._initialize_fresh_state(ddp_rank, ddp_world_size)

        def refill_buffer():
            while len(self.conv_buffer) < self.buffer_size:
                sample = self.samples[self.cursor % dataset_size]
                puzzle = sample["puzzle"]
                solution = sample["solution"]
                if self.augment:
                    puzzle, solution = augment_board(puzzle, solution, self._np_rng)
                conversation = board_to_conversation_v2(puzzle, solution, rng=self._py_rng)
                ids, _ = self.tokenizer.render_conversation(conversation)
                self.cursor += ddp_world_size
                if self.cursor >= dataset_size:
                    self.cursor = self.cursor % dataset_size
                    self.epoch += 1
                if len(ids) <= self.row_capacity:
                    self.conv_buffer.append(ids)

        while True:
            if self.split == "train" and self.it >= self.max_iter:
                break

            rows = []
            row_lengths = []

            for _ in range(self.B):
                row = []
                padded = False
                while len(row) < self.row_capacity:
                    while len(self.conv_buffer) < self.buffer_size:
                        refill_buffer()
                    remaining = self.row_capacity - len(row)

                    best_idx = -1
                    best_len = 0
                    for i, conv in enumerate(self.conv_buffer):
                        conv_len = len(conv)
                        if conv_len <= remaining and conv_len > best_len:
                            best_idx = i
                            best_len = conv_len

                    if best_idx >= 0:
                        conv = self.conv_buffer.pop(best_idx)
                        row.extend(conv)
                        self.consumed += ddp_world_size
                    else:
                        content_len = len(row)
                        remaining = self.row_capacity - len(row)
                        row.extend([bos_token] * remaining)
                        padded = True
                        break

                if padded:
                    row_lengths.append(content_len)
                else:
                    row_lengths.append(self.row_capacity)
                rows.append(row[:self.row_capacity])

            batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
            inputs = batch_tensor[:, :-1].to(device=self.device, dtype=torch.int64, non_blocking=use_cuda)
            targets = batch_tensor[:, 1:].to(device=self.device, dtype=torch.int64, non_blocking=use_cuda)
            for i, content_len in enumerate(row_lengths):
                if content_len < self.row_capacity:
                    targets[i, content_len - 1:] = -1

            self.it += 1
            self.last_state_dict = self._build_state_dict(copy_buffer=False)
            yield inputs, targets

            if self.split != "train" and self.it >= self.max_iter:
                break

    def get_dataloader_state(self):
        if self.cursor is None:
            return self.last_state_dict
        return self._build_state_dict(copy_buffer=True)

    def state_dict(self):
        state = self.get_dataloader_state()
        return {} if state is None else state

    def load_state_dict(self, state_dict):
        if self._resume_state_locked and self._can_restore(self.resume_state_dict):
            current_state = self.resume_state_dict
            incoming_state = state_dict
            if current_state != incoming_state:
                print(
                    f"[Sudoku v2 Resume] Ignore external dataloader overwrite: "
                    f"current_rank={current_state.get('rank')}, "
                    f"incoming_rank={incoming_state.get('rank') if isinstance(incoming_state, dict) else None}"
                )
                return
        self.resume_state_dict = copy.deepcopy(state_dict)
        self._runtime_initialized = False

    def __len__(self):
        return self.max_iter


class SudokuSFTV2DataLoader(DataLoader):
    """DataLoader wrapper that delegates state_dict to the underlying dataset."""

    def state_dict(self):
        dataset = getattr(self, "dataset", None)
        if dataset is None or not hasattr(dataset, "state_dict"):
            return {}
        return dataset.state_dict()

    def load_state_dict(self, state_dict):
        dataset = getattr(self, "dataset", None)
        if dataset is not None and hasattr(dataset, "load_state_dict"):
            dataset.load_state_dict(state_dict)


def generate_sudoku_sft_v2_dataloader(
    tokenizer, batch_size, max_len, max_iter, split, device,
    data_dir=None, resume_state_dict=None, augment=True, seed=None,
):
    """Factory function matching generate_sudoku_sft_dataloader signature."""
    dataset = SudokuSFTV2IterableDataset(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        split=split,
        max_iter=max_iter,
        device=device,
        data_dir=data_dir,
        resume_state_dict=resume_state_dict,
        augment=augment,
        seed=seed,
    )
    dataloader = SudokuSFTV2DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return dataloader
