"""
EBT SFT dataset adapter for NanoChat task mixtures.

This keeps the packed `(inputs, targets)` interface expected by the EBT trainer
while using NanoChat conversation tasks as the data source.
"""

import copy
import os
from functools import lru_cache

import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as _IterableDataset

from nanochat.common import get_dist_info


def _log_sft_dataset_root():
    sft_data_dir = os.environ.get("NANOCHAT_SFT_DATA_DIR", "")
    if sft_data_dir and os.path.exists(sft_data_dir):
        print(f"[EBT SFT] Using offline dataset root: {sft_data_dir}")


def _build_train_dataset():
    from tasks.common import TaskMixture
    from tasks.gsm8k import GSM8K
    from tasks.mmlu import MMLU
    from tasks.smoltalk import SmolTalk

    base_dir = os.environ.get("NANOCHAT_BASE_DIR", "")
    identity_path = os.path.join(base_dir, "identity_conversations.jsonl")

    train_tasks = [
        SmolTalk(split="train"),
        MMLU(subset="auxiliary_train", split="train"),
        GSM8K(subset="main", split="train"),
        GSM8K(subset="main", split="train"),
    ]

    try:
        from tasks.customjson import CustomJSON

        if os.path.exists(identity_path):
            train_tasks.append(CustomJSON(filepath=identity_path))
            train_tasks.append(CustomJSON(filepath=identity_path))
    except Exception:
        pass

    try:
        from tasks.spellingbee import SimpleSpelling, SpellingBee

        train_tasks.append(SimpleSpelling(size=200000, split="train"))
        train_tasks.append(SpellingBee(size=80000, split="train"))
    except Exception:
        pass

    return TaskMixture(train_tasks)


def _build_val_dataset():
    from tasks.common import TaskMixture
    from tasks.gsm8k import GSM8K
    from tasks.mmlu import MMLU
    from tasks.smoltalk import SmolTalk

    return TaskMixture(
        [
            SmolTalk(split="test"),
            MMLU(subset="all", split="test", stop=5200),
            GSM8K(subset="main", split="test", stop=420),
        ]
    )


@lru_cache(maxsize=2)
def _load_sft_dataset(split):
    """Load only the requested SFT split."""
    _log_sft_dataset_root()
    if split == "train":
        return _build_train_dataset()
    if split in ("val", "test"):
        return _build_val_dataset()
    raise ValueError(f"Unsupported SFT split: {split}")


@lru_cache(maxsize=1)
def _load_sft_datasets():
    """Backward-compatible helper for callers expecting both splits."""
    return _load_sft_dataset("train"), _load_sft_dataset("val")


class SFTIterableDataset(_IterableDataset):
    """Adapt NanoChat SFT task mixtures to the EBT iterable dataloader API."""

    STATE_VERSION = "sft_v1"

    def __init__(
        self,
        tokenizer,
        batch_size,
        max_len,
        split,
        max_iter,
        device="cuda",
        resume_state_dict=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.B = batch_size
        self.T = max_len
        self.split = split
        self.max_iter = max_iter
        self.device = device
        self.resume_state_dict = resume_state_dict
        self._resume_state_locked = self._can_restore(resume_state_dict)
        self.last_state_dict = None
        self._runtime_initialized = False

        self.row_capacity = self.T + 1
        self.buffer_size = 1000
        self.conv_buffer = []
        self.cursor = None
        self.consumed = None
        self.epoch = 1
        self.it = 0
        self.ddp_rank = None
        self.ddp_world_size = None

        # Load only the split that is actually iterated. This avoids building
        # both train and val task mixtures during trainer startup / sanity val.
        self.dataset = None

    def _can_restore(self, state):
        return isinstance(state, dict) and (
            state.get("state_version") == self.STATE_VERSION or "conv_buffer" in state
        )

    def _initialize_fresh_state(self, ddp_rank, ddp_world_size):
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.row_capacity = self.T + 1
        self.buffer_size = 1000
        self.conv_buffer = []
        self.cursor = ddp_rank
        self.consumed = ddp_rank
        self.epoch = 1
        self.it = 0

    def _restore_state(self, state, ddp_rank, ddp_world_size):
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.row_capacity = int(state.get("row_capacity", self.T + 1))
        self.buffer_size = int(state.get("buffer_size", 1000))
        self.conv_buffer = copy.deepcopy(state.get("conv_buffer", []))
        self.cursor = int(state.get("cursor", ddp_rank))
        self.consumed = int(state.get("consumed", ddp_rank))
        self.epoch = int(state.get("epoch", 1))
        self.it = int(state.get("it", 0))
        print(
            f"[SFT Resume] Rank {ddp_rank} restored state: "
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
        """BOS-aligned best-fit packing, matching NanoChat chat formatting."""
        ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
        if self.dataset is None:
            self.dataset = _load_sft_dataset(self.split)
        dataset = self.dataset
        dataset_size = len(dataset)
        assert dataset_size > 0

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
                conversation = dataset[self.cursor]
                ids, mask = self.tokenizer.render_conversation(conversation)
                self.cursor += ddp_world_size
                if self.cursor >= dataset_size:
                    self.cursor = self.cursor % dataset_size
                    self.epoch += 1
                if len(ids) <= self.row_capacity:
                    self.conv_buffer.append((ids, mask))

        while True:
            if self.split == "train" and self.it >= self.max_iter:
                break

            rows = []
            row_masks = []

            for _ in range(self.B):
                row = []
                row_mask = []
                while len(row) < self.row_capacity:
                    while len(self.conv_buffer) < self.buffer_size:
                        refill_buffer()
                    remaining = self.row_capacity - len(row)

                    best_idx = -1
                    best_len = 0
                    for i, (conv_ids, conv_mask) in enumerate(self.conv_buffer):
                        conv_len = len(conv_ids)
                        if conv_len <= remaining and conv_len > best_len:
                            best_idx = i
                            best_len = conv_len

                    if best_idx >= 0:
                        conv_ids, conv_mask = self.conv_buffer.pop(best_idx)
                        row.extend(conv_ids)
                        row_mask.extend(conv_mask)
                        self.consumed += ddp_world_size
                    else:
                        remaining = self.row_capacity - len(row)
                        row.extend([bos_token] * remaining)
                        row_mask.extend([0] * remaining)
                        break

                rows.append(row[: self.row_capacity])
                row_masks.append(row_mask[: self.row_capacity])

            batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
            mask_tensor = torch.tensor(row_masks, dtype=torch.bool, pin_memory=use_cuda)
            inputs = batch_tensor[:, :-1].to(
                device=self.device, dtype=torch.int64, non_blocking=use_cuda
            )
            targets = batch_tensor[:, 1:].to(
                device=self.device, dtype=torch.int64, non_blocking=use_cuda
            )
            target_mask = mask_tensor[:, 1:].to(
                device=self.device, dtype=torch.bool, non_blocking=use_cuda
            )

            # Use NanoChat's supervision mask directly: only assistant tokens
            # should contribute to the loss.
            targets = targets.masked_fill(~target_mask, -1)

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
                    f"[SFT Resume] Ignore external dataloader overwrite: "
                    f"current_rank={current_state.get('rank')}, "
                    f"incoming_rank={incoming_state.get('rank') if isinstance(incoming_state, dict) else None}"
                )
                return
        self.resume_state_dict = copy.deepcopy(state_dict)
        self._runtime_initialized = False

    def __len__(self):
        return self.max_iter


class SFTDataLoader(DataLoader):
    def state_dict(self):
        dataset = getattr(self, "dataset", None)
        if dataset is None or not hasattr(dataset, "state_dict"):
            return {}
        return dataset.state_dict()

    def load_state_dict(self, state_dict):
        dataset = getattr(self, "dataset", None)
        if dataset is not None and hasattr(dataset, "load_state_dict"):
            dataset.load_state_dict(state_dict)


def generate_sft_dataloader(
    tokenizer, batch_size, max_len, max_iter, split, device, resume_state_dict=None
):
    dataset = SFTIterableDataset(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        split=split,
        max_iter=max_iter,
        device=device,
        resume_state_dict=resume_state_dict,
    )
    dataloader = SFTDataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return dataloader
