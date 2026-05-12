"""EBT SFT dataset adapter built on top of NanoChat's ``TaskMixture``.

The dataset loads the NanoChat SFT task mixture (SmolTalk, MMLU, GSM8K, etc.)
and emits ``(inputs, targets)`` batches with exactly the same shape ``(B, T)``
that the EBT pretraining data loader produces, so that both pipelines can
share the same training loop.
"""

import copy
import os
import sys
from typing import Any, Iterator, Optional, Tuple

import torch
from torch.utils.data import IterableDataset as _IterableDataset, DataLoader

from nanochat.common import get_dist_info
from nanochat.tokenizer import get_tokenizer


def _load_sft_datasets() -> Tuple[Any, Any]:
    """Load the NanoChat SFT task mixture, with optional offline support.

    Respects two environment variables:

    - ``NANOCHAT_SFT_DATA_DIR``: when set, the NanoChat tasks transparently
      read their data from that directory (offline mode).
    - ``NANOCHAT_BASE_DIR``: base directory used to locate the optional
      ``identity_conversations.jsonl`` file.

    :return: Tuple ``(train_dataset, val_dataset)`` of NanoChat
        ``TaskMixture`` instances.
    :rtype: Tuple[Any, Any]
    """
    from tasks.common import TaskMixture
    from tasks.gsm8k import GSM8K
    from tasks.mmlu import MMLU
    from tasks.smoltalk import SmolTalk

    # Offline mode: ``NANOCHAT_SFT_DATA_DIR`` lets the NanoChat tasks skip
    # the network download and reuse a pre-staged dataset cache.
    sft_data_dir = os.environ.get("NANOCHAT_SFT_DATA_DIR", "")
    if sft_data_dir and os.path.exists(sft_data_dir):
        print(f"[EBT SFT] Using offline dataset dir: {sft_data_dir}")

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

    train_dataset = TaskMixture(train_tasks)

    val_dataset = TaskMixture([
        SmolTalk(split="test"),
        MMLU(subset="all", split="test", stop=5200),
        GSM8K(subset="main", split="test", stop=420),
    ])

    return train_dataset, val_dataset


class SFTIterableDataset(_IterableDataset):
    """Iterable dataset wrapping the NanoChat SFT ``TaskMixture`` for EBT.

    Applies BOS-aligned best-fit packing (identical to ``chat_sft.py`` in
    upstream NanoChat) and keeps exact-resume state across DDP ranks.
    """

    STATE_VERSION = "sft_v1"

    def __init__(
        self,
        tokenizer: Any,
        batch_size: int,
        max_len: int,
        split: str,
        max_iter: int,
        device: str = "cuda",
        resume_state_dict: Optional[dict] = None,
    ) -> None:
        """Initialize the dataset.

        :param tokenizer: Tokenizer exposing ``get_bos_token_id`` and
            ``render_conversation``.
        :type tokenizer: Any
        :param batch_size: Micro-batch size ``B``.
        :type batch_size: int
        :param max_len: Sequence length ``T``.
        :type max_len: int
        :param split: ``"train"`` or ``"val"``.
        :type split: str
        :param max_iter: Nominal iteration count.
        :type max_iter: int
        :param device: Target device.
        :type device: str
        :param resume_state_dict: Optional exact-resume state dict.
        :type resume_state_dict: Optional[dict]
        """
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

        self.train_dataset, self.val_dataset = _load_sft_datasets()

    def _can_restore(self, state: Any) -> bool:
        """Return ``True`` when ``state`` looks like a compatible resume dict.

        :param state: Candidate state dict.
        :type state: Any
        :return: Whether ``state`` can be restored with :meth:`_restore_state`.
        :rtype: bool
        """
        return isinstance(state, dict) and (
            state.get("state_version") == self.STATE_VERSION or "conv_buffer" in state
        )

    def _initialize_fresh_state(self, ddp_rank: int, ddp_world_size: int) -> None:
        """Initialize a fresh (non-resume) streaming state.

        :param ddp_rank: Rank of the current process.
        :type ddp_rank: int
        :param ddp_world_size: DDP world size.
        :type ddp_world_size: int
        """
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.row_capacity = self.T + 1
        self.buffer_size = 1000
        self.conv_buffer = []
        self.cursor = ddp_rank
        self.consumed = ddp_rank
        self.epoch = 1
        self.it = 0

    def _restore_state(self, state: dict, ddp_rank: int, ddp_world_size: int) -> None:
        """Restore streaming state from a resume dict.

        :param state: Resume state dict previously returned by
            :meth:`_build_state_dict`.
        :type state: dict
        :param ddp_rank: Rank of the current process.
        :type ddp_rank: int
        :param ddp_world_size: DDP world size.
        :type ddp_world_size: int
        """
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

    def _build_state_dict(self, copy_buffer: bool) -> dict:
        """Return a serializable snapshot of the streaming state.

        :param copy_buffer: When ``True``, deep-copy the in-flight buffer;
            otherwise share the reference (cheaper but not safe to persist).
        :type copy_buffer: bool
        :return: State dict with all fields required to resume exactly.
        :rtype: dict
        """
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

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield ``(inputs, targets)`` batches with BOS-aligned best-fit packing.

        :return: Iterator of ``(inputs, targets)`` tensors, each of shape ``(B, T)``.
        :rtype: Iterator[Tuple[torch.Tensor, torch.Tensor]]
        """
        ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
        dataset = self.train_dataset if self.split == "train" else self.val_dataset
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

        def refill_buffer() -> None:
            """Render more conversations until the sliding buffer is full."""
            while len(self.conv_buffer) < self.buffer_size:
                conversation = dataset[self.cursor]
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
                    targets[i, content_len - 1 :] = -1

            self.it += 1
            self.last_state_dict = self._build_state_dict(copy_buffer=False)
            yield inputs, targets

            if self.split != "train" and self.it >= self.max_iter:
                break

    def get_dataloader_state(self) -> Optional[dict]:
        """Return the exact-resume state of the current stream.

        :return: State dict suitable for later :meth:`load_state_dict`, or the
            cached dict when iteration has not started yet.
        :rtype: Optional[dict]
        """
        if self.cursor is None:
            return self.last_state_dict
        return self._build_state_dict(copy_buffer=True)

    def state_dict(self) -> dict:
        """Return the (possibly empty) state dict.

        :return: State dict, or ``{}`` when no state is available.
        :rtype: dict
        """
        state = self.get_dataloader_state()
        return {} if state is None else state

    def load_state_dict(self, state_dict: dict) -> None:
        """Load ``state_dict`` into the dataset.

        When the dataset was constructed with an explicit ``resume_state_dict``
        the local state takes priority — an incoming external state is logged
        and otherwise ignored. This prevents Lightning from overwriting the
        per-rank resume state during the initial setup phase.

        :param state_dict: State dict to restore.
        :type state_dict: dict
        """
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

    def __len__(self) -> int:
        """Return the nominal iteration count.

        :return: ``max_iter`` passed at construction time.
        :rtype: int
        """
        return self.max_iter


class SFTDataLoader(DataLoader):
    """``DataLoader`` that forwards ``state_dict`` to the wrapped dataset."""

    def state_dict(self) -> dict:
        """Return the underlying dataset's state dict, or ``{}``.

        :return: State dict, or ``{}`` when unavailable.
        :rtype: dict
        """
        dataset = getattr(self, "dataset", None)
        if dataset is None or not hasattr(dataset, "state_dict"):
            return {}
        return dataset.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Forward ``state_dict`` into the wrapped dataset.

        :param state_dict: State dict to restore.
        :type state_dict: dict
        """
        dataset = getattr(self, "dataset", None)
        if dataset is not None and hasattr(dataset, "load_state_dict"):
            dataset.load_state_dict(state_dict)


def generate_sft_dataloader(
    tokenizer: Any,
    batch_size: int,
    max_len: int,
    max_iter: int,
    split: str,
    device: str,
    resume_state_dict: Optional[dict] = None,
) -> SFTDataLoader:
    """Build an :class:`SFTDataLoader` around an :class:`SFTIterableDataset`.

    :param tokenizer: Tokenizer forwarded to the dataset.
    :type tokenizer: Any
    :param batch_size: Micro-batch size.
    :type batch_size: int
    :param max_len: Sequence length.
    :type max_len: int
    :param max_iter: Nominal iteration count.
    :type max_iter: int
    :param split: Dataset split.
    :type split: str
    :param device: Target device.
    :type device: str
    :param resume_state_dict: Optional exact-resume state.
    :type resume_state_dict: Optional[dict]
    :return: A ready-to-use :class:`SFTDataLoader`.
    :rtype: SFTDataLoader
    """
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
