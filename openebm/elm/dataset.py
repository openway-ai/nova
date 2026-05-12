"""Infinite streaming dataset that wraps NanoChat's stateful data loader."""

from typing import Any, Iterator, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as _IterableDataset
from torch.utils.data import IterableDataset

from nanochat.dataloader import (
    StatefulBestFitDataLoader,
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)


class IterableDataset(_IterableDataset):
    """Wrap :class:`StatefulBestFitDataLoader` as a PyTorch ``IterableDataset``.

    Preserves the key features of the underlying loader:

    - infinite streaming,
    - exact-resume state via ``doc_buffer`` + ``doc_batch_index``,
    - no padding,
    - distributed-training compatibility.
    """

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

        :param tokenizer: Tokenizer forwarded to :class:`StatefulBestFitDataLoader`.
        :type tokenizer: Any
        :param batch_size: Micro-batch size (``B``).
        :type batch_size: int
        :param max_len: Maximum sequence length (``T``).
        :type max_len: int
        :param split: Dataset split name (``"train"`` / ``"val"`` / ...).
        :type split: str
        :param max_iter: Nominal dataset length used by
            ``len(dataset)``; the underlying stream is infinite.
        :type max_iter: int
        :param device: Device on which the batches are placed.
        :type device: str
        :param resume_state_dict: Optional loader state dict used to resume
            streaming from an exact position.
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
        self.batch_idx = 0
        # Latest loader state, retained for checkpoint recovery.
        self.last_state_dict = None
        self._stateful_loader = None

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate infinitely over ``(inputs, targets)`` pairs.

        :return: Generator of ``(inputs, targets)`` tensors.
        :rtype: Iterator[Tuple[torch.Tensor, torch.Tensor]]
        """
        self._stateful_loader = StatefulBestFitDataLoader(
            tokenizer=self.tokenizer,
            B=self.B,
            T=self.T,
            split=self.split,
            device=self.device,
            resume_state_dict=self.resume_state_dict,
        )
        for inputs, targets, state_dict in self._stateful_loader:
            self.last_state_dict = state_dict
            yield inputs, targets

    def get_dataloader_state(self) -> Optional[dict]:
        """Return the exact-resume state, including the current ``doc_buffer``.

        :return: The latest state dict from the underlying stateful loader,
            or the cached one if the loader has not been created yet.
        :rtype: Optional[dict]
        """
        if self._stateful_loader is not None:
            return self._stateful_loader.state_dict()
        return self.last_state_dict

    def __len__(self) -> int:
        """Return the nominal dataset length.

        :return: ``max_iter`` passed at construction time.
        :rtype: int
        """
        return self.max_iter


def generate_dataloader(
    tokenizer: Any,
    batch_size: int,
    max_len: int,
    max_iter: int,
    split: str,
    device: str,
    resume_state_dict: Optional[dict] = None,
) -> DataLoader:
    """Build a ``DataLoader`` wrapping :class:`IterableDataset`.

    :param tokenizer: Tokenizer forwarded to the dataset.
    :type tokenizer: Any
    :param batch_size: Micro-batch size.
    :type batch_size: int
    :param max_len: Maximum sequence length.
    :type max_len: int
    :param max_iter: Nominal dataset length.
    :type max_iter: int
    :param split: Dataset split.
    :type split: str
    :param device: Target device for batches.
    :type device: str
    :param resume_state_dict: Optional exact-resume state dict.
    :type resume_state_dict: Optional[dict]
    :return: A ``DataLoader`` configured for stateful streaming
        (``batch_size=1`` because the dataset already returns batched tensors).
    :rtype: DataLoader
    """

    dataset = IterableDataset(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        split=split,
        max_iter=max_iter,
        device=device,
        resume_state_dict=resume_state_dict,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,      # IMPORTANT: the dataset already yields (B, T) batches.
        shuffle=False,
        num_workers=0,     # Keep 0 to preserve stateful streaming.
        pin_memory=False,  # Pin memory is handled internally by the loader.
    )
    return dataloader
