"""Iterable dataset wrappers around NanoChat's streaming data loader."""

import sys
from typing import Any, Iterator, Optional, Tuple

from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit

import torch
from torch.utils.data import IterableDataset as _IterableDataset, DataLoader


class NanochatIterableDataset(_IterableDataset):
    """Wrap NanoChat's stateful streaming loader as a PyTorch ``IterableDataset``.

    Design notes:

    - NanoChat's generator is infinite and yields
      ``(inputs[B, T], targets[B, T], state_dict)``. The tensors are views
      into pre-allocated GPU buffers that are overwritten on every ``next()``
      call. The ``DataLoader`` default collate (with ``batch_size=1``) calls
      :func:`torch.stack`, which implicitly copies the tensors and makes them
      safe to hold across iterations.
    - ``__len__`` returns ``max_iter`` so that ``DataLoader.__len__`` and the
      trainer's ``_limit_batches`` can compute batch counts. Actual iteration
      limits are enforced by the trainer.
    - DDP sharding is handled internally by NanoChat via ``get_dist_info()``;
      no ``DistributedSampler`` is required.
    """

    def __init__(
        self,
        tokenizer: Any,
        B: int,
        T: int,
        split: str,
        max_iter: int,
        device: str = "cuda",
        resume_state_dict: Optional[dict] = None,
    ) -> None:
        """Initialize the dataset.

        :param tokenizer: Tokenizer forwarded to NanoChat.
        :type tokenizer: Any
        :param B: Micro-batch size.
        :type B: int
        :param T: Sequence length.
        :type T: int
        :param split: Dataset split.
        :type split: str
        :param max_iter: Nominal iteration count used by ``__len__``.
        :type max_iter: int
        :param device: Target device.
        :type device: str
        :param resume_state_dict: Optional state dict for exact resume.
        :type resume_state_dict: Optional[dict]
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.B = B
        self.T = T
        self.split = split
        self.max_iter = max_iter
        self.device = device
        self.resume_state_dict = resume_state_dict

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, dict]]:
        """Return the underlying infinite NanoChat generator.

        :return: Iterator yielding ``(inputs, targets, state_dict)``.
        :rtype: Iterator[Tuple[torch.Tensor, torch.Tensor, dict]]
        """
        return tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tokenizer=self.tokenizer,
            B=self.B,
            T=self.T,
            split=self.split,
            device=self.device,
            resume_state_dict=self.resume_state_dict,
        )

    def __len__(self) -> int:
        """Return the nominal iteration count.

        :return: ``max_iter`` passed at construction time.
        :rtype: int
        """
        return self.max_iter


# Backward-compat alias: ``trainer.py`` imports ``IterableDataset`` by name.
IterableDataset = NanochatIterableDataset


def generate_dataloader(
    tokenizer: Any,
    batch_size: int,
    max_len: int,
    max_iter: int,
    split: str,
    device: str,
    resume_state_dict: Optional[dict] = None,
) -> DataLoader:
    """Build a ``DataLoader`` wrapping NanoChat's streaming data pipeline.

    ``batch_size=1`` serves two purposes:

    1. it adds a leading dimension ``[1, B, T]`` expected by the model's
       ``squeeze(dim=0)``, and
    2. the default collate's :func:`torch.stack` copies NanoChat's reused GPU
       buffer views, making the returned tensors safe to hold across
       iterations.

    ``num_workers=0`` is required because the NanoChat generator holds GPU
    state (pre-allocated CUDA buffers) that cannot be pickled into worker
    processes. ``pin_memory=False`` because the data is already on the GPU.

    :param tokenizer: Tokenizer forwarded to NanoChat.
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
    :param resume_state_dict: Optional state dict for exact resume.
    :type resume_state_dict: Optional[dict]
    :return: A ``DataLoader`` configured for stateful streaming.
    :rtype: DataLoader
    """
    dataset = NanochatIterableDataset(
        tokenizer=tokenizer,
        B=batch_size,
        T=max_len,
        split=split,
        max_iter=max_iter,
        device=device,
        resume_state_dict=resume_state_dict,
    )

    def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate NanoChat's ``(inputs, targets, state_dict)`` tuples.

        Because ``batch_size=1`` the incoming ``batch`` is a list with a
        single ``(inputs, targets, state_dict)`` triple. The ``state_dict``
        entry is discarded; the two tensors are cloned to detach them from
        NanoChat's reusable GPU buffer.

        :param batch: One-element list containing the raw triple.
        :type batch: list
        :return: ``(inputs, targets)`` with a leading batch dim of 1.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        inputs, targets, state_dict = batch[0]
        return (inputs.unsqueeze(0).clone(), targets.unsqueeze(0).clone())

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )
