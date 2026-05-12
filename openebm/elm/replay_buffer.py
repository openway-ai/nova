"""Replay buffer utilities for Energy-Based Transformer training."""

from typing import Any, Optional, Tuple

import random

import torch

# TODO(investigate): Generalize this buffer to non-causal (bidirectional)
# models; the current index slicing assumes an AR setup. A thin helper
# wrapping the indexing logic would let AR and bi-directional models share
# the same buffer implementation.


class CausalReplayBuffer:
    """FIFO replay buffer for causal (autoregressive) EBT training.

    The buffer stores tuples of ``(all_data, predicted_outputs)`` from past
    MCMC roll-outs and concatenates a random sub-sample with the freshly
    sampled batch so that the energy network sees a mix of current and
    historical negative samples.
    """

    def __init__(self, max_size: int, sample_size: int) -> None:
        """Initialize the buffer.

        :param max_size: Maximum number of records retained. Older records
            are evicted in FIFO order.
        :type max_size: int
        :param sample_size: Number of records drawn from the buffer on every
            :meth:`get_batch` call.
        :type sample_size: int
        """
        self.buffer = []
        self.max_size = max_size
        self.sample_size = int(sample_size)

    def add(self, record: Any) -> None:
        """Append ``record`` and evict the oldest entry if over capacity.

        :param record: Record to append. Typically a dict with keys
            ``"all_data"`` and ``"predicted_outputs"``.
        :type record: Any
        """
        self.buffer.append(record)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def get_batch(
        self,
        all_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Build a mixed batch of fresh and replayed samples.

        :param all_data: Tensor of shape ``(B, S)`` holding the freshly
            sampled sequences.
        :type all_data: torch.Tensor
        :return: Tuple ``(inputs, replay_predicted, gt_outputs)``. When the
            buffer is empty, ``replay_predicted`` is ``None`` and the other
            tensors contain only the fresh samples.
        :rtype: Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]
        """
        fresh_inputs = all_data[:, :-1]
        fresh_gt_outputs = all_data[:, 1:]
        if not self.buffer:
            return fresh_inputs, None, fresh_gt_outputs
        num_samples = min(self.sample_size, len(self.buffer))
        samples = random.sample(self.buffer, num_samples)
        replay_inputs = torch.cat([s["all_data"][:, :-1] for s in samples], dim=0)
        replay_gt_outputs = torch.cat([s["all_data"][:, 1:] for s in samples], dim=0)
        replay_predicted_outputs = torch.cat([s["predicted_outputs"] for s in samples], dim=0)
        # NOTE: Fresh inputs must come first in the concatenation; downstream
        # forward passes assume that ordering.
        combined_inputs = torch.cat([fresh_inputs, replay_inputs], dim=0)
        combined_gt_outputs = torch.cat([fresh_gt_outputs, replay_gt_outputs], dim=0)
        return combined_inputs, replay_predicted_outputs, combined_gt_outputs

    def update(self, all_data: torch.Tensor, final_predicted_outputs: torch.Tensor) -> None:
        """Push each sequence in the batch into the buffer.

        :param all_data: Tensor of shape ``(B, S)`` with fresh sequences.
        :type all_data: torch.Tensor
        :param final_predicted_outputs: Tensor of shape ``(B, S-1, ...)`` with
            the final MCMC predictions that will be reused as replay samples.
        :type final_predicted_outputs: torch.Tensor
        """
        for i in range(all_data.size(0)):
            record = {
                "all_data": all_data[i].unsqueeze(0),
                "predicted_outputs": final_predicted_outputs[i].detach().unsqueeze(0)
            }
            self.add(record)
