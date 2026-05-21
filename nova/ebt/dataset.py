import os, sys
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from nanochat.dataloader import StatefulBestFitDataLoader, tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit

import torch
from torch.utils.data import IterableDataset as _IterableDataset
from torch.utils.data import IterableDataset, DataLoader

class IterableDataset(_IterableDataset):
    """
    Wraps StatefulBestFitDataLoader into a PyTorch IterableDataset.

    This keeps:
    - infinite streaming
    - exact resume state support (doc_buffer + doc_batch_index)
    - no padding
    - distributed compatibility
    """

    def __init__(
        self,
        tokenizer,
        batch_size,
        max_len,
        split,
        max_iter,
        device="cuda",
        resume_state_dict=None,
        training_objective="dense_next_token",
        train_block_size=2,
        train_context_length=None,
        num_block_samples_per_window=1,
        debug_blockwise_shapes=False,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.B = batch_size
        self.T = max_len
        self.split = split
        self.max_iter = max_iter
        self.device = device
        self.resume_state_dict = resume_state_dict
        self.batch_idx = 0
        self.last_state_dict = None  # 最新的 dataloader 位置，用于 checkpoint 恢复
        self._stateful_loader = None  # holds StatefulBestFitDataLoader instance
        self.training_objective = training_objective
        self.train_block_size = int(train_block_size)
        self.train_context_length = train_context_length
        self.num_block_samples_per_window = int(num_block_samples_per_window)
        self.debug_blockwise_shapes = bool(debug_blockwise_shapes)
        self._blockwise_debug_print_count = 0

    def _build_blockwise_batch(self, inputs, targets):
        # inputs/targets shape: [B, T], where targets is next-token shifted from inputs.
        if inputs.dim() != 2 or targets.dim() != 2:
            raise ValueError(
                f"Expected 2D inputs/targets [B, T], got inputs={tuple(inputs.shape)}, targets={tuple(targets.shape)}"
            )
        if inputs.shape != targets.shape:
            raise ValueError(f"inputs/targets shape mismatch: {tuple(inputs.shape)} vs {tuple(targets.shape)}")

        total_len = inputs.shape[1]
        block_size = self.train_block_size  # K future offsets
        if block_size <= 0:
            raise ValueError(f"train_block_size must be > 0, got {block_size}")
        if block_size > total_len:
            raise ValueError(
                f"train_block_size must be <= sequence length T for dense blockwise supervision, got K={block_size}, T={total_len}"
            )

        # Recover one extra token so that offset K can still align with the same input prefix.
        # full_tokens = [x1, x2, ..., xT, x(T+1)] with length T+1.
        full_tokens = torch.cat([inputs, targets[:, -1:].clone()], dim=1)
        seq_eff = full_tokens.shape[1] - block_size  # S_eff = (T+1) - K
        if seq_eff <= 0:
            raise ValueError(
                f"Expected positive S_eff for dense block supervision, got S_eff={seq_eff}, T={total_len}, K={block_size}"
            )

        input_ids = full_tokens[:, :seq_eff]  # x_{1:S_eff}
        block_targets = full_tokens.new_empty((inputs.shape[0], block_size, seq_eff))
        for offset_idx in range(block_size):
            start = 1 + offset_idx
            end = start + seq_eff
            block_targets[:, offset_idx, :] = full_tokens[:, start:end]

        if self.debug_blockwise_shapes and self._blockwise_debug_print_count < 8:
            self._blockwise_debug_print_count += 1
            offsets = list(range(1, block_size + 1))
            print(
                "[blockwise-dataloader:dense] "
                f"T={total_len}, K={block_size}, S_eff={seq_eff}, offsets={offsets}, "
                f"input_ids.shape={tuple(input_ids.shape)}, "
                f"block_targets.shape={tuple(block_targets.shape)}, "
                f"num_block_samples_per_window(compat-only)={self.num_block_samples_per_window}",
                flush=True,
            )

        return {
            "input_ids": input_ids,
            "block_targets": block_targets,
            "target_offsets": torch.arange(1, block_size + 1, device=inputs.device, dtype=torch.long),
        }

    def __iter__(self):
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
            if self.training_objective == "blockwise":
                yield self._build_blockwise_batch(inputs, targets)
            else:
                yield inputs, targets

    def get_dataloader_state(self):
        """Return exact-resume state (includes doc_buffer)."""
        if self._stateful_loader is not None:
            return self._stateful_loader.state_dict()
        return self.last_state_dict  # fallback

    def __len__(self):
        return self.max_iter


def generate_dataloader(
    tokenizer,
    batch_size,
    max_len,
    max_iter,
    split,
    device,
    resume_state_dict=None,
    training_objective="dense_next_token",
    train_block_size=2,
    train_context_length=None,
    num_block_samples_per_window=1,
    debug_blockwise_shapes=False,
):

    dataset = IterableDataset(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        split=split,
        max_iter=max_iter,
        device=device,
        resume_state_dict=resume_state_dict,
        training_objective=training_objective,
        train_block_size=train_block_size,
        train_context_length=train_context_length,
        num_block_samples_per_window=num_block_samples_per_window,
        debug_blockwise_shapes=debug_blockwise_shapes,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,      # IMPORTANT
        shuffle=False,
        num_workers=0,        # keep 0 for stateful streaming
        pin_memory=False      # already handled internally
    )
    return dataloader
