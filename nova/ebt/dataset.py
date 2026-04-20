import sys
sys.path.append("../../")

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
            yield inputs, targets

    def get_dataloader_state(self):
        """Return exact-resume state (includes doc_buffer)."""
        if self._stateful_loader is not None:
            return self._stateful_loader.state_dict()
        return self.last_state_dict  # fallback

    def __len__(self):
        return self.max_iter


def generate_dataloader(tokenizer, batch_size, max_len, max_iter, split, device, resume_state_dict=None):

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
        batch_size=1,      # IMPORTANT
        shuffle=False,
        num_workers=0,        # keep 0 for stateful streaming
        pin_memory=False      # already handled internally
    )
    return dataloader
