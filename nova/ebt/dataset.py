import sys
sys.path.append("../../")

from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit

import torch
from torch.utils.data import IterableDataset as _IterableDataset
from torch.utils.data import IterableDataset, DataLoader

class IterableDataset(_IterableDataset):
    """
    Wraps tokenizing_distributed_data_loader_with_state_bos_bestfit
    into a PyTorch IterableDataset.

    This keeps:
    - infinite streaming
    - resume state support
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
        # tokenizer_threads=4,
        # tokenizer_batch_size=128,
        device="cuda",
        resume_state_dict=None,
        # buffer_size=1000,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.B = batch_size
        self.T = max_len
        self.split = split
        self.max_iter = max_iter
        # self.tokenizer_threads = tokenizer_threads
        # self.tokenizer_batch_size = tokenizer_batch_size
        self.device = device
        self.resume_state_dict = resume_state_dict
        # self.buffer_size = buffer_size
        self.batch_idx = 0

    def __iter__(self):
        
        return tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tokenizer=self.tokenizer,
            B=self.B,
            T=self.T,
            split=self.split,
            # tokenizer_threads=self.tokenizer_threads,
            # tokenizer_batch_size=self.tokenizer_batch_size,
            device=self.device,
            resume_state_dict=self.resume_state_dict,
            # buffer_size=1000,
        )

        # inputs, targets, state_dict = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        #     tokenizer=self.tokenizer,
        #     B=self.B,
        #     T=self.T,
        #     split=self.split,
        #     # tokenizer_threads=self.tokenizer_threads,
        #     # tokenizer_batch_size=self.tokenizer_batch_size,
        #     device=self.device,
        #     resume_state_dict=self.resume_state_dict,
        #     # buffer_size=1000,
        # )
        # state_dict["batch_idx"] = self.batch_idx
        # self.batch_idx += 1
        # yield inputs, targets, state_dict

    
    def __len__(self):
        return self.max_iter


def generate_dataloader(tokenizer, batch_size, max_len, max_iter, split, device, resume_state_dict=None):

    dataset = IterableDataset(
        tokenizer=tokenizer,
        B=batch_size, 
        T=max_len,
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