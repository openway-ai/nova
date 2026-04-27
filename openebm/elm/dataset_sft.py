"""
EBT SFT Dataset - 将 NanoChat SFT 数据集适配为 EBT 训练格式

使用 NanoChat 的 TaskMixture（SmolTalk, MMLU, GSM8K 等）作为 SFT 数据源，
输出格式与 EBT pretrain dataloader 完全一致: (inputs, targets) 形状 (B, T)。
"""

import os
import sys
import torch
from torch.utils.data import IterableDataset as _IterableDataset, DataLoader

from nanochat.common import get_dist_info
from nanochat.tokenizer import get_tokenizer


def _load_sft_datasets():
    """加载 SFT 数据集混合 (支持 offline 模式)"""
    from tasks.common import TaskMixture
    from tasks.gsm8k import GSM8K
    from tasks.mmlu import MMLU
    from tasks.smoltalk import SmolTalk

    # Offline 模式支持：设置 NANOCHAT_SFT_DATA_DIR 环境变量
    # nanochat tasks 会自动使用该路径下的离线数据集
    sft_data_dir = os.environ.get("NANOCHAT_SFT_DATA_DIR", "")
    if sft_data_dir and os.path.exists(sft_data_dir):
        print(f"[EBT SFT] 使用离线数据集: {sft_data_dir}")

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
    """将 NanoChat SFT TaskMixture 适配为 EBT IterableDataset"""

    def __init__(self, tokenizer, batch_size, max_len, split, max_iter, device="cuda"):
        super().__init__()
        self.tokenizer = tokenizer
        self.B = batch_size
        self.T = max_len
        self.split = split
        self.max_iter = max_iter
        self.device = device

        self.train_dataset, self.val_dataset = _load_sft_datasets()

    def __iter__(self):
        """BOS-aligned bestfit packing，对齐 chat_sft.py"""
        ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
        dataset = self.train_dataset if self.split == "train" else self.val_dataset
        dataset_size = len(dataset)
        assert dataset_size > 0

        row_capacity = self.T + 1
        bos_token = self.tokenizer.get_bos_token_id()
        use_cuda = "cuda" in str(self.device)

        conv_buffer = []
        cursor = ddp_rank
        consumed = ddp_rank
        buffer_size = 1000
        epoch = 1

        def refill_buffer():
            nonlocal cursor, epoch
            while len(conv_buffer) < buffer_size:
                conversation = dataset[cursor]
                ids, _ = self.tokenizer.render_conversation(conversation)
                cursor += ddp_world_size
                if cursor >= dataset_size:
                    cursor = cursor % dataset_size
                    epoch += 1
                # 跳过超长对话，只保留能放入 row_capacity 的
                if len(ids) <= row_capacity:
                    conv_buffer.append(ids)

        it = 0
        while True:
            rows = []
            row_lengths = []  # 每行实际内容长度（不含 padding）

            for _ in range(self.B):
                row = []
                padded = False
                while len(row) < row_capacity:
                    while len(conv_buffer) < buffer_size:
                        refill_buffer()
                    remaining = row_capacity - len(row)

                    best_idx = -1
                    best_len = 0
                    for i, conv in enumerate(conv_buffer):
                        conv_len = len(conv)
                        if conv_len <= remaining and conv_len > best_len:
                            best_idx = i
                            best_len = conv_len

                    if best_idx >= 0:
                        conv = conv_buffer.pop(best_idx)
                        row.extend(conv)
                        consumed += ddp_world_size
                    else:
                        content_len = len(row)
                        remaining = row_capacity - len(row)
                        row.extend([bos_token] * remaining)
                        padded = True
                        break

                if padded:
                    row_lengths.append(content_len)
                else:
                    row_lengths.append(row_capacity)
                rows.append(row[:row_capacity])

            batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
            inputs = batch_tensor[:, :-1].to(device=self.device, dtype=torch.int64, non_blocking=use_cuda)
            targets = batch_tensor[:, 1:].to(device=self.device, dtype=torch.int64, non_blocking=use_cuda)
            # 只 mask padding 位置（对齐 chat_sft.py），user + assistant token 都参与 loss
            for i, content_len in enumerate(row_lengths):
                if content_len < row_capacity:
                    targets[i, content_len-1:] = -1

            yield inputs, targets

            it += 1
            if self.split == "train" and it >= self.max_iter:
                break

    def __len__(self):
        return self.max_iter


def generate_sft_dataloader(tokenizer, batch_size, max_len, max_iter, split, device, resume_state_dict=None):
    dataset = SFTIterableDataset(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        split=split,
        max_iter=max_iter,
        device=device,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return dataloader
