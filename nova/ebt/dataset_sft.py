"""
EBT SFT Dataset - 将 NanoChat SFT 数据集适配为 EBT 训练格式

使用 NanoChat 的 TaskMixture（SmolTalk, MMLU, GSM8K 等）作为 SFT 数据源，
输出格式与 EBT pretrain dataloader 完全一致: (inputs, targets) 形状 (B, T)。
"""

import os
import sys
import torch
from torch.utils.data import IterableDataset as _IterableDataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from nanochat.common import get_dist_info
from nanochat.tokenizer import get_tokenizer


def _load_sft_datasets():
    """加载 SFT 数据集混合"""
    # 导入 nanochat 的 SFT task 类
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
    from tasks.common import TaskMixture
    from tasks.gsm8k import GSM8K
    from tasks.mmlu import MMLU
    from tasks.smoltalk import SmolTalk

    base_dir = os.environ.get("NANOCHAT_BASE_DIR", "")
    identity_path = os.path.join(base_dir, "identity_conversations.jsonl")

    # 训练集: SmolTalk + MMLU + GSM8K (与 nanochat SFT 一致)
    train_tasks = [
        SmolTalk(split="train"),        # 460K 对话
        MMLU(subset="auxiliary_train", split="train"),  # 100K 选择题
        GSM8K(subset="main", split="train"),            # 8K 数学
        GSM8K(subset="main", split="train"),            # 2 epochs GSM8K
    ]

    # 可选: 如果有 identity_conversations.jsonl 和拼写任务
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
        """BOS-aligned bestfit packing，与 nanochat chat_sft.py 逻辑一致"""
        ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
        dataset = self.train_dataset if self.split == "train" else self.val_dataset
        dataset_size = len(dataset)
        assert dataset_size > 0

        row_capacity = self.T + 1  # +1 用于 target 的最后位置
        bos_token = self.tokenizer.get_bos_token_id()
        use_cuda = "cuda" in str(self.device)

        # 对话缓冲
        conv_buffer = []
        cursor = ddp_rank
        buffer_size = 100

        def refill_buffer():
            nonlocal cursor
            while len(conv_buffer) < buffer_size:
                conversation = dataset[cursor]
                ids, _ = self.tokenizer.render_conversation(conversation)
                conv_buffer.append(ids)
                cursor += ddp_world_size
                if cursor >= dataset_size:
                    cursor = cursor % dataset_size

        it = 0
        while True:
            rows = []
            row_lengths = []

            for _ in range(self.B):
                row = []
                padded = False
                while len(row) < row_capacity:
                    refill_buffer()
                    remaining = row_capacity - len(row)

                    # Best-fit: 找最大的能放进去的对话
                    best_idx = -1
                    best_len = 0
                    for i, conv in enumerate(conv_buffer):
                        conv_len = len(conv)
                        if conv_len <= remaining and conv_len > best_len:
                            best_idx = i
                            best_len = conv_len

                    if best_idx >= 0:
                        row.extend(conv_buffer.pop(best_idx))
                    else:
                        content_len = len(row)
                        row.extend([bos_token] * remaining)
                        padded = True
                        break

                row_lengths.append(len(row) - remaining if padded else row_capacity)
                rows.append(row[:row_capacity])

            # 构建张量
            batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
            inputs = batch_tensor[:, :-1].to(device=self.device, dtype=torch.int64, non_blocking=use_cuda)
            targets = batch_tensor[:, 1:].to(device=self.device, dtype=torch.int64, non_blocking=use_cuda)

            # Mask padding 位置
            for i, content_len in enumerate(row_lengths):
                if content_len < row_capacity:
                    targets[i, content_len - 1:] = -1

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
