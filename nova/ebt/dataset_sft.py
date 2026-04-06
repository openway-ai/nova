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
    """加载 SFT 数据集混合 (支持 offline 模式)"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
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
        buffer_size = 100
        epoch = 1

        def refill_buffer():
            nonlocal cursor, epoch
            while len(conv_buffer) < buffer_size:
                conversation = dataset[cursor]
                ids, mask = self.tokenizer.render_conversation(conversation)
                # 截断超长对话：保留尾部（assistant 回复通常在末尾）
                # 这样超过 row_capacity 的对话也能参与 packing，减少 padding 浪费
                if len(ids) > row_capacity:
                    ids = ids[-row_capacity:]
                    mask = mask[-row_capacity:]
                conv_buffer.append((ids, mask))
                cursor += ddp_world_size
                if cursor >= dataset_size:
                    cursor = cursor % dataset_size
                    epoch += 1

        it = 0
        while True:
            rows = []
            row_masks = []  # SFT mask: 1 = train on this token (assistant), 0 = ignore (user/padding)

            for _ in range(self.B):
                row = []
                row_mask = []
                padded = False
                while len(row) < row_capacity:
                    while len(conv_buffer) < buffer_size:
                        refill_buffer()
                    remaining = row_capacity - len(row)

                    best_idx = -1
                    best_len = 0
                    for i, (conv, _) in enumerate(conv_buffer):
                        conv_len = len(conv)
                        if conv_len <= remaining and conv_len > best_len:
                            best_idx = i
                            best_len = conv_len

                    if best_idx >= 0:
                        conv, mask = conv_buffer.pop(best_idx)
                        row.extend(conv)
                        row_mask.extend(mask)
                        consumed += ddp_world_size
                    else:
                        # 没有对话能放进剩余空间，用 bos padding 填满
                        remaining = row_capacity - len(row)
                        row.extend([bos_token] * remaining)
                        row_mask.extend([0] * remaining)  # padding 位置不训练
                        padded = True
                        break

                rows.append(row[:row_capacity])
                row_masks.append(row_mask[:row_capacity])

            batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
            mask_tensor = torch.tensor(row_masks, dtype=torch.bool)  # (B, T+1)
            inputs = batch_tensor[:, :-1].to(device=self.device, dtype=torch.int64, non_blocking=use_cuda)
            targets = batch_tensor[:, 1:].to(device=self.device, dtype=torch.int64, non_blocking=use_cuda)
            # targets mask: 对 mask=0 的位置（user tokens 和 padding）设为 ignore_index=-1
            # mask_tensor[:, 1:] 对应 targets（shifted by 1）
            # 只训练 assistant 回复 token，忽略 user 问题 token 和 padding token
            targets_mask = mask_tensor[:, 1:].to(device=self.device, non_blocking=use_cuda)
            targets[~targets_mask] = -1

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
