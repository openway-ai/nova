"""
EBT SFT Dataset - 将 NanoChat SFT 数据集适配为 EBT 训练格式

使用 NanoChat 的 TaskMixture（SmolTalk, MMLU, GSM8K 等）作为 SFT 数据源，
输出格式与 EBT pretrain dataloader 完全一致: (inputs, targets) 形状 (B, T)。
"""

import os
import sys
import copy
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
    STATE_VERSION = "sft_v1"

    def __init__(self, tokenizer, batch_size, max_len, split, max_iter, device="cuda", resume_state_dict=None):
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

    def _can_restore(self, state):
        return isinstance(state, dict) and (
            state.get("state_version") == self.STATE_VERSION or "conv_buffer" in state
        )

    def _initialize_fresh_state(self, ddp_rank, ddp_world_size):
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.row_capacity = self.T + 1
        self.buffer_size = 1000
        self.conv_buffer = []
        self.cursor = ddp_rank
        self.consumed = ddp_rank
        self.epoch = 1
        self.it = 0

    def _restore_state(self, state, ddp_rank, ddp_world_size):
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

    def _build_state_dict(self, copy_buffer):
        state = {
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
        return state

    def __iter__(self):
        """BOS-aligned bestfit packing，对齐 chat_sft.py"""
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

        def refill_buffer():
            while len(self.conv_buffer) < self.buffer_size:
                conversation = dataset[self.cursor]
                ids, _ = self.tokenizer.render_conversation(conversation)
                self.cursor += ddp_world_size
                if self.cursor >= dataset_size:
                    self.cursor = self.cursor % dataset_size
                    self.epoch += 1
                # 跳过超长对话，只保留能放入 row_capacity 的
                if len(ids) <= self.row_capacity:
                    self.conv_buffer.append(ids)

        while True:
            if self.split == "train" and self.it >= self.max_iter:
                break

            rows = []
            row_lengths = []  # 每行实际内容长度（不含 padding）

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
            # 只 mask padding 位置（对齐 chat_sft.py），user + assistant token 都参与 loss
            for i, content_len in enumerate(row_lengths):
                if content_len < self.row_capacity:
                    targets[i, content_len-1:] = -1
            self.it += 1
            self.last_state_dict = self._build_state_dict(copy_buffer=False)
            yield inputs, targets
            if self.split != "train" and self.it >= self.max_iter:
                break

    def get_dataloader_state(self):
        if self.cursor is None:
            return self.last_state_dict
        return self._build_state_dict(copy_buffer=True)

    def state_dict(self):
        state = self.get_dataloader_state()
        return {} if state is None else state

    def load_state_dict(self, state_dict):
        # Lightning may try to restore a single legacy dataloader_state_dict after we
        # have already injected the correct per-rank state via trainer.py. Keep the
        # per-rank ctor state authoritative in that case.
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

    def __len__(self):
        return self.max_iter


class SFTDataLoader(DataLoader):
    def state_dict(self):
        dataset = getattr(self, "dataset", None)
        if dataset is None or not hasattr(dataset, "state_dict"):
            return {}
        return dataset.state_dict()

    def load_state_dict(self, state_dict):
        dataset = getattr(self, "dataset", None)
        if dataset is not None and hasattr(dataset, "load_state_dict"):
            dataset.load_state_dict(state_dict)


def generate_sft_dataloader(tokenizer, batch_size, max_len, max_iter, split, device, resume_state_dict=None):
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
