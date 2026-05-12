"""NanoChat evaluation dataset with dual-mode support.

Supports two evaluation modes on the same cached parquet shards:

- **PPL mode** — full sequence used for perplexity calculation.
- **Generation mode** — sequence split into a prompt prefix and a ground-truth
  continuation, for conditional generation benchmarks.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class NanoChatShardEvalDataset(Dataset):
    """Evaluate a NanoChat-tokenized model on selected parquet shards.

    Supports two evaluation modes:

    - PPL mode: full sequence used for perplexity calculation.
    - Generation mode: sequence split into a prompt (first half) and a target
      (second half) used for conditional generation scoring.
    """

    def __init__(
        self,
        tokenizer: Any,
        context_length: int = 256,
        shard_indices: Sequence[int] = [0, 15],
        max_samples_per_shard: int = 50,
        data_dir: str = "/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/base_data",
        enable_generation: bool = True,
        generation_split_ratio: float = 0.5,
        min_generation_length: int = 64,
    ) -> None:
        """Initialize the dataset.

        :param tokenizer: Tokenizer exposing ``encode`` and, optionally,
            ``decode`` / ``get_bos_token_id``.
        :type tokenizer: Any
        :param context_length: Maximum sequence length used for PPL.
        :type context_length: int
        :param shard_indices: Indices of the parquet shards to load
            (e.g. ``[0, 15]`` for the first and last).
        :type shard_indices: Sequence[int]
        :param max_samples_per_shard: Maximum samples drawn from each shard.
        :type max_samples_per_shard: int
        :param data_dir: Directory containing the ``shard_{idx:05d}.parquet``
            files.
        :type data_dir: str
        :param enable_generation: When ``True``, also emit a prompt/target
            split suitable for generation evaluation.
        :type enable_generation: bool
        :param generation_split_ratio: Fraction of tokens assigned to the
            prompt (``0.5`` = half/half).
        :type generation_split_ratio: float
        :param min_generation_length: Minimum token count required to keep a
            sample for generation evaluation.
        :type min_generation_length: int
        """
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.shard_indices = shard_indices
        self.max_samples_per_shard = max_samples_per_shard
        self.data_dir = data_dir
        self.enable_generation = enable_generation
        self.generation_split_ratio = generation_split_ratio
        self.min_generation_length = min_generation_length

        if hasattr(tokenizer, 'get_bos_token_id'):
            self.bos_token = tokenizer.get_bos_token_id()
        elif hasattr(tokenizer, 'bos_token_id'):
            self.bos_token = tokenizer.bos_token_id
        else:
            self.bos_token = 1

        self.samples = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Load tokenized samples from each configured shard.

        Populates :attr:`samples` in place and prints a short progress trace.
        """
        for shard_idx in self.shard_indices:
            shard_file = os.path.join(self.data_dir, f"shard_{shard_idx:05d}.parquet")

            if not os.path.exists(shard_file):
                print(f"Warning: Shard file not found: {shard_file}")
                continue

            print(f"Loading samples from {shard_file}...")
            pf = pq.ParquetFile(shard_file)

            samples_from_shard = 0
            for rg_idx in range(pf.num_row_groups):
                if samples_from_shard >= self.max_samples_per_shard:
                    break

                rg = pf.read_row_group(rg_idx)
                texts = rg.column('text').to_pylist()

                for text in texts:
                    if samples_from_shard >= self.max_samples_per_shard:
                        break

                    if hasattr(self.tokenizer, 'encode'):
                        if hasattr(self.tokenizer, 'enc'):
                            # NanoChat RustBPETokenizer path.
                            tokens = self.tokenizer.encode([text], prepend=self.bos_token, num_threads=1)[0]
                        else:
                            # HuggingFace-style tokenizer path.
                            tokens = self.tokenizer.encode(text, add_special_tokens=True)
                    else:
                        raise ValueError("Tokenizer must have encode method")

                    # Generation mode needs longer samples than PPL mode.
                    min_length = self.min_generation_length if self.enable_generation else self.context_length // 2

                    if len(tokens) >= min_length:
                        self.samples.append({
                            'tokens': tokens,
                            'text': text,
                            'shard_idx': shard_idx
                        })
                        samples_from_shard += 1

            print(f"  Loaded {samples_from_shard} samples from shard {shard_idx}")

        print(f"Total samples loaded: {len(self.samples)}")

    def __len__(self) -> int:
        """Return the number of samples loaded across all shards.

        :return: Sample count.
        :rtype: int
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return the sample at ``idx``.

        The returned dict always contains PPL fields. When
        ``enable_generation`` is set and the sample is long enough, the dict
        also contains generation fields.

        :param idx: Sample index.
        :type idx: int
        :return: Dict with keys ``input_ids`` (full sequence tensor),
            ``text``, ``shard_idx``, ``num_tokens``, and optionally
            ``prompt_ids``, ``target_ids``, ``prompt_text``, ``target_text``.
        :rtype: Dict[str, Any]
        """
        sample = self.samples[idx]
        tokens = sample['tokens']

        if len(tokens) > self.context_length:
            tokens = tokens[:self.context_length]

        input_ids = torch.tensor(tokens, dtype=torch.long)

        result = {
            'input_ids': input_ids,
            'text': sample['text'][:500],
            'shard_idx': sample['shard_idx'],
            'num_tokens': len(tokens)
        }

        if self.enable_generation and len(tokens) >= self.min_generation_length:
            split_point = int(len(tokens) * self.generation_split_ratio)

            # Enforce at least 10 tokens on each side of the split.
            split_point = max(split_point, 10)
            split_point = min(split_point, len(tokens) - 10)

            prompt_tokens = tokens[:split_point]
            target_tokens = tokens[split_point:]

            result['prompt_ids'] = torch.tensor(prompt_tokens, dtype=torch.long)
            result['target_ids'] = torch.tensor(target_tokens, dtype=torch.long)

            if hasattr(self.tokenizer, 'decode'):
                try:
                    result['prompt_text'] = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
                    result['target_text'] = self.tokenizer.decode(target_tokens, skip_special_tokens=True)
                except:
                    # Decode failures fall back to a raw character-level split.
                    result['prompt_text'] = sample['text'][:len(sample['text'])//2]
                    result['target_text'] = sample['text'][len(sample['text'])//2:]
            else:
                text_split = int(len(sample['text']) * self.generation_split_ratio)
                result['prompt_text'] = sample['text'][:text_split]
                result['target_text'] = sample['text'][text_split:]

        return result


def collate_fn_nanochat_eval(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate :class:`NanoChatShardEvalDataset` samples into a batch dict.

    :param batch: List of sample dicts produced by :meth:`NanoChatShardEvalDataset.__getitem__`.
    :type batch: List[Dict[str, Any]]
    :return: Dict with PPL fields ``input_ids`` (shape ``[B, 1, S]``),
        ``attention_mask``, ``texts``, ``shard_indices``, ``num_tokens``.
        When generation data is present in the batch, also includes
        ``prompt_ids``, ``prompt_attention_mask``, ``prompt_texts``,
        ``target_texts`` and ``target_ids``.
    :rtype: Dict[str, Any]
    """
    max_len = max(item['input_ids'].shape[0] for item in batch)

    input_ids_list = []
    attention_mask_list = []

    for item in batch:
        seq_len = item['input_ids'].shape[0]
        padding_len = max_len - seq_len

        padded_input_ids = torch.cat([
            item['input_ids'],
            torch.zeros(padding_len, dtype=torch.long)
        ])

        attention_mask = torch.cat([
            torch.ones(seq_len, dtype=torch.long),
            torch.zeros(padding_len, dtype=torch.long)
        ])

        input_ids_list.append(padded_input_ids)
        attention_mask_list.append(attention_mask)

    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)

    # Add a channel dimension for compatibility with downstream code: [B, S] -> [B, 1, S].
    input_ids = input_ids.unsqueeze(1)

    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': [item['text'] for item in batch],
        'shard_indices': [item['shard_idx'] for item in batch],
        'num_tokens': [item['num_tokens'] for item in batch]
    }

    if 'prompt_ids' in batch[0]:
        max_prompt_len = max(item['prompt_ids'].shape[0] for item in batch)
        prompt_ids_list = []
        prompt_attention_mask_list = []

        for item in batch:
            prompt_len = item['prompt_ids'].shape[0]
            padding_len = max_prompt_len - prompt_len

            padded_prompt = torch.cat([
                item['prompt_ids'],
                torch.zeros(padding_len, dtype=torch.long)
            ])

            prompt_mask = torch.cat([
                torch.ones(prompt_len, dtype=torch.long),
                torch.zeros(padding_len, dtype=torch.long)
            ])

            prompt_ids_list.append(padded_prompt)
            prompt_attention_mask_list.append(prompt_mask)

        result['prompt_ids'] = torch.stack(prompt_ids_list)
        result['prompt_attention_mask'] = torch.stack(prompt_attention_mask_list)
        result['prompt_texts'] = [item['prompt_text'] for item in batch]
        result['target_texts'] = [item['target_text'] for item in batch]

        # Targets are kept as a list (no padding required for logging/scoring).
        result['target_ids'] = [item['target_ids'] for item in batch]

    return result
