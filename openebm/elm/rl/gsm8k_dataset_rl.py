"""RL-specific GSM8K dataset — mirrors sudoku_dataset_rl.py structure.

Loads GSM8K problems from the HuggingFace `openai/gsm8k` cache used by
run_ebt_sft.sh (`${NANOCHAT_SFT_DATA_DIR}/gsm8k`). Falls back to the
nanochat eval_bundle JSONL if HF cache is unavailable.

Each item:
  - prompt_ids:       list[int]  — token IDs up to <|assistant_start|>
  - question:         str        — raw question text
  - answer:           str        — numerical answer string (e.g. "18")
  - chain_of_thought: str        — full CoT reference solution

Data layout when loaded from HF cache:
  ${NANOCHAT_SFT_DATA_DIR}/gsm8k/openai___gsm8k/main/0.0.0/<hash>/
      gsm8k-train.arrow   (7473 rows)
      gsm8k-test.arrow    (1319 rows)

Columns: 'question', 'answer'. The 'answer' field contains the full CoT
plus a trailing '#### <N>' line — we extract the numeric answer below.
"""

import glob
import json
import os
import random
import re

import numpy as np
import torch
from torch.utils.data import IterableDataset

from nanochat.common import get_dist_info

# ── Default data paths ────────────────────────────────────────────────────────
_HF_CACHE_ROOT_DEFAULT = os.path.join(
    os.environ.get("NANOCHAT_SFT_DATA_DIR",
                   "/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/sft_data"),
    "gsm8k",
)
_FALLBACK_JSONL = (
    "/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/"
    "eval_bundle/eval_data/symbolic_problem_solving/gsm8k.jsonl"
)

_GSM_HASH_RE = re.compile(r"####\s*(-?[\d,\.]+)")

# Prompt templates (single-turn math). Keep concise: EBT context = 2048.
GSM8K_PROMPT_TEMPLATES = [
    "Solve the following math problem step by step.\n\nProblem: {question}\n\nSolution:",
    "Please solve this math problem:\n\n{question}\n\nShow your work:",
    "Work through this problem carefully:\n\n{question}",
    "{question}\n\nPlease solve step by step.",
    "Math problem: {question}\n\nAnswer with step-by-step reasoning:",
]


def _split_answer_field(answer_text: str):
    """Split GSM8K official 'answer' field into (chain_of_thought, final_answer).

    Format example:
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n
         Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n
         #### 72"
    """
    m = _GSM_HASH_RE.search(answer_text)
    if m:
        final = m.group(1).replace(",", "").strip()
        cot = answer_text[:m.start()].rstrip()
    else:
        final = answer_text.strip().split()[-1] if answer_text.strip() else ""
        cot = answer_text
    return cot, final


def _find_arrow_file(hf_cache_root: str, split: str):
    """Locate gsm8k-<split>.arrow inside the HF dataset cache layout.

    HF caches under: {root}/openai___gsm8k/main/<version>/<hash>/gsm8k-<split>.arrow
    """
    pattern = os.path.join(
        hf_cache_root, "openai___gsm8k", "main", "*", "*", f"gsm8k-{split}.arrow"
    )
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def _load_arrow(path: str):
    """Read a HF .arrow file using pyarrow IPC streaming."""
    import pyarrow.ipc as ipc
    with open(path, "rb") as f:
        table = ipc.open_stream(f).read_all()
    questions = table["question"].to_pylist()
    answers = table["answer"].to_pylist()
    samples = []
    for q, a in zip(questions, answers):
        cot, final = _split_answer_field(a)
        samples.append({
            "question": q,
            "chain_of_thought": cot,
            "answer": final,
        })
    return samples


def _load_jsonl(path: str):
    """Fallback loader for nanochat eval_bundle JSONL."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append({
                "question": obj.get("context", obj.get("question", "")),
                "chain_of_thought": obj.get("chain_of_thought", ""),
                "answer": str(obj.get("answer", "")),
            })
    return samples


def _load_gsm8k(data_path: str = None, split: str = "train", seed: int = 42):
    """Load GSM8K. Resolution order:
      1. data_path (if it ends with .arrow or .jsonl, load directly)
      2. data_path treated as HF cache root (find gsm8k-{split}.arrow inside)
      3. env GSM8K_DATA_PATH / GSM8K_HF_CACHE
      4. ${NANOCHAT_SFT_DATA_DIR}/gsm8k
      5. nanochat eval_bundle JSONL fallback (test only — uses 10% holdout for val)
    """
    candidate_path = data_path or os.environ.get("GSM8K_DATA_PATH", "")
    samples = None

    # 1. Direct file path
    if candidate_path and os.path.isfile(candidate_path):
        if candidate_path.endswith(".arrow"):
            samples = _load_arrow(candidate_path)
        elif candidate_path.endswith(".jsonl"):
            samples = _load_jsonl(candidate_path)

    # 2. Treat as HF cache root
    if samples is None:
        roots = []
        if candidate_path and os.path.isdir(candidate_path):
            roots.append(candidate_path)
        roots.append(os.environ.get("GSM8K_HF_CACHE", ""))
        roots.append(_HF_CACHE_ROOT_DEFAULT)
        for root in roots:
            if not root:
                continue
            arrow_file = _find_arrow_file(root, split)
            if arrow_file:
                samples = _load_arrow(arrow_file)
                print(f"[GSM8K-RL] Loaded {split} from HF arrow: {arrow_file}", flush=True)
                break

    # 3. JSONL fallback
    if samples is None:
        if not os.path.exists(_FALLBACK_JSONL):
            raise FileNotFoundError(
                f"GSM8K data not found.\n"
                f"  Tried HF cache at {_HF_CACHE_ROOT_DEFAULT}\n"
                f"  Tried JSONL at  {_FALLBACK_JSONL}\n"
                f"Set GSM8K_DATA_PATH (file or HF cache root) to a valid location."
            )
        print(f"[GSM8K-RL] HF cache not found, using JSONL fallback: {_FALLBACK_JSONL}", flush=True)
        all_samples = _load_jsonl(_FALLBACK_JSONL)
        rng = random.Random(seed)
        rng.shuffle(all_samples)
        n_val = max(1, int(len(all_samples) * 0.1))
        samples = all_samples[:n_val] if split == "val" else all_samples[n_val:]

    # Deterministic shuffle for reproducibility
    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples


class GSM8KRLPromptDataset(IterableDataset):
    """Iterable dataset yielding GSM8K prompts for RL rollout.

    Reuses the HF `openai/gsm8k` cache populated by `run_ebt_sft.sh`.
    Mirrors :class:`SudokuRLPromptDataset` so EBMGRPOTrainer can swap tasks
    with minimal change.

    Args:
        split: 'train' (7473 examples) or 'val' (uses HF 'test' = 1319 examples).
        data_path: HF cache root, .arrow file path, or .jsonl fallback path.
    """

    # Map our 'val' notion → HF dataset's 'test' split.
    _SPLIT_MAP = {"train": "train", "val": "test", "test": "test"}

    def __init__(
        self,
        tokenizer,
        max_prompt_length: int = 512,
        split: str = "train",
        data_path: str = None,
        seed: int = 42,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.split = split
        self._seed = seed
        self._inner_tok = getattr(tokenizer, "tokenizer", tokenizer)

        # Sanity-check special token encoding
        asst_start_id = self._inner_tok.encode_special("<|assistant_start|>")
        asst_end_id = self._inner_tok.encode_special("<|assistant_end|>")
        assert isinstance(asst_start_id, int) and asst_start_id > 32000, (
            f"Tokenizer encode_special broken: <|assistant_start|>={asst_start_id}"
        )
        assert isinstance(asst_end_id, int) and asst_end_id > 32000, (
            f"Tokenizer encode_special broken: <|assistant_end|>={asst_end_id}"
        )

        hf_split = self._SPLIT_MAP.get(split, split)
        self.samples = _load_gsm8k(data_path, split=hf_split, seed=seed)
        print(
            f"[GSM8K-RL] {split} (HF split={hf_split}): {len(self.samples)} examples",
            flush=True,
        )

    def __iter__(self):
        is_ddp, rank, local_rank, world_size = get_dist_info()
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0

        seed = self._seed + rank * 1000 + worker_id
        rng = np.random.default_rng(seed)
        py_rng = random.Random(seed)

        indices = list(range(len(self.samples)))
        rng.shuffle(indices)

        while True:
            for idx in indices:
                sample = self.samples[idx]
                question = sample["question"]

                tmpl = py_rng.choice(GSM8K_PROMPT_TEMPLATES)
                user_content = tmpl.format(question=question)

                conv = {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": ""},
                    ]
                }
                prompt_ids = self._inner_tok.render_for_completion(conv)
                if len(prompt_ids) > self.max_prompt_length:
                    prompt_ids = prompt_ids[-self.max_prompt_length:]

                yield {
                    "prompt_ids": prompt_ids,
                    "question": question,
                    "answer": sample["answer"],
                    "chain_of_thought": sample["chain_of_thought"],
                }

            rng.shuffle(indices)


def collate_gsm8k_prompts(batch, tokenizer, max_prompt_length: int):
    """Collate GSM8K RL prompt dicts into tensors (left-padded).

    Returns dict with:
      - prompt_ids:        (B, max_len) long tensor
      - prompt_lengths:    (B,) long tensor
      - questions:         list[str]
      - answers:           list[str]
      - chain_of_thoughts: list[str]
    """
    all_ids = [item["prompt_ids"] for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    cots = [item.get("chain_of_thought", "") for item in batch]

    all_ids = [ids[-max_prompt_length:] for ids in all_ids]
    max_len = max(len(ids) for ids in all_ids)
    pad_id = int(getattr(tokenizer, "bos_token_id", 0) or 0)

    prompt_ids = torch.full((len(all_ids), max_len), pad_id, dtype=torch.long)
    prompt_lengths = torch.zeros(len(all_ids), dtype=torch.long)
    for i, ids in enumerate(all_ids):
        length = len(ids)
        prompt_lengths[i] = length
        prompt_ids[i, max_len - length:] = torch.tensor(ids, dtype=torch.long)

    return {
        "prompt_ids": prompt_ids,
        "prompt_lengths": prompt_lengths,
        "questions": questions,
        "answers": answers,
        "chain_of_thoughts": cots,
    }
