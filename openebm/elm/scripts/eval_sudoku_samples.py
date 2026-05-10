#!/usr/bin/env python3
"""Quick evaluation script for SFT-finetuned EBT on the Sudoku v2 dataset.

For each split (train/val/test) it runs puzzles through the chat engine,
prints the puzzle / ground-truth / model output, and reports cell accuracy.
Supports per-split sample counts, full-split evaluation, and automatic
multi-GPU sharding.

Examples:
    # 1) Single GPU, 2 samples per split (default).
    python -m openebm.elm.scripts.eval_sudoku_samples -c CKPT --num_samples 2 --single_gpu

    # 2) Different sample count per split.
    python -m openebm.elm.scripts.eval_sudoku_samples -c CKPT \
        --num_samples_train 8 --num_samples_val 4 --num_samples_test 16 --single_gpu

    # 3) Auto multi-GPU + full evaluation.
    python -m openebm.elm.scripts.eval_sudoku_samples -c CKPT --full

    # 4) Force GPU count.
    python -m openebm.elm.scripts.eval_sudoku_samples -c CKPT --full --gpus 4

    # 5) Direct torchrun launch.mu
    torchrun --standalone --nproc_per_node=8 \
        -m openebm.elm.scripts.eval_sudoku_samples -c CKPT --full
"""
from __future__ import annotations

import argparse
import os
import socket

# Only clear distributed env vars if we are NOT a worker spawned by
# torchrun / mp.spawn (those set WORLD_SIZE).
if not int(os.environ.get('WORLD_SIZE', '0')):
    for _var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']:
        if _var in os.environ:
            del os.environ[_var]

import random
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.multiprocessing as mp

from openebm.elm.scripts.chat_ebt import EBTChatEngine

DEFAULT_DATA_DIR = Path('/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/data/sudoku_cache_v2')
DEFAULT_TOKENIZER = '/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/elm_d26_ctx2048/tokenizer.json'
DEFAULT_RUN_DIR = Path('/mnt/shared-storage-user/puyuan/code/OpenEBM/openebm/elm/runs/d26-ctx2048-sudoku-mixed-0.6-20260508')

PROMPT_TEMPLATE = (
    "请解决以下数独谜题。"
    "在每个9×9的网格中，请用1-9填充空格（用0表示），"
    "确保每行、每列以及每个3×3小方格中的数字都不重复。\n\n"
    "数独谜题：\n{puzzle}\n\n"
    "请直接给出完整的解答（每行用空格分隔数字）："
)


class _Tee:
    """Duplicate writes to multiple streams (e.g., stdout + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


class _PrefixStream:
    """Prefix every line written to a stream (used for non-rank-0 workers)."""

    def __init__(self, stream, prefix):
        self.stream = stream
        self.prefix = prefix
        self._at_line_start = True

    def write(self, data):
        if not data:
            return
        out = []
        for ch in data:
            if self._at_line_start and ch != '\n':
                out.append(self.prefix)
                self._at_line_start = False
            out.append(ch)
            if ch == '\n':
                self._at_line_start = True
        try:
            self.stream.write(''.join(out))
        except Exception:
            pass

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass


def _resolve_log_path(checkpoint: str, run_dir: Path) -> Path:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_stem = Path(checkpoint).stem if checkpoint else 'nockpt'
    log_dir = run_dir / 'eval_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f'{ckpt_stem}_{ts}.log'


def _format_board(values):
    return "\n".join(" ".join(str(v) for v in values[r * 9:(r + 1) * 9]) for r in range(9))


def parse_board(text):
    nums = [int(x) for x in text.split() if x.isdigit() and len(x) == 1]
    if len(nums) < 81:
        return None
    return nums[:81]


def evaluate_sample(engine, sample):
    flat_puzzle = sample['puzzle']
    if hasattr(flat_puzzle, 'tolist'):
        flat_puzzle = flat_puzzle.tolist()
    flat_solution = sample['solution']
    if hasattr(flat_solution, 'tolist'):
        flat_solution = flat_solution.tolist()
    if hasattr(flat_puzzle[0], '__iter__'):
        flat_puzzle = [v for row in flat_puzzle for v in row]
    if hasattr(flat_solution[0], '__iter__'):
        flat_solution = [v for row in flat_solution for v in row]
    puzzle_text = _format_board(flat_puzzle)
    solution_text = _format_board(flat_solution)
    prompt = PROMPT_TEMPLATE.format(puzzle=puzzle_text)

    response, _ = engine.generate_response(prompt, max_new_tokens=400, temperature=0.0)
    pred = parse_board(response)

    correct, total = 0, 81
    if pred is not None:
        for r in range(9):
            for c in range(9):
                if pred[r * 9 + c] == flat_solution[r * 9 + c]:
                    correct += 1
    return puzzle_text, solution_text, response, correct, total, pred is not None


SPLIT_FILES = [
    ('train', 'rrn_train.pt'),
    ('val',   'rrn_val.pt'),
    ('test',  'satnet_test.pt'),
]


def load_splits(data_dir):
    splits = {}
    for name, fname in SPLIT_FILES:
        path = Path(data_dir) / fname
        if path.is_file():
            splits[name] = torch.load(path, weights_only=False)
    return splits


def _resolve_split_sizes(args, splits):
    raw = {
        'train': args.num_samples_train if args.num_samples_train is not None else args.num_samples,
        'val':   args.num_samples_val   if args.num_samples_val   is not None else args.num_samples,
        'test':  args.num_samples_test  if args.num_samples_test  is not None else args.num_samples,
    }
    full_flags = {
        'train': args.full or args.full_train,
        'val':   args.full or args.full_val,
        'test':  args.full or args.full_test,
    }
    out = {}
    for name, samples in splits.items():
        total = len(samples)
        n = raw[name]
        if full_flags[name] or n is None or n < 0:
            n = total
        out[name] = min(n, total)
    return out


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _decide_world_size(args, ngpu: int) -> int:
    if args.single_gpu:
        return 1
    if args.gpus and args.gpus > 0:
        return min(args.gpus, max(ngpu, 1))
    # auto
    return max(ngpu, 1)


def _setup_logging_if_needed(args, is_main: bool, rank: int):
    if is_main:
        log_path = args.log_file or _resolve_log_path(args.checkpoint, args.run_dir)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, 'a', buffering=1)
        sys.stdout = _Tee(sys.__stdout__, log_file)
        sys.stderr = _Tee(sys.__stderr__, log_file)
        print(f"[eval_sudoku_samples] checkpoint={args.checkpoint}")
        print(f"[eval_sudoku_samples] log_file={log_path}")
    else:
        prefix = f"[rank={rank}] "
        sys.stdout = _PrefixStream(sys.__stdout__, prefix)
        sys.stderr = _PrefixStream(sys.__stderr__, prefix)


def _run_eval(args, rank: int, world_size: int):
    import torch.distributed as dist

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    splits = load_splits(args.data_dir)
    if not splits:
        if rank == 0:
            print(f"No splits found under {args.data_dir}")
        return

    dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}[args.dtype]
    engine = EBTChatEngine(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device,
        dtype=dtype,
        use_compile=args.use_compile,
        eval_mode=args.eval_mode,
        verbose=(rank == 0),
    )

    split_n = _resolve_split_sizes(args, splits)
    if rank == 0:
        print(f"[eval_sudoku_samples] world_size={world_size} split sizes: {split_n}")

    reduce_device = args.device if str(args.device).startswith('cuda') else 'cpu'
    global_correct = 0.0
    global_total = 0.0

    for split_name, samples in splits.items():
        n = split_n[split_name]
        all_indices = list(range(n))
        my_indices = all_indices[rank::world_size]
        if rank == 0:
            print(f"\n========== Split: {split_name} ({n} samples, world_size={world_size}) ==========")
        local_correct, local_total, local_parsed = 0, 0, 0
        for i in my_indices:
            sample = samples[i]
            puzzle_text, solution_text, response, c, t, ok = evaluate_sample(engine, sample)
            tag = f"--- {split_name} sample {i} (rank={rank}) ---"
            print(f"\n{tag}")
            print('[PUZZLE]')
            print(puzzle_text)
            print('[GROUND TRUTH]')
            print(solution_text)
            print('[MODEL OUTPUT]')
            print(response)
            print(f'[CELL ACCURACY] {c}/{t} parsed={ok}')
            local_correct += c
            local_total += t
            local_parsed += int(ok)

        ct = torch.tensor([local_correct, local_total, local_parsed, len(my_indices)],
                          device=reduce_device, dtype=torch.float64)
        if world_size > 1:
            dist.all_reduce(ct, op=dist.ReduceOp.SUM)
        if rank == 0:
            agg_c = int(ct[0].item())
            agg_t = int(ct[1].item())
            agg_ok = int(ct[2].item())
            agg_n = int(ct[3].item())
            acc = (agg_c / agg_t * 100) if agg_t else 0.0
            print(f"\n>>> {split_name} aggregate cell accuracy: {agg_c}/{agg_t} = "
                  f"{acc:.2f}% (parsed_ok={agg_ok}/{agg_n})")
        global_correct += float(ct[0].item())
        global_total += float(ct[1].item())

    if rank == 0:
        acc = (global_correct / global_total * 100) if global_total else 0.0
        print(f"\n===== SUMMARY (all splits): {int(global_correct)}/{int(global_total)} = {acc:.2f}% =====")


def _run_distributed_worker(args):
    import torch.distributed as dist
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(local_rank)
    args.device = f'cuda:{local_rank}'
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    _setup_logging_if_needed(args, is_main=(rank == 0), rank=rank)
    try:
        _run_eval(args, rank=rank, world_size=world_size)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _mp_worker_entry(local_rank, world_size, port, args):
    os.environ['RANK'] = str(local_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    _run_distributed_worker(args)


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', required=True)
    parser.add_argument('--tokenizer', default=DEFAULT_TOKENIZER)
    parser.add_argument('--data_dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--run_dir', type=Path, default=DEFAULT_RUN_DIR,
                        help='Run directory; logs will be written to <run_dir>/eval_logs/.')
    parser.add_argument('--log_file', type=Path, default=None,
                        help='Override the log file path.')
    parser.add_argument('--num_samples', type=int, default=2,
                        help='Default samples per split (used if --num_samples_<split> not set).')
    parser.add_argument('--num_samples_train', type=int, default=None)
    parser.add_argument('--num_samples_val', type=int, default=None)
    parser.add_argument('--num_samples_test', type=int, default=None)
    parser.add_argument('--full', action='store_true', help='Evaluate all samples in every split.')
    parser.add_argument('--full_train', action='store_true')
    parser.add_argument('--full_val', action='store_true')
    parser.add_argument('--full_test', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', default='float32', choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--use_compile', action='store_true')
    parser.add_argument('--no-eval_mode', dest='eval_mode', action='store_false')
    parser.set_defaults(eval_mode=True)
    parser.add_argument('--gpus', type=int, default=0,
                        help='Number of GPUs to use (0=auto, 1=force single GPU).')
    parser.add_argument('--single_gpu', action='store_true', help='Disable auto multi-GPU.')
    parser.add_argument('--master_port', type=int, default=0,
                        help='Master port for mp.spawn (0=auto).')
    return parser


def main():
    args = _build_parser().parse_args()

    in_distributed = int(os.environ.get('WORLD_SIZE', '0')) > 0
    if in_distributed:
        # Scenario A: launched by torchrun or our own mp.spawn worker.
        _run_distributed_worker(args)
        return

    ngpu = torch.cuda.device_count()
    want = _decide_world_size(args, ngpu)

    if want > 1 and ngpu > 1:
        # Scenario B: spawn workers ourselves.
        port = args.master_port if args.master_port > 0 else _find_free_port()
        mp.spawn(_mp_worker_entry, args=(want, port, args), nprocs=want, join=True)
        return

    # Scenario C: single GPU / CPU.
    _setup_logging_if_needed(args, is_main=True, rank=0)
    _run_eval(args, rank=0, world_size=1)


if __name__ == '__main__':
    main()
