#!/usr/bin/env python3
"""Evaluate local LLMs on the SATNet Sudoku test split.

Run with:
    python -m openebm.elm.scripts.sudoku_compare.eval_llm_sudoku --models qwen3_1p7b
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from .common import (
        DEFAULT_DATA_DIR,
        DEFAULT_MODEL_PARAMS,
        DEFAULT_MODEL_PATHS,
        DEFAULT_RESULTS_ROOT,
        dedupe_sort_rows,
        format_board,
        infer_response_format,
        load_split,
        model_spec_to_name_path,
        params_for_model,
        parse_llm_board,
        read_jsonl,
        resolve_hf_snapshot,
        sample_boards,
        sanitize_model_name,
        score_prediction,
        select_indices,
        summarize_rows,
        write_json,
        write_jsonl,
    )
except ImportError:  # pragma: no cover - direct file execution fallback.
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[4]))
    from openebm.elm.scripts.sudoku_compare.common import (  # type: ignore
        DEFAULT_DATA_DIR,
        DEFAULT_MODEL_PARAMS,
        DEFAULT_MODEL_PATHS,
        DEFAULT_RESULTS_ROOT,
        dedupe_sort_rows,
        format_board,
        infer_response_format,
        load_split,
        model_spec_to_name_path,
        params_for_model,
        parse_llm_board,
        read_jsonl,
        resolve_hf_snapshot,
        sample_boards,
        sanitize_model_name,
        score_prediction,
        select_indices,
        summarize_rows,
        write_json,
        write_jsonl,
    )


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful Sudoku solver. Solve the puzzle and put the final answer "
    "as a completed 9x9 grid. The final grid must contain only digits 1-9."
)

SUDOKU_FEW_SHOTS = [
    (
        "0 3 4 6 7 8 9 1 2\n"
        "6 0 2 1 9 5 3 4 8\n"
        "1 9 0 3 4 2 5 6 7\n"
        "8 5 9 0 6 1 4 2 3\n"
        "4 2 6 8 0 3 7 9 1\n"
        "7 1 3 9 2 0 8 5 6\n"
        "9 6 1 5 3 7 0 8 4\n"
        "2 8 7 4 1 9 6 0 5\n"
        "3 4 5 2 8 6 1 7 0",
        "5 3 4 6 7 8 9 1 2\n"
        "6 7 2 1 9 5 3 4 8\n"
        "1 9 8 3 4 2 5 6 7\n"
        "8 5 9 7 6 1 4 2 3\n"
        "4 2 6 8 5 3 7 9 1\n"
        "7 1 3 9 2 4 8 5 6\n"
        "9 6 1 5 3 7 2 8 4\n"
        "2 8 7 4 1 9 6 3 5\n"
        "3 4 5 2 8 6 1 7 9",
    ),
    (
        "8 0 0 0 0 0 0 0 0\n"
        "0 0 3 6 0 0 0 0 0\n"
        "0 7 0 0 9 0 2 0 0\n"
        "0 5 0 0 0 7 0 0 0\n"
        "0 0 0 0 4 5 7 0 0\n"
        "0 0 0 1 0 0 0 3 0\n"
        "0 0 1 0 0 0 0 6 8\n"
        "0 0 8 5 0 0 0 1 0\n"
        "0 9 0 0 0 0 4 0 0",
        "8 1 2 7 5 3 6 4 9\n"
        "9 4 3 6 8 2 1 7 5\n"
        "6 7 5 4 9 1 2 8 3\n"
        "1 5 4 2 3 7 8 9 6\n"
        "3 6 9 8 4 5 7 2 1\n"
        "2 8 7 1 6 9 5 3 4\n"
        "5 2 1 9 7 4 3 6 8\n"
        "4 3 8 5 2 6 9 1 7\n"
        "7 9 6 3 1 8 4 5 2",
    ),
]


def build_user_prompt(puzzle_text: str) -> str:
    return (
        "Solve this Sudoku puzzle. 0 denotes an empty cell.\n"
        "Return only the completed grid as 9 lines of 9 digits, separated by spaces.\n"
        "Do not include explanations in the final answer.\n\n"
        f"Puzzle:\n{puzzle_text}\n\n"
        "Final answer:"
    )


def build_messages(puzzle_text: str, few_shot: int, system_prompt: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for shot_puzzle, shot_solution in SUDOKU_FEW_SHOTS[: max(0, few_shot)]:
        messages.append({"role": "user", "content": build_user_prompt(shot_puzzle)})
        messages.append({"role": "assistant", "content": shot_solution})
    messages.append({"role": "user", "content": build_user_prompt(puzzle_text)})
    return messages


def render_prompt(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    use_chat_template: bool,
    thinking: str,
) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if thinking == "enable":
            kwargs["enable_thinking"] = True
        elif thinking == "disable":
            kwargs["enable_thinking"] = False
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            pass

    chunks = []
    for msg in messages:
        chunks.append(f"{msg['role'].upper()}:\n{msg['content']}")
    chunks.append("ASSISTANT:")
    return "\n\n".join(chunks)


class VLLMRunner:
    def __init__(self, model_dir: Path, args: argparse.Namespace):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=args.trust_remote_code,
            local_files_only=True,
        )

        llm_kwargs: Dict[str, Any] = {
            "model": str(model_dir),
            "tokenizer": str(model_dir),
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "trust_remote_code": args.trust_remote_code,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "seed": args.seed,
        }
        if args.pipeline_parallel_size > 1:
            llm_kwargs["pipeline_parallel_size"] = args.pipeline_parallel_size
        if args.quantization:
            llm_kwargs["quantization"] = args.quantization
        if args.max_model_len:
            llm_kwargs["max_model_len"] = args.max_model_len
        if args.enforce_eager:
            llm_kwargs["enforce_eager"] = True

        self.llm = LLM(**llm_kwargs)
        sampling_kwargs: Dict[str, Any] = {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        try:
            self.sampling_params = SamplingParams(**sampling_kwargs, seed=args.seed)
        except TypeError:
            self.sampling_params = SamplingParams(**sampling_kwargs)

    def generate(self, prompts: Sequence[str]) -> List[Tuple[str, int]]:
        try:
            outputs = self.llm.generate(list(prompts), self.sampling_params, use_tqdm=False)
        except TypeError:
            outputs = self.llm.generate(list(prompts), self.sampling_params)
        rows: List[Tuple[str, int]] = []
        for out in outputs:
            choice = out.outputs[0]
            token_ids = getattr(choice, "token_ids", None)
            rows.append((choice.text, len(token_ids) if token_ids is not None else 0))
        return rows


class TransformersRunner:
    def __init__(self, model_dir: Path, args: argparse.Namespace):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "auto": "auto",
        }
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=args.trust_remote_code,
            local_files_only=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=dtype_map.get(args.dtype, "auto"),
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            local_files_only=True,
        )
        self.args = args

    def generate(self, prompts: Sequence[str]) -> List[Tuple[str, int]]:
        inputs = self.tokenizer(list(prompts), return_tensors="pt", padding=True)
        first_device = next(self.model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
        do_sample = self.args.temperature > 0
        gen_kwargs: Dict[str, Any] = {
            **inputs,
            "max_new_tokens": self.args.max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.args.temperature
            gen_kwargs["top_p"] = self.args.top_p
        with self.torch.no_grad():
            out = self.model.generate(**gen_kwargs)
        prompt_width = int(inputs["input_ids"].shape[1])
        rows: List[Tuple[str, int]] = []
        for seq in out:
            gen_ids = seq[prompt_width:]
            rows.append((self.tokenizer.decode(gen_ids, skip_special_tokens=True), len(gen_ids)))
        return rows


def make_runner(model_dir: Path, args: argparse.Namespace):
    if args.backend == "vllm":
        return VLLMRunner(model_dir, args)
    return TransformersRunner(model_dir, args)


def params_for_run(name: str, args: argparse.Namespace) -> str:
    per_model = getattr(args, "_model_params_map", {})
    return params_for_model(name, per_model.get(name) or args.params)


def parse_model_params(items: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--model-params entries must be NAME=PARAMS, got: {item}")
        name, params = item.split("=", 1)
        out[sanitize_model_name(name)] = params.strip()
    return out


def validate_model_dir(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"Resolved model directory lacks config.json: {model_dir}. "
            "Pass a concrete snapshot path or --revision if auto-resolution picked the wrong folder."
        )


def response_for_log(response: str, args: argparse.Namespace) -> Dict[str, Any]:
    chars = len(response)
    digest = hashlib.sha256(response.encode("utf-8", errors="replace")).hexdigest()
    if args.response_log == "none":
        logged = ""
    elif args.response_log == "full" or chars <= args.response_log_chars:
        logged = response
    else:
        # Keep the tail because final-answer grids usually appear after CoT/thinking.
        logged = response[-args.response_log_chars :]
    return {
        "response": logged,
        "response_chars": chars,
        "response_sha256": digest,
        "response_log_mode": args.response_log,
        "response_was_truncated": logged != response,
    }


def make_invalid_generation_row(
    idx: int,
    name: str,
    raw_model_path: Path,
    resolved_model_dir: Path,
    puzzle: List[int],
    solution: List[int],
    puzzle_text: str,
    prompt: str,
    error: Exception,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    scored = score_prediction(puzzle, solution, None, elapsed_s=0.0, tokens_generated=0)
    return {
        "split": "test",
        "idx": int(idx),
        "model": name,
        "model_path": str(raw_model_path),
        "resolved_model_dir": str(resolved_model_dir),
        "puzzle_text": puzzle_text,
        "solution_text": format_board(solution),
        "prompt_used": prompt if args.save_prompts else "",
        "response": f"<generation error: {type(error).__name__}: {error}>",
        "response_chars": 0,
        "response_sha256": "",
        "response_log_mode": "error",
        "response_was_truncated": False,
        "pred": None,
        "parser_strategy": "generation_error",
        "solution_format": "unknown",
        "puzzle_format": "grid",
        **scored,
    }


def evaluate_one_model(
    name: str,
    raw_model_path: Path,
    samples: Sequence[dict],
    indices: Sequence[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    resolved_model_dir = resolve_hf_snapshot(raw_model_path, revision=args.revision)
    validate_model_dir(resolved_model_dir)
    model_out_dir = Path(args.out_dir) / name
    model_out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = model_out_dir / "test.jsonl"
    summary_path = model_out_dir / "summary.json"

    if args.overwrite and jsonl_path.exists():
        jsonl_path.write_text("")

    done = set()
    existing_rows = []
    if args.resume and jsonl_path.is_file():
        selected = set(indices)
        existing_rows = [row for row in read_jsonl(jsonl_path) if row.get("idx") in selected]
        done = {int(r["idx"]) for r in existing_rows if isinstance(r.get("idx"), int)}

    pending = [idx for idx in indices if idx not in done]
    config = {
        "model": name,
        "model_path": str(raw_model_path),
        "resolved_model_dir": str(resolved_model_dir),
        "params": params_for_run(name, args),
        "backend": args.backend,
        "data_dir": str(args.data_dir),
        "out_dir": str(model_out_dir),
        "indices": {"start": min(indices) if indices else None, "count": len(indices)},
        "generation": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "few_shot": args.few_shot,
            "thinking": args.thinking,
            "use_chat_template": args.use_chat_template,
            "save_prompts": args.save_prompts,
            "response_log": args.response_log,
            "response_log_chars": args.response_log_chars,
        },
    }
    write_json(model_out_dir / "config.json", config)

    print(f"[eval_llm_sudoku] model={name} path={resolved_model_dir}")
    print(f"[eval_llm_sudoku] pending={len(pending)} already_done={len(done)} out={jsonl_path}")
    if not pending:
        rows = dedupe_sort_rows(existing_rows)
        summary = summarize_rows(rows, model=name, params=params_for_run(name, args))
        write_json(summary_path, summary)
        return summary

    runner = make_runner(resolved_model_dir, args)
    all_new_rows: List[Dict[str, Any]] = []

    iterator = range(0, len(pending), args.batch_size)
    if tqdm is not None:
        iterator = tqdm(iterator, total=(len(pending) + args.batch_size - 1) // args.batch_size, desc=name)

    with open(jsonl_path, "a", buffering=1) as fp:
        for start in iterator:
            batch_indices = pending[start : start + args.batch_size]
            prompts: List[str] = []
            prompt_texts: List[str] = []
            puzzles: List[List[int]] = []
            solutions: List[List[int]] = []

            for idx in batch_indices:
                puzzle, solution = sample_boards(samples[idx])
                puzzle_text = format_board(puzzle, blank=args.blank)
                messages = build_messages(puzzle_text, args.few_shot, args.system_prompt)
                prompt = render_prompt(
                    runner.tokenizer,
                    messages,
                    use_chat_template=args.use_chat_template,
                    thinking=args.thinking,
                )
                prompts.append(prompt)
                prompt_texts.append(puzzle_text)
                puzzles.append(puzzle)
                solutions.append(solution)

            try:
                t0 = time.time()
                outputs = runner.generate(prompts)
                batch_elapsed = time.time() - t0
                if len(outputs) != len(batch_indices):
                    raise RuntimeError(
                        f"generation returned {len(outputs)} outputs for {len(batch_indices)} prompts"
                    )
            except Exception as e:
                if not args.continue_on_generation_error:
                    raise
                print(f"[eval_llm_sudoku] WARNING: generation failed for batch {batch_indices}: {e}")
                for idx, puzzle, solution, puzzle_text, prompt in zip(
                    batch_indices, puzzles, solutions, prompt_texts, prompts
                ):
                    row = make_invalid_generation_row(
                        idx, name, raw_model_path, resolved_model_dir,
                        puzzle, solution, puzzle_text, prompt, e, args,
                    )
                    fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                    all_new_rows.append(row)
                continue

            per_sample_elapsed = batch_elapsed / max(len(outputs), 1)

            for idx, puzzle, solution, puzzle_text, prompt, (response, token_count) in zip(
                batch_indices, puzzles, solutions, prompt_texts, prompts, outputs
            ):
                parse_result = parse_llm_board(response)
                scored = score_prediction(
                    puzzle,
                    solution,
                    parse_result.pred,
                    elapsed_s=per_sample_elapsed,
                    tokens_generated=token_count,
                )
                row = {
                    "split": "test",
                    "idx": int(idx),
                    "model": name,
                    "model_path": str(raw_model_path),
                    "resolved_model_dir": str(resolved_model_dir),
                    "puzzle_text": puzzle_text,
                    "solution_text": format_board(solution),
                    "prompt_used": prompt if args.save_prompts else "",
                    **response_for_log(response, args),
                    "pred": parse_result.pred,
                    "parser_strategy": parse_result.strategy,
                    "solution_format": infer_response_format(response, parse_result.pred),
                    "puzzle_format": "grid",
                    **scored,
                }
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                all_new_rows.append(row)

    rows = dedupe_sort_rows(existing_rows + all_new_rows + read_jsonl(jsonl_path))
    selected = set(indices)
    rows = [row for row in rows if row.get("idx") in selected]
    write_jsonl(jsonl_path, rows)
    summary = summarize_rows(rows, model=name, params=params_for_run(name, args))
    write_json(summary_path, summary)
    print(
        f"[eval_llm_sudoku] {name}: n={summary['n']} "
        f"board_acc={summary['board_acc']:.4f} "
        f"cell_acc={summary['cell_acc']:.4f} "
        f"valid_rate={summary['valid_rate']:.4f} "
        f"avg_time={summary['avg_time']:.3f}s"
    )

    del runner
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen3_1p7b"],
        help=(
            "Model aliases or NAME=PATH specs. Known aliases: "
            + ", ".join(DEFAULT_MODEL_PATHS.keys())
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_ROOT / "llm")
    parser.add_argument("--revision", default=None, help="Snapshot hash under snapshots/, optional.")
    parser.add_argument("--params", default=None, help="Override params string for all selected models.")
    parser.add_argument(
        "--model-params",
        nargs="*",
        default=[],
        help="Per-model params labels, e.g. qwen3_1p7b=1.7B deepseek_r1_0528=671B.",
    )
    parser.add_argument("--num-samples", type=int, default=-1, help="-1 means full test split.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--few-shot", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--blank", choices=["0", "."], default="0")
    parser.add_argument("--thinking", choices=["auto", "enable", "disable"], default="auto")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--save-prompts", action="store_true", help="Store full rendered prompts in JSONL.")
    parser.add_argument(
        "--response-log",
        choices=["truncated", "full", "none"],
        default="truncated",
        help="How much raw model output to keep in JSONL. Parsing always uses the full in-memory output.",
    )
    parser.add_argument(
        "--response-log-chars",
        type=int,
        default=12000,
        help="Tail characters kept when --response-log=truncated.",
    )
    chat_group = parser.add_mutually_exclusive_group()
    chat_group.add_argument("--use-chat-template", dest="use_chat_template", action="store_true")
    chat_group.add_argument("--raw-prompt", dest="use_chat_template", action="store_false")
    parser.set_defaults(use_chat_template=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--quantization", default=None, help="vLLM quantization, e.g. awq/fp8.")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--device-map", default="auto", help="transformers backend only.")
    trust_group = parser.add_mutually_exclusive_group()
    trust_group.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true")
    trust_group.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.set_defaults(trust_remote_code=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--continue-on-generation-error",
        action="store_true",
        help="Record failed batches as invalid rows instead of aborting. Use cautiously for final results.",
    )
    parser.add_argument("--list-default-models", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.list_default_models:
        for name, path in DEFAULT_MODEL_PATHS.items():
            print(f"{name}\t{DEFAULT_MODEL_PARAMS.get(name, '')}\t{path}")
        return
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be > 0")
    if args.response_log_chars <= 0:
        raise SystemExit("--response-log-chars must be > 0")
    args._model_params_map = parse_model_params(args.model_params)

    random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
    except Exception:
        pass

    samples = load_split(args.data_dir, split="test")
    indices = select_indices(len(samples), args.start_index, args.num_samples)
    print(
        f"[eval_llm_sudoku] data_dir={args.data_dir} split=test "
        f"total={len(samples)} selected={len(indices)}"
    )

    summaries = []
    for spec in args.models:
        name, model_path = model_spec_to_name_path(spec)
        summaries.append(evaluate_one_model(name, model_path, samples, indices, args))

    write_json(Path(args.out_dir) / "run_summary.json", {"models": summaries})


if __name__ == "__main__":
    main()
