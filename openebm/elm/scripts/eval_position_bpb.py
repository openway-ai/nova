"""
Position-wise BPB evaluation for EBT and nanochat GPT checkpoints.

This script computes BPB independently at each context position:
    bpb[pos] = sum_nats[pos] / (log(2) * sum_bytes[pos])

The sample/batch dimension is accumulated, while the context dimension is kept
as a length-T list in the output JSON.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = None
F = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-context-position BPB for an EBT checkpoint and a nanochat GPT checkpoint."
    )
    parser.add_argument("--ebt-ckpt", type=Path, required=True, help="Path to the Lightning EBT .ckpt file.")
    parser.add_argument(
        "--nanochat-checkpoint-dir",
        type=Path,
        default=None,
        help="Directory containing nanochat model_*.pt and meta_*.json files.",
    )
    parser.add_argument(
        "--nanochat-model-tag",
        type=str,
        default=None,
        help="Optional nanochat base model tag under $NANOCHAT_BASE_DIR/base_checkpoints, e.g. d26.",
    )
    parser.add_argument(
        "--nanochat-step",
        type=int,
        default=None,
        help="nanochat checkpoint step. Defaults to the largest model_*.pt step in the checkpoint dir.",
    )
    parser.add_argument(
        "--nanochat-base-dir",
        type=Path,
        default=None,
        help="Optional NANOCHAT_BASE_DIR override. Needed when tokenizer/base_checkpoints are not under ~/.cache/nanochat.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fineweb",
        choices=("fineweb", "climbmix", "dclm"),
        help="Base pretraining dataset name.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional explicit parquet directory. Overrides the dataset default directory.",
    )
    parser.add_argument("--split", type=str, default="val", choices=("train", "val"))
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--device-batch-size", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device string, e.g. auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--ebt-dtype",
        type=str,
        default="auto",
        choices=("auto", "bfloat16", "float32"),
        help="Dtype used when moving the EBT model to device. auto = bfloat16 on CUDA, float32 otherwise.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Where to write the per-position BPB JSON.",
    )
    return parser.parse_args()


def resolve_ebt_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def resolve_nanochat_checkpoint_dir(args: argparse.Namespace) -> Path:
    if args.nanochat_checkpoint_dir is not None:
        return args.nanochat_checkpoint_dir.expanduser().resolve()
    if args.nanochat_model_tag is None:
        raise ValueError("Provide either --nanochat-checkpoint-dir or --nanochat-model-tag.")

    base_dir = Path(os.environ.get("NANOCHAT_BASE_DIR", Path.home() / ".cache" / "nanochat"))
    return (base_dir / "base_checkpoints" / args.nanochat_model_tag).resolve()


def find_last_nanochat_step(checkpoint_dir: Path) -> int:
    model_paths = sorted(checkpoint_dir.glob("model_*.pt"))
    if not model_paths:
        raise FileNotFoundError(f"No model_*.pt files found in {checkpoint_dir}")
    return max(int(path.stem.split("_")[-1]) for path in model_paths)


def collect_eval_batches(
    tokenizer,
    *,
    batch_size: int,
    context_length: int,
    split: str,
    device: torch.device,
    eval_batches: int,
    dataset: str,
    data_dir: Path | None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer,
        batch_size,
        context_length,
        split,
        device=str(device),
        dataset_name=dataset,
        data_dir=str(data_dir) if data_dir is not None else None,
    )
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(eval_batches):
        x, y = next(loader)
        batches.append((x.detach().cpu(), y.detach().cpu()))
    return batches


def move_batch(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def _flatten_if_lightning_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def ebt_final_step_loss2d(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mirror forward_loss_wrapper's final-step CE path, but keep [B, T]."""
    input_ids = _flatten_if_lightning_batch(x)
    targets = _flatten_if_lightning_batch(y)
    learning = False
    no_randomness = True

    if getattr(model, "use_tf_head", False):
        predicted_distributions, _, predicted_pred_hiddens = model(
            input_ids,
            return_raw_logits=True,
            no_randomness=no_randomness,
            learning=learning,
            return_pred_hiddens=True,
        )
        prev_embed = model.embeddings(input_ids)
        if getattr(model, "post_update_state_detach_prev_embed", False):
            prev_embed = prev_embed.detach()
        predicted_distributions = [
            None if pred_hidden_step is None else model.tf_head(pred_hidden_step, prev_embed)
            for pred_hidden_step in predicted_pred_hiddens
        ]
    else:
        predicted_distributions, _ = model(
            input_ids,
            return_raw_logits=True,
            no_randomness=no_randomness,
            learning=learning,
        )

    if not predicted_distributions:
        raise RuntimeError("EBT forward returned no predicted distributions.")

    final_logits = predicted_distributions[-1]
    if final_logits is None:
        raise RuntimeError("Final EBT predicted distribution is None; check TF-head/free-embedding settings.")

    targets_flat = targets.reshape(-1)
    if getattr(model.hparams, "soften_target_prob_dist", 0.0) != 0.0:
        per_token_ce = F.cross_entropy(
            final_logits.reshape(-1, model.vocab_size),
            targets_flat,
            label_smoothing=0.0,
            ignore_index=-1,
            reduction="none",
        )
    else:
        log_probs = model.log_softmax(final_logits).reshape(-1, model.vocab_size)
        per_token_ce = F.nll_loss(log_probs, targets_flat, ignore_index=-1, reduction="none")
    return per_token_ce.reshape_as(targets)


def gpt_loss2d(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        loss = model(x, y, loss_reduction="none")
    return loss.reshape_as(y)


def add_position_sums(
    loss2d: torch.Tensor,
    targets: torch.Tensor,
    token_bytes: torch.Tensor,
    nats_by_pos: torch.Tensor,
    bytes_by_pos: torch.Tensor,
) -> None:
    if loss2d.shape != targets.shape:
        raise ValueError(f"loss/target shape mismatch: {tuple(loss2d.shape)} vs {tuple(targets.shape)}")
    if loss2d.shape[1] != nats_by_pos.numel():
        raise ValueError(f"Expected context length {nats_by_pos.numel()}, got {loss2d.shape[1]}")

    valid = targets >= 0
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    num_bytes = token_bytes[safe_targets]
    counted = valid & (num_bytes > 0)

    nats_by_pos += (loss2d.detach().to(torch.float64) * counted).sum(dim=0)
    bytes_by_pos += (num_bytes.to(torch.int64) * counted).sum(dim=0)


def compute_position_bpb(
    model_name: str,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    loss2d_fn,
    device: torch.device,
    context_length: int,
    token_bytes: torch.Tensor,
    autocast_ctx,
) -> dict[str, object]:
    nats_by_pos = torch.zeros(context_length, dtype=torch.float64, device=device)
    bytes_by_pos = torch.zeros(context_length, dtype=torch.int64, device=device)
    num_samples = 0
    num_batches = 0

    for batch in batches:
        x, y = move_batch(batch, device)
        with autocast_ctx:
            loss2d = loss2d_fn(x, y)
        add_position_sums(loss2d, y, token_bytes, nats_by_pos, bytes_by_pos)
        num_samples += y.shape[0]
        num_batches += 1
        del x, y, loss2d

    nats_cpu = nats_by_pos.cpu()
    bytes_cpu = bytes_by_pos.cpu()
    bpb = [
        (float(nats_cpu[i].item()) / (math.log(2) * int(bytes_cpu[i].item())))
        if int(bytes_cpu[i].item()) > 0
        else None
        for i in range(context_length)
    ]
    total_nats = float(nats_cpu.sum().item())
    total_bytes = int(bytes_cpu.sum().item())
    overall_bpb = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else None

    return {
        "model": model_name,
        "num_batches": num_batches,
        "num_samples": num_samples,
        "overall_bpb": overall_bpb,
        "position_bpb": bpb,
        "position_nats": [float(v) for v in nats_cpu.tolist()],
        "position_bytes": [int(v) for v in bytes_cpu.tolist()],
    }


def load_ebt(args: argparse.Namespace, device: torch.device):
    from openebm.elm.scripts.ebt_core_eval import load_ebt_model

    dtype = resolve_ebt_dtype(args.ebt_dtype, device)
    wrapper, tokenizer, hparams = load_ebt_model(args.ebt_ckpt, tokenizer_path=None, device=device, dtype=dtype)
    wrapper.model.requires_grad_(False)
    wrapper.model.eval()
    return wrapper.model, tokenizer, hparams


def load_nanochat_gpt(args: argparse.Namespace, device: torch.device):
    from nanochat.checkpoint_manager import build_model

    checkpoint_dir = resolve_nanochat_checkpoint_dir(args)
    step = args.nanochat_step if args.nanochat_step is not None else find_last_nanochat_step(checkpoint_dir)
    model, tokenizer, meta = build_model(str(checkpoint_dir), step, device, phase="eval")
    model.requires_grad_(False)
    model.eval()
    return model, tokenizer, meta, checkpoint_dir, step


def main() -> None:
    args = parse_args()
    global torch, F
    import torch as _torch
    import torch.nn.functional as _F

    torch = _torch
    F = _F

    if args.nanochat_base_dir is not None:
        os.environ["NANOCHAT_BASE_DIR"] = str(args.nanochat_base_dir.expanduser().resolve())

    if args.eval_batches <= 0:
        raise ValueError("--eval-batches must be positive.")
    if args.device_batch_size <= 0:
        raise ValueError("--device-batch-size must be positive.")

    device_arg = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_arg == "auto":
        device_arg = "cpu"
    device = torch.device(device_arg)
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    from nanochat.tokenizer import get_token_bytes, get_tokenizer

    token_bytes = get_token_bytes(device=device)
    tokenizer = get_tokenizer()

    print(f"Collecting {args.eval_batches} {args.split} batch(es) from dataset={args.dataset}")
    batches = collect_eval_batches(
        tokenizer,
        batch_size=args.device_batch_size,
        context_length=args.context_length,
        split=args.split,
        device=torch.device("cpu"),
        eval_batches=args.eval_batches,
        dataset=args.dataset,
        data_dir=args.data_dir,
    )

    print(f"Loading EBT from {args.ebt_ckpt}")
    ebt_model, _, ebt_hparams = load_ebt(args, device)
    if args.context_length != int(ebt_hparams.get("context_length", args.context_length)):
        print(
            f"[warning] requested context_length={args.context_length}, "
            f"EBT checkpoint context_length={ebt_hparams.get('context_length')}"
        )

    print("Evaluating EBT position-wise BPB")
    ebt_result = compute_position_bpb(
        "ebt",
        batches,
        loss2d_fn=lambda x, y: ebt_final_step_loss2d(ebt_model, x, y),
        device=device,
        context_length=args.context_length,
        token_bytes=token_bytes,
        autocast_ctx=autocast_ctx,
    )
    del ebt_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Loading nanochat GPT")
    gpt_model, _, gpt_meta, gpt_checkpoint_dir, gpt_step = load_nanochat_gpt(args, device)
    print("Evaluating nanochat GPT position-wise BPB")
    gpt_result = compute_position_bpb(
        "nanochat_gpt",
        batches,
        loss2d_fn=lambda x, y: gpt_loss2d(gpt_model, x, y),
        device=device,
        context_length=args.context_length,
        token_bytes=token_bytes,
        autocast_ctx=autocast_ctx,
    )

    out = {
        "context_length": args.context_length,
        "split": args.split,
        "dataset": args.dataset,
        "data_dir": str(args.data_dir.resolve()) if args.data_dir is not None else None,
        "device_batch_size": args.device_batch_size,
        "eval_batches": args.eval_batches,
        "num_samples": ebt_result["num_samples"],
        "models": {
            "ebt": {
                "checkpoint": str(args.ebt_ckpt.resolve()),
                "model_name": ebt_hparams.get("model_name"),
                "model_size": ebt_hparams.get("model_size"),
                "overall_bpb": ebt_result["overall_bpb"],
            },
            "nanochat_gpt": {
                "checkpoint_dir": str(gpt_checkpoint_dir),
                "step": gpt_step,
                "model_config": gpt_meta.get("model_config", {}),
                "overall_bpb": gpt_result["overall_bpb"],
            },
        },
        "position_bpb": {
            "ebt": ebt_result["position_bpb"],
            "nanochat_gpt": gpt_result["position_bpb"],
        },
        "position_nats": {
            "ebt": ebt_result["position_nats"],
            "nanochat_gpt": gpt_result["position_nats"],
        },
        "position_bytes": {
            "ebt": ebt_result["position_bytes"],
            "nanochat_gpt": gpt_result["position_bytes"],
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"Wrote {args.output_json}")
    print(f"EBT overall BPB: {ebt_result['overall_bpb']}")
    print(f"nanochat GPT overall BPB: {gpt_result['overall_bpb']}")


if __name__ == "__main__":
    main()
