"""EBT CORE evaluation harness.

Adapts nanochat's ``base_eval.py`` CORE evaluator to run against an EBT
checkpoint. Supports single- and multi-GPU execution, greedy load-balanced task
scheduling, and optional per-sample trajectory logging.
"""
import os
import sys
import csv
import time
import json
import yaml
import heapq
import random
import argparse
import glob as glob_module
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm

from openebm.elm.trainer import ModelTrainer
from openebm.elm.nanochat_tokenizer_adapter import NanoChatTokenizerWrapper
from nanochat.core_eval import (
    evaluate_example,
    evaluate_task as _nanochat_evaluate_task,
    forward_model,
    render_prompts_mc,
    render_prompts_schema,
    render_prompts_lm,
    batch_sequences_mc,
    batch_sequences_schema,
    batch_sequences_lm,
    stack_sequences,
)


class EBTModelWrapper:
    """Wrap an EBT model so it exposes a nanochat-compatible eval interface."""

    def __init__(self, model: "Any", tokenizer: "Any", device: "Any", max_seq_len: int = 256) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids: "Any", targets: "Optional[Any]" = None, loss_reduction: str = 'mean') -> "Any":
        """Run a forward pass compatible with nanochat's eval helpers.

        :param input_ids: token ids of shape ``[batch_size, seq_len]``.
        :type input_ids: Any
        :param targets: optional target ids of shape ``[batch_size, seq_len]``;
            when ``None`` raw logits are returned.
        :type targets: Optional[Any]
        :param loss_reduction: ``'mean'`` or ``'none'`` passed to
            ``F.cross_entropy``.
        :type loss_reduction: str
        :return: logits ``[batch_size, seq_len, vocab_size]`` when ``targets``
            is ``None``, otherwise the computed loss.
        :rtype: Any
        """
        with torch.no_grad():
            # EBT.forward returns (logits_list, energies).
            outputs = self.model.forward(input_ids, start_pos=0, learning=False, return_raw_logits=True)

            # Take the final MCMC step's logits.
            if isinstance(outputs, tuple):
                logits = outputs[0]
                if isinstance(logits, list):
                    logits = logits[-1]
            else:
                logits = outputs

            if targets is None:
                return logits

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction
            )
            return loss

    def get_device(self) -> "Any":
        """Return the torch device this wrapper runs on.

        :return: the underlying torch device.
        :rtype: Any
        """
        return self.device


def load_ebt_model(ckpt_path: str, tokenizer_path: str, device: "Any", dtype: "torch.dtype" = torch.bfloat16) -> "Tuple[Any, Any, Any]":
    """Load an EBT checkpoint ready for inference.

    :param ckpt_path: path to the Lightning-style checkpoint.
    :type ckpt_path: str
    :param tokenizer_path: path to the tokenizer artefacts.
    :type tokenizer_path: str
    :param device: torch device to place the model on.
    :type device: Any
    :param dtype: inference dtype (``bfloat16`` by default).
    :type dtype: torch.dtype
    :return: tuple of ``(wrapped_model, tokenizer, hparams)``.
    :rtype: Tuple[Any, Any, Any]
    """
    print(f"Loading EBT checkpoint from: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams = checkpoint['hyper_parameters']

    # Switch to inference mode.
    hparams['execution_mode'] = 'inference'
    hparams['no_wandb'] = True
    # NOTE: torch.compile is disabled here; compile + MCMC grad + mp.spawn
    # combination triggers CUBLAS errors, and eval does not need compilation.
    hparams['compile_model'] = False

    model_trainer = ModelTrainer(hparams)

    # Reuse the ``_orig_mod`` prefix fix-up from on_load_checkpoint.
    state_dict = checkpoint['state_dict']
    has_orig_mod_keys = any('_orig_mod.' in k for k in state_dict)
    model_has_orig_mod = any('_orig_mod.' in k for k in model_trainer.state_dict())

    if has_orig_mod_keys and not model_has_orig_mod:
        state_dict = {k.replace('._orig_mod.', '.'): v for k, v in state_dict.items()}
        print(f"[load_ebt_model] Stripped '_orig_mod' prefix from {len(checkpoint['state_dict'])} keys")
    elif not has_orig_mod_keys and model_has_orig_mod:
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_sd['model._orig_mod.' + k[len('model.'):]] = v
            else:
                new_sd[k] = v
        state_dict = new_sd
        print(f"[load_ebt_model] Added '_orig_mod' prefix to {len(checkpoint['state_dict'])} keys")

    model_trainer.load_state_dict(state_dict)
    model_trainer.eval()
    model = model_trainer.model
    model.to(device=device, dtype=dtype)
    model.eval()

    from nanochat.tokenizer import get_tokenizer
    tokenizer_obj = get_tokenizer()
    tokenizer_wrapper = NanoChatTokenizerWrapper(tokenizer_obj=tokenizer_obj)

    max_seq_len = hparams.get('context_length', 256)

    wrapped_model = EBTModelWrapper(model, tokenizer_wrapper, device, max_seq_len=max_seq_len)

    print(f"✓ Model loaded: {hparams['model_name']} (size: {hparams.get('model_size', 'unknown')})")
    print(f"✓ Context length: {max_seq_len}")

    # Return the raw RustBPETokenizer so nanochat's core_eval (which expects
    # ``get_bos_token_id`` and ``__call__(prompts, prepend=...)``) works directly.
    return wrapped_model, tokenizer_obj, hparams


def evaluate_task(model: "Any", tokenizer: "Any", data: "Any", device: "Any", task_meta: "Any", max_seq_len: "Optional[int]" = None) -> "Any":
    """Wrapper around nanochat's ``evaluate_task`` that ignores ``max_seq_len``.

    The max sequence length is already threaded through ``model.max_seq_len``,
    so the extra argument is accepted only for API symmetry.

    :param model: wrapped EBT model.
    :type model: Any
    :param tokenizer: tokenizer used for rendering prompts.
    :type tokenizer: Any
    :param data: eval examples.
    :type data: Any
    :param device: torch device.
    :type device: Any
    :param task_meta: nanochat task metadata dict.
    :type task_meta: Any
    :param max_seq_len: ignored; kept for compatibility.
    :type max_seq_len: Optional[int]
    :return: whatever nanochat's ``evaluate_task`` returns.
    :rtype: Any
    """
    return _nanochat_evaluate_task(model, tokenizer, data, device, task_meta)


@torch.no_grad()
def evaluate_example_with_trajectory(idx: int, model: "Any", tokenizer: "Any", data: "Any", device: "Any", task_meta: "Any") -> "Tuple[bool, Dict[str, Any]]":
    """Evaluate a single example and return its per-sample trajectory.

    Mirrors nanochat's ``evaluate_example`` but additionally records a
    trajectory dictionary (prompt, ground truth, prediction, per-option
    losses) without modifying nanochat's own code.

    :param idx: index of the example in ``data``.
    :type idx: int
    :param model: wrapped EBT model.
    :type model: Any
    :param tokenizer: tokenizer used for rendering prompts.
    :type tokenizer: Any
    :param data: list of eval examples.
    :type data: Any
    :param device: torch device.
    :type device: Any
    :param task_meta: nanochat task metadata dict.
    :type task_meta: Any
    :return: ``(is_correct, trajectory)``.
    :rtype: Tuple[bool, Dict[str, Any]]
    :raises ValueError: if ``task_meta['task_type']`` is unsupported.
    """
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    # Sample few-shot examples (excluding current item) -- matches nanochat.
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    # Render prompts and batch sequences based on task type.
    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # max_seq_len truncation -- matches nanochat.
    if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:])
                new_start_idxs.append(s - num_to_crop)
                new_end_idxs.append(e - num_to_crop)
                assert s - num_to_crop >= 0
                assert e - num_to_crop >= 0
            else:
                new_tokens.append(t)
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    # Stack up sequences into a batch
    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)

    # Forward the model
    losses, predictions = forward_model(model, input_ids)

    # Build trajectory dict
    trajectory = {
        "sample_id": idx,
        "is_correct": False,
        "metadata": {
            "task_type": task_type,
            "num_fewshot": num_fewshot,
        },
    }

    if task_type == 'language_modeling':
        si = start_idxs[0]
        ei = end_idxs[0]
        predicted_tokens = predictions[0, si-1:ei-1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()

        # Trajectory: decode ground truth continuation and model prediction
        try:
            gt_text = tokenizer.decode(actual_tokens.tolist())
        except Exception:
            gt_text = str(actual_tokens.tolist())
        try:
            pred_text = tokenizer.decode(predicted_tokens.tolist())
        except Exception:
            pred_text = str(predicted_tokens.tolist())

        trajectory["input"] = prompts[0]
        trajectory["ground_truth"] = gt_text
        trajectory["prediction"] = pred_text
        trajectory["parsed_answer"] = pred_text

    elif task_type in ['multiple_choice', 'schema']:
        mean_losses = [losses[i, si-1:ei-1].mean().item()
                       for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
        pred_idx = mean_losses.index(min(mean_losses))
        gold_idx = item['gold']
        is_correct = pred_idx == gold_idx

        # Trajectory: record choices and losses.
        if task_type == 'multiple_choice':
            choices = item.get('choices', [])
            trajectory["input"] = prompts[gold_idx] if gold_idx < len(prompts) else prompts[0]
            trajectory["ground_truth"] = choices[gold_idx] if gold_idx < len(choices) else str(gold_idx)
            trajectory["prediction"] = choices[pred_idx] if pred_idx < len(choices) else str(pred_idx)
            trajectory["parsed_answer"] = pred_idx
            trajectory["metadata"]["num_choices"] = len(choices)
        else:
            context_options = item.get('context_options', [])
            continuation = item.get('continuation', '')
            trajectory["input"] = prompts[gold_idx] if gold_idx < len(prompts) else prompts[0]
            trajectory["ground_truth"] = context_options[gold_idx] if gold_idx < len(context_options) else str(gold_idx)
            trajectory["prediction"] = context_options[pred_idx] if pred_idx < len(context_options) else str(pred_idx)
            trajectory["parsed_answer"] = pred_idx
            trajectory["metadata"]["num_choices"] = len(context_options)
            trajectory["metadata"]["continuation"] = continuation

        trajectory["metadata"]["mean_losses"] = mean_losses
        trajectory["metadata"]["gold_idx"] = gold_idx
        trajectory["metadata"]["pred_idx"] = pred_idx
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    trajectory["is_correct"] = bool(is_correct)
    return is_correct, trajectory


def evaluate_task_with_trajectory(model: "Any", tokenizer: "Any", data: "Any", device: "Any", task_meta: "Any",
                                  output_dir: "Optional[str]" = None, task_label: "Optional[str]" = None,
                                  num_trajectory_samples: int = 10,
                                  show_progress: bool = True, progress_position: int = 1) -> "Tuple[float, List[Dict[str, Any]]]":
    """Evaluate a task while optionally persisting per-sample trajectories.

    Replaces ``_nanochat_evaluate_task``. The first ``num_trajectory_samples``
    examples are evaluated with :func:`evaluate_example_with_trajectory` and
    their trajectories are streamed to ``samples.jsonl``; remaining examples
    fall back to the faster ``evaluate_example``.

    :param model: wrapped EBT model.
    :type model: Any
    :param tokenizer: tokenizer used for rendering prompts.
    :type tokenizer: Any
    :param data: eval examples.
    :type data: Any
    :param device: torch device.
    :type device: Any
    :param task_meta: nanochat task metadata dict.
    :type task_meta: Any
    :param output_dir: directory under which to write trajectories.
    :type output_dir: Optional[str]
    :param task_label: label of the task (used in the output path).
    :type task_label: Optional[str]
    :param num_trajectory_samples: number of samples to log per task; ``0`` disables.
    :type num_trajectory_samples: int
    :param show_progress: whether to render a tqdm progress bar.
    :type show_progress: bool
    :param progress_position: tqdm ``position`` argument.
    :type progress_position: int
    :return: ``(mean_correct, trajectories)``.
    :rtype: Tuple[float, List[Dict[str, Any]]]
    """
    n = len(data)
    correct = 0

    # Prepare trajectory output
    trajectories = []
    traj_file = None
    if num_trajectory_samples > 0 and output_dir and task_label:
        traj_dir = os.path.join(output_dir, "trajectories", task_label)
        os.makedirs(traj_dir, exist_ok=True)
        traj_path = os.path.join(traj_dir, "samples.jsonl")
        traj_file = open(traj_path, 'w', encoding='utf-8')

    # Sample-level progress bar (only for single-GPU path)
    iterator = range(n)
    if show_progress:
        iterator = tqdm(iterator, desc=f"  samples", position=progress_position,
                        leave=False, unit="ex")

    try:
        for idx in iterator:
            if num_trajectory_samples > 0 and idx < num_trajectory_samples:
                is_correct, traj = evaluate_example_with_trajectory(
                    idx, model, tokenizer, data, device, task_meta)
                trajectories.append(traj)
                if traj_file:
                    traj_file.write(json.dumps(traj, ensure_ascii=False) + '\n')
                    traj_file.flush()
            else:
                is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
            correct += int(is_correct)
    finally:
        if traj_file:
            traj_file.close()

    mean_correct = correct / n if n > 0 else 0.0
    return mean_correct, trajectories


def evaluate_core(model: "Any", tokenizer: "Any", device: "Any", eval_bundle_dir: str, max_per_task: int = -1,
                  task_samples: "Optional[Dict[str, int]]" = None, output_dir: "Optional[str]" = None,
                  num_trajectory_samples: int = 10) -> "Dict[str, Any]":
    """Run the full CORE evaluation suite on a single device.

    :param model: wrapped EBT model.
    :type model: Any
    :param tokenizer: tokenizer used for rendering prompts.
    :type tokenizer: Any
    :param device: torch device.
    :type device: Any
    :param eval_bundle_dir: directory containing ``core.yaml`` and
        ``eval_meta_data.csv``.
    :type eval_bundle_dir: str
    :param max_per_task: global default cap on examples per task; ``-1`` means
        use all samples.
    :type max_per_task: int
    :param task_samples: per-task overrides, e.g. ``{"hellaswag_zeroshot": 100}``.
    :type task_samples: Optional[Dict[str, int]]
    :param output_dir: directory to write trajectories into.
    :type output_dir: Optional[str]
    :param num_trajectory_samples: number of trajectories per task; ``0`` disables.
    :type num_trajectory_samples: int
    :return: dictionary with ``results``, ``centered_results``, ``core_metric`` and ``timing``.
    :rtype: Dict[str, Any]
    """
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    # Load config.
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # Load random baselines for centered-accuracy normalisation.
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    # Per-task result accumulators.
    results = {}
    centered_results = {}
    timing_records = []

    print("\n" + "="*80)
    print("Running CORE Evaluation")
    print("="*80 + "\n")

    task_pbar = tqdm(tasks, desc="Tasks", position=0, unit="task")
    for task_idx, task in enumerate(task_pbar):
        start_time = time.time()
        task_start_iso = datetime.now().isoformat()
        label = task['label']
        task_pbar.set_description(f"[{task_idx+1}/{len(tasks)}] {label}")

        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }

        tqdm.write(f"[{task_idx+1}/{len(tasks)}] Evaluating: {label}")
        tqdm.write(f"  Type: {task_meta['task_type']} | Few-shot: {task_meta['num_fewshot']}")

        # Load task data.
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # Decide how many samples to use for this task.
        task_max_samples = max_per_task
        if task_samples and label in task_samples:
            task_max_samples = task_samples[label]
            tqdm.write(f"  Using task-specific sample limit: {task_max_samples}")

        # Subsample for quick smoke tests.
        if task_max_samples > 0:
            shuffle_rng = random.Random(1337)
            shuffle_rng.shuffle(data)
            data = data[:task_max_samples]
            tqdm.write(f"  Using {len(data)} samples (subsampled)")
        else:
            tqdm.write(f"  Total samples: {len(data)} (evaluating ALL samples)")

        # Evaluate (trajectory-aware variant).
        accuracy, _ = evaluate_task_with_trajectory(
            model, tokenizer, data, device, task_meta,
            output_dir=output_dir, task_label=label,
            num_trajectory_samples=num_trajectory_samples,
            show_progress=True, progress_position=1)
        results[label] = accuracy

        # Compute centered (random-baseline-adjusted) accuracy.
        random_baseline = random_baselines.get(label, 50.0)
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result

        elapsed = time.time() - start_time
        task_end_iso = datetime.now().isoformat()
        timing_records.append({
            "task": label,
            "start_time": task_start_iso,
            "end_time": task_end_iso,
            "duration_seconds": round(elapsed, 2),
            "num_samples": len(data),
        })
        tqdm.write(f"  -> Accuracy: {accuracy:.4f} | Centered: {centered_result:.4f} | Time: {elapsed:.2f}s\n")

    task_pbar.close()

    # Aggregate the CORE metric across tasks.
    core_metric = sum(centered_results.values()) / len(centered_results)

    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
        "timing": timing_records,
    }


def _greedy_assign_tasks(all_tasks, num_gpus, history_timing):
    """Assign tasks to GPUs with greedy load balancing.

    Tasks are sorted by estimated duration (descending) and repeatedly placed
    on the GPU with the smallest accumulated cost. When no historical timing
    information is available the function falls back to round-robin.

    :param all_tasks: list of tasks from ``core.yaml`` (``icl_tasks``).
    :type all_tasks: list
    :param num_gpus: number of GPUs participating in the eval.
    :type num_gpus: int
    :param history_timing: parsed historical ``timing_summary.json`` content
        (dict or flat list), or ``None`` to force round-robin.
    :type history_timing: Any
    :return: tuple ``(assignment, used_greedy)`` where ``assignment`` maps
        rank to a list of task indices and ``used_greedy`` indicates whether
        greedy scheduling was applied.
    :rtype: tuple
    """
    # Attempt to read per-task durations from historical timing.
    per_task_duration = {}
    if history_timing is not None:
        per_task = history_timing.get("per_task", {})
        if not per_task:
            # Try flat list / records format
            if isinstance(history_timing, list):
                per_task = {rec["task"]: rec["duration_seconds"]
                            for rec in history_timing if "task" in rec}
            elif "timing" in history_timing:
                per_task = {rec["task"]: rec["duration_seconds"]
                            for rec in history_timing["timing"] if "task" in rec}
            elif "records" in history_timing:
                per_task = {rec["task"]: rec["duration_seconds"]
                            for rec in history_timing["records"] if "task" in rec}

        for label, val in per_task.items():
            if isinstance(val, dict):
                per_task_duration[label] = val.get("duration_seconds", 0)
            else:
                per_task_duration[label] = float(val)

    # Fall back to round-robin if no historical timing is available.
    if not per_task_duration:
        assignment = {r: [] for r in range(num_gpus)}
        for i in range(len(all_tasks)):
            assignment[i % num_gpus].append(i)
        return assignment, False

    # Greedy scheduling.
    # 1. Estimate each task's duration (unseen tasks get the median).
    known_durations = list(per_task_duration.values())
    median_duration = sorted(known_durations)[len(known_durations) // 2] if known_durations else 60.0

    task_costs = []
    for i, task in enumerate(all_tasks):
        label = task['label']
        cost = per_task_duration.get(label, median_duration)
        task_costs.append((i, label, cost))

    # 2. Sort by estimated cost, heaviest first.
    task_costs.sort(key=lambda x: x[2], reverse=True)

    # 3. Greedy placement using a (cumulative_cost, rank) min-heap.
    gpu_heap = [(0.0, r) for r in range(num_gpus)]
    heapq.heapify(gpu_heap)
    assignment = {r: [] for r in range(num_gpus)}

    for task_idx, label, cost in task_costs:
        min_cost, min_rank = heapq.heappop(gpu_heap)
        assignment[min_rank].append(task_idx)
        heapq.heappush(gpu_heap, (min_cost + cost, min_rank))

    return assignment, True


def _eval_worker(rank, world_size, ckpt_path, tokenizer_path, dtype,
                 eval_bundle_dir, max_per_task, task_samples,
                 all_tasks, random_baselines, result_dict,
                 timing_list, output_dir, num_trajectory_samples,
                 per_rank_indices=None):
    """Worker entry point for multi-GPU CORE evaluation.

    :param rank: this worker's GPU rank.
    :type rank: int
    :param world_size: total number of GPUs.
    :type world_size: int
    :param per_rank_indices: pre-computed ``{rank: [task_index, ...]}`` from the
        main process (greedy schedule). When ``None`` the worker falls back to
        round-robin.
    :type per_rank_indices: Any
    """
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('medium')

    # Use the pre-computed assignment, else fall back to round-robin.
    if per_rank_indices is not None and rank in per_rank_indices:
        task_indices = list(per_rank_indices[rank])
    else:
        task_indices = [i for i in range(len(all_tasks)) if i % world_size == rank]

    print(f"[GPU {rank}] Loading model... ({len(task_indices)} tasks assigned)")
    sys.stdout.flush()
    model, tokenizer, hparams = load_ebt_model(ckpt_path, tokenizer_path, device, dtype=dtype)

    data_base_path = os.path.join(eval_bundle_dir, "eval_data")

    for task_idx in task_indices:
        task = all_tasks[task_idx]
        label = task['label']
        start_time = time.time()
        task_start_iso = datetime.now().isoformat()

        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }

        print(f"[GPU {rank}] [{task_idx+1}/{len(all_tasks)}] Evaluating: {label}")
        sys.stdout.flush()

        # Load task data.
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # Decide sample count for this task.
        task_max_samples = max_per_task
        if task_samples and label in task_samples:
            task_max_samples = task_samples[label]

        if task_max_samples > 0:
            shuffle_rng = random.Random(1337)
            shuffle_rng.shuffle(data)
            data = data[:task_max_samples]

        # Evaluate (trajectory-aware); trajectories are written to rank-suffixed
        # files and later merged by the main process. tqdm is suppressed here.
        traj_file = None
        if num_trajectory_samples > 0 and output_dir:
            traj_dir = os.path.join(output_dir, "trajectories", label)
            os.makedirs(traj_dir, exist_ok=True)
            traj_path = os.path.join(traj_dir, f"samples_rank{rank}.jsonl")
            traj_file = open(traj_path, 'w', encoding='utf-8')

        n = len(data)
        correct = 0
        try:
            for idx in range(n):
                if num_trajectory_samples > 0 and idx < num_trajectory_samples:
                    is_correct, traj = evaluate_example_with_trajectory(
                        idx, model, tokenizer, data, device, task_meta)
                    if traj_file:
                        traj_file.write(json.dumps(traj, ensure_ascii=False) + '\n')
                        traj_file.flush()
                else:
                    is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
                correct += int(is_correct)
        finally:
            if traj_file:
                traj_file.close()

        accuracy = correct / n if n > 0 else 0.0

        # Centered (random-baseline-adjusted) accuracy.
        random_baseline = random_baselines.get(label, 50.0)
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)

        elapsed = time.time() - start_time
        task_end_iso = datetime.now().isoformat()
        print(f"[GPU {rank}]   -> {label}: acc={accuracy:.4f} centered={centered_result:.4f} ({elapsed:.2f}s)")
        sys.stdout.flush()

        result_dict[label] = (accuracy, centered_result)
        timing_list.append({
            "task": label,
            "start_time": task_start_iso,
            "end_time": task_end_iso,
            "duration_seconds": round(elapsed, 2),
            "num_samples": n,
            "gpu_rank": rank,
        })


def evaluate_core_multigpu(num_gpus, ckpt_path, tokenizer_path, dtype,
                           eval_bundle_dir, max_per_task, task_samples,
                           output_dir=None, num_trajectory_samples=10,
                           history_timing_path=None):
    """Run the CORE evaluation suite in parallel across multiple GPUs.

    Uses :func:`_greedy_assign_tasks` to spread tasks across ranks based on
    estimated duration and spawns one worker per GPU via ``mp.spawn``.

    :param num_gpus: number of GPUs to use.
    :type num_gpus: int
    :param ckpt_path: EBT checkpoint path.
    :type ckpt_path: str
    :param tokenizer_path: tokenizer path.
    :type tokenizer_path: str
    :param dtype: inference dtype.
    :type dtype: torch.dtype
    :param eval_bundle_dir: directory containing ``core.yaml``.
    :type eval_bundle_dir: str
    :param max_per_task: global cap on examples per task (``-1`` for all).
    :type max_per_task: int
    :param task_samples: per-task sample overrides.
    :type task_samples: Any
    :param output_dir: directory to write trajectories into.
    :type output_dir: Any
    :param num_trajectory_samples: trajectories per task (``0`` disables).
    :type num_trajectory_samples: int
    :param history_timing_path: path to historical ``timing_summary.json`` for
        greedy scheduling (optional).
    :type history_timing_path: Any
    :return: aggregated results dict matching :func:`evaluate_core`.
    :rtype: dict
    """
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    all_tasks = config['icl_tasks']

    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            random_baselines[row['Eval Task']] = float(row['Random baseline'])

    # Load historical timing for greedy scheduling.
    history_timing = None
    if history_timing_path and os.path.isfile(history_timing_path):
        try:
            with open(history_timing_path, 'r', encoding='utf-8') as f:
                history_timing = json.load(f)
        except Exception as e:
            print(f"  Warning: could not load history timing for scheduling: {e}")

    # Greedy scheduling (or fall back to round-robin).
    assignment, used_greedy = _greedy_assign_tasks(all_tasks, num_gpus, history_timing)

    print(f"\n{'='*80}")
    scheduling_method = "greedy load-balanced" if used_greedy else "round-robin (no history timing)"
    print(f"Running CORE Evaluation (multi-GPU: {num_gpus} GPUs, {len(all_tasks)} tasks)")
    print(f"Task scheduling: {scheduling_method}")
    print(f"{'='*80}")
    for rank in range(num_gpus):
        task_names = [all_tasks[i]['label'] for i in assignment[rank]]
        # Show estimated duration when historical timing is available.
        if used_greedy and history_timing is not None:
            per_task = history_timing.get("per_task", {})
            if not per_task and "records" in history_timing:
                per_task = {rec["task"]: rec for rec in history_timing["records"] if "task" in rec}
            total_est = 0
            for idx in assignment[rank]:
                label = all_tasks[idx]['label']
                val = per_task.get(label, {})
                if isinstance(val, dict):
                    total_est += val.get("duration_seconds", 0)
                else:
                    total_est += float(val)
            print(f"  GPU {rank}: {len(task_names)} tasks (est. {total_est:.0f}s = {total_est/60:.1f}min) -- {', '.join(task_names)}")
        else:
            print(f"  GPU {rank}: {len(task_names)} tasks -- {', '.join(task_names)}")
    print()

    # Shared result dict and timing list across worker processes.
    manager = mp.Manager()
    result_dict = manager.dict()
    timing_list = manager.list()

    # Convert each rank's task list into a manager.dict entry for IPC.
    per_rank_indices = manager.dict()
    for rank in range(num_gpus):
        per_rank_indices[rank] = assignment[rank]

    # Spawn worker processes with the precomputed task assignment.
    mp.spawn(
        _eval_worker,
        args=(num_gpus, ckpt_path, tokenizer_path, dtype,
              eval_bundle_dir, max_per_task, task_samples,
              all_tasks, random_baselines, result_dict,
              timing_list, output_dir, num_trajectory_samples,
              per_rank_indices),
        nprocs=num_gpus,
        join=True,
    )

    # Merge per-rank trajectory JSONL files into a single samples.jsonl per task.
    if num_trajectory_samples > 0 and output_dir:
        traj_base = os.path.join(output_dir, "trajectories")
        if os.path.isdir(traj_base):
            for task_dir_name in os.listdir(traj_base):
                task_dir = os.path.join(traj_base, task_dir_name)
                if not os.path.isdir(task_dir):
                    continue
                rank_files = sorted(glob_module.glob(os.path.join(task_dir, "samples_rank*.jsonl")))
                if rank_files:
                    merged_path = os.path.join(task_dir, "samples.jsonl")
                    with open(merged_path, 'w', encoding='utf-8') as out_f:
                        for rf in rank_files:
                            with open(rf, 'r', encoding='utf-8') as in_f:
                                for line in in_f:
                                    out_f.write(line)
                    # Clean up per-rank files
                    for rf in rank_files:
                        os.remove(rf)

    # Collect aggregate results.
    results = {}
    centered_results = {}
    for label, (acc, centered) in result_dict.items():
        results[label] = acc
        centered_results[label] = centered

    core_metric = sum(centered_results.values()) / len(centered_results) if centered_results else 0.0

    # Collect timing records across ranks.
    timing_records = list(timing_list)

    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
        "timing": timing_records,
    }


def _find_latest_timing(ebt_dir):
    """Return the newest ``timing_summary.json`` under ``{ebt_dir}/logs/core_eval/``.

    :param ebt_dir: ELM package directory whose ``logs/core_eval`` subtree is
        searched.
    :type ebt_dir: str
    :return: path to the latest ``timing_summary.json`` by mtime, or ``None``
        if none exists.
    :rtype: Any
    """
    search_dir = os.path.join(ebt_dir, "logs", "core_eval")
    if not os.path.isdir(search_dir):
        return None
    candidates = sorted(glob_module.glob(os.path.join(search_dir, "**/timing_summary.json"), recursive=True))
    if not candidates:
        return None
    # Return the latest by modification time.
    return max(candidates, key=os.path.getmtime)


def _print_eta_table(history_timing_path, tasks):
    """Load historical timing and print an ETA estimation table.

    :param history_timing_path: path to a ``timing_summary.json`` file.
    :type history_timing_path: str
    :param tasks: list of task dicts from ``core.yaml``.
    :type tasks: list
    """
    try:
        with open(history_timing_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except Exception as e:
        print(f"  Warning: could not load history timing: {e}")
        return

    per_task = history.get("per_task", {})
    if not per_task:
        # Try flat list format
        if isinstance(history, list):
            per_task = {rec["task"]: rec["duration_seconds"] for rec in history if "task" in rec}
        elif "timing" in history:
            per_task = {rec["task"]: rec["duration_seconds"] for rec in history["timing"] if "task" in rec}

    if not per_task:
        print(f"  Warning: no per-task timing found in {history_timing_path}")
        return

    task_labels = [t['label'] for t in tasks]
    total_est = 0
    print(f"\n  ETA Estimation (based on {history_timing_path}):")
    print(f"  {'Task':<40s} {'Est. Time':>10s}")
    print(f"  {'-'*40} {'-'*10}")
    for label in task_labels:
        est = per_task.get(label, {})
        if isinstance(est, dict):
            dur = est.get("duration_seconds", 0)
        else:
            dur = float(est)
        total_est += dur
        print(f"  {label:<40s} {dur:>8.1f}s")
    print(f"  {'-'*40} {'-'*10}")
    print(f"  {'TOTAL':<40s} {total_est:>8.1f}s  ({total_est/60:.1f}min)")
    print()


def main():
    """CLI entry point for running the EBT CORE evaluation."""
    parser = argparse.ArgumentParser(
        description="EBT CORE Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

1. Evaluate all samples (full CORE score):
   python -m scripts.ebt_core_eval --ckpt-path model.ckpt --tokenizer-path tokenizer/ \\
       --eval-bundle-dir eval_bundle/ --output-dir output/ --max-per-task -1

2. Quick test (100 samples per task):
   python -m scripts.ebt_core_eval --ckpt-path model.ckpt --tokenizer-path tokenizer/ \\
       --eval-bundle-dir eval_bundle/ --output-dir output/ --max-per-task 100

3. Set different sample counts for specific tasks:
   python -m scripts.ebt_core_eval --ckpt-path model.ckpt --tokenizer-path tokenizer/ \\
       --eval-bundle-dir eval_bundle/ --output-dir output/ --max-per-task 100 \\
       --task-samples "hellaswag_zeroshot:500,arc_easy:200"

4. Use multiple GPUs:
   python -m scripts.ebt_core_eval --ckpt-path model.ckpt --tokenizer-path tokenizer/ \\
       --eval-bundle-dir eval_bundle/ --output-dir output/ --gpus 4

5. Save trajectory (first 20 samples per task):
   python -m scripts.ebt_core_eval --ckpt-path model.ckpt --tokenizer-path tokenizer/ \\
       --eval-bundle-dir eval_bundle/ --output-dir output/ --num-trajectory-samples 20
        """
    )
    parser.add_argument('--ckpt-path', type=str, required=True, help='EBT checkpoint path')
    parser.add_argument('--tokenizer-path', type=str, required=True, help='Tokenizer path')
    parser.add_argument('--eval-bundle-dir', type=str, required=True, help='Path to eval_bundle directory')
    parser.add_argument('--eval-modes', type=str, default='core', help='Evaluation modes (default: core)')
    parser.add_argument('--max-per-task', type=int, default=-1,
                        help='Max examples per task globally (-1 = all samples for complete CORE score)')
    parser.add_argument('--task-samples', type=str, default='',
                        help='Task-specific sample limits (format: "task1:num1,task2:num2")')
    parser.add_argument('--device-batch-size', type=int, default=16, help='Batch size per device')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--gpus', type=int, default=-1,
                        help='Number of GPUs to use (-1 = auto-detect all available GPUs)')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'],
                        help='Model dtype for inference (default: bfloat16)')
    parser.add_argument('--num-trajectory-samples', type=int, default=10,
                        help='Number of trajectory samples to save per task (0 = disable)')
    parser.add_argument('--history-timing', type=str, default='',
                        help='Path to historical timing_summary.json for ETA estimation (auto-detect if empty)')
    args = parser.parse_args()

    # Parse per-task sample overrides.
    task_samples_dict = {}
    if args.task_samples:
        for item in args.task_samples.split(','):
            if ':' in item:
                task_name, num_samples = item.split(':', 1)
                task_samples_dict[task_name.strip()] = int(num_samples.strip())

    # Resolve device / GPU configuration.
    if args.gpus == -1:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"Auto-detected {num_gpus} GPU(s)")
        else:
            num_gpus = 0
            print("No GPU detected, using CPU")
    else:
        num_gpus = args.gpus
        print(f"Using {num_gpus} GPU(s) as specified")

    dtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16

    # Enable TF32 for Tensor Core acceleration on H200/A100.
    torch.set_float32_matmul_precision('medium')

    # Ensure the output directory exists.
    os.makedirs(args.output_dir, exist_ok=True)

    # ETA estimation: load historical timing if available.
    ebt_dir = str(Path(__file__).parent.parent)
    history_timing_path = args.history_timing
    if not history_timing_path:
        history_timing_path = _find_latest_timing(ebt_dir)
    if history_timing_path and os.path.isfile(history_timing_path):
        print(f"\nFound historical timing: {history_timing_path}")
        # Load task list for the ETA table.
        config_path = os.path.join(args.eval_bundle_dir, "core.yaml")
        if os.path.isfile(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                _config = yaml.safe_load(f)
            _print_eta_table(history_timing_path, _config['icl_tasks'])
    else:
        print("\nNo historical timing found (first run or --history-timing not specified)")

    eval_start_time = time.time()
    eval_start_iso = datetime.now().isoformat()

    # Multi-GPU path.
    if num_gpus > 1 and 'core' in args.eval_modes:
        print(f"\nUsing multi-GPU evaluation with {num_gpus} GPUs")
        core_results = evaluate_core_multigpu(
            num_gpus=num_gpus,
            ckpt_path=args.ckpt_path,
            tokenizer_path=args.tokenizer_path,
            dtype=dtype,
            eval_bundle_dir=args.eval_bundle_dir,
            max_per_task=args.max_per_task,
            task_samples=task_samples_dict if task_samples_dict else None,
            output_dir=args.output_dir,
            num_trajectory_samples=args.num_trajectory_samples,
            history_timing_path=history_timing_path,
        )
    else:
        # Single-GPU / CPU path.
        if num_gpus >= 1:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        print(f"Using device: {device}\n")

        print("="*80)
        print("Loading EBT Model")
        print("="*80 + "\n")

        model, tokenizer, hparams = load_ebt_model(args.ckpt_path, args.tokenizer_path, device, dtype=dtype)

        core_results = None
        if 'core' in args.eval_modes:
            core_results = evaluate_core(
                model, tokenizer, device, args.eval_bundle_dir,
                max_per_task=args.max_per_task,
                task_samples=task_samples_dict if task_samples_dict else None,
                output_dir=args.output_dir,
                num_trajectory_samples=args.num_trajectory_samples,
            )

    eval_end_time = time.time()
    eval_end_iso = datetime.now().isoformat()
    total_seconds = round(eval_end_time - eval_start_time, 2)

    # Print and persist results.
    if core_results:
        print("\n" + "="*80)
        print("CORE Evaluation Results")
        print("="*80 + "\n")

        print(f"CORE Metric: {core_results['core_metric']:.4f}\n")

        print("Task Results:")
        for task_name, accuracy in sorted(core_results['results'].items()):
            centered = core_results['centered_results'][task_name]
            print(f"  {task_name:40s}: {accuracy:.4f} (centered: {centered:.4f})")

        # Save results (timing is persisted separately below).
        results_to_save = {
            "results": core_results["results"],
            "centered_results": core_results["centered_results"],
            "core_metric": core_results["core_metric"],
        }
        results_file = os.path.join(args.output_dir, "core_results.json")
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"\n-> Results saved to: {results_file}")

        # Save CSV.
        csv_file = os.path.join(args.output_dir, "core_results.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Task', 'Accuracy', 'Centered', 'CORE'])
            for task_name in sorted(core_results['results'].keys()):
                writer.writerow([
                    task_name,
                    f"{core_results['results'][task_name]:.4f}",
                    f"{core_results['centered_results'][task_name]:.4f}",
                    ''
                ])
            writer.writerow(['OVERALL', '', '', f"{core_results['core_metric']:.4f}"])
        print(f"-> CSV saved to: {csv_file}")

        # Save timing_summary.json.
        timing_records = core_results.get("timing", [])
        per_task_timing = {}
        for rec in timing_records:
            per_task_timing[rec["task"]] = {
                "duration_seconds": rec["duration_seconds"],
                "num_samples": rec.get("num_samples", -1),
                "start_time": rec.get("start_time", ""),
                "end_time": rec.get("end_time", ""),
            }
        timing_summary = {
            "total_seconds": total_seconds,
            "total_start": eval_start_iso,
            "total_end": eval_end_iso,
            "num_gpus": num_gpus if num_gpus > 1 else 1,
            "per_task": per_task_timing,
            "records": timing_records,
        }
        timing_file = os.path.join(args.output_dir, "timing_summary.json")
        with open(timing_file, 'w') as f:
            json.dump(timing_summary, f, indent=2)
        print(f"-> Timing saved to: {timing_file}")

        # Hint at the trajectory output location.
        traj_dir = os.path.join(args.output_dir, "trajectories")
        if os.path.isdir(traj_dir):
            traj_tasks = [d for d in os.listdir(traj_dir) if os.path.isdir(os.path.join(traj_dir, d))]
            print(f"-> Trajectories saved for {len(traj_tasks)} tasks in: {traj_dir}/")

        # Machine-parseable summary line.
        num_tasks = len(core_results['results'])
        avg_acc = np.mean(list(core_results['results'].values()))
        print(f"\n[EVAL_SUMMARY] dataset=core core_metric={core_results['core_metric']:.4f} avg_accuracy={avg_acc:.4f} num_tasks={num_tasks} total_time={total_seconds:.1f}s")

    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
