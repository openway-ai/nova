"""Evaluation metrics used across EBT training and QA-style eval scripts."""

from typing import Any, List

import math
import string

import nltk
import torch
import torch.distributed as dist
import torchmetrics
import ipdb

custom_nltk_path = "/mnt/shared-storage-user/lixueyan/datasets/nltk_data"
if custom_nltk_path not in nltk.data.path:
    nltk.data.path.insert(0, custom_nltk_path)

def get_torchmetrics(
    metric: str,
    metrics_average_type: str,
    num_classes: int,
    metrics_task: str,
) -> Any:
    """Return the ``torchmetrics`` instance that matches ``metric``.

    :param metric: Metric name substring (must contain ``accuracy``,
        ``f1_score``, ``precision`` or ``recall``).
    :type metric: str
    :param metrics_average_type: Averaging strategy forwarded to torchmetrics.
    :type metrics_average_type: str
    :param num_classes: Number of classes for the metric.
    :type num_classes: int
    :param metrics_task: Task kind forwarded to torchmetrics
        (``"binary"`` / ``"multiclass"`` / ``"multilabel"``).
    :type metrics_task: str
    :return: Configured torchmetrics metric instance.
    :rtype: Any
    :raises ValueError: When ``metric`` does not match any supported name.
    """
    if 'accuracy' in metric:
        return torchmetrics.Accuracy(average=metrics_average_type, num_classes=num_classes, task=metrics_task)
    elif 'f1_score' in metric:
        return torchmetrics.F1Score(average=metrics_average_type, num_classes=num_classes, task=metrics_task)
    elif 'precision' in metric:
        return torchmetrics.Precision(average=metrics_average_type, num_classes=num_classes, task=metrics_task)
    elif 'recall' in metric:
        return torchmetrics.Recall(average=metrics_average_type, num_classes=num_classes, task=metrics_task)
    else:
        raise ValueError(f"metric {metric} unimplemented")


def calculate_em(predictions: List[str], references: List[str]) -> float:
    """Exact-match score for QA / text-generation tasks.

    :param predictions: Model predictions.
    :type predictions: List[str]
    :param references: Ground-truth answers.
    :type references: List[str]
    :return: Fraction of predictions that exactly match the reference after
        :func:`normalize_text` normalization.
    :rtype: float
    :raises AssertionError: If the inputs are empty or of different lengths.
    """
    assert len(predictions) > 0, "Predictions list cannot be empty"
    assert len(predictions) == len(references), "Predictions and references must have same length"

    total = len(predictions)
    correct = sum(
        normalize_text(pred) == normalize_text(ref)
        for pred, ref in zip(predictions, references)
    )
    return correct / total


def calculate_f1_score(predictions: List[str], references: List[str]) -> float:
    """Token-level F1 score for QA / text-generation tasks.

    :param predictions: Model predictions.
    :type predictions: List[str]
    :param references: Ground-truth answers.
    :type references: List[str]
    :return: Average token-level F1 across the provided pairs.
    :rtype: float
    :raises AssertionError: If the inputs are empty or of different lengths.
    """
    assert len(predictions) > 0, "Predictions list cannot be empty"
    assert len(predictions) == len(references), "Predictions and references must have same length"

    total_f1 = 0
    for pred, ref in zip(predictions, references):
        pred_tokens = nltk.word_tokenize(normalize_text(pred))
        ref_tokens = nltk.word_tokenize(normalize_text(ref))

        pred_counter = {}
        ref_counter = {}
        for token in pred_tokens:
            pred_counter[token] = pred_counter.get(token, 0) + 1
        for token in ref_tokens:
            ref_counter[token] = ref_counter.get(token, 0) + 1

        matches = 0
        for token in pred_counter:
            if token in ref_counter:
                matches += min(pred_counter[token], ref_counter[token])

        precision = matches / len(pred_tokens) if pred_tokens else 0
        recall = matches / len(ref_tokens) if ref_tokens else 0

        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        total_f1 += f1

    return total_f1 / len(predictions)


def normalize_text(text: str) -> str:
    """Lowercase the text, strip punctuation, and collapse whitespace.

    :param text: Input text.
    :type text: str
    :return: Normalized text suitable for EM/F1 comparison.
    :rtype: str
    """
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in set(string.punctuation))
    text = ' '.join(text.split())
    return text


@torch.no_grad()
def calculate_bpb_score(
    next_token_indices: torch.Tensor,
    per_token_loss: torch.Tensor,
    token_bytes: torch.Tensor,
) -> float:
    """Compute bits-per-byte (BPB), a vocab-size-agnostic evaluation metric.

    Mirrors :func:`forward_loss_wrapper`: normalizes the per-token
    cross-entropy (in nats) by the byte length of each target token, following
    ``nanochat/loss_eval.py::evaluate_bpb``.

    :param next_token_indices: Flat tensor of target token ids
        (shape ``(B*S,)``). Negative entries are treated as ``ignore_index``.
    :type next_token_indices: torch.Tensor
    :param per_token_loss: Per-token cross-entropy in nats (shape ``(B*S,)``).
    :type per_token_loss: torch.Tensor
    :param token_bytes: 1-D ``LongTensor`` of shape ``(vocab_size,)`` with the
        UTF-8 byte length of each token id. Entries equal to ``0`` mark
        special tokens that are excluded from the metric.
    :type token_bytes: torch.Tensor
    :return: Bits-per-byte score aggregated across ranks when DDP is active,
        or ``float('inf')`` if there are no valid bytes in the batch.
    :rtype: float
    """
    # Map target tokens to byte lengths; explicitly handle ``ignore_index``
    # (entries with ``y < 0`` introduced by fine-tune masking).

    if (next_token_indices.int() < 0).any():
        valid = next_token_indices >= 0
        y_safe = torch.where(valid, next_token_indices, torch.zeros_like(next_token_indices))
        num_bytes = torch.where(
            valid,
            token_bytes[y_safe],
            torch.zeros_like(next_token_indices, dtype=token_bytes.dtype),
        )
    else:
        num_bytes = token_bytes[next_token_indices]

    # Only count positions whose byte length is positive (excludes specials).
    total_nats = (per_token_loss * (num_bytes > 0)).sum()
    total_bytes = num_bytes.sum().to(torch.int64)

    # DDP aggregation across ranks.
    if dist.is_initialized():
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)

    total_nats_val = total_nats.item()
    total_bytes_val = total_bytes.item()
    bpb = total_nats_val / (math.log(2) * total_bytes_val) if total_bytes_val > 0 else float('inf')

    return bpb, total_nats_val, total_bytes_val
