import torch
import torchmetrics
try:
    import nltk
except ImportError:
    nltk = None
import string
from typing import List
import torch.distributed as dist
try:
    import ipdb
except ImportError:
    ipdb = None
import math

custom_nltk_path = "/mnt/shared-storage-user/lixueyan/datasets/nltk_data"
if nltk is not None and custom_nltk_path not in nltk.data.path:
    nltk.data.path.insert(0, custom_nltk_path)

def get_torchmetrics(metric, metrics_average_type, num_classes, metrics_task):
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
    '''
    Method for calculating the Exact Match (EM) score for Question Answering / Text Generation tasks
    '''
    assert len(predictions) > 0, "Predictions list cannot be empty"
    assert len(predictions) == len(references), "Predictions and references must have same length"
    
    total = len(predictions)
    correct = sum(
        normalize_text(pred) == normalize_text(ref)
        for pred, ref in zip(predictions, references)
    )
    return correct / total


def calculate_f1_score(predictions: List[str], references: List[str]) -> float:
    '''
    if nltk is None:
        raise ImportError("calculate_f1_score requires nltk")
    Method for calculating the F1 score for Question Answering / Text Generation tasks
    '''
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
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in set(string.punctuation))
    text = ' '.join(text.split())
    return text


@torch.no_grad()
def calculate_bpb_score(next_token_indices, per_token_loss,token_bytes):
    """
      Compute bits-per-byte (BPB) loss, a vocab-size agnostic evaluation metric.
      Mirrors forward_loss_wrapper's structure. Normalizes cross-entropy (nats) by
      the byte length of each target token, following nanochat/loss_eval.py:evaluate_bpb().

      Args:
          x: same tuple format as forward_loss_wrapper — (input_ids_batch, targets_batch, ...)
          token_bytes: 1D LongTensor of shape (vocab_size,), byte count per token id;
                       0 marks special tokens (e.g. <|bos|>) excluded from the metric.
          phase: "val" or "test" — always uses no_randomness=True

      Returns:
          log_dict with 'bpb' and auxiliary keys matching forward_loss_wrapper.
    """
    # Map target tokens → byte lengths; explicitly handle ignore_index (y < 0) from finetune masking
    
    if (next_token_indices.int() < 0).any():
        valid = next_token_indices >= 0
        y_safe = torch.where(valid, next_token_indices, torch.zeros_like(next_token_indices))
        num_bytes = torch.where(
            valid,
            token_bytes[y_safe],
            torch.zeros_like(next_token_indices, dtype=token_bytes.dtype),
        )
    else:
        num_bytes = token_bytes[next_token_indices]  # [B*S]

    # Only count positions where token has bytes > 0 (excludes special tokens)
    total_nats = (per_token_loss * (num_bytes > 0)).sum()
    total_bytes = num_bytes.sum().to(torch.int64)

    # DDP aggregation across ranks
    if dist.is_initialized():
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)

    total_nats_val = total_nats.item()
    total_bytes_val = total_bytes.item()
    bpb = total_nats_val / (math.log(2) * total_bytes_val) if total_bytes_val > 0 else float('inf')
    
    return bpb
