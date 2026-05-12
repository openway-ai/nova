"""Post-hoc evaluation utilities for generated answer files."""

from typing import Tuple

import json

from openebm.elm import metrics


def nlp_eval_acc(answer_file: str) -> Tuple[float, float]:
    """Compute Exact-Match and F1 accuracy for a generated answer file.

    The answer file is expected to contain one JSON object per line with a
    ``generation`` field and either a ``target`` or (legacy) ``gt_answer``
    field. Anything after ``"\\n####"`` in the ground truth is treated as the
    canonical answer (GSM8K convention).

    :param answer_file: Path to the JSONL file of predictions.
    :type answer_file: str
    :return: ``(exact_match, f1)`` both in the ``[0, 1]`` range.
    :rtype: Tuple[float, float]
    """
    data_name = answer_file.split("/")[-1].split("_")[-1].split(".")[0]
    model_name = '_'.join(answer_file.split("/")[-1].split("_")[:-1])
    predictions = []
    references = []

    with open(answer_file, 'r', encoding='utf-8') as f:
        skipped = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            output = data['generation']
            # Support both old and new field names for backward compatibility
            # (new: ``target``, old: ``gt_answer``).
            gt_ans = data.get('target', data.get('gt_answer', '')).split("\n####")[-1].strip()

            predictions.append(output)
            references.append(gt_ans)
        if skipped > 0:
            print(f"WARNING: Skipped {skipped} corrupted JSON lines in {answer_file} (possibly caused by concurrent writes)")

    em_score = metrics.calculate_em(predictions, references)
    f1_score = metrics.calculate_f1_score(predictions, references)

    print(f"{model_name} Results on {data_name}:")
    print(f"Exact Match Accuracy: {100*em_score:.3f}%")
    print(f"F1 Score: {100*f1_score:.3f}%")
    return em_score, f1_score
