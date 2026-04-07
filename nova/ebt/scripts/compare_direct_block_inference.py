import argparse
import copy
import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EBT_DIR = os.path.dirname(THIS_DIR)
if EBT_DIR not in sys.path:
    sys.path.insert(0, EBT_DIR)

from generate import generate_text, _get_tokenizer, get_ppl
from trainer import ModelTrainer


def _encode_text(tokenizer, text):
    ids = tokenizer.encode(text)
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return [int(x) for x in ids]


def _build_generation_batch(prompt_ids, eos_id):
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)
    prompt_mask = torch.ones_like(prompt_tensor, dtype=torch.long)
    questions = {"input_ids": prompt_tensor, "attention_mask": prompt_mask}
    answers = {"input_ids": torch.tensor([[eos_id]], dtype=torch.long)}
    return questions, answers


def _token_overlap_metrics(tokenizer, baseline_text, candidate_text):
    baseline_ids = _encode_text(tokenizer, baseline_text)
    cand_ids = _encode_text(tokenizer, candidate_text)
    token_em = baseline_ids == cand_ids

    prefix = 0
    for a, b in zip(baseline_ids, cand_ids):
        if a != b:
            break
        prefix += 1
    denom = max(1, len(baseline_ids))
    prefix_match_ratio = prefix / denom
    return token_em, prefix_match_ratio


def _repetition_ratio(tokenizer, text):
    ids = _encode_text(tokenizer, text)
    if len(ids) == 0:
        return 0.0
    return 1.0 - (len(set(ids)) / len(ids))


def _maybe_eval_ppl(model, hparams, tokenizer, prompt_text, target_text):
    if target_text is None:
        return None, None

    full_ids = _encode_text(tokenizer, prompt_text + target_text)
    if len(full_ids) < 2:
        return None, None
    full_tensor = torch.tensor([full_ids], dtype=torch.long, device="cuda")
    batch_dict = {"input_ids": full_tensor.unsqueeze(1)}
    outputs = get_ppl(model, batch_dict, hparams)
    return outputs["loss"].item(), outputs["perplexity"].item()


def _run_once(model, hparams, tokenizer, prompt_ids, eos_id, mode, block_size):
    run_hparams = copy.deepcopy(hparams)
    run_hparams.infer_block_mode = mode
    run_hparams.infer_block_size = block_size
    run_hparams.infer_logprobs = False

    batch = _build_generation_batch(prompt_ids, eos_id)
    out = generate_text(model, batch, run_hparams)[0]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text to compare decoding modes")
    parser.add_argument("--target_text", type=str, default=None, help="Optional target continuation to compute loss/ppl")
    parser.add_argument("--max_gen_len", type=int, default=64)
    parser.add_argument("--refine_steps", type=int, default=1)
    parser.add_argument("--init_logit_scale", type=float, default=8.0)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hparams = ckpt["hyper_parameters"]
    hparams["execution_mode"] = "inference"
    hparams["infer_max_gen_len"] = args.max_gen_len
    hparams["infer_temp"] = 0.0
    hparams["infer_topp"] = 1.0
    hparams["infer_block_use_refine"] = True
    hparams["infer_block_refine_steps"] = args.refine_steps
    hparams["infer_block_init_logit_scale"] = args.init_logit_scale
    hparams["infer_block_mode"] = "sequential"
    hparams["infer_block_size"] = 1
    hparams["infer_logprobs"] = False

    wrapper = ModelTrainer(hparams)
    wrapper.load_state_dict(ckpt["state_dict"], strict=True)
    model = wrapper.model.cuda().eval()
    tokenizer = _get_tokenizer(wrapper.hparams)

    eos_id = getattr(tokenizer, "eos_token_id", 0)
    prompt_ids = _encode_text(tokenizer, args.prompt)

    block_k_half = max(2, int(wrapper.hparams.context_length) // 2)
    configs = [
        ("sequential", 1),
        ("direct_block", 2),
        ("direct_block", block_k_half),
    ]

    print("=" * 100)
    print("Direct-Block Inference Comparison")
    print("=" * 100)
    print(f"Prompt: {args.prompt}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Context length: {wrapper.hparams.context_length}")
    print(f"Configs: {configs}")
    print()

    results = []
    for mode, k in configs:
        out = _run_once(model, wrapper.hparams, tokenizer, prompt_ids, eos_id, mode, k)
        loss, ppl = _maybe_eval_ppl(model, wrapper.hparams, tokenizer, args.prompt, args.target_text)
        out["loss"] = loss
        out["ppl"] = ppl
        out["repetition_ratio"] = _repetition_ratio(tokenizer, out["generation"])
        results.append(((mode, k), out))

    baseline_text = results[0][1]["generation"]
    for (mode, k), out in results:
        token_em, prefix_ratio = _token_overlap_metrics(tokenizer, baseline_text, out["generation"])
        out["token_em_vs_sequential"] = token_em
        out["prefix_match_ratio_vs_sequential"] = prefix_ratio

    for (mode, k), out in results:
        print("-" * 100)
        print(f"Mode={mode}, block_size={k}")
        print(f"decode_time_sec: {out.get('decode_time_sec')}")
        print(f"block_energy_before_refine: {out.get('block_energy_before_refine')}")
        print(f"block_energy_after_refine: {out.get('block_energy_after_refine')}")
        print(f"loss: {out.get('loss')}, ppl: {out.get('ppl')}")
        print(f"token_em_vs_sequential: {out.get('token_em_vs_sequential')}")
        print(f"prefix_match_ratio_vs_sequential: {out.get('prefix_match_ratio_vs_sequential'):.4f}")
        print(f"repetition_ratio: {out.get('repetition_ratio'):.4f}")
        print("generation:")
        print(out["generation"])
        print()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
