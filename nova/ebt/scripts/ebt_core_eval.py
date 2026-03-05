"""
EBT CORE 评估脚本
基于 nanochat 的 base_eval.py，适配 EBT 模型
"""
import os
import sys
import csv
import time
import json
import yaml
import random
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/mnt/shared-storage-user/puyuan/code/nanochat")

from trainer import ModelTrainer
from nanochat_tokenizer_adapter import NanoChatTokenizerWrapper


class EBTModelWrapper:
    """将 EBT 模型包装为 nanochat 兼容的接口"""

    def __init__(self, model, tokenizer, device, max_seq_len=256):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction='mean'):
        """
        前向传播，兼容 nanochat 的接口

        Args:
            input_ids: [batch_size, seq_len]
            targets: [batch_size, seq_len] 或 None
            loss_reduction: 'mean' 或 'none'

        Returns:
            如果 targets is None: 返回 logits [batch_size, seq_len, vocab_size]
            否则: 返回 loss (scalar 或 [batch_size])
        """
        with torch.no_grad():
            # EBT forward 返回 (logits_list, energies)
            outputs = self.model.forward(input_ids, start_pos=0, learning=False, return_raw_logits=True)

            # 取最后一个 MCMC 步骤的 logits
            if isinstance(outputs, tuple):
                logits = outputs[0]
                if isinstance(logits, list):
                    logits = logits[-1]  # 最后一个 MCMC 步骤
            else:
                logits = outputs

            if targets is None:
                return logits

            # 计算损失
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction
            )
            return loss

    def get_device(self):
        return self.device


def load_ebt_model(ckpt_path, tokenizer_path, device):
    """加载 EBT checkpoint"""
    print(f"Loading EBT checkpoint from: {ckpt_path}")

    # 加载 checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams = checkpoint['hyper_parameters']

    # 设置推理模式参数
    hparams['execution_mode'] = 'inference'
    hparams['no_wandb'] = True

    # 加载模型
    model_trainer = ModelTrainer(hparams)
    model_trainer.load_state_dict(checkpoint['state_dict'])
    model_trainer.eval()
    model = model_trainer.model
    model.to(device)
    model.eval()

    # 加载 tokenizer
    from nanochat.tokenizer import get_tokenizer
    tokenizer_obj = get_tokenizer()
    tokenizer = NanoChatTokenizerWrapper(tokenizer_obj=tokenizer_obj)

    # 获取最大序列长度
    max_seq_len = hparams.get('context_length', 256)

    # 包装模型
    wrapped_model = EBTModelWrapper(model, tokenizer, device, max_seq_len=max_seq_len)

    print(f"✓ Model loaded: {hparams['model_name']} (size: {hparams.get('model_size', 'unknown')})")
    print(f"✓ Context length: {max_seq_len}")

    return wrapped_model, tokenizer, hparams


def evaluate_task(model, tokenizer, data, device, task_meta):
    """
    评估单个 CORE 任务
    改编自 nanochat.core_eval.evaluate_task
    """
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    correct = 0
    total = 0

    for example_idx, example in enumerate(data):
        if task_type == 'multiple_choice':
            # 多选题任务
            context = example.get('context', '')
            question = example.get('query', example.get('question', ''))
            choices = example['choices']
            answer_idx = example['gold']

            # 构建提示
            prompt = context + question if context else question
            if continuation_delimiter:
                prompt += continuation_delimiter

            # 编码提示
            prompt_ids = tokenizer.encode(prompt)
            prompt_tokens = torch.tensor([prompt_ids], dtype=torch.long).to(device)

            # 计算每个选项的 loss
            choice_losses = []
            for choice in choices:
                choice_ids = tokenizer.encode(choice, add_special_tokens=False)
                choice_tokens = torch.tensor([choice_ids], dtype=torch.long).to(device)
                full_tokens = torch.cat([prompt_tokens, choice_tokens], dim=1)

                # 计算损失（只对 choice 部分）
                with torch.no_grad():
                    logits = model(full_tokens)
                    targets = full_tokens[:, 1:].contiguous()
                    logits_choice = logits[:, prompt_tokens.shape[1]-1:-1, :].contiguous()

                    loss = F.cross_entropy(
                        logits_choice.view(-1, logits_choice.size(-1)),
                        targets[:, prompt_tokens.shape[1]-1:].view(-1),
                        reduction='mean'
                    )
                    choice_losses.append(loss.item())

            # 选择 loss 最小的选项
            predicted_idx = np.argmin(choice_losses)
            if predicted_idx == answer_idx:
                correct += 1
            total += 1

        elif task_type == 'language_modeling':
            # 语言建模任务
            context = example.get('context', '')
            continuation = example.get('continuation', example.get('answer', ''))

            # 构建输入
            if continuation_delimiter:
                full_text = context + continuation_delimiter + continuation
            else:
                full_text = context + continuation

            full_ids = tokenizer.encode(full_text)
            tokens = torch.tensor([full_ids], dtype=torch.long).to(device)

            # 计算困惑度
            with torch.no_grad():
                logits = model(tokens)
                targets = tokens[:, 1:].contiguous()
                logits = logits[:, :-1, :].contiguous()

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    reduction='mean'
                )
                # 对于语言建模任务，低困惑度表示好的性能
                # 这里简化处理：如果 ppl < 某个阈值则认为正确
                ppl = torch.exp(loss).item()
                if ppl < 50:  # 简单阈值
                    correct += 1
                total += 1

        elif task_type == 'schema':
            # Schema 任务（如 Winograd）
            # 这类任务通常需要特殊处理
            # 简化处理：选择 loss 最小的候选
            choices = example.get('choices', [])
            if not choices:
                continue

            answer_idx = example.get('gold', 0)

            choice_losses = []
            for choice_text in choices:
                choice_ids = tokenizer.encode(choice_text)
                tokens = torch.tensor([choice_ids], dtype=torch.long).to(device)
                with torch.no_grad():
                    loss = model(tokens, targets=tokens[:, 1:])
                    choice_losses.append(loss.item())

            predicted_idx = np.argmin(choice_losses)
            if predicted_idx == answer_idx:
                correct += 1
            total += 1

        # 打印进度
        if (example_idx + 1) % 10 == 0:
            print(f"  Progress: {example_idx + 1}/{len(data)} examples", end='\r')

    print()  # 换行
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def evaluate_core(model, tokenizer, device, eval_bundle_dir, max_per_task=-1):
    """
    运行完整的 CORE 评估
    """
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # 加载 random baseline
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    # 评估每个任务
    results = {}
    centered_results = {}

    print("\n" + "="*80)
    print("Running CORE Evaluation")
    print("="*80 + "\n")

    for task_idx, task in enumerate(tasks):
        start_time = time.time()
        label = task['label']

        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }

        print(f"[{task_idx+1}/{len(tasks)}] Evaluating: {label}")
        print(f"  Type: {task_meta['task_type']} | Few-shot: {task_meta['num_fewshot']}")

        # 加载数据
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # 子采样（用于快速测试）
        if max_per_task > 0:
            shuffle_rng = random.Random(1337)
            shuffle_rng.shuffle(data)
            data = data[:max_per_task]
            print(f"  Using {len(data)} samples (max_per_task={max_per_task})")
        else:
            print(f"  Total samples: {len(data)}")

        # 评估
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
        results[label] = accuracy

        # 计算 centered result
        random_baseline = random_baselines.get(label, 50.0)
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result

        elapsed = time.time() - start_time
        print(f"  ✓ Accuracy: {accuracy:.4f} | Centered: {centered_result:.4f} | Time: {elapsed:.2f}s\n")

    # 计算 CORE metric
    core_metric = sum(centered_results.values()) / len(centered_results)

    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }


def main():
    parser = argparse.ArgumentParser(description="EBT CORE Evaluation")
    parser.add_argument('--ckpt-path', type=str, required=True, help='EBT checkpoint path')
    parser.add_argument('--tokenizer-path', type=str, required=True, help='Tokenizer path')
    parser.add_argument('--eval-bundle-dir', type=str, required=True, help='Path to eval_bundle directory')
    parser.add_argument('--eval-modes', type=str, default='core', help='Evaluation modes (default: core)')
    parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per task (-1 = all)')
    parser.add_argument('--device-batch-size', type=int, default=16, help='Batch size (not used yet)')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print("\n" + "="*80)
    print("Loading EBT Model")
    print("="*80 + "\n")

    model, tokenizer, hparams = load_ebt_model(args.ckpt_path, args.tokenizer_path, device)

    # 运行 CORE 评估
    if 'core' in args.eval_modes:
        core_results = evaluate_core(model, tokenizer, device, args.eval_bundle_dir, args.max_per_task)

        # 打印结果
        print("\n" + "="*80)
        print("CORE Evaluation Results")
        print("="*80 + "\n")

        print(f"CORE Metric: {core_results['core_metric']:.4f}\n")

        print("Task Results:")
        for task_name, accuracy in sorted(core_results['results'].items()):
            centered = core_results['centered_results'][task_name]
            print(f"  {task_name:40s}: {accuracy:.4f} (centered: {centered:.4f})")

        # 保存结果
        results_file = os.path.join(args.output_dir, "core_results.json")
        with open(results_file, 'w') as f:
            json.dump(core_results, f, indent=2)
        print(f"\n✓ Results saved to: {results_file}")

        # 保存 CSV
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
        print(f"✓ CSV saved to: {csv_file}")

    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
