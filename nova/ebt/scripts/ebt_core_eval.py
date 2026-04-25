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
import torch.multiprocessing as mp
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/mnt/shared-storage-user/puyuan/code/nanochat")

from trainer import ModelTrainer
from nanochat_tokenizer_adapter import NanoChatTokenizerWrapper
from nanochat.core_eval import (
    evaluate_example,
    evaluate_task as _nanochat_evaluate_task,
    forward_model,
)


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


def load_ebt_model(ckpt_path, tokenizer_path, device, dtype=torch.bfloat16):
    """加载 EBT checkpoint"""
    print(f"Loading EBT checkpoint from: {ckpt_path}")

    # 加载 checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams = checkpoint['hyper_parameters']

    # 设置推理模式参数
    hparams['execution_mode'] = 'inference'
    hparams['no_wandb'] = True
    # 禁用 torch.compile — eval 不需要编译，且 compile + MCMC grad + mp.spawn 会导致 CUBLAS 错误
    hparams['compile_model'] = False

    # 加载模型
    model_trainer = ModelTrainer(hparams)

    # 复用 on_load_checkpoint 的 _orig_mod 前缀修复逻辑
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

    # 加载 tokenizer
    from nanochat.tokenizer import get_tokenizer
    tokenizer_obj = get_tokenizer()
    tokenizer_wrapper = NanoChatTokenizerWrapper(tokenizer_obj=tokenizer_obj)

    # 获取最大序列长度
    max_seq_len = hparams.get('context_length', 256)

    # 包装模型
    wrapped_model = EBTModelWrapper(model, tokenizer_wrapper, device, max_seq_len=max_seq_len)

    print(f"✓ Model loaded: {hparams['model_name']} (size: {hparams.get('model_size', 'unknown')})")
    print(f"✓ Context length: {max_seq_len}")

    # 返回原始 RustBPETokenizer 给 eval 函数（nanochat core_eval 需要 .get_bos_token_id() 和 __call__(prompts, prepend=...) 接口）
    return wrapped_model, tokenizer_obj, hparams


def evaluate_task(model, tokenizer, data, device, task_meta, max_seq_len=None):
    """适配 nanochat 的 evaluate_task，忽略 max_seq_len 参数（已通过 model.max_seq_len 传递）"""
    return _nanochat_evaluate_task(model, tokenizer, data, device, task_meta)


def evaluate_core(model, tokenizer, device, eval_bundle_dir, max_per_task=-1, task_samples=None):
    """
    运行完整的 CORE 评估

    Args:
        model: 模型
        tokenizer: tokenizer
        device: 设备
        eval_bundle_dir: 评估数据目录
        max_per_task: 全局默认每个任务的最大样本数，-1 表示全部
        task_samples: 字典，指定每个任务的样本数，例如 {"hellaswag_zeroshot": 100}
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

        # 确定该任务使用的样本数
        task_max_samples = max_per_task
        if task_samples and label in task_samples:
            task_max_samples = task_samples[label]
            print(f"  Using task-specific sample limit: {task_max_samples}")

        # 子采样（用于快速测试）
        if task_max_samples > 0:
            shuffle_rng = random.Random(1337)
            shuffle_rng.shuffle(data)
            data = data[:task_max_samples]
            print(f"  Using {len(data)} samples (limited from {len(data)} total)")
        else:
            print(f"  Total samples: {len(data)} (evaluating ALL samples)")

        # 评估
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta,
                                 max_seq_len=getattr(model, 'max_seq_len', None))
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


def _eval_worker(rank, world_size, ckpt_path, tokenizer_path, dtype,
                 eval_bundle_dir, max_per_task, task_samples,
                 all_tasks, random_baselines, result_dict):
    """多 GPU 评估的 worker 进程，根据 rank 自动计算分配的任务"""
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('medium')

    # 根据 rank 计算自己负责的任务 (round-robin)
    task_indices = [i for i in range(len(all_tasks)) if i % world_size == rank]

    print(f"[GPU {rank}] Loading model... ({len(task_indices)} tasks assigned)")
    sys.stdout.flush()
    model, tokenizer, hparams = load_ebt_model(ckpt_path, tokenizer_path, device, dtype=dtype)

    data_base_path = os.path.join(eval_bundle_dir, "eval_data")

    for task_idx in task_indices:
        task = all_tasks[task_idx]
        label = task['label']
        start_time = time.time()

        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }

        print(f"[GPU {rank}] [{task_idx+1}/{len(all_tasks)}] Evaluating: {label}")
        sys.stdout.flush()

        # 加载数据
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # 确定样本数
        task_max_samples = max_per_task
        if task_samples and label in task_samples:
            task_max_samples = task_samples[label]

        if task_max_samples > 0:
            shuffle_rng = random.Random(1337)
            shuffle_rng.shuffle(data)
            data = data[:task_max_samples]

        # 评估
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta,
                                 max_seq_len=getattr(model, 'max_seq_len', None))

        # centered result
        random_baseline = random_baselines.get(label, 50.0)
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)

        elapsed = time.time() - start_time
        print(f"[GPU {rank}]   ✓ {label}: acc={accuracy:.4f} centered={centered_result:.4f} ({elapsed:.2f}s)")
        sys.stdout.flush()

        result_dict[label] = (accuracy, centered_result)


def evaluate_core_multigpu(num_gpus, ckpt_path, tokenizer_path, dtype,
                           eval_bundle_dir, max_per_task, task_samples):
    """多 GPU 并行评估：按任务 round-robin 分片到各 GPU"""
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

    print(f"\n{'='*80}")
    print(f"Running CORE Evaluation (multi-GPU: {num_gpus} GPUs, {len(all_tasks)} tasks)")
    print(f"{'='*80}")
    for rank in range(num_gpus):
        task_names = [all_tasks[i]['label'] for i in range(len(all_tasks)) if i % num_gpus == rank]
        print(f"  GPU {rank}: {len(task_names)} tasks — {', '.join(task_names)}")
    print()

    # 共享结果字典
    manager = mp.Manager()
    result_dict = manager.dict()

    # 使用 mp.spawn — worker 内部根据 rank 自动计算任务分配
    mp.spawn(
        _eval_worker,
        args=(num_gpus, ckpt_path, tokenizer_path, dtype,
              eval_bundle_dir, max_per_task, task_samples,
              all_tasks, random_baselines, result_dict),
        nprocs=num_gpus,
        join=True,
    )

    # 汇总结果
    results = {}
    centered_results = {}
    for label, (acc, centered) in result_dict.items():
        results[label] = acc
        centered_results[label] = centered

    core_metric = sum(centered_results.values()) / len(centered_results) if centered_results else 0.0

    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }


def main():
    parser = argparse.ArgumentParser(
        description="EBT CORE Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 评估所有样本（完整 CORE score）:
   python -m scripts.ebt_core_eval --ckpt-path model.ckpt --tokenizer-path tokenizer/ \\
       --eval-bundle-dir eval_bundle/ --output-dir output/ --max-per-task -1

2. 快速测试（每个任务 100 个样本）:
   python -m scripts.ebt_core_eval --ckpt-path model.ckpt --tokenizer-path tokenizer/ \\
       --eval-bundle-dir eval_bundle/ --output-dir output/ --max-per-task 100

3. 为特定任务设置不同样本数:
   python -m scripts.ebt_core_eval --ckpt-path model.ckpt --tokenizer-path tokenizer/ \\
       --eval-bundle-dir eval_bundle/ --output-dir output/ --max-per-task 100 \\
       --task-samples "hellaswag_zeroshot:500,arc_easy:200"

4. 使用多个 GPU:
   python -m scripts.ebt_core_eval --ckpt-path model.ckpt --tokenizer-path tokenizer/ \\
       --eval-bundle-dir eval_bundle/ --output-dir output/ --gpus 4
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
    args = parser.parse_args()

    # 解析任务特定样本数
    task_samples_dict = {}
    if args.task_samples:
        for item in args.task_samples.split(','):
            if ':' in item:
                task_name, num_samples = item.split(':', 1)
                task_samples_dict[task_name.strip()] = int(num_samples.strip())

    # 设置设备和 GPU
    if args.gpus == -1:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"🔍 Auto-detected {num_gpus} GPU(s)")
        else:
            num_gpus = 0
            print("⚠️  No GPU detected, using CPU")
    else:
        num_gpus = args.gpus
        print(f"📊 Using {num_gpus} GPU(s) as specified")

    dtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16

    # 性能优化: 启用 TF32 (H200/A100 Tensor Core 加速)
    torch.set_float32_matmul_precision('medium')

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 多 GPU 路径
    if num_gpus > 1 and 'core' in args.eval_modes:
        print(f"\n🚀 Using multi-GPU evaluation with {num_gpus} GPUs")
        core_results = evaluate_core_multigpu(
            num_gpus=num_gpus,
            ckpt_path=args.ckpt_path,
            tokenizer_path=args.tokenizer_path,
            dtype=dtype,
            eval_bundle_dir=args.eval_bundle_dir,
            max_per_task=args.max_per_task,
            task_samples=task_samples_dict if task_samples_dict else None,
        )
    else:
        # 单 GPU / CPU 路径
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
                task_samples=task_samples_dict if task_samples_dict else None
            )

    # 打印和保存结果
    if core_results:
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

        # 机器可解析汇总行
        num_tasks = len(core_results['results'])
        avg_acc = np.mean(list(core_results['results'].values()))
        print(f"\n[EVAL_SUMMARY] dataset=core core_metric={core_results['core_metric']:.4f} avg_accuracy={avg_acc:.4f} num_tasks={num_tasks}")

    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
