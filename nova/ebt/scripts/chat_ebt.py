#!/usr/bin/env python3
"""
EBT 交互式对话脚本 - 与训练好的 EBT 模型进行交互式对话

该脚本加载 EBT checkpoint，支持交互式对话，并可选择性地展示 MCMC 步骤的迭代过程，
便于用户直观评估 EBT 模型的效果和各项指标。

使用方法:
    python -m scripts.chat_ebt                           # 默认参数
    python -m scripts.chat_ebt --show-mcmc               # 展示 MCMC 步骤
    python -m scripts.chat_ebt --show-mcmc --verbose     # 详细展示每步信息
    python -m scripts.chat_ebt -c /path/to/checkpoint    # 指定 checkpoint

参数:
    -c, --checkpoint      Checkpoint 路径
    --tokenizer           Tokenizer 路径
    -t, --temperature     生成温度 (默认: 0.8)
    --top-p               Top-P 采样 (默认: 0.9)
    -m, --max-tokens      最大生成 tokens (默认: 256)
    --show-mcmc           展示 MCMC 步骤过程
    --verbose             详细模式，展示更多指标
    --show-energy         展示能量值变化
    --show-distribution   展示概率分布变化
"""

import argparse
import sys
import os
import time
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

# 清除分布式训练环境变量
for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']:
    if var in os.environ:
        del os.environ[var]

# 设置离线模式
os.environ['NANOCHAT_OFFLINE_MODE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['NANOCHAT_BASE_DIR'] = "/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat"

# 颜色代码
class Colors:
    BLUE = '\033[1;34m'
    GREEN = '\033[1;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[1;31m'
    CYAN = '\033[1;36m'
    MAGENTA = '\033[1;35m'
    GRAY = '\033[0;90m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    # MCMC 步骤颜色渐变
    MCMC_COLORS = [
        '\033[38;5;196m',  # 红
        '\033[38;5;208m',  # 橙
        '\033[38;5;226m',  # 黄
        '\033[38;5;118m',  # 绿
        '\033[38;5;51m',   # 青
    ]


def print_colored(text: str, color: str):
    """打印带颜色的文本"""
    print(f"{color}{text}{Colors.RESET}")


def print_banner():
    """打印启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███████╗██████╗ ████████╗     ██████╗██╗  ██╗ █████╗ ████████╗             ║
║   ██╔════╝██╔══██╗╚══██╔══╝    ██╔════╝██║  ██║██╔══██╗╚══██╔══╝             ║
║   █████╗  ██████╔╝   ██║       ██║     ███████║███████║   ██║                ║
║   ██╔══╝  ██╔══██╗   ██║       ██║     ██╔══██║██╔══██║   ██║                ║
║   ███████╗██████╔╝   ██║       ╚██████╗██║  ██║██║  ██║   ██║                ║
║   ╚══════╝╚═════╝    ╚═╝        ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝                ║
║                                                                              ║
║           Energy-Based Transformer Interactive Chat Terminal                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print_colored(banner, Colors.CYAN)


def format_energy(energy: float, width: int = 10) -> str:
    """格式化能量值显示"""
    if energy > 0:
        return f"{Colors.RED}+{energy:>{width-1}.6f}{Colors.RESET}"
    else:
        return f"{Colors.GREEN}{energy:>{width}.6f}{Colors.RESET}"


def format_energy_bar(energy: float, min_energy: float, max_energy: float, width: int = 20) -> str:
    """生成能量条形图"""
    if max_energy == min_energy:
        ratio = 0.5
    else:
        ratio = (energy - min_energy) / (max_energy - min_energy)
    ratio = max(0, min(1, ratio))

    filled = int(ratio * width)
    empty = width - filled

    # 使用渐变色
    if ratio < 0.3:
        color = Colors.GREEN
    elif ratio < 0.7:
        color = Colors.YELLOW
    else:
        color = Colors.RED

    bar = f"{color}{'█' * filled}{Colors.GRAY}{'░' * empty}{Colors.RESET}"
    return bar


def format_top_tokens(probs: torch.Tensor, tokenizer, top_k: int = 5) -> str:
    """格式化显示 top-k tokens"""
    top_probs, top_indices = torch.topk(probs, min(top_k, probs.shape[-1]))
    tokens_str = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        token_text = tokenizer.decode([idx])
        token_text = repr(token_text)[1:-1]  # 显示特殊字符
        if len(token_text) > 10:
            token_text = token_text[:10] + "..."
        tokens_str.append(f"'{token_text}':{prob:.3f}")
    return " | ".join(tokens_str)


class EBTChatEngine:
    """EBT 对话引擎"""

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        show_mcmc: bool = False,
        verbose: bool = False,
        show_energy: bool = False,
        show_distribution: bool = False,
        override_mcmc_steps: Optional[int] = None,
        override_noise_std: Optional[float] = None,
        override_alpha: Optional[float] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.show_mcmc = show_mcmc
        self.verbose = verbose
        self.show_energy = show_energy
        self.show_distribution = show_distribution
        self.override_mcmc_steps = override_mcmc_steps
        self.override_noise_std = override_noise_std
        self.override_alpha = override_alpha

        self.model = None
        self.tokenizer = None
        self.hparams = None

        self._load_model(checkpoint_path, tokenizer_path)

    def _load_model(self, checkpoint_path: str, tokenizer_path: str):
        """加载模型和 tokenizer"""
        print_colored("正在加载模型...", Colors.YELLOW)

        # 加载 tokenizer
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from nanochat.tokenizer import get_tokenizer
        self.tokenizer = get_tokenizer()
        print(f"  Tokenizer vocab size: {self.tokenizer.get_vocab_size()}")

        # 加载 checkpoint
        print(f"  Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # 提取 hparams
        if 'hyper_parameters' in checkpoint:
            self.hparams = checkpoint['hyper_parameters']
        elif 'hparams' in checkpoint:
            self.hparams = checkpoint['hparams']
        else:
            raise ValueError("Cannot find hyperparameters in checkpoint")

        # 转换为 namespace 风格访问
        class HParamsNamespace:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)

        if isinstance(self.hparams, dict):
            self.hparams = HParamsNamespace(self.hparams)

        # 设置 tokenizer_obj
        self.hparams.tokenizer_obj = self.tokenizer

        # 创建模型
        from modeling_ebt import EBT_NLP
        self.model = EBT_NLP(self.hparams)

        # 加载权重
        state_dict = checkpoint['state_dict']
        # 移除 'model.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict, strict=False)
        self.model = self.model.to(self.device).to(self.dtype)
        self.model.eval()

        print_colored("✓ 模型加载完成", Colors.GREEN)
        self._print_model_info()

    def _print_model_info(self):
        """打印模型信息"""
        # 使用 getattr 配合备用名称，防止因 Checkpoint 参数命名不一致导致崩溃
        embed_dim = getattr(self.hparams, 'embedding_dim', getattr(self.hparams, 'dim', '未知'))
        n_layers = getattr(self.hparams, 'num_layers', getattr(self.hparams, 'n_layers', '未知'))
        n_heads = getattr(self.hparams, 'num_heads', getattr(self.hparams, 'n_heads', '未知'))
        mcmc_steps = getattr(self.hparams, 'mcmc_num_steps', '未知')
        ctx_len = getattr(self.hparams, 'context_length', getattr(self.hparams, 'max_seq_len', '未知'))
        

        # 安全获取模型内部的张量参数
        alpha_val = '未知'
        if hasattr(self.model, 'alpha'):
            alpha_val = self.model.alpha.item() if isinstance(self.model.alpha, torch.Tensor) else self.model.alpha

        noise_std_val = '未知'
        if hasattr(self.model, 'langevin_dynamics_noise_std'):
            noise_std_val = self.model.langevin_dynamics_noise_std.item() if isinstance(self.model.langevin_dynamics_noise_std, torch.Tensor) else self.model.langevin_dynamics_noise_std

        # 使用训练时的参数,而不是硬编码覆盖
        # NOTE: 保持推理与训练一致性至关重要!

        # 应用用户指定的覆盖参数
        if self.override_mcmc_steps is not None:
            original_steps = getattr(self.hparams, 'mcmc_num_steps', 2)
            original_steps_model = getattr(self.model.hparams, 'mcmc_num_steps', original_steps)

            if self.override_mcmc_steps > original_steps:
                # 模型使用 time_embed 类型时，mcmc_num_steps 是能量景观的数量，不是迭代次数。
                # 训练时只学习了 original_steps 个能量景观（time embeddings），
                # 不能简单增加景观数。正确做法是保持景观数不变，
                # 通过 randomize_mcmc_num_steps 在最终景观上多做迭代。
                extra = self.override_mcmc_steps - original_steps
                self.hparams.randomize_mcmc_num_steps = extra
                self.model.hparams.randomize_mcmc_num_steps = extra
                self.hparams.randomize_mcmc_num_steps_final_landscape = True
                self.model.hparams.randomize_mcmc_num_steps_final_landscape = True
                self.hparams.randomize_mcmc_num_steps_min = extra + 1
                self.model.hparams.randomize_mcmc_num_steps_min = extra + 1
                mcmc_steps = self.override_mcmc_steps

                # 当扩展步数超过训练景观数时，自动启用 Langevin 噪声，
                # 否则重复的最终景观迭代因梯度为零而完全无效。
                # 仅当用户未显式指定 --override-noise-std 时才自动启用。
                trained_noise = getattr(self.hparams, 'langevin_dynamics_noise', 0.0)
                if self.override_noise_std is None and trained_noise == 0:
                    # 推理使用 float32 autocast，σ=0.01 的噪声不会被量化截断。
                    # auto_noise = 0.01
                    auto_noise = 0.0
                    self.hparams.langevin_dynamics_noise = auto_noise
                    self.model.hparams.langevin_dynamics_noise = auto_noise
                    if hasattr(self.model, 'langevin_dynamics_noise_std'):
                        self.model.langevin_dynamics_noise_std.data.fill_(auto_noise)
                    noise_std_val = auto_noise

                print()
                print(f"    {Colors.YELLOW}{'='*60}")
                print(f"    ⚠ MCMC 步数覆盖: {original_steps} → {self.override_mcmc_steps}")
                print(f"    {'='*60}{Colors.RESET}")
                print(f"    {Colors.CYAN}模型训练时使用 {original_steps} 个能量景观 (time embeddings)。")
                print(f"    无法创建新的景观，改为在最终景观上重复迭代 {extra + 1} 次。")
                print(f"    步骤序列: {list(range(original_steps - 1))} + [{original_steps - 1}] × {extra + 1}{Colors.RESET}")
                if self.override_noise_std is None and trained_noise == 0:
                    print(f"    {Colors.GREEN}✓ 已自动启用 Langevin 噪声 (σ={auto_noise})，")
                    print(f"      使重复迭代能通过随机扰动继续优化能量。")
                    print(f"      可通过 --override-noise-std 手动调整。{Colors.RESET}")
                print(f"    {Colors.GRAY}注意: 推理使用 float32 精度以保证 MCMC 梯度和噪声的有效性。")
                print(f"    超出训练景观数的额外步骤效果有限，")
                print(f"    建议训练时增加 mcmc_num_steps 以获得更好效果。{Colors.RESET}")
                print()
            elif self.override_mcmc_steps < original_steps:
                # 减少步数：直接修改 mcmc_num_steps
                self.hparams.mcmc_num_steps = mcmc_steps = self.override_mcmc_steps
                self.model.hparams.mcmc_num_steps = self.override_mcmc_steps
                print(f"    {Colors.YELLOW}⚠ 覆盖 MCMC 步数: {original_steps} → {self.override_mcmc_steps}{Colors.RESET}")
            else:
                mcmc_steps = original_steps
                print(f"    {Colors.GRAY}MCMC 步数已是 {original_steps}，无需覆盖{Colors.RESET}")

        if self.override_noise_std is not None:
            if hasattr(self.model, 'langevin_dynamics_noise_std'):
                self.model.langevin_dynamics_noise_std.data.fill_(self.override_noise_std)
                # 同时更新 hparam，否则 modeling_ebt.py:124 的条件检查不通过
                self.hparams.langevin_dynamics_noise = self.override_noise_std
                self.model.hparams.langevin_dynamics_noise = self.override_noise_std
                noise_std_val = self.override_noise_std
                print(f"    {Colors.YELLOW}⚠ 覆盖 Langevin 噪声: {self.override_noise_std:.6f}{Colors.RESET}")

        if self.override_alpha is not None:
            if hasattr(self.model, 'alpha'):
                self.model.alpha = torch.tensor(
                    self.override_alpha,
                    dtype=self.model.alpha.dtype,
                    device=self.model.alpha.device
                )
                alpha_val = self.override_alpha
                print(f"    {Colors.YELLOW}⚠ 覆盖 Alpha 步长: {self.override_alpha:.6f}{Colors.RESET}")

        print()
        print(f"  {Colors.BOLD}模型配置:{Colors.RESET}")
        print(f"    - 嵌入维度: {embed_dim}")
        print(f"    - 层数: {n_layers}")
        print(f"    - 注意力头数: {n_heads}")
        print(f"    - MCMC 步数: {mcmc_steps}")
        
        if isinstance(alpha_val, float):
            print(f"    - MCMC 步长 (alpha): {alpha_val:.6f}")
        else:
            print(f"    - MCMC 步长 (alpha): {alpha_val}")
            
        if isinstance(noise_std_val, float):
            print(f"    - Langevin 噪声: {noise_std_val:.6f}")
        else:
            print(f"    - Langevin 噪声: {noise_std_val}")
            
        print(f"    - 上下文长度: {ctx_len}")
        print()

    def _run_mcmc_with_tracking(
        self,
        input_ids: torch.Tensor,
        return_all_steps: bool = True
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """运行 MCMC 推理并追踪每一步的状态

        直接调用 model.forward() 以保证与训练/评估时完全一致的 MCMC 逻辑。
        """

        mcmc_history = []
        start_time = time.time()

        # 使用 float32 autocast：模型权重是 bfloat16，但 autocast(float32) 会将
        # 中间计算提升到 float32，避免 bfloat16 精度不足导致：
        # 1. Langevin 噪声 (σ~0.1) 被量化截断
        # 2. 能量值变化 (~0.01) 无法区分
        # 3. 梯度更新被吞掉
        with torch.amp.autocast('cuda', dtype=torch.float32):
            predicted_distributions, predicted_energies = self.model.forward(
                input_ids, start_pos=0, learning=False, return_raw_logits=True, no_randomness=True
            )

        total_time = time.time() - start_time

        # 从 forward 返回的结果构建 mcmc_history 用于显示
        num_steps = len(predicted_distributions)
        step_time = total_time / max(num_steps, 1)

        for step_idx in range(num_steps):
            logits = predicted_distributions[step_idx]  # (B, S, V)
            energy = predicted_energies[step_idx].float()  # (B*S, 1) → float32 避免 bfloat16 精度不足

            probs = F.softmax(logits.detach().float(), dim=-1)
            step_info = {
                'step': step_idx,
                'energy_mean': energy.mean().item(),
                'energy_std': energy.std().item(),
                'energy_min': energy.min().item(),
                'energy_max': energy.max().item(),
                'grad_norm': 0.0,  # 不再单独追踪梯度
                'prob_entropy': -(probs * (probs + 1e-10).log()).sum(-1).mean().item(),
                'prob_max': probs.max(-1).values.mean().item(),
                'step_time': step_time,
            }

            if return_all_steps:
                step_info['logits'] = logits.detach().clone()
                step_info['energies'] = energy.detach().clone()

            mcmc_history.append(step_info)

        # 返回最终步骤的 logits
        final_logits = predicted_distributions[-1]  # (B, S, V)
        return final_logits, mcmc_history

    def _display_mcmc_progress(self, history: List[Dict[str, Any]], position: int = 0):
        """展示 MCMC 迭代过程"""
        if not self.show_mcmc:
            return

        num_steps = len(history)
        trained_landscapes = getattr(self.hparams, 'mcmc_num_steps', num_steps)

        # 重建 mcmc_steps 列表以了解每步对应的景观索引
        # 与 modeling_ebt.py forward() 中的逻辑对应
        landscape_indices = []
        extra_steps = getattr(self.model.hparams, 'randomize_mcmc_num_steps', 0)
        if extra_steps > 0:
            for s in range(trained_landscapes):
                if s == trained_landscapes - 1:
                    landscape_indices.extend([s] * (extra_steps + 1))
                else:
                    landscape_indices.append(s)
        else:
            landscape_indices = list(range(trained_landscapes))
        # 截断/填充到实际步数
        while len(landscape_indices) < num_steps:
            landscape_indices.append(landscape_indices[-1] if landscape_indices else 0)

        print()
        print(f"  {Colors.BOLD}═══ MCMC 迭代过程 (位置 {position}, {num_steps} 步, "
              f"{trained_landscapes} 个能量景观) ═══{Colors.RESET}")

        # 收集能量范围
        all_energies = [h['energy_mean'] for h in history]
        min_e, max_e = min(all_energies), max(all_energies)

        for i, step_info in enumerate(history):
            color_idx = i % len(Colors.MCMC_COLORS)
            step_color = Colors.MCMC_COLORS[color_idx]

            energy_bar = format_energy_bar(step_info['energy_mean'], min_e, max_e)
            energy_str = format_energy(step_info['energy_mean'])

            # 景观标识
            li = landscape_indices[i] if i < len(landscape_indices) else '?'
            landscape_tag = f"L{li}"

            # Per-step energy delta
            if i > 0:
                energy_delta = step_info['energy_mean'] - history[i-1]['energy_mean']
                delta_color = Colors.GREEN if energy_delta < 0 else (Colors.GRAY if abs(energy_delta) < 1e-7 else Colors.RED)
                delta_str = f"{delta_color}ΔE={energy_delta:+.6f}{Colors.RESET}"
            else:
                delta_str = f"{Colors.GRAY}ΔE=  ------{Colors.RESET}"

            # 添加熵和最大概率显示以诊断问题
            entropy = step_info['prob_entropy']
            max_prob = step_info['prob_max']

            # 熵值颜色: 使用 cyan (信息性指标，不直接表示好/坏)
            entropy_color = Colors.CYAN

            # 最大概率颜色: 高概率(好) = 绿色, 低概率(差) = 红色
            if max_prob > 0.5:
                prob_color = Colors.GREEN
            elif max_prob > 0.1:
                prob_color = Colors.YELLOW
            else:
                prob_color = Colors.RED

            print(f"  {step_color}Step {i:2d}{Colors.RESET} {Colors.GRAY}[{landscape_tag}]{Colors.RESET} │ "
                  f"E={energy_str} │ {energy_bar} │ "
                  f"{delta_str} │ "
                  f"{entropy_color}H={entropy:.2f}{Colors.RESET} │ "
                  f"{prob_color}P_max={max_prob:.3f}{Colors.RESET} │ "
                  f"t={step_info['step_time']*1000:.1f}ms")

            if self.verbose and 'logits' in step_info:
                probs = F.softmax(step_info['logits'][0, position], dim=-1)
                top_tokens = format_top_tokens(probs, self.tokenizer, top_k=5)
                print(f"         └─ Top tokens: {top_tokens}")

        # 显示能量和熵的变化
        if len(history) > 1:
            energy_change = history[-1]['energy_mean'] - history[0]['energy_mean']
            entropy_change = history[-1]['prob_entropy'] - history[0]['prob_entropy']
            prob_change = history[-1]['prob_max'] - history[0]['prob_max']

            change_color = Colors.GREEN if energy_change < 0 else Colors.RED
            prob_change_color = Colors.GREEN if prob_change > 0 else Colors.RED

            print(f"  {Colors.BOLD}收敛统计:{Colors.RESET}")
            print(f"    能量变化 (↓好): {change_color}{energy_change:+.6f}{Colors.RESET} "
                  f"({history[0]['energy_mean']:.6f} → {history[-1]['energy_mean']:.6f})")
            print(f"    熵值变化 (信息): {Colors.CYAN}{entropy_change:+.2f}{Colors.RESET} "
                  f"({history[0]['prob_entropy']:.2f} → {history[-1]['prob_entropy']:.2f})")
            print(f"    最大概率 (↑好): {prob_change_color}{prob_change:+.3f}{Colors.RESET} "
                  f"({history[0]['prob_max']:.3f} → {history[-1]['prob_max']:.3f})")

            # 当使用扩展步数时，显示额外诊断信息
            extra_steps = getattr(self.model.hparams, 'randomize_mcmc_num_steps', 0)
            if extra_steps > 0 and num_steps > trained_landscapes:
                # 计算前 N 个景观阶段和扩展阶段的能量变化
                landscape_phase_end = trained_landscapes - 1  # 最后一个景观首次出现的位置
                if landscape_phase_end < num_steps:
                    landscape_energy_change = history[landscape_phase_end]['energy_mean'] - history[0]['energy_mean']
                    repeat_energy_change = history[-1]['energy_mean'] - history[landscape_phase_end]['energy_mean']
                    print(f"    {Colors.GRAY}--- 分阶段分析 ---{Colors.RESET}")
                    print(f"    景观切换阶段 (Step 0→{landscape_phase_end}): ΔE={landscape_energy_change:+.6f}")
                    print(f"    最终景观重复 (Step {landscape_phase_end}→{num_steps-1}): ΔE={repeat_energy_change:+.6f}")
                    noise_val = getattr(self.model.hparams, 'langevin_dynamics_noise', 0)
                    if noise_val > 0:
                        print(f"    Langevin 噪声: σ={noise_val} (启用随机扰动)")
                    else:
                        print(f"    {Colors.YELLOW}Langevin 噪声: 0 (无随机扰动，重复步骤可能无效){Colors.RESET}")
        print()

    def generate_token(
        self,
        input_ids: torch.Tensor,
        temperature: float = 0.8,
        top_p: float = 0.9,
        show_position: bool = True
    ) -> Tuple[int, Dict[str, Any]]:
        """生成下一个 token"""

        # 运行 MCMC
        final_logits, mcmc_history = self._run_mcmc_with_tracking(input_ids, return_all_steps=self.show_mcmc)

        # 获取最后一个位置的 logits
        last_logits = final_logits[0, -1]  # (V,)

        # 应用温度
        if temperature > 0:
            probs = F.softmax(last_logits / temperature, dim=-1)
        else:
            probs = F.softmax(last_logits, dim=-1)

        # Top-p 采样
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum()
            next_token_idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices[next_token_idx].item()
        else:
            next_token = torch.multinomial(probs, num_samples=1).item()

        # 展示 MCMC 过程
        if show_position and self.show_mcmc:
            self._display_mcmc_progress(mcmc_history, position=input_ids.shape[1]-1)

        # 返回生成信息
        gen_info = {
            'token': next_token,
            'token_prob': probs[next_token].item(),
            'final_energy': mcmc_history[-1]['energy_mean'] if mcmc_history else 0,
            'energy_change': (mcmc_history[-1]['energy_mean'] - mcmc_history[0]['energy_mean']) if len(mcmc_history) > 1 else 0,
            'mcmc_steps': len(mcmc_history),
        }

        return next_token, gen_info

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        stop_tokens: Optional[List[int]] = None,
        stream: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """生成文本"""

        # Encode prompt
        prompt_tokens = self.tokenizer.encode(prompt)
        if len(prompt_tokens) == 0:
            prompt_tokens = [self.tokenizer.get_bos_token_id()]

        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)

        # 生成统计
        stats = {
            'tokens_generated': 0,
            'total_time': 0,
            'avg_energy_change': [],
            'token_probs': [],
        }

        generated_tokens = []
        start_time = time.time()

        # 获取 EOS token
        try:
            eos_token_id = self.tokenizer.encode_special("<|eos|>")
        except:
            eos_token_id = None

        if stop_tokens is None:
            stop_tokens = [eos_token_id] if eos_token_id else []

        for i in range(max_tokens):
            # 检查上下文长度
            if input_ids.shape[1] >= self.hparams.context_length - 1:
                print_colored(f"\n[达到最大上下文长度 {self.hparams.context_length}]", Colors.YELLOW)
                break

            # 生成下一个 token
            # 第一个 token 展示完整 MCMC 过程; verbose 模式下每 10 个 token 也展示
            show_pos = self.show_mcmc and (i == 0 or (self.verbose and i % 10 == 0))
            next_token, gen_info = self.generate_token(
                input_ids,
                temperature=temperature,
                top_p=top_p,
                show_position=show_pos
            )

            # 检查停止条件
            if next_token in stop_tokens:
                break

            generated_tokens.append(next_token)
            stats['token_probs'].append(gen_info['token_prob'])
            stats['avg_energy_change'].append(gen_info['energy_change'])

            # 更新输入
            input_ids = torch.cat([
                input_ids,
                torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            ], dim=1)

            # 流式输出
            if stream:
                token_text = self.tokenizer.decode([next_token])
                print(token_text, end='', flush=True)

        stats['tokens_generated'] = len(generated_tokens)
        stats['total_time'] = time.time() - start_time
        stats['avg_energy_change'] = np.mean(stats['avg_energy_change']) if stats['avg_energy_change'] else 0
        stats['avg_token_prob'] = np.mean(stats['token_probs']) if stats['token_probs'] else 0
        stats['tokens_per_second'] = stats['tokens_generated'] / stats['total_time'] if stats['total_time'] > 0 else 0

        generated_text = self.tokenizer.decode(generated_tokens)

        return generated_text, stats


def print_help():
    """显示帮助信息"""
    print()
    print(f"{Colors.BOLD}可用命令:{Colors.RESET}")
    print("  /quit, /exit        - 退出对话")
    print("  /clear              - 清空对话历史")
    print("  /temp <值>          - 设置温度 (0.0-2.0)")
    print("  /topp <值>          - 设置 top-p (0.0-1.0)")
    print("  /tokens <值>        - 设置最大 tokens (1-4096)")
    print("  /mcmc               - 切换 MCMC 显示")
    print("  /verbose            - 切换详细模式")
    print("  /energy             - 切换能量显示")
    print("  /status             - 显示当前设置")
    print("  /info               - 显示模型信息")
    print("  /help               - 显示此帮助")
    print()


def print_status(engine: EBTChatEngine, temperature: float, top_p: float, max_tokens: int):
    """显示当前状态"""
    print()
    print(f"{Colors.BOLD}当前设置:{Colors.RESET}")
    print(f"  温度: {temperature}")
    print(f"  Top-P: {top_p}")
    print(f"  最大 Tokens: {max_tokens}")
    print(f"  显示 MCMC: {'是' if engine.show_mcmc else '否'}")
    print(f"  详细模式: {'是' if engine.verbose else '否'}")
    print(f"  显示能量: {'是' if engine.show_energy else '否'}")
    print()


def print_generation_stats(stats: Dict[str, Any]):
    """打印生成统计信息"""
    print()
    print(f"{Colors.GRAY}────────────────────────────────────────{Colors.RESET}")
    print(f"{Colors.BOLD}生成统计:{Colors.RESET}")
    print(f"  生成 tokens: {stats['tokens_generated']}")
    print(f"  总时间: {stats['total_time']:.2f}s")
    print(f"  速度: {stats['tokens_per_second']:.2f} tokens/s")
    print(f"  平均 token 概率: {stats['avg_token_prob']:.4f}")
    print(f"  平均能量变化: {stats['avg_energy_change']:.4f}")
    print(f"{Colors.GRAY}────────────────────────────────────────{Colors.RESET}")
    print()


def main():
    parser = argparse.ArgumentParser(description='EBT 交互式对话终端')
    parser.add_argument('-c', '--checkpoint', type=str,
                       default="/mnt/shared-storage-user/puyuan/code/nova/logs/checkpoints/ebt-d26-stable_20260313_123203_2026-03-13_12-32-54_/last.ckpt",
                       help='Checkpoint 路径')
    parser.add_argument('--tokenizer', type=str,
                       default="/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/tokenizer",
                       help='Tokenizer 路径')
    parser.add_argument('-t', '--temperature', type=float, default=0.8,
                       help='生成温度 (默认: 0.8)')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-P 采样 (默认: 0.9)')
    parser.add_argument('-m', '--max-tokens', type=int, default=256,
                       help='最大生成 tokens (默认: 256)')
    parser.add_argument('--show-mcmc', action='store_true',
                       help='展示 MCMC 步骤过程')
    parser.add_argument('--verbose', action='store_true',
                       help='详细模式')
    parser.add_argument('--show-energy', action='store_true',
                       help='展示能量值变化')
    parser.add_argument('--show-distribution', action='store_true',
                       help='展示概率分布变化')
    parser.add_argument('-d', '--dtype', type=str, default='bfloat16',
                       choices=['float32', 'bfloat16'],
                       help='数据类型 (默认: bfloat16)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (默认: cuda)')

    # 高级推理参数 (覆盖训练配置)
    parser.add_argument('--override-mcmc-steps', type=int, default=None,
                       help='覆盖训练时的 MCMC 步数 (默认: 使用训练值)')
    parser.add_argument('--override-noise-std', type=float, default=None,
                       help='覆盖 Langevin 噪声标准差 (默认: 使用训练值)')
    parser.add_argument('--override-alpha', type=float, default=None,
                       help='覆盖 MCMC 步长 alpha (默认: 使用训练值)')

    args = parser.parse_args()

    # 打印横幅
    print_banner()

    # 验证 checkpoint
    if not os.path.exists(args.checkpoint):
        print_colored(f"错误: 找不到 checkpoint: {args.checkpoint}", Colors.RED)
        return 1

    # 初始化引擎
    dtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16

    try:
        engine = EBTChatEngine(
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
            device=args.device,
            dtype=dtype,
            show_mcmc=args.show_mcmc,
            verbose=args.verbose,
            show_energy=args.show_energy,
            show_distribution=args.show_distribution,
            override_mcmc_steps=args.override_mcmc_steps,
            override_noise_std=args.override_noise_std,
            override_alpha=args.override_alpha,
        )
    except Exception as e:
        print_colored(f"错误: 加载模型失败 - {str(e)}", Colors.RED)
        import traceback
        traceback.print_exc()
        return 1

    # 显示使用说明
    print("=" * 70)
    print_colored("EBT 对话终端已就绪!", Colors.GREEN)
    print("=" * 70)
    print("使用说明:")
    print("  - 输入文本，按 Enter 开始生成")
    print("  - 输入 /quit 或 /exit 退出")
    print("  - 输入 /help 查看所有命令")
    print("  - 输入 /mcmc 切换 MCMC 步骤显示")
    print("=" * 70)
    print()

    # 参数
    temperature = args.temperature
    top_p = args.top_p
    max_tokens = args.max_tokens

    # 对话循环
    while True:
        try:
            user_input = input(f'{Colors.BLUE}你:{Colors.RESET} ')

            # 处理命令
            cmd = user_input.strip().lower()

            if cmd in ['/quit', '/exit']:
                print_colored('\n再见!', Colors.GREEN)
                break

            if cmd == '/clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                print_banner()
                continue

            if cmd == '/help':
                print_help()
                continue

            if cmd == '/status':
                print_status(engine, temperature, top_p, max_tokens)
                continue

            if cmd == '/info':
                engine._print_model_info()
                continue

            if cmd == '/mcmc':
                engine.show_mcmc = not engine.show_mcmc
                status = "开启" if engine.show_mcmc else "关闭"
                print_colored(f'\n✓ MCMC 显示已{status}\n', Colors.GREEN)
                continue

            if cmd == '/verbose':
                engine.verbose = not engine.verbose
                status = "开启" if engine.verbose else "关闭"
                print_colored(f'\n✓ 详细模式已{status}\n', Colors.GREEN)
                continue

            if cmd == '/energy':
                engine.show_energy = not engine.show_energy
                status = "开启" if engine.show_energy else "关闭"
                print_colored(f'\n✓ 能量显示已{status}\n', Colors.GREEN)
                continue

            if cmd.startswith('/temp '):
                try:
                    new_temp = float(cmd.split()[1])
                    if 0.0 <= new_temp <= 2.0:
                        temperature = new_temp
                        print_colored(f'\n✓ 温度已设置为: {temperature}\n', Colors.GREEN)
                    else:
                        print_colored('\n✗ 温度必须在 0.0-2.0 之间\n', Colors.RED)
                except:
                    print_colored('\n✗ 无效的温度值\n', Colors.RED)
                continue

            if cmd.startswith('/topp '):
                try:
                    new_topp = float(cmd.split()[1])
                    if 0.0 <= new_topp <= 1.0:
                        top_p = new_topp
                        print_colored(f'\n✓ Top-P 已设置为: {top_p}\n', Colors.GREEN)
                    else:
                        print_colored('\n✗ Top-P 必须在 0.0-1.0 之间\n', Colors.RED)
                except:
                    print_colored('\n✗ 无效的 Top-P 值\n', Colors.RED)
                continue

            if cmd.startswith('/tokens '):
                try:
                    new_tokens = int(cmd.split()[1])
                    if 1 <= new_tokens <= 4096:
                        max_tokens = new_tokens
                        print_colored(f'\n✓ 最大 Tokens 已设置为: {max_tokens}\n', Colors.GREEN)
                    else:
                        print_colored('\n✗ 最大 Tokens 必须在 1-4096 之间\n', Colors.RED)
                except:
                    print_colored('\n✗ 无效的 Tokens 值\n', Colors.RED)
                continue

            # 空输入
            if not user_input.strip():
                continue

            # 生成回复
            print(f'{Colors.GREEN}EBT:{Colors.RESET} ', end='', flush=True)

            try:
                generated_text, stats = engine.generate(
                    prompt=user_input,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                )
                print()  # 换行

                # 显示统计
                if engine.verbose or engine.show_mcmc:
                    print_generation_stats(stats)

            except Exception as e:
                print()
                print_colored(f'✗ 生成错误: {str(e)}', Colors.RED)
                import traceback
                traceback.print_exc()

        except KeyboardInterrupt:
            print_colored('\n\n检测到 Ctrl+C，退出...', Colors.YELLOW)
            break
        except EOFError:
            print_colored('\n\n检测到 EOF，退出...', Colors.YELLOW)
            break

    print_colored('\n对话已结束。', Colors.GREEN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
