#!/usr/bin/env python3
"""Interactive terminal chat client for trained EBT checkpoints.

This script loads an EBT checkpoint and runs an interactive chat REPL.
It can optionally surface MCMC iteration information so the user can
inspect sampling behaviour and qualitative model effects.

Example usage::

    python -m scripts.chat_ebt
    python -m scripts.chat_ebt --show-mcmc
    python -m scripts.chat_ebt --show-mcmc --verbose
    python -m scripts.chat_ebt -c /path/to/checkpoint
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

# Reuse core decoding logic from generate.py to keep behaviour identical.
from openebm.elm.generate import call_model_forward_decode, _get_tokenizer, sample_top_p

# Clear any distributed training env vars so this single-process script does not
# inherit a stale rendezvous config from the caller.
for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']:
    if var in os.environ:
        del os.environ[var]

# Force offline mode so the tokenizer does not try to hit HuggingFace Hub.
os.environ['NANOCHAT_OFFLINE_MODE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['NANOCHAT_BASE_DIR'] = "/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat"


class Colors:
    """ANSI colour escape codes used for terminal output."""

    BLUE = '\033[1;34m'
    GREEN = '\033[1;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[1;31m'
    CYAN = '\033[1;36m'
    MAGENTA = '\033[1;35m'
    GRAY = '\033[0;90m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    # Gradient palette used when rendering the MCMC trajectory.
    MCMC_COLORS = [
        '\033[38;5;196m',
        '\033[38;5;208m',
        '\033[38;5;226m',
        '\033[38;5;118m',
        '\033[38;5;51m',
    ]


def print_colored(text: str, color: str) -> None:
    """Print ``text`` wrapped with the given ANSI ``color`` escape.

    :param text: text to print.
    :type text: str
    :param color: ANSI colour escape sequence from :class:`Colors`.
    :type color: str
    """
    print(f"{color}{text}{Colors.RESET}")


def print_banner() -> None:
    """Print the ASCII art startup banner."""
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


class EBTChatEngine:
    """Interactive chat engine wrapping a loaded EBT checkpoint."""

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
    ) -> None:
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

    def _load_model(self, checkpoint_path: str, tokenizer_path: str) -> None:
        """Load the EBT model and tokenizer from ``checkpoint_path``.

        :param checkpoint_path: path to a Lightning-style ``.ckpt`` file.
        :type checkpoint_path: str
        :param tokenizer_path: path to the tokenizer artefacts (kept for API
            symmetry; the actual tokenizer is resolved through ``_get_tokenizer``).
        :type tokenizer_path: str
        :raises ValueError: if the checkpoint does not contain hyperparameters.
        """
        print_colored("Loading model...", Colors.YELLOW)

        print(f"  Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'hyper_parameters' in checkpoint:
            self.hparams = checkpoint['hyper_parameters']
        elif 'hparams' in checkpoint:
            self.hparams = checkpoint['hparams']
        else:
            raise ValueError("Cannot find hyperparameters in checkpoint")

        # Convert dict hparams to an attribute-style namespace for downstream code.
        class HParamsNamespace:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)

        if isinstance(self.hparams, dict):
            self.hparams = HParamsNamespace(self.hparams)

        # NOTE: resolve tokenizer via generate.py helper to stay consistent with training.
        self.tokenizer = _get_tokenizer(self.hparams)

        # Some tokenizer wrappers expose ``get_vocab_size`` while others only
        # support ``len``; normalise to a single call pattern.
        vocab_size = self.tokenizer.get_vocab_size() if hasattr(self.tokenizer, 'get_vocab_size') else len(self.tokenizer)
        print(f"  Raw tokenizer vocab size: {vocab_size}")

        # Back-fill ``get_vocab_size`` on the wrapper if missing.
        if not hasattr(self.tokenizer, 'get_vocab_size'):
            if hasattr(self.tokenizer, 'tokenizer_obj') and hasattr(self.tokenizer.tokenizer_obj, 'get_vocab_size'):
                self.tokenizer.get_vocab_size = self.tokenizer.tokenizer_obj.get_vocab_size
            else:
                self.tokenizer.get_vocab_size = lambda: len(self.tokenizer)

        self.hparams.tokenizer_obj = self.tokenizer

        from openebm.elm.modeling_ebt import EBT_NLP
        self.model = EBT_NLP(self.hparams)

        # NOTE: scrub state_dict keys so both ``model.`` (Lightning/DDP) and
        # ``_orig_mod.`` (torch.compile) prefixes map cleanly onto the bare model.
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if new_key.startswith('model.'):
                new_key = new_key[6:]
            if new_key.startswith('_orig_mod.'):
                new_key = new_key[10:]

            new_state_dict[new_key] = v

        # Prefer strict loading so any missing parameter is surfaced immediately.
        try:
            self.model.load_state_dict(new_state_dict, strict=True)
            print_colored("  ✓ Weights loaded successfully (strict=True, all parameters matched)", Colors.GREEN)
        except Exception as e:
            print_colored(f"  ⚠ Strict loading failed, falling back to strict=False. Details:\n{e}", Colors.YELLOW)
            self.model.load_state_dict(new_state_dict, strict=False)

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        print_colored("✓ Model loaded and initialized", Colors.GREEN)
        self._print_model_info()

    def _print_model_info(self) -> None:
        """Print a summary of model hyperparameters and apply inference overrides."""
        embed_dim = getattr(self.hparams, 'embedding_dim', getattr(self.hparams, 'dim', 'unknown'))
        n_layers = getattr(self.hparams, 'num_layers', getattr(self.hparams, 'n_layers', 'unknown'))
        n_heads = getattr(self.hparams, 'num_heads', getattr(self.hparams, 'n_heads', 'unknown'))
        mcmc_steps = getattr(self.hparams, 'mcmc_num_steps', 'unknown')
        ctx_len = getattr(self.hparams, 'context_length', getattr(self.hparams, 'max_seq_len', 'unknown'))

        alpha_val = 'unknown'
        if hasattr(self.model, 'alpha'):
            alpha_val = self.model.alpha.item() if isinstance(self.model.alpha, torch.Tensor) else self.model.alpha

        noise_std_val = 'unknown'
        if hasattr(self.model, 'langevin_dynamics_noise_std'):
            noise_std_val = self.model.langevin_dynamics_noise_std.item() if isinstance(self.model.langevin_dynamics_noise_std, torch.Tensor) else self.model.langevin_dynamics_noise_std

        if self.override_mcmc_steps is not None:
            original_steps = getattr(self.hparams, 'mcmc_num_steps', 2)
            if self.override_mcmc_steps > original_steps:
                extra = self.override_mcmc_steps - original_steps
                self.hparams.randomize_mcmc_num_steps = extra
                self.model.hparams.randomize_mcmc_num_steps = extra
                self.hparams.randomize_mcmc_num_steps_final_landscape = True
                self.model.hparams.randomize_mcmc_num_steps_final_landscape = True
                self.hparams.randomize_mcmc_num_steps_min = extra + 1
                self.model.hparams.randomize_mcmc_num_steps_min = extra + 1
                mcmc_steps = self.override_mcmc_steps

                trained_noise = getattr(self.hparams, 'langevin_dynamics_noise', 0.0)
                if self.override_noise_std is None and trained_noise == 0:
                    auto_noise = 0.0
                    self.hparams.langevin_dynamics_noise = auto_noise
                    self.model.hparams.langevin_dynamics_noise = auto_noise
                    if hasattr(self.model, 'langevin_dynamics_noise_std'):
                        self.model.langevin_dynamics_noise_std.data.fill_(auto_noise)
                    noise_std_val = auto_noise
            elif self.override_mcmc_steps < original_steps:
                self.hparams.mcmc_num_steps = mcmc_steps = self.override_mcmc_steps
                self.model.hparams.mcmc_num_steps = self.override_mcmc_steps

        if self.override_noise_std is not None:
            if hasattr(self.model, 'langevin_dynamics_noise_std'):
                self.model.langevin_dynamics_noise_std.data.fill_(self.override_noise_std)
                self.hparams.langevin_dynamics_noise = self.override_noise_std
                self.model.hparams.langevin_dynamics_noise = self.override_noise_std
                noise_std_val = self.override_noise_std

        if self.override_alpha is not None:
            if hasattr(self.model, 'alpha'):
                self.model.alpha = torch.tensor(
                    self.override_alpha,
                    dtype=self.model.alpha.dtype,
                    device=self.model.alpha.device
                )
                alpha_val = self.override_alpha

        print()
        print(f"  {Colors.BOLD}Model configuration:{Colors.RESET}")
        print(f"    - Embedding dim: {embed_dim}")
        print(f"    - Layer count: {n_layers}")
        print(f"    - Attention head count: {n_heads}")
        print(f"    - MCMC steps: {mcmc_steps}")

        if isinstance(alpha_val, float):
            print(f"    - MCMC step size (alpha): {alpha_val:.6f}")
        else:
            print(f"    - MCMC step size (alpha): {alpha_val}")

        if isinstance(noise_std_val, float):
            print(f"    - Langevin noise: {noise_std_val:.6f}")
        else:
            print(f"    - Langevin noise: {noise_std_val}")

        print(f"    - Context length: {ctx_len}")
        print()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        stop_tokens: Optional[List[int]] = None,
        stream: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a continuation for ``prompt`` using the loaded EBT model.

        The prompt is rendered using the nanochat chat template
        (``<|bos|> <|user_start|> ... <|user_end|> <|assistant_start|>``) so it
        matches what ``dataset_sft.py`` produces during SFT training.

        :param prompt: user message to respond to.
        :type prompt: str
        :param max_tokens: maximum number of new tokens to sample.
        :type max_tokens: int
        :param temperature: sampling temperature; ``0`` enables greedy decoding.
        :type temperature: float
        :param top_p: nucleus-sampling cumulative probability threshold.
        :type top_p: float
        :param stop_tokens: optional list of additional stop token ids.
        :type stop_tokens: Optional[List[int]]
        :param stream: whether to print tokens incrementally as they are sampled.
        :type stream: bool
        :return: the decoded continuation and a statistics dictionary.
        :rtype: Tuple[str, Dict[str, Any]]
        """

        # Build the prompt with nanochat's render_conversation template so that
        # tokenisation matches SFT training (dataset_sft.py::render_conversation).
        inner_tok = getattr(self.tokenizer, 'tokenizer', None)  # NanoChatTokenizerWrapper.tokenizer = RustBPETokenizer
        if inner_tok is not None and hasattr(inner_tok, 'encode_special'):
            bos_id       = inner_tok.get_bos_token_id()
            user_start   = inner_tok.encode_special("<|user_start|>")
            user_end     = inner_tok.encode_special("<|user_end|>")
            asst_start   = inner_tok.encode_special("<|assistant_start|>")
            content_ids  = inner_tok.encode(prompt)
            prompt_tokens_list = [bos_id, user_start] + content_ids + [user_end, asst_start]
        else:
            # Fallback for base models with no chat template.
            encoded = self.tokenizer.encode(prompt)
            prompt_tokens_list = encoded if isinstance(encoded, list) else encoded.tolist()
            bos_id = getattr(self.tokenizer, 'bos_token_id', None)
            if bos_id is not None and (not prompt_tokens_list or prompt_tokens_list[0] != bos_id):
                prompt_tokens_list = [bos_id] + prompt_tokens_list

        # NOTE: pad_id is only used for tensor padding, not as a stop token.
        # NanoChatTokenizerWrapper aliases eos/pad to bos_id, but bos must not be
        # a stop token: packed SFT sequences place bos right after assistant_end,
        # so the model may legitimately emit bos right after assistant_start.
        # Treating bos as a stop token would yield an immediate empty output.
        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            pad_id = self.tokenizer.bos_token_id
        elif hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            pad_id = self.tokenizer.eos_token_id
        else:
            pad_id = 0

        bsz = 1
        min_prompt_len = len(prompt_tokens_list)
        max_prompt_len = len(prompt_tokens_list)

        ctx_len = getattr(self.hparams, 'context_length', getattr(self.hparams, 'max_seq_len', 2048))
        total_len = min(ctx_len, max_tokens + max_prompt_len)

        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=self.device)
        tokens[0, :len(prompt_tokens_list)] = torch.tensor(prompt_tokens_list, dtype=torch.long, device=self.device)

        # NOTE: use a position-based mask instead of a value-based (==pad_id)
        # mask because bos_id == pad_id would otherwise mis-classify the first
        # prompt token as padding.
        input_text_mask = torch.zeros(bsz, total_len, dtype=torch.bool, device=self.device)
        input_text_mask[0, :len(prompt_tokens_list)] = True

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=self.device)
        start_time = time.time()

        # NOTE: stop_token_ids contains only <|assistant_end|>, excluding bos/pad,
        # since bos_id == pad_id and emitting bos at assistant_start is valid.
        stop_token_ids = set()
        inner_tok = getattr(self.tokenizer, 'tokenizer', None)
        if inner_tok is not None and hasattr(inner_tok, 'encode_special'):
            asst_end_id = inner_tok.encode_special("<|assistant_end|>")
            if asst_end_id is not None:
                stop_token_ids.add(asst_end_id)
        # Fallback: if <|assistant_end|> is missing, fall back to pad_id.
        if not stop_token_ids:
            stop_token_ids.add(pad_id)

        with torch.no_grad():
            if min_prompt_len == total_len:
                logits = call_model_forward_decode(self.hparams, self.model, tokens, prev_pos, bsz)

            for cur_pos in range(min_prompt_len, total_len):
                input_tokens = tokens[:, :cur_pos]

                logits = call_model_forward_decode(self.hparams, self.model, input_tokens, prev_pos, bsz)

                if temperature > 0:
                    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                    next_token = sample_top_p(probs, top_p)
                else:
                    next_token = torch.argmax(logits[:, -1], dim=-1)

                next_token = next_token.reshape(-1)

                next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
                tokens[:, cur_pos] = next_token

                # Streaming output
                if stream and cur_pos >= min_prompt_len:
                    token_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
                    print(token_text, end='', flush=True)

                # EOS check: <|assistant_end|> only (bos/pad excluded to avoid
                # false positives for SFT models emitting bos).
                is_stop = torch.zeros(bsz, dtype=torch.bool, device=self.device)
                for sid in stop_token_ids:
                    is_stop |= (next_token == sid)
                eos_reached |= (~input_text_mask[:, cur_pos]) & is_stop
                prev_pos = cur_pos

                if all(eos_reached):
                    break

        # Extract generated tokens.
        toks = tokens[0].tolist()
        start = len(prompt_tokens_list)
        toks = toks[start : len(prompt_tokens_list) + max_tokens]

        # cut to first stop token
        for sid in stop_token_ids:
            if sid in toks:
                toks = toks[:toks.index(sid)]

        generated_text = self.tokenizer.decode(toks, skip_special_tokens=True)

        stats = {
            'tokens_generated': len(toks),
            'total_time': time.time() - start_time,
            'tokens_per_second': len(toks) / (time.time() - start_time) if (time.time() - start_time) > 0 else 0,
            'avg_energy_change': 0,
            'avg_token_prob': 0,
        }

        return generated_text, stats


def print_help() -> None:
    """Print the list of interactive slash commands."""
    print()
    print(f"{Colors.BOLD}Available commands:{Colors.RESET}")
    print("  /quit, /exit        - Exit the chat")
    print("  /clear              - Clear the conversation history")
    print("  /temp <value>       - Set temperature (0.0-2.0)")
    print("  /topp <value>       - Set top-p (0.0-1.0)")
    print("  /tokens <value>     - Set max tokens (1-4096)")
    print("  /mcmc               - Toggle MCMC display")
    print("  /verbose            - Toggle verbose mode")
    print("  /energy             - Toggle energy display")
    print("  /status             - Show current settings")
    print("  /info               - Show model info")
    print("  /help               - Show this help")
    print()


def print_status(engine: "EBTChatEngine", temperature: float, top_p: float, max_tokens: int) -> None:
    """Print the current runtime configuration.

    :param engine: the active chat engine.
    :type engine: EBTChatEngine
    :param temperature: current sampling temperature.
    :type temperature: float
    :param top_p: current nucleus-sampling threshold.
    :type top_p: float
    :param max_tokens: current maximum generation length.
    :type max_tokens: int
    """
    print()
    print(f"{Colors.BOLD}Current settings:{Colors.RESET}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-P: {top_p}")
    print(f"  Max Tokens: {max_tokens}")
    print(f"  Show MCMC: {'yes' if engine.show_mcmc else 'no'}")
    print(f"  Verbose mode: {'yes' if engine.verbose else 'no'}")
    print(f"  Show energy: {'yes' if engine.show_energy else 'no'}")
    print()


def print_generation_stats(stats: Dict[str, Any]) -> None:
    """Print generation statistics produced by :meth:`EBTChatEngine.generate`.

    :param stats: dictionary returned alongside the generated text.
    :type stats: Dict[str, Any]
    """
    print()
    print(f"{Colors.GRAY}────────────────────────────────────────{Colors.RESET}")
    print(f"{Colors.BOLD}Generation stats:{Colors.RESET}")
    print(f"  Tokens generated: {stats['tokens_generated']}")
    print(f"  Total time: {stats['total_time']:.2f}s")
    print(f"  Speed: {stats['tokens_per_second']:.2f} tokens/s")
    print(f"  Average token probability: {stats['avg_token_prob']:.4f}")
    print(f"  Average energy change: {stats['avg_energy_change']:.4f}")
    print(f"{Colors.GRAY}────────────────────────────────────────{Colors.RESET}")
    print()


def main() -> int:
    """Entry point for the interactive chat terminal.

    :return: process exit code (0 on success, 1 on failure).
    :rtype: int
    """
    parser = argparse.ArgumentParser(description='EBT interactive chat terminal')
    parser.add_argument('-c', '--checkpoint', type=str,
                       default="/mnt/shared-storage-user/puyuan/code/OpenEBM/logs/checkpoints/ebt-d26-stable_20260313_123203_2026-03-13_12-32-54_/last.ckpt",
                       help='Checkpoint path')
    parser.add_argument('--tokenizer', type=str,
                       default="/mnt/shared-storage-user/puyuan/code/nanochat/.cache/nanochat/tokenizer",
                       help='Tokenizer path')
    parser.add_argument('-t', '--temperature', type=float, default=0.8,
                       help='Generation temperature (default: 0.8)')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-P sampling (default: 0.9)')
    parser.add_argument('-m', '--max-tokens', type=int, default=256,
                       help='Maximum generated tokens (default: 256)')
    parser.add_argument('--show-mcmc', action='store_true',
                       help='Show MCMC step process')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose mode')
    parser.add_argument('--show-energy', action='store_true',
                       help='Show energy value changes')
    parser.add_argument('--show-distribution', action='store_true',
                       help='Show probability distribution changes')
    parser.add_argument('-d', '--dtype', type=str, default='bfloat16',
                       choices=['float32', 'bfloat16'],
                       help='Data type (default: bfloat16)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (default: cuda)')

    # Advanced inference overrides (override training-time values).
    parser.add_argument('--override-mcmc-steps', type=int, default=None,
                       help='Override MCMC steps from training time (default: use trained value)')
    parser.add_argument('--override-noise-std', type=float, default=None,
                       help='Override Langevin noise std (default: use trained value)')
    parser.add_argument('--override-alpha', type=float, default=None,
                       help='Override MCMC step size alpha (default: use trained value)')

    args = parser.parse_args()

    print_banner()

    if not os.path.exists(args.checkpoint):
        print_colored(f"Error: checkpoint not found: {args.checkpoint}", Colors.RED)
        return 1

    dtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16

    # Enable TF32 for Tensor Core acceleration on H200/A100.
    torch.set_float32_matmul_precision('medium')

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
        print_colored(f"Error: failed to load model - {str(e)}", Colors.RED)
        import traceback
        traceback.print_exc()
        return 1

    print("=" * 70)
    print_colored("EBT chat terminal ready!", Colors.GREEN)
    print("=" * 70)
    print("Usage:")
    print("  - Type text and press Enter to start generation")
    print("  - Type /quit or /exit to exit")
    print("  - Type /help to view all commands")
    print("  - Type /mcmc to toggle MCMC step display")
    print("=" * 70)
    print()

    temperature = args.temperature
    top_p = args.top_p
    max_tokens = args.max_tokens

    session_total_tokens = 0
    session_total_time = 0.0
    session_turns = 0

    while True:
        try:
            user_input = input(f'{Colors.BLUE}You:{Colors.RESET} ')

            cmd = user_input.strip().lower()

            if cmd in ['/quit', '/exit']:
                print_colored('\nGoodbye!', Colors.GREEN)
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
                status = "enabled" if engine.show_mcmc else "disabled"
                print_colored(f'\n✓ MCMC display {status}\n', Colors.GREEN)
                continue

            if cmd == '/verbose':
                engine.verbose = not engine.verbose
                status = "enabled" if engine.verbose else "disabled"
                print_colored(f'\n✓ Verbose mode {status}\n', Colors.GREEN)
                continue

            if cmd == '/energy':
                engine.show_energy = not engine.show_energy
                status = "enabled" if engine.show_energy else "disabled"
                print_colored(f'\n✓ Energy display {status}\n', Colors.GREEN)
                continue

            if cmd.startswith('/temp '):
                try:
                    new_temp = float(cmd.split()[1])
                    if 0.0 <= new_temp <= 2.0:
                        temperature = new_temp
                        print_colored(f'\n✓ Temperature set to: {temperature}\n', Colors.GREEN)
                    else:
                        print_colored('\n✗ Temperature must be between 0.0-2.0\n', Colors.RED)
                except:
                    print_colored('\n✗ Invalid temperature value\n', Colors.RED)
                continue

            if cmd.startswith('/topp '):
                try:
                    new_topp = float(cmd.split()[1])
                    if 0.0 <= new_topp <= 1.0:
                        top_p = new_topp
                        print_colored(f'\n✓ Top-P set to: {top_p}\n', Colors.GREEN)
                    else:
                        print_colored('\n✗ Top-P must be between 0.0-1.0\n', Colors.RED)
                except:
                    print_colored('\n✗ Invalid Top-P value\n', Colors.RED)
                continue

            if cmd.startswith('/tokens '):
                try:
                    new_tokens = int(cmd.split()[1])
                    if 1 <= new_tokens <= 4096:
                        max_tokens = new_tokens
                        print_colored(f'\n✓ Max Tokens set to: {max_tokens}\n', Colors.GREEN)
                    else:
                        print_colored('\n✗ Max Tokens must be between 1-4096\n', Colors.RED)
                except:
                    print_colored('\n✗ Invalid Tokens value\n', Colors.RED)
                continue

            if not user_input.strip():
                continue

            print(f'{Colors.GREEN}EBT:{Colors.RESET} ', end='', flush=True)

            try:
                generated_text, stats = engine.generate(
                    prompt=user_input,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                )
                print()

                session_total_tokens += stats['tokens_generated']
                session_total_time += stats['total_time']
                session_turns += 1

                # Always emit a short summary; verbose/show-mcmc prints the full block.
                if engine.verbose or engine.show_mcmc:
                    print_generation_stats(stats)
                else:
                    print(f"{Colors.GRAY}  [{stats['tokens_generated']} tokens, {stats['total_time']:.1f}s, {stats['tokens_per_second']:.1f} tok/s]{Colors.RESET}")
                    print()

            except Exception as e:
                print()
                print_colored(f'✗ Generation error: {str(e)}', Colors.RED)
                import traceback
                traceback.print_exc()

        except KeyboardInterrupt:
            print_colored('\n\nCtrl+C detected, exiting...', Colors.YELLOW)
            break
        except EOFError:
            print_colored('\n\nEOF detected, exiting...', Colors.YELLOW)
            break

    # Final session summary.
    if session_turns > 0:
        avg_tps = session_total_tokens / session_total_time if session_total_time > 0 else 0
        print()
        print("=" * 50)
        print(f"  Session summary")
        print("=" * 50)
        print(f"  Turns:           {session_turns}")
        print(f"  Total tokens:    {session_total_tokens}")
        print(f"  Total time:      {session_total_time:.2f}s")
        print(f"  Avg throughput:  {avg_tps:.2f} tokens/s")
        print(f"  Avg per turn:    {session_total_tokens/session_turns:.0f} tokens, {session_total_time/session_turns:.1f}s")
        print("=" * 50)
        print(f"[SESSION_SUMMARY] turns={session_turns} tokens={session_total_tokens} time={session_total_time:.1f}s throughput={avg_tps:.2f}tok/s")

    print_colored('\nSession ended.', Colors.GREEN)
    return 0


if __name__ == "__main__":
    sys.exit(main())