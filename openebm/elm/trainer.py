"""PyTorch Lightning trainer for Energy-Based Transformers (EBT).

Wraps :class:`openebm.elm.modeling_ebt.EBT_NLP` in a :class:`LightningModule`
that covers pretraining, SFT, and inference flows with DDP, wandb logging,
``torch.compile``, MCMC replay buffers, and exact-resume state across ranks.
"""

import gc
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import ipdb
import torch
import wandb
from torch import nn
from torch.distributed import all_reduce
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets import load_dataset, load_from_disk
from openebm.elm.collector import NLP_HF_Collator
from openebm.elm.modeling_ebt import EBT_NLP

try:
    from lightning.pytorch import LightningModule
except ImportError:
    from pytorch_lightning import LightningModule
from openebm.elm.dataset import IterableDataset, generate_dataloader
from openebm.elm.dataset_sft import generate_sft_dataloader


class GSM8KDataset(torch.utils.data.Dataset):
    """Minimal GSM8K dataset wrapper for inference.

    Loads either a local offline copy or the HuggingFace hub version, and
    formats each row as plain text according to ``hparams.execution_mode``.
    """

    def __init__(self, hparams: Any, split: str) -> None:
        """Initialize the dataset.

        :param hparams: Hparams providing ``execution_mode`` and optionally
            ``dataset_dir``.
        :type hparams: Any
        :param split: HuggingFace split name (e.g. ``"train"``, ``"test"``).
        :type split: str
        """
        self.hparams = hparams
        local_dataset_path = "/mnt/shared-storage-user/puyuan/code/EBT/data/gsm8k_offline"

        if os.path.exists(local_dataset_path):
            dataset = load_from_disk(local_dataset_path)
            self.dataset = dataset[split]
        else:
            hf_token = os.getenv('HF_TOKEN')
            hf_home = os.getenv('HF_HOME')
            dataset_dir = self.hparams.dataset_dir if hasattr(self.hparams, 'dataset_dir') and self.hparams.dataset_dir != "" else hf_home
            self.dataset = load_dataset("openai/gsm8k", "main", cache_dir=dataset_dir, token=hf_token, trust_remote_code=True)[split]

    def __len__(self) -> int:
        """Return the number of samples in the split.

        :return: Sample count.
        :rtype: int
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        """Return the formatted sample at ``idx``.

        :param idx: Sample index.
        :type idx: int
        :return: Formatted string or ``(question, answer)`` tuple depending on
            ``hparams.execution_mode``.
        :rtype: Any
        :raises ValueError: When ``execution_mode`` is not supported.
        """
        if self.hparams.execution_mode == "inference":
            return f"[[Question]]: {self.dataset[idx]['question']}\n[[Answer]]: ", self.dataset[idx]['answer']
        elif self.hparams.execution_mode == "pretrain":
            return f"Question: {self.dataset[idx]['question']}\nAnswer: {self.dataset[idx]['answer']}"
        elif self.hparams.execution_mode == "finetune":
            return f"[[Question]]: {self.dataset[idx]['question']}\n[[Answer]]: {self.dataset[idx]['answer']}"
        else:
            raise ValueError(f"Execution mode not supported: {self.hparams.execution_mode}")

from openebm.elm.generate import generate_text, get_ppl
from openebm.elm.optimization import WarmUpCosineAnnealingLR, WarmUpLinearWarmdownLR, LARS, exclude_bias_and_norm, StableAdamW, StableAdamWUnfused
from openebm.elm import logger as text_logger
from openebm.elm.metrics import get_torchmetrics

from nanochat.tokenizer import get_tokenizer, get_token_bytes


class ModelTrainer(LightningModule):
    """LightningModule wrapping :class:`EBT_NLP` for EBT training and eval."""

    def __init__(self, hparams: Any, trained_model: Optional[Any] = None) -> None:
        """Initialize the trainer.

        :param hparams: Training hparams (argparse Namespace or similar).
        :type hparams: Any
        :param trained_model: Optional pre-initialized model to wrap.
        :type trained_model: Optional[Any]
        """
        super().__init__()
        if isinstance(hparams, dict):
            # Passed in from model checkpoint reload.
            self.hparams.update(hparams)
        else:
            self.hparams.update(vars(hparams))

        # Test-time metric tracking.
        self.test_losses = []
        self.test_perplexities = []

        # Training throughput tracking.
        self._train_step_start_time = None
        self._train_start_time = None

        # Dataloader resume state used when restoring from checkpoint.
        self._dataloader_resume_state = None

        if self.hparams.modality == "NLP":
            # The pair of attribute checks guards loading of older ckpts that
            # may predate these fields.
            if "execution_mode" in self.hparams and "save_generation_logs_dir" in self.hparams and self.hparams.execution_mode == "inference":
                print("setting up infer logger")
                self.infer_logger = text_logger.setup_jsonl_logger(log_filename = "results.jsonl", base_log_dir=self.hparams.save_generation_logs_dir)
        self.full_ds = None

        # Tokenizer configuration for the NanoChat dataset.
        # ``get_tokenizer()`` loads the NanoChat custom BPE tokenizer from
        # ``$NANOCHAT_BASE_DIR/tokenizer/`` (vocab_size=32768).
        # The ``--tokenizer`` CLI parameter is IGNORED for NanoChat.
        self.hparams.tokenizer_obj = tokenizer = get_tokenizer()
        # Keep tokenizer path as a string for ``generate_text`` compatibility.
        if not hasattr(self.hparams, 'tokenizer_path'):
            self.hparams.tokenizer_path = self.hparams.tokenizer if isinstance(self.hparams.tokenizer, str) else "EleutherAI/gpt-neox-20b"

        # Load token_bytes for BPB (bits-per-byte) metric. The table maps each
        # token id to its byte length; used in validation/test metrics. Moved
        # to GPU on demand.
        try:
            self.token_bytes = get_token_bytes(device="cpu")
            print(f"  Token bytes loaded: shape={self.token_bytes.shape}")
        except Exception as e:
            print(f"  Warning: Could not load token_bytes: {e}")
            print(f"  BPB metrics will not be available")
            self.token_bytes = None

        print(f"=" * 80)
        print(f"TOKENIZER INFO:")
        print(f"  Actual tokenizer used: NanoChat custom BPE tokenizer")
        print(f"  Tokenizer location: $NANOCHAT_BASE_DIR/tokenizer/")
        print(f"  Vocab size: {tokenizer.get_vocab_size()}")
        print(f"  Command-line --tokenizer parameter: {self.hparams.tokenizer} (IGNORED)")
        print(f"=" * 80)

        if trained_model is not None:
            self.model = trained_model
        else:
            self.model = EBT_NLP(self.hparams)

        # torch.compile support.
        # EBT training uses ``autograd.grad(create_graph=True)`` to produce
        # second-order gradients. ``torch.compile`` (aot_autograd) does not
        # support double backward, so compilation is skipped during training.
        # Inference uses ``learning=False`` → ``create_graph=False``, which is
        # safe to compile.
        if self.hparams.compile_model:
            compile_mode = getattr(self.hparams, 'compile_mode', 'transformer_only')
            compile_backend = getattr(self.hparams, 'compile_backend', 'inductor')
            compile_dynamic = getattr(self.hparams, 'compile_dynamic', False)

            if compile_mode == 'full':
                # Compile the whole model (may be incompatible with autograd.grad).
                print(f"\n{'='*80}")
                print(f"[torch.compile] compiling whole model...")
                print(f"[torch.compile] mode: full | backend: {compile_backend} | dynamic: {compile_dynamic}")
                print(f"[torch.compile] warning: EBT MCMC uses autograd.grad, compilation may fail")
                print(f"[torch.compile] first compile can take 5-15 min")
                print(f"{'='*80}\n")
                import time
                start_time = time.time()
                self.model = torch.compile(self.model, backend=compile_backend, dynamic=compile_dynamic)
                compile_time = time.time() - start_time
                print(f"\n{'='*80}")
                print(f"[torch.compile] model compile done (took: {compile_time:.1f}s)")
                print(f"{'='*80}\n")

            elif compile_mode == 'transformer_only':
                # Compile only the transformer (keeps MCMC in eager mode).
                # Keep an eager reference used when ``_mcmc_step_excluded`` needs
                # ``create_graph=True``.
                print(f"[torch.compile] compiling transformer only (mode=transformer_only, backend={compile_backend})")
                if hasattr(self.model, 'transformer'):
                    self.model.transformer_eager = self.model.transformer
                    self.model.transformer = torch.compile(
                        self.model.transformer,
                        backend=compile_backend,
                        dynamic=compile_dynamic
                    )
                    print(f"[torch.compile] transformer compiled; transformer_eager retained for MCMC")
                else:
                    print(f"[torch.compile] warning: model has no transformer attribute, skipping compile")

            elif compile_mode == 'disabled':
                print(f"[torch.compile] disabled")

            elif (self.hparams.execution_mode == "inference") or getattr(self.hparams, 'only_test', False):
                # Inference mode: ``learning=False`` means no double backward,
                # so compilation is safe.
                if compile_mode == 'full':
                    print(f"[torch.compile] inference mode: compiling whole model (backend={compile_backend})")
                    self.model = torch.compile(self.model, backend=compile_backend, dynamic=compile_dynamic)
                elif compile_mode == 'transformer_only':
                    if hasattr(self.model, 'transformer'):
                        print(f"[torch.compile] inference mode: compiling transformer (backend={compile_backend})")
                        self.model.transformer = torch.compile(
                            self.model.transformer, backend=compile_backend, dynamic=compile_dynamic
                        )
                    else:
                        print(f"[torch.compile] warning: model has no 'transformer' attribute, skipping")
                else:
                    raise ValueError(f"Unknown compile_mode: {compile_mode}")

            else:
                # Training mode: skip compile because EBT MCMC needs double backward.
                print(f"[torch.compile] training mode: skipping compile (EBT MCMC needs create_graph=True; aot_autograd does not support double backward)")

        phases = ['train', 'valid', 'test']
        self.torchmetrics_dict = nn.ModuleDict()
        self.metrics = []
        for metric in self.hparams.metrics_list:
            self.metrics.append(metric)
        if len(self.metrics) > 0:
            assert self.hparams.num_classes != -1, "please set num_classes to the appropriate amount for the in use metrics. if are using accuracy and num_classes varies just set it to something that makes sense (shouldnt matter in that case)"
            assert self.hparams.metrics_task != "", "please set metrics_task to the appropriate value for your metrics"
        for phase in phases:
            for metric in self.metrics:
                self.torchmetrics_dict[f"{phase}_{metric}"] = get_torchmetrics(metric, self.hparams.metrics_average_type, self.hparams.num_classes, self.hparams.metrics_task)

        if self.hparams.wandb_watch:
            # Tag every submodule with its name for activation logging.
            for name, module in self.model.named_modules():
                module.name = name


    def on_train_start(self) -> None:
        """Restore RNG state and optionally arm parameter-usage hooks.

        Runs after the val-sanity-check and before the first training step.
        """
        import random
        rng = getattr(self, '_rng_resume_state', None)
        if rng is not None:
            torch.random.set_rng_state(rng['torch_cpu'])
            if rng.get('torch_cuda') is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(rng['torch_cuda'])
            random.setstate(rng['python'])
            self._rng_resume_state = None
            print(f"[Exact Resume] RNG states restored for rank {self.global_rank}")

        if self.hparams.debug_unused_parameters:
            for name, param in self.model.named_parameters():
                # Excludes image_encoder-style frozen sub-nets; extend this
                # guard when additional frozen modules are introduced.
                if param.requires_grad and "image_encoder" not in name:
                    print(f"registering param - {name}")
                    param.register_hook(self.create_hook(name))
                else:
                    self.model.parameters_not_to_check.add(name)

    def create_hook(self, name: str):
        """Create a backward hook that records parameter usage.

        Only used when ``debug_unused_parameters`` is set.

        :param name: Parameter name to record on use.
        :type name: str
        :return: A backward hook callable.
        :rtype: Callable
        """
        def hook(grad):
            self.model.used_parameters.add(name)
        return hook

    @staticmethod
    def wandb_activation_hook(run: Any, step: int):
        """Return a forward hook that logs activation stats to W&B.

        Logs mean/std/min/max of each activation; tuple outputs are skipped.

        :param run: W&B run (typically ``self.logger``).
        :type run: Any
        :param step: Global step index used as the W&B ``step`` key.
        :type step: int
        :return: A forward hook callable.
        :rtype: Callable
        """
        def hook(module, input, output):
            if isinstance(output, tuple):
                pass
            else:
                try:
                    # Aggregate on-device rather than copying the tensor to CPU.
                    data = output.detach().float()
                    run.experiment.log(
                        {
                            f"activations/{module.name}_mean": data.mean().item(),
                            f"activations/{module.name}_std": data.std().item(),
                            f"activations/{module.name}_min": data.min().item(),
                            f"activations/{module.name}_max": data.max().item(),
                        },
                        step=step
                    )
                except RuntimeError:
                    # Tensors inside ``torch.func.grad`` have no storage.
                    pass

        return hook

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Run one training step and return the scalar loss.

        :param batch: Training batch.
        :type batch: Any
        :param batch_idx: Global batch index.
        :type batch_idx: int
        :return: Loss tensor used by Lightning for backward.
        :rtype: torch.Tensor
        """
        # Activation logging only when wandb_watch is on AND level is "all".
        if not self.hparams.no_wandb and self.hparams.wandb_watch and getattr(self.hparams, 'wandb_watch_level', 'parameters') == 'all' and self.global_step % self.hparams.wandb_watch_log_freq == 0:
            hook_handles = []
            hook_function = self.wandb_activation_hook(run=self.logger, step=self.global_step)
            for module in self.model.modules():
                # Only hook modules with trainable parameters.
                if any(param.requires_grad for param in module.parameters(recurse=False)):
                    handle = module.register_forward_hook(hook_function)
                    hook_handles.append(handle)

            eval_step_dict = self.eval_step(batch, "train")
            for handle in hook_handles:
                handle.remove()

        else:
            eval_step_dict = self.eval_step(batch, "train")

        self.log_metrics(eval_step_dict, "train")
        return eval_step_dict['loss']

    def on_after_backward(self) -> None:
        """Log gradient norms and clip-rate when ``log_gradients`` is set."""
        if self.hparams.log_gradients:
            total_norm = 0.0
            num_parameters = 0
            num_grads_exceeding_clip_val = 0
            # ``total_gradients`` counts individual scalar grads, not param tensors.
            total_gradients = 0
            for param in self.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm
                    num_parameters += 1

                    total_gradients += torch.numel(param.grad)
                    num_grads_exceeding_clip_val += torch.sum(param.grad.abs() > self.hparams.gradient_clip_val)

            assert num_parameters > 0, "no gradients after backwards detected please investigate"
            average_norm = (total_norm / num_parameters).detach()
            percentage_clipped = ((num_grads_exceeding_clip_val / total_gradients) * 100).detach()

            things_to_log = {}
            things_to_log['avg_gradient_norms'] = average_norm
            things_to_log['pct_gradient_clipped'] = percentage_clipped
            self.log_metrics(things_to_log, "train", log_torchmetrics = False)

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Post-step housekeeping: unused-param debug, GC, and timing tracking.

        Note: when extending this hook, explicitly exclude frozen modules
        (``requires_grad == False``) to avoid false positives.

        :param outputs: Outputs returned by :meth:`training_step`.
        :type outputs: Any
        :param batch: Training batch.
        :type batch: Any
        :param batch_idx: Global batch index.
        :type batch_idx: int
        """
        if self.hparams.debug_unused_parameters:
            all_parameters = {name for name, _ in self.model.named_parameters()}
            unused_parameters = all_parameters - self.model.used_parameters - self.model.parameters_not_to_check

            print(f"number of parameters total: {len(all_parameters)}")
            print(f"number of unused_parameters: {len(unused_parameters)}")
            print(f"Unused parameters: {unused_parameters}")
            print(f"Used parameters: {self.model.used_parameters}")

        if self.hparams.manual_gc_collect_every_n_steps != -1:
            if self.global_step > 0 and self.global_step % self.hparams.manual_gc_collect_every_n_steps == 0:
                print("calling GC manually")
                gc.collect()
                torch.cuda.empty_cache()

        # Record step end time for dt calculation.
        import time as _time
        now = _time.time()
        if self._train_step_start_time is not None:
            self._last_dt = now - self._train_step_start_time
        else:
            self._last_dt = None
        self._train_step_start_time = now
        if self._train_start_time is None:
            self._train_start_time = now

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Persist per-rank dataloader and RNG state for exact resume.

        :param checkpoint: Checkpoint dict that Lightning is about to write.
        :type checkpoint: Dict[str, Any]
        """
        import torch.distributed as dist
        import random

        # 1) Snapshot this rank's dataloader state (exact, including buffers).
        local_dl_state = None
        try:
            train_dl = self.trainer.train_dataloader
            if train_dl is not None:
                dataset = train_dl.dataset
                if hasattr(dataset, 'get_dataloader_state'):
                    local_dl_state = dataset.get_dataloader_state()
                elif hasattr(dataset, 'last_state_dict') and dataset.last_state_dict is not None:
                    local_dl_state = dataset.last_state_dict
        except Exception:
            # ``trainer.train_dataloader`` can be missing outside of training.
            pass

        # 2) Snapshot this rank's RNG state.
        local_rng_state = {
            'torch_cpu': torch.random.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'python': random.getstate(),
        }

        # 3) DDP: gather states from all ranks onto rank 0.
        if dist.is_initialized() and dist.get_world_size() > 1:
            all_dl_states = [None] * dist.get_world_size()
            dist.all_gather_object(all_dl_states, local_dl_state)
            all_rng_states = [None] * dist.get_world_size()
            dist.all_gather_object(all_rng_states, local_rng_state)
        else:
            all_dl_states = [local_dl_state]
            all_rng_states = [local_rng_state]

        # 4) Write into the checkpoint dict.
        checkpoint['dataloader_state_dict_by_rank'] = all_dl_states
        # Legacy key (kept for backwards compatibility).
        checkpoint['dataloader_state_dict'] = all_dl_states[0]
        checkpoint['rng_states_by_rank'] = all_rng_states

        print(f"[Checkpoint] saved per-rank dataloader state ({len(all_dl_states)} ranks) + RNG states")

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load dataloader/RNG state from ``checkpoint`` with compat shims.

        :param checkpoint: Checkpoint dict loaded from disk.
        :type checkpoint: Dict[str, Any]
        """
        # Compatibility: strip/add ``_orig_mod.`` prefixes introduced by
        # ``torch.compile`` so either side of the mismatch loads cleanly.
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            has_orig_mod_keys = any('_orig_mod.' in k for k in state_dict)
            model_has_orig_mod = any('_orig_mod.' in k for k in self.state_dict())

            if has_orig_mod_keys and not model_has_orig_mod:
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_state_dict[k.replace('._orig_mod.', '.')] = v
                checkpoint['state_dict'] = new_state_dict
                print(f"[Checkpoint] Stripped '_orig_mod' prefix from {len(state_dict)} keys")
            elif not has_orig_mod_keys and model_has_orig_mod:
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        new_state_dict['model._orig_mod.' + k[len('model.'):]] = v
                    else:
                        new_state_dict[k] = v
                checkpoint['state_dict'] = new_state_dict
                print(f"[Checkpoint] Added '_orig_mod' prefix to {len(state_dict)} keys")

        # Restore per-rank dataloader position + RNG state.
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Prefer the per-rank format; fall back to the legacy single-entry key.
        if 'dataloader_state_dict_by_rank' in checkpoint:
            states = checkpoint['dataloader_state_dict_by_rank']
            self._dataloader_resume_state = states[rank] if rank < len(states) else None
        elif 'dataloader_state_dict' in checkpoint:
            self._dataloader_resume_state = checkpoint['dataloader_state_dict']

        # Per-rank RNG state.
        if 'rng_states_by_rank' in checkpoint:
            rng_states = checkpoint['rng_states_by_rank']
            self._rng_resume_state = rng_states[rank] if rank < len(rng_states) else None
        else:
            self._rng_resume_state = None

        if self._dataloader_resume_state:
            if 'cursor' in self._dataloader_resume_state:
                print(
                    f"[Checkpoint] Rank {rank} restoring SFT dataloader state: "
                    f"cursor={self._dataloader_resume_state.get('cursor')}, "
                    f"consumed={self._dataloader_resume_state.get('consumed')}, "
                    f"epoch={self._dataloader_resume_state.get('epoch')}, "
                    f"it={self._dataloader_resume_state.get('it')}, "
                    f"state_version={self._dataloader_resume_state.get('state_version', 'legacy')}"
                )
            else:
                print(f"[Checkpoint] Rank {rank} restoring dataloader state: "
                      f"pq_idx={self._dataloader_resume_state.get('pq_idx')}, "
                      f"rg_idx={self._dataloader_resume_state.get('rg_idx')}, "
                      f"state_version={self._dataloader_resume_state.get('state_version', 'legacy')}")
        if self._rng_resume_state:
            print(f"[Checkpoint] Rank {rank} restoring RNG state: keys={list(self._rng_resume_state.keys())}")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Run one validation step and cache metrics for the progress bar.

        :param batch: Validation batch.
        :type batch: Any
        :param batch_idx: Batch index.
        :type batch_idx: int
        """
        token_bytes = self.token_bytes
        if token_bytes is not None and token_bytes.device != self.device:
            token_bytes = token_bytes.to(self.device)
        eval_step_dict = self.eval_step(batch, "valid", token_bytes)
        self.log_metrics(eval_step_dict, "valid")
        # Cache the latest validation metrics so the training progress bar
        # can display them.
        if not hasattr(self, '_last_valid_metrics'):
            self._last_valid_metrics = {}
        for k, v in eval_step_dict.items():
            if isinstance(v, torch.Tensor) and v.dim() == 0:
                self._last_valid_metrics[k] = v.detach().item()
            elif isinstance(v, (int, float)):
                self._last_valid_metrics[k] = v

    def on_test_epoch_start(self) -> None:
        """Reset per-epoch test metrics and print a header banner."""
        import time
        self.test_losses = []
        self.test_perplexities = []
        self.test_energies = {}
        self.test_start_time = time.time()
        self.test_generation_count = 0

        import sys
        sys.stdout.write(f"\n{'='*100}\n")
        sys.stdout.write(f"{'STARTING EVALUATION':^100}\n")
        sys.stdout.write(f"{'='*100}\n\n")
        sys.stdout.flush()

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """Run one test step (PPL and/or generation).

        :param batch: Test batch.
        :type batch: Any
        :param batch_idx: Batch index.
        :type batch_idx: int
        """
        if self.hparams.execution_mode == "inference":
            if self.hparams.modality == "NLP":
                # For GSM8K and other generation tasks that use DataLoader with collate_fn
                if self.hparams.dataset_name == "gsm8k":
                    outputs = generate_text(self.model, batch, self.hparams)
                    self.test_generation_count += len(outputs)
                    for output in outputs:
                        self.infer_logger.log_data(output)

                # For nanochat_shard_eval with dual-mode evaluation (PPL + Generation)
                elif self.hparams.dataset_name == "nanochat_shard_eval":
                    # Check if generation is enabled
                    enable_generation = getattr(self.hparams, 'enable_nanochat_generation', True)

                    # 1. Always compute PPL on full sequence
                    batch_dict = batch  # Already a dict from DataLoader
                    ppl_outputs = get_ppl(self.model, batch_dict, self.hparams)

                    # Track metrics for averaging
                    self.test_losses.append(ppl_outputs['loss'].item())
                    self.test_perplexities.append(ppl_outputs['perplexity'].item())

                    # Track energy metrics if available
                    if not hasattr(self, 'test_energies'):
                        self.test_energies = {}
                    for key, value in ppl_outputs.items():
                        if 'energy' in key:
                            if key not in self.test_energies:
                                self.test_energies[key] = []
                            self.test_energies[key].append(value)

                    # 2. If generation is enabled and prompt data is available, do text generation
                    if enable_generation and 'prompt_ids' in batch:
                        # Prepare batch for generation (similar to GSM8K format)
                        # questions = prompts, answers = targets
                        questions = {
                            'input_ids': batch['prompt_ids'],
                            'attention_mask': batch['prompt_attention_mask']
                        }
                        # Pad target_ids to same length before stacking
                        target_list = batch['target_ids']
                        max_target_len = max(t.shape[0] for t in target_list)
                        padded_targets = []
                        for t in target_list:
                            pad_len = max_target_len - t.shape[0]
                            if pad_len > 0:
                                padded_t = torch.cat([t, torch.zeros(pad_len, dtype=t.dtype, device=t.device)])
                            else:
                                padded_t = t
                            padded_targets.append(padded_t)
                        answers = {
                            'input_ids': torch.stack(padded_targets)
                        }

                        generation_batch = (questions, answers)
                        generation_outputs = generate_text(self.model, generation_batch, self.hparams)

                        # Log generation results with additional context
                        for i, output in enumerate(generation_outputs):
                            # Add PPL info and shard index
                            output['loss'] = ppl_outputs['loss'].item()
                            output['ppl'] = ppl_outputs['perplexity'].item()
                            output['shard_idx'] = batch['shard_indices'][i]
                            # Note: prompt and target are already in output from generate_text()
                            # No need to add duplicate fields

                            self.infer_logger.log_data(output)

                    # Print progress every 5 batches
                    if batch_idx % 5 == 0:
                        import sys
                        import numpy as np
                        import time

                        # Calculate statistics
                        current_avg_loss = np.mean(self.test_losses)
                        current_avg_ppl = np.mean(self.test_perplexities)
                        current_std_loss = np.std(self.test_losses) if len(self.test_losses) > 1 else 0.0
                        current_std_ppl = np.std(self.test_perplexities) if len(self.test_perplexities) > 1 else 0.0

                        # Estimate time remaining
                        if not hasattr(self, 'test_start_time'):
                            self.test_start_time = time.time()
                        elapsed = time.time() - self.test_start_time
                        batches_done = batch_idx + 1
                        batches_total = self.hparams.limit_test_batches if self.hparams.limit_test_batches != 1 else 100
                        eta = elapsed / batches_done * (batches_total - batches_done) if batches_done > 0 else 0

                        sys.stdout.write(f"\n{'─'*100}\n")
                        sys.stdout.write(f"📊 Batch {batch_idx:3d}/{batches_total} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s\n")
                        sys.stdout.write(f"{'─'*100}\n")
                        sys.stdout.write(f"  Current Batch:  Loss={ppl_outputs['loss'].item():.4f}  PPL={ppl_outputs['perplexity'].item():.2f}\n")
                        sys.stdout.write(f"  Running Avg:    Loss={current_avg_loss:.4f} (±{current_std_loss:.4f})  PPL={current_avg_ppl:.2f} (±{current_std_ppl:.2f})\n")

                        # Show energy metrics if available
                        if self.test_energies:
                            energy_strs = []
                            for key, values in self.test_energies.items():
                                avg_energy = np.mean(values)
                                energy_strs.append(f"{key}={avg_energy:.2f}")
                            sys.stdout.write(f"  Energy Metrics: {' | '.join(energy_strs)}\n")

                        if enable_generation and 'prompt_ids' in batch:
                            sys.stdout.write(f"  Generation: Enabled (prompt→target)\n")

                        sys.stdout.write(f"{'─'*100}\n")
                        sys.stdout.flush()

                    # Log metrics
                    filtered_outputs = {k: v for k, v in ppl_outputs.items() if 'energy' not in k}
                    self.log_metrics(filtered_outputs, "test")

                else:
                    # For nanochat and other PPL evaluation tasks
                    # nanochat uses IterableDataset which returns (x, y) tuples
                    # During training: x[0].squeeze(dim=0) is used (see modeling_ebt.py:189)
                    # Convert to dict format for get_ppl
                    if isinstance(batch, tuple):
                        # batch is (x, y) from IterableDataset
                        x, y = batch
                        # x might have shape [1, B, S] or [B, S], squeeze to ensure [B, S]
                        if x.dim() == 3 and x.shape[0] == 1:
                            x = x.squeeze(0)  # [1, B, S] -> [B, S]
                        # Add channel dimension for get_ppl: [B, S] -> [B, 1, S]
                        batch_dict = {'input_ids': x.unsqueeze(1)}
                    else:
                        # batch is already a dict from DataLoader
                        batch_dict = batch

                    # Compute PPL and save sample outputs
                    ppl_outputs = get_ppl(self.model, batch_dict, self.hparams)

                    # Track metrics for averaging
                    self.test_losses.append(ppl_outputs['loss'].item())
                    self.test_perplexities.append(ppl_outputs['perplexity'].item())

                    # Track energy metrics if available
                    if not hasattr(self, 'test_energies'):
                        self.test_energies = {}
                    for key, value in ppl_outputs.items():
                        if 'energy' in key:
                            if key not in self.test_energies:
                                self.test_energies[key] = []
                            self.test_energies[key].append(value)

                    # Print progress every 5 batches (more frequent)
                    if batch_idx % 5 == 0:
                        import sys
                        import numpy as np
                        import time

                        # Calculate statistics
                        current_avg_loss = np.mean(self.test_losses)
                        current_avg_ppl = np.mean(self.test_perplexities)
                        current_std_loss = np.std(self.test_losses) if len(self.test_losses) > 1 else 0.0
                        current_std_ppl = np.std(self.test_perplexities) if len(self.test_perplexities) > 1 else 0.0

                        # Estimate time remaining
                        if not hasattr(self, 'test_start_time'):
                            self.test_start_time = time.time()
                        elapsed = time.time() - self.test_start_time
                        batches_done = batch_idx + 1
                        batches_total = self.hparams.limit_test_batches if self.hparams.limit_test_batches != 1 else 100
                        eta = elapsed / batches_done * (batches_total - batches_done) if batches_done > 0 else 0

                        sys.stdout.write(f"\n{'─'*100}\n")
                        sys.stdout.write(f"📊 Batch {batch_idx:3d}/{batches_total} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s\n")
                        sys.stdout.write(f"{'─'*100}\n")
                        sys.stdout.write(f"  Current Batch:  Loss={ppl_outputs['loss'].item():.4f}  PPL={ppl_outputs['perplexity'].item():.2f}\n")
                        sys.stdout.write(f"  Running Avg:    Loss={current_avg_loss:.4f} (±{current_std_loss:.4f})  PPL={current_avg_ppl:.2f} (±{current_std_ppl:.2f})\n")

                        # Show energy metrics if available
                        if self.test_energies:
                            energy_strs = []
                            for key, values in self.test_energies.items():
                                avg_energy = np.mean(values)
                                energy_strs.append(f"{key}={avg_energy:.2f}")
                            sys.stdout.write(f"  Energy Metrics: {' | '.join(energy_strs)}\n")

                        sys.stdout.write(f"{'─'*100}\n")
                        sys.stdout.flush()

                    # Save sample inputs/outputs to inference logger for first few batches
                    if batch_idx < 5 and hasattr(self, 'infer_logger'):
                        tokenizer = self.hparams.tokenizer_obj if hasattr(self.hparams, 'tokenizer_obj') else None
                        if tokenizer is not None:
                            # Wrap nanochat tokenizer if needed
                            from openebm.elm.nanochat_tokenizer_adapter import NanoChatTokenizerWrapper
                            if hasattr(tokenizer, 'enc') and hasattr(tokenizer.enc, 'encode'):
                                # It's a RustBPETokenizer, wrap it for HF compatibility
                                tokenizer = NanoChatTokenizerWrapper(tokenizer_obj=tokenizer)

                            # Log samples from this batch
                            sample_ids = batch_dict['input_ids'][:2]  # First 2 samples
                            for i, ids in enumerate(sample_ids):
                                # Decode full sequence
                                full_text = tokenizer.decode(ids.squeeze().tolist(), skip_special_tokens=True)

                                # Create output record
                                output_record = {
                                    "batch_idx": batch_idx,
                                    "sample_idx": i,
                                    "text": full_text[:500],  # First 500 chars
                                    "loss": ppl_outputs['loss'].item(),
                                    "perplexity": ppl_outputs['perplexity'].item(),
                                }
                                self.infer_logger.log_data(output_record)

                                # Also print to console for first few samples
                                if batch_idx < 3:
                                    import sys
                                    sys.stdout.write(f"\n{'─'*100}\n")
                                    sys.stdout.write(f"📄 Sample Text [Batch {batch_idx}, Sample {i}]:\n")
                                    sys.stdout.write(f"{'─'*100}\n")
                                    sys.stdout.write(f"{full_text[:300]}...\n")
                                    sys.stdout.write(f"{'─'*100}\n")
                                    sys.stdout.write(f"   Loss: {ppl_outputs['loss'].item():.4f} | PPL: {ppl_outputs['perplexity'].item():.2f}\n")
                                    sys.stdout.write(f"{'─'*100}\n\n")
                                    sys.stdout.flush()

                    # Log metrics (filter out energy metrics to avoid clutter)
                    filtered_outputs = {k: v for k, v in ppl_outputs.items() if 'energy' not in k}
                    self.log_metrics(filtered_outputs, "test")

            else:
                raise NotImplementedError(f"Inference mode not supported for modality {self.hparams.modality} yet")
        else:
            # Non-inference mode: just compute metrics.
            if self.hparams.modality == "NLP" and self.hparams.model_name == "ebt" and self.hparams.infer_ebt_advanced:
                # Special case: avoid inference mode but still use EBT advanced
                # inference to score log PPL / energies without emitting text
                # one token at a time.
                outputs = get_ppl(self.model, batch, self.hparams)
                self.log_metrics(outputs, "test")
            else:
                eval_step_dict = self.eval_step(batch, "test")
                self.log_metrics(eval_step_dict, "test")

    def on_test_epoch_end(self) -> None:
        """Print a comprehensive summary of the test epoch."""
        import sys
        import numpy as np
        import time

        total_time = time.time() - self.test_start_time
        has_ppl = len(self.test_losses) > 0
        has_generation = getattr(self, 'test_generation_count', 0) > 0

        if not has_ppl and not has_generation:
            return

        sys.stdout.write(f"\n\n")
        sys.stdout.write(f"{'='*100}\n")
        sys.stdout.write(f"{'EVALUATION RESULTS SUMMARY':^100}\n")
        sys.stdout.write(f"{'='*100}\n\n")

        # Dataset info
        sys.stdout.write(f"Dataset Information:\n")
        sys.stdout.write(f"   Dataset:           {self.hparams.dataset_name}\n")
        if has_ppl:
            sys.stdout.write(f"   Total Batches:     {len(self.test_losses)}\n")
            sys.stdout.write(f"   Total Samples:     ~{len(self.test_losses) * self.hparams.batch_size_per_device}\n")
        if has_generation:
            sys.stdout.write(f"   Generated Samples: {self.test_generation_count}\n")
        sys.stdout.write(f"   Batch Size:        {self.hparams.batch_size_per_device}\n")
        sys.stdout.write(f"   Context Length:    {self.hparams.context_length}\n\n")

        # Model info
        sys.stdout.write(f"Model Information:\n")
        sys.stdout.write(f"   Model Type:        {self.hparams.model_name.upper()}\n")
        sys.stdout.write(f"   Model Size:        {self.hparams.model_size}\n")
        if self.hparams.model_name == "ebt":
            sys.stdout.write(f"   MCMC Steps:        {self.hparams.mcmc_num_steps}\n")
            sys.stdout.write(f"   MCMC Step Size:    {self.hparams.mcmc_step_size}\n")
            sys.stdout.write(f"   EBT Type:          {self.hparams.ebt_type}\n")
        sys.stdout.write(f"\n")

        # Timing info
        sys.stdout.write(f"Performance:\n")
        sys.stdout.write(f"   Total Time:        {total_time:.2f}s\n")
        if has_ppl:
            samples_per_sec = len(self.test_losses) * self.hparams.batch_size_per_device / total_time
            sys.stdout.write(f"   Time per Batch:    {total_time/len(self.test_losses):.3f}s\n")
            sys.stdout.write(f"   Throughput:        {samples_per_sec:.2f} samples/s\n")
        if has_generation:
            gen_per_sec = self.test_generation_count / total_time
            sys.stdout.write(f"   Generation Speed:  {gen_per_sec:.2f} samples/s\n")
        sys.stdout.write(f"\n")

        if has_ppl:
            # Calculate statistics
            avg_loss = np.mean(self.test_losses)
            avg_ppl = np.mean(self.test_perplexities)
            std_loss = np.std(self.test_losses)
            std_ppl = np.std(self.test_perplexities)
            min_loss = np.min(self.test_losses)
            max_loss = np.max(self.test_losses)
            min_ppl = np.min(self.test_perplexities)
            max_ppl = np.max(self.test_perplexities)
            median_loss = np.median(self.test_losses)
            median_ppl = np.median(self.test_perplexities)
            p25_loss, p75_loss = np.percentile(self.test_losses, [25, 75])
            p25_ppl, p75_ppl = np.percentile(self.test_perplexities, [25, 75])

            # Loss statistics
            sys.stdout.write(f"Cross-Entropy Loss Statistics:\n")
            sys.stdout.write(f"   Mean:              {avg_loss:.4f}\n")
            sys.stdout.write(f"   Median:            {median_loss:.4f}\n")
            sys.stdout.write(f"   Std Dev:           {std_loss:.4f}\n")
            sys.stdout.write(f"   Min:               {min_loss:.4f}\n")
            sys.stdout.write(f"   Max:               {max_loss:.4f}\n")
            sys.stdout.write(f"   25th Percentile:   {p25_loss:.4f}\n")
            sys.stdout.write(f"   75th Percentile:   {p75_loss:.4f}\n\n")

            # Perplexity statistics
            sys.stdout.write(f"Perplexity (PPL) Statistics:\n")
            sys.stdout.write(f"   Mean:              {avg_ppl:.2f}\n")
            sys.stdout.write(f"   Median:            {median_ppl:.2f}\n")
            sys.stdout.write(f"   Std Dev:           {std_ppl:.2f}\n")
            sys.stdout.write(f"   Min:               {min_ppl:.2f}\n")
            sys.stdout.write(f"   Max:               {max_ppl:.2f}\n")
            sys.stdout.write(f"   25th Percentile:   {p25_ppl:.2f}\n")
            sys.stdout.write(f"   75th Percentile:   {p75_ppl:.2f}\n\n")

            # Energy statistics if available
            if hasattr(self, 'test_energies') and self.test_energies:
                sys.stdout.write(f"Energy Landscape Statistics:\n")
                for key, values in sorted(self.test_energies.items()):
                    avg_energy = np.mean(values)
                    std_energy = np.std(values)
                    min_energy = np.min(values)
                    max_energy = np.max(values)
                    sys.stdout.write(f"   {key}:\n")
                    sys.stdout.write(f"      Mean: {avg_energy:.4f} +/- {std_energy:.4f}\n")
                    sys.stdout.write(f"      Range: [{min_energy:.4f}, {max_energy:.4f}]\n")
                sys.stdout.write(f"\n")

            # ASCII histogram for PPL distribution
            sys.stdout.write(f"Perplexity Distribution (Histogram):\n")
            hist, bin_edges = np.histogram(self.test_perplexities, bins=10)
            max_count = max(hist)
            for i in range(len(hist)):
                bar_length = int(40 * hist[i] / max_count) if max_count > 0 else 0
                bar = '#' * bar_length
                sys.stdout.write(f"   [{bin_edges[i]:6.2f}-{bin_edges[i+1]:6.2f}]: {bar} ({hist[i]})\n")
            sys.stdout.write(f"\n")

            # Quality assessment
            sys.stdout.write(f"Quality Assessment:\n")
            if avg_ppl < 20:
                quality = "Excellent"
            elif avg_ppl < 40:
                quality = "Good"
            elif avg_ppl < 60:
                quality = "Fair"
            else:
                quality = "Needs Improvement"
            sys.stdout.write(f"   Overall: {quality} (PPL={avg_ppl:.2f})\n\n")

        # GSM8K / generation-only summary
        if has_generation and not has_ppl:
            sys.stdout.write(f"Generation Summary:\n")
            sys.stdout.write(f"   Total Generated:   {self.test_generation_count}\n")
            sys.stdout.write(f"   Total Time:        {total_time:.2f}s\n")
            sys.stdout.write(f"   Avg Time/Sample:   {total_time/self.test_generation_count:.2f}s\n\n")

        # Output files
        if hasattr(self.hparams, 'save_generation_logs_dir'):
            import os
            results_file = os.path.join(self.hparams.save_generation_logs_dir, "results.jsonl")
            if os.path.exists(results_file):
                num_samples = sum(1 for _ in open(results_file))
                sys.stdout.write(f"Output Files:\n")
                sys.stdout.write(f"   Results:           {results_file}\n")
                sys.stdout.write(f"   Num Samples:       {num_samples}\n\n")

        # Machine-parseable summary block for bash grep
        sys.stdout.write(f"{'='*100}\n")
        sys.stdout.write(f"[EVAL_SUMMARY] dataset={self.hparams.dataset_name}")
        if has_ppl:
            sys.stdout.write(f" loss={avg_loss:.4f} ppl={avg_ppl:.2f}")
        if has_generation:
            sys.stdout.write(f" generated={self.test_generation_count}")
        sys.stdout.write(f" time={total_time:.1f}s\n")
        sys.stdout.write(f"{'='*100}\n\n")
        sys.stdout.flush()

    def eval_step(self, batch: Any, phase: str, token_bytes: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Run the model's loss wrapper and return a log dict.

        :param batch: Input batch.
        :type batch: Any
        :param phase: ``"train"`` / ``"valid"`` / ``"test"``.
        :type phase: str
        :param token_bytes: Optional per-token byte-length table used for BPB.
        :type token_bytes: Optional[torch.Tensor]
        :return: Dict with a ``"loss"`` key (required for backward) plus any
            additional metrics the model chose to log.
        :rtype: Dict[str, Any]
        """
        things_to_log = self.model.forward_loss_wrapper(batch, phase, token_bytes=token_bytes)

        if len(self.metrics) > 0:
            raise NotImplementedError("Need to implement torchmetrics stuff, i.e. looping through self.torchmetrics_dict.keys(), checking to make sure 'phase in key', and updating based off predicted and labels i.e. self.torchmetrics_dict[key].update(logits, labels), more info https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html (just be careful make sure to detach logits before using them and only update current phase). recommended to possibly return things_to_log and logits from forward_loss_wrapper to do this easily")

        return things_to_log

    def forward(self, batch: Any) -> Any:
        """Call the wrapped model directly.

        :param batch: Input batch.
        :type batch: Any
        :return: Model output.
        :rtype: Any
        """
        return self.model(batch)

    def configure_optimizers(self) -> Any:
        """Lightning hook returning the optimizer and LR scheduler.

        :return: Whatever :meth:`configure_optimizers_nlp` returns.
        :rtype: Any
        """
        return self.configure_optimizers_nlp()

    def get_optimizer(self, optimizer_parameters: Any) -> torch.optim.Optimizer:
        """Build an optimizer matching ``self.hparams.optimizer``.

        :param optimizer_parameters: Parameter groups / iterable.
        :type optimizer_parameters: Any
        :return: The configured optimizer.
        :rtype: torch.optim.Optimizer
        """
        if self.hparams.optimizer == "lars":
            lars_exclude_bias_and_norm = None if not self.hparams.lars_exclude_bias_bn_wd else exclude_bias_and_norm
            optimizer = LARS(optimizer_parameters, lr=self.hparams.peak_learning_rate, weight_decay=self.hparams.weight_decay, momentum=self.hparams.beta1, eta=self.hparams.lars_trust_coeff, weight_decay_filter=lars_exclude_bias_and_norm, lars_adaptation_filter=lars_exclude_bias_and_norm)
        elif self.hparams.optimizer == "stableadamw":
            optimizer = StableAdamWUnfused(optimizer_parameters, betas=[self.hparams.beta1, self.hparams.beta2])
        else:
            optimizer = torch.optim.AdamW(optimizer_parameters, betas=[self.hparams.beta1, self.hparams.beta2])
        return optimizer

    def on_warm_up_finished(self) -> None:
        """Forward the warmup-finished signal to the model when supported."""
        if hasattr(self.model, 'warm_up_finished'):
            self.model.warm_up_finished()
            print("Warm up finished, calling self.model.warm_up_finished()")
        else:
            print("Warm up finished, no self.model.warm_up_finished() exists so not doing anything")

    def get_lr_scheduler(self, optimizer: torch.optim.Optimizer) -> Any:
        """Build an LR scheduler based on ``self.hparams`` flags.

        :param optimizer: Optimizer to attach the scheduler to.
        :type optimizer: torch.optim.Optimizer
        :return: The configured scheduler (raw or wrapped with warmup).
        :rtype: Any
        """
        # Dynamic weight decay paired with the LR schedule.
        enable_wd_decay = getattr(self.hparams, 'dynamic_wd', False)

        # Linear warmup / constant / linear warmdown schedule toggle.
        use_linear_warmdown = getattr(self.hparams, 'linear_warmdown', False)

        if use_linear_warmdown:
            warmup_ratio = getattr(self.hparams, 'warmup_ratio', 0.0)
            warmdown_ratio = getattr(self.hparams, 'warmdown_ratio', 0.5)
            final_lr_frac = getattr(self.hparams, 'final_lr_frac', 0.0)
            resume_warmup_steps = getattr(self.hparams, 'resume_warmup_steps', 0)

            lr_scheduler = WarmUpLinearWarmdownLR(
                optimizer,
                warmup_ratio=warmup_ratio,
                warmdown_ratio=warmdown_ratio,
                final_lr_frac=final_lr_frac,
                total_steps=self.hparams.max_scheduling_steps,
                warm_up_finished_func=self.on_warm_up_finished,
                enable_wd_decay=enable_wd_decay,
                resume_warmup_steps=resume_warmup_steps
            )
        else:
            # Cosine annealing schedule with optional linear warmup.
            cosine_annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_scheduling_steps - self.hparams.warm_up_steps,
                eta_min=self.hparams.peak_learning_rate / self.hparams.min_lr_scale
            )
            total_steps = self.hparams.max_scheduling_steps if enable_wd_decay else None

            lr_scheduler = WarmUpCosineAnnealingLR(
                optimizer,
                warm_up_steps=self.hparams.warm_up_steps,
                warm_up_base_lr_divider=self.hparams.warm_up_base_lr_divider,
                cosine_scheduler=cosine_annealing_scheduler,
                warm_up_finished_func=self.on_warm_up_finished,
                total_steps=total_steps,
                enable_wd_decay=enable_wd_decay
            )
        return lr_scheduler

    def get_optimizer_scheduler_dict(self, optimizer_parameters: Any) -> Dict[str, Any]:
        """Package optimizer + scheduler into Lightning's expected dict.

        :param optimizer_parameters: Parameter groups / iterable.
        :type optimizer_parameters: Any
        :return: Dict with ``optimizer`` and ``lr_scheduler`` entries
            (stepped every training step).
        :rtype: Dict[str, Any]
        """
        optimizer = self.get_optimizer(optimizer_parameters)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def _configure_muon_adamw_optimizer(self) -> Any:
        """Build the Muon + AdamW hybrid optimizer from ``nanochat/optim.py``.

        Parameter grouping strategy (mirrors NanoChat's ``setup_optimizer``):

        - ``alpha``: AdamW, high LR (``mcmc_step_size_lr_multiplier * peak_lr``),
          no weight decay. EBT-specific.
        - ``embeddings``: AdamW with an independent absolute LR that matches
          NanoChat's ``embedding_lr``; no weight decay.
        - ``vocab_to_embed``: AdamW with a conservative independent LR
          (EBT-specific); no weight decay.
        - Transformer scalar params (``ndim < 2``): AdamW with independent LR;
          no weight decay.
        - Transformer matrix params (``ndim >= 2``): Muon, grouped by tensor
          shape (Muon requires uniform shapes for stacking).

        LR design notes:

        - ``embeddings`` live outside the MCMC loop, so their gradient behavior
          matches NanoChat and they can safely use a larger LR.
        - ``vocab_to_embed`` is inside the MCMC loop
          (``autograd.grad(create_graph=True)`` second-order gradients) so we
          use a conservative LR.
        - Transformer scalars (RMSNorm) are inside the MCMC loop and need a
          moderately conservative LR.
        - When ``adamw_*_lr > 0``, an absolute LR is used; otherwise it falls
          back to ``peak_lr * mult``.

        Uses ``MuonAdamW`` (single-GPU variant) rather than ``DistMuonAdamW``
        because PyTorch Lightning DDP already handles gradient synchronization
        and ``DistMuonAdamW`` would double-manage distributed communication.

        :return: The configured optimizer instance.
        :rtype: Any
        """
        from nanochat.optim import MuonAdamW

        muon_lr = getattr(self.hparams, 'muon_lr', 0.02)
        muon_momentum = getattr(self.hparams, 'muon_momentum', 0.95)
        muon_ns_steps = getattr(self.hparams, 'muon_ns_steps', 5)
        muon_beta2 = getattr(self.hparams, 'muon_beta2', 0.95)
        adam_betas = (self.hparams.beta1, self.hparams.beta2)

        # AdamW LR: use the absolute value when provided, otherwise fall back
        # to ``peak_lr * mult``.
        adamw_embedding_lr = getattr(self.hparams, 'adamw_embedding_lr', -1)
        adamw_vocab_to_embed_lr = getattr(self.hparams, 'adamw_vocab_to_embed_lr', -1)
        adamw_scalar_lr = getattr(self.hparams, 'adamw_scalar_lr', -1)
        use_dmodel_scaling = getattr(self.hparams, 'adamw_dmodel_lr_scaling', False)

        # ``dmodel`` scaling: ``lr * (dim / 768) ** -0.5`` (NanoChat gpt.py:362).
        dmodel_scale = 1.0
        if use_dmodel_scaling:
            model_dim = self.hparams.embedding_dim
            dmodel_scale = (model_dim / 768) ** -0.5
            print(f"[Muon+AdamW] dmodel LR scaling: (dim={model_dim}/768)^-0.5 = {dmodel_scale:.4f}")

        # Compute per-group LR.
        if adamw_embedding_lr > 0:
            embedding_lr = adamw_embedding_lr * dmodel_scale
        else:
            embedding_lr_mult = getattr(self.hparams, 'embedding_lr_mult', 0.3)
            embedding_lr = self.hparams.peak_learning_rate * embedding_lr_mult

        if adamw_vocab_to_embed_lr > 0:
            vocab_to_embed_lr = adamw_vocab_to_embed_lr * dmodel_scale
        else:
            vocab_to_embed_lr_mult = getattr(self.hparams, 'vocab_to_embed_lr_mult', 0.1)
            vocab_to_embed_lr = self.hparams.peak_learning_rate * vocab_to_embed_lr_mult

        if adamw_scalar_lr > 0:
            scalar_lr = adamw_scalar_lr * dmodel_scale
        else:
            scalar_lr_mult = getattr(self.hparams, 'scalar_lr_mult', 0.5)
            scalar_lr = self.hparams.peak_learning_rate * scalar_lr_mult

        alpha_lr = self.hparams.mcmc_step_size_lr_multiplier * self.hparams.peak_learning_rate

        # Parameter collection.
        alpha_params = [self.model.alpha]
        embedding_params = list(self.model.embeddings.parameters())

        vocab_to_embed_params = []
        if hasattr(self.model, 'vocab_to_embed') and self.model.vocab_to_embed is not None:
            vocab_to_embed_params = list(self.model.vocab_to_embed.parameters())

        # VE parameters are collected separately and routed to AdamW (they
        # cannot be placed in a Muon group).
        ve_embed_params = []
        ve_gate_params = []
        transformer_matrix_params = []
        transformer_scalar_params = []
        for name, param in self.model.transformer.named_parameters():
            if 'value_embeds.' in name:
                ve_embed_params.append(param)
            elif 've_gate.' in name:
                ve_gate_params.append(param)
            elif param.ndim >= 2:
                transformer_matrix_params.append(param)
            else:
                transformer_scalar_params.append(param)

        # Build param_groups.
        param_groups = []

        if alpha_params:
            param_groups.append(dict(
                kind='adamw', params=alpha_params,
                lr=alpha_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0,
            ))
        if embedding_params:
            param_groups.append(dict(
                kind='adamw', params=embedding_params,
                lr=embedding_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0,
            ))
        if vocab_to_embed_params:
            param_groups.append(dict(
                kind='adamw', params=vocab_to_embed_params,
                lr=vocab_to_embed_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0,
            ))
        if transformer_scalar_params:
            param_groups.append(dict(
                kind='adamw', params=transformer_scalar_params,
                lr=scalar_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0,
            ))
        # VE embedding params: AdamW at ``embedding_lr`` (matches NanoChat).
        if ve_embed_params:
            param_groups.append(dict(
                kind='adamw', params=ve_embed_params,
                lr=embedding_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0,
            ))
        # VE gate params: AdamW at ``scalar_lr``.
        if ve_gate_params:
            param_groups.append(dict(
                kind='adamw', params=ve_gate_params,
                lr=scalar_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0,
            ))

        # Muon groups: shape-bucketed so Muon can stack parameters.
        shape_groups = {}
        for p in transformer_matrix_params:
            shape_groups.setdefault(p.shape, []).append(p)

        for shape in sorted(shape_groups.keys()):
            group_params = shape_groups[shape]
            param_groups.append(dict(
                kind='muon', params=group_params,
                lr=muon_lr, momentum=muon_momentum,
                ns_steps=muon_ns_steps, beta2=muon_beta2,
                weight_decay=self.hparams.weight_decay,
            ))

        # Lightning calls ``optimizer.step(closure=closure)`` but
        # ``MuonAdamW.step()`` does not accept a closure kwarg. Wrap to
        # bridge the calling convention.
        use_cpu_offload = getattr(self.hparams, 'cpu_offload_optimizer', False)
        if use_cpu_offload:
            class PLMuonAdamW(MuonAdamW):
                """MuonAdamW variant keeping optimizer state on CPU (AdamW + Muon)."""

                @torch.no_grad()
                def step(self, closure=None):
                    if closure is not None:
                        with torch.enable_grad():
                            closure()

                    # Dispatch per param group by kind.
                    for group in self.param_groups:
                        kind = group.get('kind')

                        if kind == 'adamw':
                            # AdamW: move exp_avg / exp_avg_sq per parameter.
                            for p in group['params']:
                                if p.grad is None:
                                    continue
                                state = self.state[p]
                                if not state:
                                    continue
                                for k in ('exp_avg', 'exp_avg_sq'):
                                    if k in state and state[k].device.type == 'cpu':
                                        state[k] = state[k].to(p.device, non_blocking=False)

                        elif kind == 'muon':
                            # Muon: group-level buffers live on params[0]'s state.
                            if not group['params']:
                                continue
                            p0 = group['params'][0]
                            state = self.state[p0]
                            if not state:
                                continue
                            for k in ('momentum_buffer', 'second_momentum_buffer'):
                                if k in state and state[k].device.type == 'cpu':
                                    state[k] = state[k].to(p0.device, non_blocking=False)

                    # Actual optimizer step — fused kernels require state on GPU.
                    super().step()

                    # Move all state back to CPU once the step is done.
                    for group in self.param_groups:
                        kind = group.get('kind')

                        if kind == 'adamw':
                            for p in group['params']:
                                state = self.state[p]
                                for k in ('exp_avg', 'exp_avg_sq'):
                                    if k in state and state[k].device.type != 'cpu':
                                        cpu_t = state[k].to('cpu', non_blocking=False)
                                        del state[k]
                                        state[k] = cpu_t

                        elif kind == 'muon':
                            if not group['params']:
                                continue
                            p0 = group['params'][0]
                            state = self.state[p0]
                            for k in ('momentum_buffer', 'second_momentum_buffer'):
                                if k in state and state[k].device.type != 'cpu':
                                    cpu_t = state[k].to('cpu', non_blocking=False)
                                    del state[k]
                                    state[k] = cpu_t

                    torch.cuda.synchronize()
        else:
            class PLMuonAdamW(MuonAdamW):
                """MuonAdamW wrapper compatible with Lightning's ``optimizer.step(closure=closure)``."""
                @torch.no_grad()
                def step(self, closure=None):
                    if closure is not None:
                        with torch.enable_grad():
                            closure()
                    super().step()

        optimizer = PLMuonAdamW(param_groups)

        # Lightning's LR scheduler needs ``initial_lr`` on every group.
        for group in optimizer.param_groups:
            group['initial_lr'] = group['lr']

        lr_scheduler = self.get_lr_scheduler(optimizer)

        # Logging.
        num_muon_params = sum(p.numel() for p in transformer_matrix_params)
        num_ve_params = (
            sum(p.numel() for p in ve_embed_params) +
            sum(p.numel() for p in ve_gate_params)
        )
        num_adamw_params = (
            sum(p.numel() for p in alpha_params) +
            sum(p.numel() for p in embedding_params) +
            sum(p.numel() for p in vocab_to_embed_params) +
            sum(p.numel() for p in transformer_scalar_params) +
            num_ve_params
        )
        print(f"=" * 80)
        print(f"[Muon+AdamW] hybrid optimizer enabled:")
        print(f"  Muon groups: {len(shape_groups)} (grouped by shape)")
        print(f"  Muon params: {num_muon_params:,} ({num_muon_params/(num_muon_params+num_adamw_params)*100:.1f}%)")
        print(f"  AdamW params: {num_adamw_params:,} ({num_adamw_params/(num_muon_params+num_adamw_params)*100:.1f}%)")
        if num_ve_params > 0:
            print(f"  VE params: {num_ve_params:,} (AdamW, embedding_lr)")
        print(f"  Muon LR: {muon_lr}, momentum: {muon_momentum}, ns_steps: {muon_ns_steps}, beta2: {muon_beta2}")
        print(f"  Alpha LR: {alpha_lr} (AdamW) [EBT-specific]")
        print(f"  Embedding LR: {embedding_lr} (AdamW)")
        print(f"  vocab_to_embed LR: {vocab_to_embed_lr} (AdamW) [EBT-specific, inside MCMC]")
        print(f"  Scalar LR: {scalar_lr} (AdamW)")
        if use_dmodel_scaling:
            print(f"  dmodel scaling: {dmodel_scale:.4f} (dim={self.hparams.embedding_dim})")
        for shape, params in sorted(shape_groups.items()):
            print(f"  Muon group shape={shape}: {len(params)} params")
        print(f"=" * 80)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def configure_optimizers_nlp(self) -> Any:
        """Build the optimizer + LR scheduler for NLP training.

        Supports three paths:

        - Muon + AdamW hybrid (``--optimizer muon_adamw``).
        - Layered LR (``--layered_lr``) with separate groups for
          ``alpha`` / embedding / ``vocab_to_embed`` / matrix / scalar params.
        - Baseline single-group AdamW.

        :return: Dict accepted by Lightning's ``configure_optimizers``.
        :rtype: Any
        """
        if self.hparams.model_name == "ebt":
            # Muon + AdamW hybrid, gated on ``--optimizer muon_adamw``.
            use_muon = getattr(self.hparams, 'optimizer', 'adamw') == 'muon_adamw'

            if use_muon:
                return self._configure_muon_adamw_optimizer()

            # Layered-LR path, gated on ``--layered_lr``.
            use_layered_lr = getattr(self.hparams, 'layered_lr', False)

            if use_layered_lr:
                # Layered parameter groups (mirrors NanoChat base_train.py).
                alpha_param = [self.model.alpha]
                embedding_params = list(self.model.embeddings.parameters())

                # ``vocab_to_embed`` acts as an unembedding projection.
                vocab_to_embed_params = []
                if hasattr(self.model, 'vocab_to_embed') and self.model.vocab_to_embed is not None:
                    vocab_to_embed_params = list(self.model.vocab_to_embed.parameters())

                # Split transformer params into matrix vs. scalar/vector.
                transformer_matrix_params = []
                transformer_scalar_params = []
                for name, param in self.model.transformer.named_parameters():
                    if param.ndim >= 2:
                        transformer_matrix_params.append(param)
                    else:
                        transformer_scalar_params.append(param)

                # Multipliers are overridable from the CLI.
                embedding_lr_mult = getattr(self.hparams, 'embedding_lr_mult', 0.3)
                vocab_to_embed_lr_mult = getattr(self.hparams, 'vocab_to_embed_lr_mult', 0.1)
                scalar_lr_mult = getattr(self.hparams, 'scalar_lr_mult', 0.5)

                optimizer_parameters = [
                    # Alpha: high LR, no weight decay.
                    {'params': alpha_param, 'weight_decay': 0.0,
                     'lr': self.hparams.mcmc_step_size_lr_multiplier * self.hparams.peak_learning_rate},
                    # Embedding: medium LR, no weight decay.
                    {'params': embedding_params, 'weight_decay': 0.0,
                     'lr': self.hparams.peak_learning_rate * embedding_lr_mult},
                    # vocab_to_embed: conservative LR, no weight decay.
                    {'params': vocab_to_embed_params, 'weight_decay': 0.0,
                     'lr': self.hparams.peak_learning_rate * vocab_to_embed_lr_mult},
                    # Transformer matrices: base LR.
                    {'params': transformer_matrix_params, 'weight_decay': self.hparams.weight_decay,
                     'lr': self.hparams.peak_learning_rate},
                    # Transformer scalars: higher LR, no weight decay.
                    {'params': transformer_scalar_params, 'weight_decay': 0.0,
                     'lr': self.hparams.peak_learning_rate * scalar_lr_mult},
                ]

                # Drop empty groups.
                optimizer_parameters = [p for p in optimizer_parameters if len(p['params']) > 0]

                print(f"[Layered LR] enabled:")
                print(f"  - Alpha LR: {self.hparams.mcmc_step_size_lr_multiplier * self.hparams.peak_learning_rate}")
                print(f"  - Embedding LR: {self.hparams.peak_learning_rate * embedding_lr_mult}")
                print(f"  - vocab_to_embed LR: {self.hparams.peak_learning_rate * vocab_to_embed_lr_mult}")
                print(f"  - Transformer Matrix LR: {self.hparams.peak_learning_rate}")
                print(f"  - Transformer Scalar LR: {self.hparams.peak_learning_rate * scalar_lr_mult}")
            else:
                # Default single-group path.
                alpha_param = self.model.alpha
                other_params = [param for name, param in self.model.named_parameters() if not any(keyword in name for keyword in ['alpha'])]
                assert len(other_params) > 1, "Could not gather model params correctly please investigate"

                optimizer_parameters = [
                    {'params': alpha_param, 'weight_decay': 0.0, 'lr': self.hparams.mcmc_step_size_lr_multiplier*self.hparams.peak_learning_rate},
                    {'params': other_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate}
                ]

            return self.get_optimizer_scheduler_dict(optimizer_parameters)

        elif self.hparams.model_name == "baseline_transformer":
            all_params = [param for _, param in self.model.named_parameters()]
            optimizer_parameters = [
                {'params': all_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate}
            ]
            return self.get_optimizer_scheduler_dict(optimizer_parameters)

        else:
            raise NotImplementedError(f"havent implemented configure optimizers for model {self.hparams.model_name}")


    def configure_optimizers_vid(self) -> Any:
        """Build an optimizer for the legacy VID modality.

        :return: Optimizer/scheduler dict.
        :rtype: Any
        """
        if self.hparams.model_name == "ebt":
            alpha_param = self.model.alpha
            encoder_params = list(self.model.image_encoder.parameters())
            other_params = [param for name, param in self.model.named_parameters() if not any(keyword in name for keyword in ['alpha', 'image_encoder'])]
            assert len(other_params) > 1, "Could not gather model params correctly please investigate"
            
            optimizer_parameters = [
                {'params': alpha_param, 'weight_decay': 0.0, 'lr': self.hparams.mcmc_step_size_lr_multiplier*self.hparams.peak_learning_rate},  # No weight decay for alpha
                {'params': encoder_params, 'weight_decay': 0.0, 'lr': 0.0},
                {'params': other_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate}  # Weight decay for other parameters
            ]
            return self.get_optimizer_scheduler_dict(optimizer_parameters)
            
        elif self.hparams.model_name == "baseline_transformer":
            encoder_params = list(self.model.image_encoder.parameters())
            other_params = [param for name, param in self.model.named_parameters() if not any(keyword in name for keyword in ['image_encoder'])]

            optimizer_parameters = [
                {'params': encoder_params, 'weight_decay': 0, 'lr': 0},
                {'params': other_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate}  # Weight decay for other parameters
            ]
            return self.get_optimizer_scheduler_dict(optimizer_parameters)
        
        else:
            raise NotImplementedError(f"havent implemented configure optimizers for model {self.hparams.model_name}")
        
    def configure_optimizers_img(self) -> Any:
        """Build an optimizer for the legacy IMG modality.

        :return: Optimizer/scheduler dict.
        :rtype: Any
        """
        if self.hparams.model_name == "ebt":
            alpha_param = self.model.alpha
            other_params = [param for name, param in self.model.named_parameters() if not any(keyword in name for keyword in ['alpha', 'image_encoder', 'text_encoder'])]
            assert len(other_params) > 1, "Could not gather model params correctly please investigate"

            optimizer_parameters = [
                {'params': alpha_param, 'weight_decay': 0.0, 'lr': self.hparams.mcmc_step_size_lr_multiplier*self.hparams.peak_learning_rate},  # No weight decay for alpha
                {'params': other_params, 'weight_decay': self.hparams.weight_decay, 'lr': self.hparams.peak_learning_rate} # Weight decay for other parameters
            ]

            return self.get_optimizer_scheduler_dict(optimizer_parameters)

        else:
            raise NotImplementedError(f"havent implemented configure optimizers for model {self.hparams.model_name}")

    def get_collate_fn(self) -> Optional[Any]:
        """Return the dataset-specific collate function, if any.

        :return: Collator instance or ``None`` when padding is unnecessary.
        :rtype: Optional[Any]
        """
        collate_fn = None if not self.hparams.modality == "NLP" else NLP_HF_Collator(self.hparams) #NOTE this assumes all modalities except NLP DONT have collator, may not be true in the future
        if self.hparams.dataset_name == "nlp_synthetic": #NOTE this is a hack to get around the fact that synthetic dataset cant return real text and thus cant use collate_fn
            collate_fn = None
        return collate_fn
    
    def  train_dataloader(self) -> Any:
        """Build the training dataloader.

        Routes to the NanoChat pretrain / NanoChat SFT / Sudoku SFT /
        mixed-sudoku dataloaders depending on ``hparams.dataset_name``.
        Applies a one-shot resume-state dict that is consumed on the first
        call after checkpoint load.

        :return: A dataloader producing ``(inputs, targets)`` batches.
        :rtype: Any
        """
        # Use tokenizer_obj for dataloader
        tokenizer = self.hparams.tokenizer_obj if hasattr(self.hparams, 'tokenizer_obj') else self.hparams.tokenizer

        # Dataloader position restored from checkpoint (used at most once).
        resume_state = getattr(self, '_dataloader_resume_state', None)
        self._dataloader_resume_state = None

        if getattr(self.hparams, 'dataset_name', 'nanochat') == 'nanochat_sft':
            train_dataloader = generate_sft_dataloader(
                tokenizer=tokenizer,
                batch_size=self.hparams.batch_size_per_device,
                max_len=self.hparams.context_length,
                max_iter=self.hparams.max_steps * self.hparams.accumulate_grad_batches,
                split="train",
                device=self.device,
                resume_state_dict=resume_state,
            )
        elif getattr(self.hparams, 'dataset_name', 'nanochat') == 'sudoku_sft':
            from openebm.elm.data.sudoku_dataset import generate_sudoku_sft_dataloader
            train_dataloader = generate_sudoku_sft_dataloader(
                tokenizer=tokenizer,
                batch_size=self.hparams.batch_size_per_device,
                max_len=self.hparams.context_length,
                max_iter=self.hparams.max_steps * self.hparams.accumulate_grad_batches,
                split="train",
                device=self.device,
                resume_state_dict=resume_state,
            )
        else:
            train_dataloader = generate_dataloader(
                tokenizer=tokenizer,
                batch_size=self.hparams.batch_size_per_device,
                max_len=self.hparams.context_length,
                max_iter=self.hparams.max_steps * self.hparams.accumulate_grad_batches, # one displayed epoch corresponds to ``self.hparams.max_steps`` training steps
                split="train",
                device=self.device,
                resume_state_dict=resume_state,
            )
        return train_dataloader

    def val_dataloader(self) -> Any:
        """Build the validation dataloader.

        .. note::

            The NanoChat train/val split is hardcoded in
            ``nanochat/dataloader.py`` — training uses
            ``parquet_paths[:-1]`` and validation uses the final shard
            (``parquet_paths[-1:]``). With 370 shards this is roughly
            369 train + 1 val (~0.27% validation). The
            ``--validation_split_pct`` flag does not control this split.

        :return: A dataloader producing validation batches.
        :rtype: Any
        """
        tokenizer = self.hparams.tokenizer_obj if hasattr(self.hparams, 'tokenizer_obj') else self.hparams.tokenizer

        if getattr(self.hparams, 'dataset_name', 'nanochat') == 'nanochat_sft':
            val_dataloader = generate_sft_dataloader(
                tokenizer=tokenizer,
                batch_size=self.hparams.batch_size_per_device,
                max_len=self.hparams.context_length,
                max_iter=self.hparams.val_steps,
                split="val",
                device=self.device,
            )
        elif getattr(self.hparams, 'dataset_name', 'nanochat') == 'sudoku_sft':
            from openebm.elm.data.sudoku_dataset import generate_sudoku_sft_dataloader
            val_dataloader = generate_sudoku_sft_dataloader(
                tokenizer=tokenizer,
                batch_size=self.hparams.batch_size_per_device,
                max_len=self.hparams.context_length,
                max_iter=self.hparams.val_steps,
                split="val",
                device=self.device,
            )
        else:
            val_dataloader = generate_dataloader(
                tokenizer=tokenizer,
                batch_size=self.hparams.batch_size_per_device,
                max_len=self.hparams.context_length,
                max_iter=self.hparams.val_steps,
                split="val",
                device=self.device,
                resume_state_dict=None,
            )

        return val_dataloader

    def test_dataloader(self) -> Any:
        """Build the test dataloader.

        Returns a map-style ``DataLoader`` for inference on GSM8K or the
        NanoChat shard evaluation dataset; otherwise falls back to
        :meth:`val_dataloader` (pretrain-mode default).

        :return: Dataloader for the configured test/inference dataset.
        :rtype: Any
        """
        # For inference mode with specific datasets, use DataLoader with collate_fn
        if self.hparams.execution_mode == "inference" and self.hparams.dataset_name == "gsm8k":
            test_ds = GSM8KDataset(self.hparams, split="test")
            return DataLoader(
                test_ds,
                batch_size=self.hparams.batch_size_per_device,
                num_workers=0,  # Keep 0 for simplicity
                collate_fn=self.get_collate_fn(),
                pin_memory=True,
                drop_last=False,
                shuffle=False
            )
        elif self.hparams.execution_mode == "inference" and self.hparams.dataset_name == "nanochat_shard_eval":
            # Custom NanoChat shard evaluation dataset
            from openebm.elm.dataset_nanochat_eval import NanoChatShardEvalDataset, collate_fn_nanochat_eval

            # Parse shard indices from comma-separated string
            shard_indices_str = getattr(self.hparams, 'eval_shard_indices', '0,15')
            if isinstance(shard_indices_str, str):
                shard_indices = [int(x.strip()) for x in shard_indices_str.split(',')]
            else:
                shard_indices = shard_indices_str

            max_samples_per_shard = getattr(self.hparams, 'max_samples_per_shard', 50)
            enable_generation = getattr(self.hparams, 'enable_nanochat_generation', True)
            generation_split_ratio = getattr(self.hparams, 'generation_split_ratio', 0.5)
            min_generation_length = getattr(self.hparams, 'min_generation_length', 64)

            test_ds = NanoChatShardEvalDataset(
                tokenizer=self.hparams.tokenizer_obj,
                context_length=self.hparams.context_length,
                shard_indices=shard_indices,
                max_samples_per_shard=max_samples_per_shard,
                enable_generation=enable_generation,
                generation_split_ratio=generation_split_ratio,
                min_generation_length=min_generation_length
            )

            return DataLoader(
                test_ds,
                batch_size=self.hparams.batch_size_per_device,
                num_workers=0,
                collate_fn=collate_fn_nanochat_eval,
                pin_memory=True,
                drop_last=False,
                shuffle=False
            )
        else:
            # Default: use val_dataloader for pretrain mode
            return self.val_dataloader()

    def log_metrics(self, metrics_dict: Dict[str, Any], phase: str, log_torchmetrics: bool = True) -> None:
        """Log metrics to W&B / Lightning with train vs. val semantics.

        Scalars from ``metrics_dict`` are logged; tensor values with more
        than one element are summarized as per-step mean/std to avoid a CPU
        sync. Train-phase entries are logged per step, validation entries
        per epoch so ``ModelCheckpoint`` sees an epoch-level ``valid_loss``.

        :param metrics_dict: Mapping of metric name to value (scalar or
            tensor).
        :type metrics_dict: Dict[str, Any]
        :param phase: ``"train"``, ``"valid"`` or ``"test"``.
        :type phase: str
        :param log_torchmetrics: When ``True`` and any torchmetrics are
            registered, also log their phase-filtered values.
        :type log_torchmetrics: bool
        :raises ValueError: If a value has an unsupported type.
        """
        # first log torchmetrics if there are any
        if log_torchmetrics and len(self.metrics) > 0:
            phase_dict = {key : value for key, value in self.torchmetrics_dict.items() if phase in key}
            self.log_dict(phase_dict, on_step = False, on_epoch = True) # for these always do on_epoch

        # log all other metrics in metrics_dict
        scalar_metrics = {}
        keys = list(metrics_dict.keys()) # Iterate over a copy of the keys to avoid modification issues during iteration
        for key in keys:
            value = metrics_dict[key]

            if isinstance(value, torch.Tensor) and value.numel() > 1: # histogram
                # Optimize: Log stats instead of Histogram to avoid CPU sync/copy
                self.logger.experiment.log({
                    f"{phase}_{key}_mean": value.detach().mean(),
                    f"{phase}_{key}_std": value.detach().std(),
                })

            elif isinstance(value, torch.Tensor) and value.dim() == 0: # two types of scalar, tensor (here) and int/float (below)
                scalar_metrics[f"{phase}_{key}"] = value.detach()
            elif isinstance(value, (int, float)):
                scalar_metrics[f"{phase}_{key}"] = value
            else:
                raise ValueError(f"unsupported type/format in log_metrics, type:, {type(value)}, key: {key}")

        if scalar_metrics:
            if phase == "train":
                # Train phase: on_step=True, on_epoch=False. Each train step
                # reports independently without cross-step accumulation.
                self.log_dict(scalar_metrics, sync_dist=True, prog_bar=True,
                              on_step=True, on_epoch=False)
            else:
                # Validation/test: on_step=False, on_epoch=True. Lightning
                # treats each val loop as an independent epoch, so
                # on_epoch=True averages only within the current val loop's
                # batches and does not carry over across val_check_interval
                # cycles. ``ModelCheckpoint`` reads epoch-level metrics in
                # ``on_validation_end``, so on_epoch=True is required to
                # surface ``valid_loss``.
                self.log_dict(scalar_metrics, sync_dist=True, prog_bar=True,
                              on_step=False, on_epoch=True)

        # === Extra diagnostic logging (train phase only) ===
        # lr/wd/alpha are logged only at training steps to avoid the
        # "log on epoch level in distributed setting" warning during
        # validation.
        if phase == "train" and len(self.trainer.optimizers) > 0:
            optimizer = self.trainer.optimizers[0]

            # Log the learning rate / weight decay of every param group.
            for i, group in enumerate(optimizer.param_groups):
                group_lr = group['lr']
                group_wd = group.get('weight_decay', 0)
                self.log(f"lr/param_group_{i}", group_lr, prog_bar=False,
                         on_step=True, on_epoch=False)
                self.log(f"wd/param_group_{i}", group_wd, prog_bar=False,
                         on_step=True, on_epoch=False)

            # Primary LR is the last param group (usually transformer params).
            current_lr = optimizer.param_groups[-1]['lr']
            self.log("Global_LR", current_lr, on_step=True, on_epoch=False)

            # Alpha parameter LR is the first param group.
            if len(optimizer.param_groups) > 1:
                alpha_lr = optimizer.param_groups[0]['lr']
                self.log("Alpha_LR", alpha_lr, on_step=True, on_epoch=False)

        # Alpha (MCMC step size) value — train only, to avoid validation warning.
        if phase == "train" and self.hparams.mcmc_step_size_learnable:
            self.log("Alpha_MCMC_Step_Size", self.model.alpha.detach(),
                     on_step=True, on_epoch=False)

        # Langevin dynamics noise — train only.
        if phase == "train" and self.hparams.langevin_dynamics_noise_learnable:
            self.log("Langevin_dynamics_noise", self.model.langevin_dynamics_noise_std.detach(),
                     on_step=True, on_epoch=False)

        # Training progress information (train phase only, rank-0 printing only).
        if phase == "train" and hasattr(self, 'trainer') and self.trainer is not None:
            import time as _time

            current_step = self.global_step
            max_steps = self.hparams.max_steps
            progress_pct = 100.0 * current_step / max_steps if max_steps > 0 else 0
            self.log("step", float(current_step), prog_bar=True, on_step=True, on_epoch=False)
            self.log("progress_pct", progress_pct, prog_bar=False, on_step=True, on_epoch=False)

            # GPU memory usage (when available).
            if torch.cuda.is_available():
                gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                self.log("gpu_mem_allocated_gb", gpu_mem_allocated, prog_bar=False,
                         on_step=True, on_epoch=False)
                self.log("gpu_mem_reserved_gb", gpu_mem_reserved, prog_bar=False,
                         on_step=True, on_epoch=False)

            # === Rich training log (rank-0 only to avoid DDP duplication) ===
            if self.trainer.is_global_zero:
                # --- Wall-clock statistics ---
                dt_ms = (getattr(self, '_last_dt', None) or 0.0) * 1000.0
                wall_elapsed = 0.0
                if self._train_start_time is not None:
                    wall_elapsed = _time.time() - self._train_start_time
                total_min = wall_elapsed / 60.0

                # --- LR ratio relative to peak_lr ---
                lrm = 1.0
                if len(self.trainer.optimizers) > 0:
                    opt = self.trainer.optimizers[0]
                    # Take the last param group (usually the transformer/muon main group).
                    cur_lr = opt.param_groups[-1]['lr']
                    peak_lr = self.hparams.peak_learning_rate
                    lrm = cur_lr / peak_lr if peak_lr > 0 else 1.0

                # --- tok/sec: tokens processed per second (global) ---
                # Tokens per optimizer step = num_gpus × batch_per_device × context_length × grad_accum
                num_gpus = getattr(self.hparams, 'num_gpus', 1)
                tokens_per_step = (num_gpus
                                   * self.hparams.batch_size_per_device
                                   * self.hparams.context_length
                                   * self.hparams.accumulate_grad_batches)
                tok_per_sec = tokens_per_step / (dt_ms / 1000.0) if dt_ms > 0 else 0.0

                # --- MFU (Model FLOP Utilization) ---
                # Following PaLM / nanoGPT conventions:
                # FLOPs per token ≈ 6 × num_params (forward + backward)
                # MFU = actual_tok_per_sec × flops_per_token / peak_flops_per_sec
                # H200 peak bfloat16 FLOPS ≈ 989 TFLOPS per GPU.
                try:
                    num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    flops_per_token = 6 * num_params  # forward + backward
                    # Peak FLOPS: H200=989T, A100=312T. Default to H200 here.
                    # peak_flops_per_gpu = 312e12
                    peak_flops_per_gpu = 989e12
                    gpu_peak_flops = num_gpus * peak_flops_per_gpu
                    actual_flops_per_sec = tok_per_sec * flops_per_token
                    mfu = 100.0 * actual_flops_per_sec / gpu_peak_flops if gpu_peak_flops > 0 else 0.0
                except Exception:
                    mfu = 0.0

                # --- Epoch ---
                epoch = self.current_epoch + 1

                # --- ETA ---
                if current_step > 0 and wall_elapsed > 0 and max_steps > 0:
                    steps_remaining = max_steps - current_step
                    sec_per_step = wall_elapsed / current_step
                    eta_min = steps_remaining * sec_per_step / 60.0
                    eta_str = f" | eta: {eta_min:.1f}m"
                else:
                    eta_str = ""

                # --- Current loss ---
                loss_val = metrics_dict.get('loss', 0.0)
                if isinstance(loss_val, torch.Tensor):
                    loss_val = loss_val.item()

                # --- Latest validation metrics ---
                last_valid = getattr(self, '_last_valid_metrics', {})
                valid_loss_val = last_valid.get('loss', None)
                valid_bpb_val = last_valid.get('bpb', None)
                valid_ppl_val = last_valid.get('perplexity', None)
                valid_str = ""
                if valid_loss_val is not None:
                    valid_str += f" | valid_loss: {valid_loss_val:.4f}"
                if valid_bpb_val is not None:
                    valid_str += f" | valid_bpb: {valid_bpb_val:.4f}"
                if valid_ppl_val is not None:
                    valid_str += f" | valid_ppl: {valid_ppl_val:.2f}"

                # --- Print ---
                print(
                    f"step {current_step:05d}/{max_steps} ({progress_pct:.2f}%) | "
                    f"loss: {loss_val:.6f}"
                    f"{valid_str} | "
                    f"lrm: {lrm:.2f} | "
                    f"dt: {dt_ms:.2f}ms | "
                    f"tok/sec: {tok_per_sec:,.0f} | "
                    f"mfu: {mfu:.2f} | "
                    f"epoch: {epoch} | "
                    f"total time: {total_min:.2f}m"
                    f"{eta_str}",
                    flush=True,
                )
