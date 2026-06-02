"""First-stage FSDP2 integration for EBT training.

MVP policy:
- Use native composable FSDP2 when ``--train_engine fsdp2`` is selected.
- Wrap transformer blocks only by default.
- Keep EBT-specific small/high-order modules replicated: embeddings,
  vocab_to_embed, alpha, Langevin noise scalar, replay buffer, and transformer
  root-only parameters such as VE embeddings/final norm/final energy head.
- Guard against combinations that are risky for
  ``torch.autograd.grad(..., create_graph=True)`` in the MCMC loop.
"""

from __future__ import annotations

import os
from typing import Any


def _as_bool_string(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _warn(message: str) -> None:
    print(f"[train_engine=fsdp2] WARNING: {message}", flush=True)


def prepare_fsdp2_args(args) -> None:
    """Mutate args before ``ModelTrainer`` is constructed.

    This is deliberately conservative because EBT's MCMC loop has a second-order
    autograd path. The current default Lightning/DDP path does not call this.
    """
    _set_local_cuda_device()

    if getattr(args, "compile_model", False) and getattr(args, "fsdp_disable_compile", True):
        _warn("Disabling torch.compile for FSDP2 MVP; EBT MCMC uses create_graph=True.")
        args.compile_model = False
        args.compile_mode = "disabled"

    if getattr(args, "fsdp_force_truncate_mcmc", False) and not getattr(args, "truncate_mcmc", False):
        _warn("Forcing truncate_mcmc=True to keep only the final MCMC step on the high-order path.")
        args.truncate_mcmc = True

    if getattr(args, "no_mcmc_detach", False):
        _warn(
            "no_mcmc_detach=True keeps graph history across MCMC steps and is high risk under FSDP2. "
            "Proceed only for explicit experiments."
        )

    randomized_steps = int(getattr(args, "randomize_mcmc_num_steps", 0) or 0)
    if randomized_steps > 0:
        _warn(
            "randomize_mcmc_num_steps>0 creates variable retained graph length under FSDP2. "
            "MVP validation should use randomize_mcmc_num_steps=0."
        )

    if int(getattr(args, "mcmc_num_steps", 0) or 0) > 2:
        _warn("mcmc_num_steps>2 may cause repeated FSDP all-gather and large retained graphs.")

    if getattr(args, "fsdp_first_order_mcmc_debug", False):
        _warn(
            "fsdp_first_order_mcmc_debug=True will force MCMC autograd.grad(create_graph=False). "
            "This is only for isolating FSDP2 second-order autograd failures and is not equivalent "
            "to the normal EBT training objective."
        )

    if _as_bool_string(getattr(args, "fsdp_reshard_after_forward", "false")):
        _warn(
            "Forcing fsdp_reshard_after_forward=false for the FSDP2 MVP. "
            "EBT training uses torch.autograd.grad(create_graph=True); resharding "
            "wrapped transformer blocks immediately after forward can invalidate the "
            "higher-order backward path and has produced CUDA illegal memory accesses."
        )
        args.fsdp_reshard_after_forward = "false"

    if getattr(args, "fsdp_activation_checkpointing", "off") != "off":
        _warn(
            "FSDP2 activation checkpointing is not enabled by this MVP because create_graph=True "
            "can make checkpoint recompute retain more graph state."
        )
        args.fsdp_activation_checkpointing = "off"

    if getattr(args, "fsdp_state_dict_type", "sharded") != "sharded":
        _warn(
            "FSDP2 MVP only supports the normal Lightning/FSDP2 sharded checkpoint path. "
            "Full export for eval/chat is documented as follow-up."
        )
        args.fsdp_state_dict_type = "sharded"

    if getattr(args, "optimizer", "adamw") == "muon_adamw" and not getattr(args, "fsdp_allow_muon_adamw", False):
        _warn(
            "MuonAdamW is not enabled for FSDP2 MVP because its shape-stacked parameter groups "
            "need DTensor validation. Falling back to layered AdamW param groups."
        )
        args.optimizer = "adamw"
        args.layered_lr = True

    # Native composable FSDP2 is already distributed. In torchrun mode, each
    # process should run a single-device Lightning trainer instead of being
    # wrapped again by Lightning DDP.
    # Lightning does not accept ``strategy=None`` when the keyword is passed.
    # Use "auto" so the Trainer stays single-device per torchrun rank while the
    # native composable FSDP2 wrapper handles distributed parameter sharding.
    args.distributed_strategy = "auto"
    if os.getenv("WORLD_SIZE") is not None:
        # Lightning's ``devices=1`` means "use visible cuda:0" in every
        # torchrun child process. That fights native FSDP2, whose current device
        # is cuda:LOCAL_RANK, and leaves unwrapped root params on cuda:0. Pass an
        # explicit device list so Lightning moves replicated parameters to the
        # same local device used by FSDP2.
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        args.gpus = [local_rank]
        args.num_nodes = 1


def _init_process_group_for_fsdp2() -> None:
    import torch
    import torch.distributed as dist

    if not dist.is_available() or dist.is_initialized():
        return

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size <= 1:
        return

    _set_local_cuda_device()

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    print(f"[train_engine=fsdp2] Initialized torch.distributed process group backend={backend}", flush=True)


def _set_local_cuda_device() -> None:
    """Set the current CUDA device before model construction on torchrun ranks.

    Several OpenEBM modules construct parameters with ``device=self.device`` or
    default CUDA context semantics. If rank>0 builds the model before calling
    ``torch.cuda.set_device(local_rank)``, replicated root parameters can remain
    on cuda:0 while activations are on cuda:local_rank.
    """
    if os.getenv("LOCAL_RANK") is None:
        return
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))


def _local_cuda_device():
    import torch

    if torch.cuda.is_available() and os.getenv("LOCAL_RANK") is not None:
        return torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))
    return None


def _move_to_local_device(module) -> None:
    device = _local_cuda_device()
    if device is None:
        return
    module.to(device)
    print(f"[train_engine=fsdp2] Moved model to local device {device} before FSDP2 wrapping.", flush=True)


def _warn_if_param_devices_mismatch(module, expected_device) -> None:
    if expected_device is None:
        return
    bad = []
    for name, param in module.named_parameters():
        if param.device != expected_device:
            bad.append((name, str(param.device)))
            if len(bad) >= 8:
                break
    if bad:
        _warn(
            "Some parameters are not on the local FSDP2 device "
            f"{expected_device}: {bad}. This may indicate Lightning/device placement drift."
        )


def _fsdp2_kwargs(args) -> dict[str, Any]:
    """Build conservative kwargs accepted by the composable FSDP2 API.

    API support varies across PyTorch versions. The MVP uses only stable core
    arguments and lets Lightning precision control dtype. CPU offload and mixed
    precision flags are parsed and logged, but not passed unless the local
    PyTorch FSDP2 API is extended and tested.
    """
    kwargs: dict[str, Any] = {}
    reshard = getattr(args, "fsdp_reshard_after_forward", "false")
    kwargs["reshard_after_forward"] = _as_bool_string(reshard)

    if getattr(args, "fsdp_cpu_offload", False):
        _warn("fsdp_cpu_offload is parsed but not enabled in the MVP wrapper.")
    if getattr(args, "fsdp_mixed_precision", "bf16") != "bf16":
        _warn("fsdp_mixed_precision is parsed for future use; current dtype is controlled by Lightning precision.")
    if getattr(args, "fsdp_sharding_strategy", "full_shard") != "full_shard":
        _warn("Only full_shard-style FSDP2 wrapping is implemented in the MVP.")
    return kwargs


def _param_dtypes(module, recurse: bool = True) -> set:
    return {p.dtype for p in module.parameters(recurse=recurse) if p.requires_grad}


def apply_fsdp2_wrapping(model_trainer, args) -> None:
    """Apply native composable FSDP2 wrapping after any legacy finetune load."""
    try:
        from torch.distributed._composable.fsdp import fully_shard
    except Exception as exc:
        raise RuntimeError(
            "train_engine=fsdp2 requires a PyTorch build with "
            "torch.distributed._composable.fsdp.fully_shard. "
            "Use train_engine=lightning_ddp or install a newer PyTorch."
        ) from exc

    _init_process_group_for_fsdp2()

    model = model_trainer.model
    _move_to_local_device(model_trainer)
    expected_device = _local_cuda_device()
    transformer = getattr(model, "transformer", None)
    if transformer is None:
        raise ValueError("FSDP2 wrapping expected model_trainer.model.transformer to exist.")

    layers = getattr(transformer, "layers", None)
    if layers is None:
        raise ValueError("FSDP2 wrap_policy=transformer_block requires transformer.layers.")

    kwargs = _fsdp2_kwargs(args)
    wrap_policy = getattr(args, "fsdp_wrap_policy", "transformer_block")

    if wrap_policy == "transformer_block":
        for layer in layers:
            fully_shard(layer, **kwargs)
        print(
            "[train_engine=fsdp2] Wrapped transformer blocks only. "
            "Kept transformer root leftovers replicated because use_ve can mix bf16 value_embeds "
            "with fp32 norm/final_layer parameters, and FSDP requires uniform dtype per group. "
            "Also kept embeddings/vocab_to_embed/alpha/MCMC replay buffer replicated for high-order autograd safety.",
            flush=True,
        )
    elif wrap_policy == "transformer_root":
        dtypes = _param_dtypes(transformer, recurse=True)
        if len(dtypes) > 1:
            raise ValueError(
                "fsdp_wrap_policy=transformer_root is not safe for the current transformer because "
                f"it contains mixed trainable parameter dtypes: {sorted(str(dtype) for dtype in dtypes)}. "
                "Use fsdp_wrap_policy=transformer_block so VE/root parameters stay replicated, "
                "or make transformer parameters uniform dtype before root wrapping."
            )
        fully_shard(transformer, **kwargs)
        print(
            "[train_engine=fsdp2] Wrapped transformer root only. "
            "Kept embeddings/vocab_to_embed/alpha/MCMC replay buffer replicated.",
            flush=True,
        )
    elif wrap_policy == "none":
        _warn("fsdp_wrap_policy=none selected; no FSDP2 wrapping was applied.")
    else:
        raise ValueError(f"Unsupported fsdp_wrap_policy={wrap_policy!r}")

    _warn_if_param_devices_mismatch(model_trainer, expected_device)
    model_trainer._openebm_train_engine = "fsdp2"
