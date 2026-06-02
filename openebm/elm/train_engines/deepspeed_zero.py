"""Lightning DeepSpeed ZeRO integration for OpenEBM EBT training.

This module intentionally implements ZeRO-1/2 through Lightning's
``DeepSpeedStrategy`` instead of replacing the training loop with native
DeepSpeed. That keeps the existing ``ModelTrainer`` optimizer, scheduler,
checkpoint, validation, and logging hooks intact.

Compatibility policy:
- ZeRO-1/2 keep model parameters replicated, so they are the safest DeepSpeed
  stages for EBT's exact MCMC ``autograd.grad(..., create_graph=True)`` path.
- ZeRO-3 is reserved for future surrogate/first-order MCMC because parameter
  sharding can interact badly with higher-order autograd, similar to FSDP2.
"""

from __future__ import annotations

import inspect
import importlib.util
import json
import os
import platform
import sys
import types
from typing import Any


def _warn(message: str) -> None:
    print(f"[train_engine=deepspeed_zero] WARNING: {message}", flush=True)


def _zero_stage(engine: str) -> int:
    if engine == "zero-1":
        return 1
    if engine == "zero-2":
        return 2
    if engine == "zero-3":
        return 3
    raise ValueError(f"Unsupported DeepSpeed ZeRO engine={engine!r}")


def _import_deepspeed_strategy():
    local_repo = getattr(_import_deepspeed_strategy, "_local_repo", "")
    candidate_repos = [
        local_repo,
        "/mnt/shared-storage-user/puyuan/code/DeepSpeed",
        "/mnt/shared-storage-user/puyuan/code/deepspeed",
    ]
    for repo in candidate_repos:
        if repo and os.path.isdir(os.path.join(repo, "deepspeed")) and repo not in sys.path:
            sys.path.insert(0, repo)
            break

    if importlib.util.find_spec("deepspeed") is None:
        repo_hint = local_repo or "/mnt/shared-storage-user/puyuan/code/DeepSpeed"
        raise RuntimeError(
            "train_engine=zero-1/zero-2 requires the Python package `deepspeed`. "
            f"It is not importable, and deepspeed_repo_path={repo_hint!r} is not a usable source checkout. "
            "Install deepspeed in the training environment or provide a valid local DeepSpeed repo path."
        )

    _ensure_cpuinfo_compat()
    _ensure_hjson_compat()
    _ensure_einops_compat()
    _ensure_msgpack_compat()
    _ensure_distutils_hack_compat()

    try:
        import deepspeed  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "DeepSpeed is visible on sys.path but failed to import. This usually means the local "
            "DeepSpeed checkout is missing runtime dependencies in the active training environment. "
            "For the current checkout, install its requirements or at least the missing package shown "
            f"by the traceback. deepspeed_repo_path={local_repo!r}."
        ) from exc

    _patch_lightning_deepspeed_requirement_cache()

    try:
        from lightning.pytorch.strategies import DeepSpeedStrategy

        return DeepSpeedStrategy
    except Exception:
        try:
            from pytorch_lightning.strategies import DeepSpeedStrategy

            return DeepSpeedStrategy
        except Exception as exc:
            raise RuntimeError(
                "train_engine=zero-1/zero-2 requires Lightning with DeepSpeedStrategy. "
                "Install lightning[pytorch] and deepspeed in the training environment, "
                "or use train_engine=lightning_ddp."
            ) from exc


def _patch_lightning_deepspeed_requirement_cache() -> None:
    """Let Lightning accept a local DeepSpeed source checkout.

    Lightning checks package metadata via ``RequirementCache("deepspeed")`` in
    ``DeepSpeedStrategy.__init__``. A source checkout added to ``sys.path`` can
    be importable but still fail that metadata check because it was not
    installed as a distribution. We only patch after ``import deepspeed``
    succeeds, so subsequent DeepSpeed calls still use the real package.
    """
    try:
        import lightning.fabric.strategies.deepspeed as fabric_deepspeed
        import lightning.pytorch.strategies.deepspeed as pytorch_deepspeed
    except Exception:
        return

    fabric_deepspeed._DEEPSPEED_AVAILABLE = True
    fabric_deepspeed._DEEPSPEED_GREATER_EQUAL_0_16 = True
    pytorch_deepspeed._DEEPSPEED_AVAILABLE = True
    pytorch_deepspeed._DEEPSPEED_GREATER_EQUAL_0_16 = True


def _ensure_cpuinfo_compat() -> None:
    """Provide a tiny ``cpuinfo`` fallback when py-cpuinfo is unavailable.

    The local DeepSpeed checkout imports ``from cpuinfo import get_cpu_info`` at
    package import time. Some internal package indexes do not expose
    ``py-cpuinfo``. For ZeRO-1/2 GPU training, DeepSpeed only needs this import
    to succeed; the vendor string is mainly used by CPU optimizer paths.
    """
    if importlib.util.find_spec("cpuinfo") is not None:
        return

    def get_cpu_info() -> dict[str, str]:
        vendor = "unknown"
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("vendor_id"):
                        vendor = line.split(":", 1)[1].strip()
                        break
        except OSError:
            vendor = platform.processor() or "unknown"
        return {"vendor_id_raw": vendor}

    module = types.ModuleType("cpuinfo")
    module.get_cpu_info = get_cpu_info
    sys.modules["cpuinfo"] = module
    _warn("py-cpuinfo is unavailable; installed a minimal in-process cpuinfo.get_cpu_info fallback for DeepSpeed import.")


def _ensure_hjson_compat() -> None:
    """Provide a JSON-backed hjson fallback for standard JSON configs."""
    if importlib.util.find_spec("hjson") is not None:
        return

    module = types.ModuleType("hjson")
    module.load = json.load
    module.loads = json.loads
    module.dump = json.dump
    module.dumps = json.dumps
    sys.modules["hjson"] = module
    _warn("hjson is unavailable; installed a minimal JSON-backed hjson fallback for DeepSpeed import.")


def _ensure_einops_compat() -> None:
    """Provide a fail-fast einops fallback for unused DeepSpeed import paths."""
    if importlib.util.find_spec("einops") is not None:
        return

    def _unsupported(*args, **kwargs):
        raise RuntimeError(
            "einops is required for the DeepSpeed sequence-parallel/profiler path that was invoked. "
            "Install einops in the training environment before using that DeepSpeed feature."
        )

    module = types.ModuleType("einops")
    module.rearrange = _unsupported
    module.reduce = _unsupported
    module.repeat = _unsupported
    module.einsum = _unsupported
    sys.modules["einops"] = module
    _warn("einops is unavailable; installed a fail-fast fallback so unused DeepSpeed import paths can load.")


def _ensure_msgpack_compat() -> None:
    """Provide a fail-fast msgpack fallback for unused pipeline import paths."""
    if importlib.util.find_spec("msgpack") is not None:
        return

    def _unsupported(*args, **kwargs):
        raise RuntimeError(
            "msgpack is required for the DeepSpeed pipeline communication path that was invoked. "
            "Install msgpack in the training environment before using pipeline parallelism."
        )

    module = types.ModuleType("msgpack")
    module.packb = _unsupported
    module.unpackb = _unsupported
    module.pack = _unsupported
    module.unpack = _unsupported
    module.dumps = _unsupported
    module.loads = _unsupported
    sys.modules["msgpack"] = module
    _warn("msgpack is unavailable; installed a fail-fast fallback so unused DeepSpeed pipeline imports can load.")


def _ensure_distutils_hack_compat() -> None:
    """Provide the minimal package imported by setuptools in this env."""
    try:
        if importlib.util.find_spec("_distutils_hack.override") is not None:
            return
    except ModuleNotFoundError:
        pass

    package = types.ModuleType("_distutils_hack")
    package.__path__ = []
    override = types.ModuleType("_distutils_hack.override")
    sys.modules["_distutils_hack"] = package
    sys.modules["_distutils_hack.override"] = override
    _warn("setuptools is missing _distutils_hack.override; installed a minimal in-process fallback.")


def _load_zero_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _filter_strategy_kwargs(strategy_cls, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs not accepted by the installed Lightning version."""
    try:
        signature = inspect.signature(strategy_cls)
    except (TypeError, ValueError):
        return kwargs

    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return kwargs

    accepted = {name for name in params if name != "self"}
    dropped = sorted(set(kwargs) - accepted)
    if dropped:
        _warn(
            "Installed Lightning DeepSpeedStrategy does not expose kwargs "
            f"{dropped}; dropping them. Check the runtime Lightning version if this is unexpected."
        )
    return {key: value for key, value in kwargs.items() if key in accepted}


def _build_deepspeed_strategy(args, stage: int):
    strategy_cls = _import_deepspeed_strategy()

    config_path = getattr(args, "zero_config", "") or ""
    if config_path:
        config = _load_zero_config(config_path)
        if not isinstance(config.get("zero_optimization"), dict):
            config["zero_optimization"] = {}
        configured_stage = (
            config.get("zero_optimization", {}).get("stage")
            if isinstance(config.get("zero_optimization"), dict)
            else None
        )
        if configured_stage is not None and int(configured_stage) != stage:
            raise ValueError(
                f"--train_engine requests ZeRO stage {stage}, but zero_config={config_path} "
                f"contains zero_optimization.stage={configured_stage}."
            )
        config["zero_optimization"]["stage"] = stage
        kwargs = {"config": config}
    else:
        kwargs = {
            "stage": stage,
            "offload_optimizer": getattr(args, "zero_cpu_offload_optimizer", False),
            "offload_parameters": getattr(args, "zero_cpu_offload_parameters", False) if stage == 3 else False,
        }
        if int(getattr(args, "zero_allgather_bucket_size", 0) or 0) > 0:
            kwargs["allgather_bucket_size"] = int(args.zero_allgather_bucket_size)
        if int(getattr(args, "zero_reduce_bucket_size", 0) or 0) > 0:
            kwargs["reduce_bucket_size"] = int(args.zero_reduce_bucket_size)
        if getattr(args, "zero_contiguous_gradients", False):
            kwargs["contiguous_gradients"] = True

    kwargs = _filter_strategy_kwargs(strategy_cls, kwargs)
    return strategy_cls(**kwargs)


def prepare_deepspeed_zero_args(args, engine: str) -> None:
    """Mutate args before ``ModelTrainer`` construction for ZeRO engines."""
    stage = _zero_stage(engine)
    args.train_engine = engine

    mcmc_gradient_mode = getattr(args, "mcmc_gradient_mode", "second_order")
    if stage == 3 and mcmc_gradient_mode != "first_order_cd":
        raise NotImplementedError(
            "train_engine=zero-3 shards model parameters and is only enabled with "
            "--mcmc_gradient_mode first_order_cd in this phase. Use zero-1/zero-2 "
            "for exact second-order EBT training."
        )
    if stage == 3:
        _warn(
            "Enabling experimental ZeRO-3 with first_order_cd. Parameters are sharded, "
            "so exact second-order MCMC remains unsupported on this path."
        )

    _import_deepspeed_strategy._local_repo = getattr(args, "deepspeed_repo_path", "")

    if getattr(args, "compile_model", False) and (
        getattr(args, "zero_disable_compile", False) or not getattr(args, "zero_allow_compile", False)
    ):
        _warn(
            "Disabling torch.compile for DeepSpeed ZeRO. ZeRO-1/2 keep parameters replicated, "
            "but EBT MCMC still uses create_graph=True and retained activation graphs; compile is opt-in "
            "via --zero_allow_compile."
        )
        args.compile_model = False
        args.compile_mode = "disabled"

    if getattr(args, "zero_force_truncate_mcmc", False) and not getattr(args, "truncate_mcmc", False):
        _warn("Forcing truncate_mcmc=True to reduce retained high-order graph memory under ZeRO.")
        args.truncate_mcmc = True

    if getattr(args, "optimizer", "adamw") == "muon_adamw" and not getattr(args, "zero_allow_muon_adamw", False):
        _warn(
            "MuonAdamW is not enabled for DeepSpeed ZeRO-1/2 by default because its custom "
            "optimizer state layout has not been validated with ZeRO partitioning. "
            "Falling back to layered AdamW param groups."
        )
        args.optimizer = "adamw"
        args.layered_lr = True

    if stage in {1, 2} and getattr(args, "zero_cpu_offload_parameters", False):
        _warn("zero_cpu_offload_parameters is parsed for the reserved ZeRO-3 path and ignored for ZeRO-1/2.")
        args.zero_cpu_offload_parameters = False

    if getattr(args, "no_mcmc_detach", False):
        _warn(
            "no_mcmc_detach=True keeps graph history across MCMC steps. ZeRO-1/2 should be "
            "more compatible than parameter-sharded engines, but memory can still grow quickly."
        )

    if int(getattr(args, "mcmc_num_steps", 0) or 0) > 2:
        _warn("mcmc_num_steps>2 may dominate memory with retained second-order activations; monitor peak memory.")

    args.distributed_strategy = _build_deepspeed_strategy(args, stage)
    print(
        f"[train_engine={engine}] Using Lightning DeepSpeedStrategy ZeRO stage {stage}; "
        "parameters remain replicated for exact second-order EBT compatibility.",
        flush=True,
    )
