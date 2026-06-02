"""Shared training-engine dispatch.

This layer keeps engine-specific policy out of ``train.py`` while preserving the
current Lightning/DDP behavior unless ``--train_engine`` is explicitly changed.
"""

from __future__ import annotations


DEFAULT_TRAIN_ENGINE = "lightning_ddp"
SUPPORTED_TRAIN_ENGINES = {"lightning_ddp", "fsdp2", "megatron", "deepspeed-zero3"}


def _engine_name(args) -> str:
    return getattr(args, "train_engine", DEFAULT_TRAIN_ENGINE) or DEFAULT_TRAIN_ENGINE


def prepare_train_engine(args) -> None:
    """Apply pre-model-construction guardrails for the selected engine."""
    engine = _engine_name(args)
    if engine not in SUPPORTED_TRAIN_ENGINES:
        raise ValueError(f"Unsupported train_engine={engine!r}; supported={sorted(SUPPORTED_TRAIN_ENGINES)}")

    if engine == DEFAULT_TRAIN_ENGINE:
        return
    if engine == "fsdp2":
        from openebm.elm.train_engines.fsdp2 import prepare_fsdp2_args

        prepare_fsdp2_args(args)
        return
    raise NotImplementedError(
        f"train_engine={engine!r} is reserved for future work. "
        "First-stage implementation only supports train_engine=fsdp2."
    )


def apply_train_engine(model_trainer, args) -> None:
    """Apply post-model-construction wrapping for the selected engine."""
    engine = _engine_name(args)
    if engine == DEFAULT_TRAIN_ENGINE:
        return
    if engine == "fsdp2":
        from openebm.elm.train_engines.fsdp2 import apply_fsdp2_wrapping

        apply_fsdp2_wrapping(model_trainer, args)
        return
    raise NotImplementedError(
        f"train_engine={engine!r} is reserved for future work. "
        "First-stage implementation only supports train_engine=fsdp2."
    )
