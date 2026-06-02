"""Shared training-engine dispatch.

This layer keeps engine-specific policy out of ``train.py`` while preserving the
current Lightning/DDP behavior unless ``--train_engine`` is explicitly changed.
"""

from __future__ import annotations


DEFAULT_TRAIN_ENGINE = "lightning_ddp"
SUPPORTED_TRAIN_ENGINES = {
    "lightning_ddp",
    "fsdp2",
    "zero-1",
    "zero-2",
    "zero-3",
    "deepspeed-zero1",
    "deepspeed-zero2",
    "deepspeed-zero3",
    "megatron",
}


def _engine_name(args) -> str:
    engine = getattr(args, "train_engine", DEFAULT_TRAIN_ENGINE) or DEFAULT_TRAIN_ENGINE
    aliases = {
        "deepspeed-zero1": "zero-1",
        "deepspeed-zero2": "zero-2",
        "deepspeed-zero3": "zero-3",
    }
    return aliases.get(engine, engine)


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
    if engine in {"zero-1", "zero-2", "zero-3"}:
        from openebm.elm.train_engines.deepspeed_zero import prepare_deepspeed_zero_args

        prepare_deepspeed_zero_args(args, engine)
        return
    raise NotImplementedError(
        f"train_engine={engine!r} is reserved for future work. "
        "Implemented engines: lightning_ddp, fsdp2, zero-1, zero-2."
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
    if engine in {"zero-1", "zero-2", "zero-3"}:
        model_trainer._openebm_train_engine = engine
        return
    raise NotImplementedError(
        f"train_engine={engine!r} is reserved for future work. "
        "Implemented engines: lightning_ddp, fsdp2, zero-1, zero-2."
    )
