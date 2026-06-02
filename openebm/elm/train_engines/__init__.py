"""Training engine integration points.

The default engine intentionally preserves the existing Lightning/DDP path.
Optional engines live behind explicit CLI flags so importing this package does
not require FSDP, DeepSpeed, or Megatron dependencies on the default path.
"""

from openebm.elm.train_engines.base import apply_train_engine, prepare_train_engine

__all__ = ["apply_train_engine", "prepare_train_engine"]
