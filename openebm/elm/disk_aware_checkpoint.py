"""Disk-space-aware ``ModelCheckpoint`` that evicts old checkpoints on demand.

Wraps Lightning's :class:`ModelCheckpoint` so that every save is preceded by
a free-space check. If the remaining space is below ``min_free_gb`` the
callback force-evicts every checkpoint except ``last.ckpt`` before saving.
"""
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from lightning.pytorch.callbacks import ModelCheckpoint


class DiskAwareCheckpoint(ModelCheckpoint):
    """``ModelCheckpoint`` that evicts old checkpoints when disk is full."""

    def __init__(self, *args: Any, min_free_gb: int = 50, **kwargs: Any) -> None:
        """Initialize the checkpoint callback.

        :param min_free_gb: Minimum free space (in gigabytes) required before
            writing a new checkpoint. When the target directory has less free
            space, every ``*.ckpt`` file except ``last.ckpt`` is deleted
            first.
        :type min_free_gb: int
        """
        super().__init__(*args, **kwargs)
        self.min_free_gb = min_free_gb

    def _check_disk_space(self) -> float:
        """Return the free space available under ``self.dirpath`` in GiB.

        :return: Free space in gigabytes, or ``inf`` when no directory is
            configured.
        :rtype: float
        """
        if self.dirpath:
            Path(self.dirpath).mkdir(parents=True, exist_ok=True)
            stat = shutil.disk_usage(self.dirpath)
            free_gb = stat.free / (1024**3)
            return free_gb
        return float('inf')

    def _cleanup_old_checkpoints(self) -> None:
        """Delete every ``*.ckpt`` under ``self.dirpath`` except ``last.ckpt``."""
        if not self.dirpath:
            return

        ckpt_dir = Path(self.dirpath)
        if not ckpt_dir.exists():
            return

        ckpts = [f for f in ckpt_dir.glob("*.ckpt") if f.name != "last.ckpt"]

        for ckpt in ckpts:
            try:
                ckpt.unlink()
                print(f"[DiskAware] Removed old checkpoint: {ckpt.name}")
            except Exception as e:
                print(f"[DiskAware] Failed to remove {ckpt.name}: {e}")

    def _save_checkpoint(self, trainer: Any, filepath: str) -> None:
        """Check disk space, evict if needed, then delegate to the parent.

        :param trainer: Lightning trainer driving the save.
        :type trainer: Any
        :param filepath: Destination path for the checkpoint.
        :type filepath: str
        """
        free_gb = self._check_disk_space()

        if free_gb < self.min_free_gb:
            print(f"[DiskAware] Low disk space: {free_gb:.1f}GB < {self.min_free_gb}GB")
            print(f"[DiskAware] Cleaning up old checkpoints...")
            self._cleanup_old_checkpoints()

            free_gb = self._check_disk_space()
            print(f"[DiskAware] Free space after cleanup: {free_gb:.1f}GB")

        super()._save_checkpoint(trainer, filepath)

    def _temporarily_align_completed_for_save(self, trainer: Any) -> bool:
        """Temporarily bump ``batch_progress.completed`` so the checkpoint
        captures the post-step boundary.

        This avoids a one-batch drift on resume: when saving mid-step, the
        checkpoint would otherwise see the pre-increment state and rewind by
        one batch on reload.

        :param trainer: Lightning trainer whose progress is inspected.
        :type trainer: Any
        :return: ``True`` if the counters were adjusted and need to be rolled
            back by the caller, ``False`` otherwise.
        :rtype: bool
        """
        epoch_loop = getattr(getattr(trainer, "fit_loop", None), "epoch_loop", None)
        batch_progress = getattr(epoch_loop, "batch_progress", None)
        if batch_progress is None:
            return False

        total = getattr(batch_progress, "total", None)
        current = getattr(batch_progress, "current", None)
        if total is None or current is None:
            return False

        should_align = (
            total.processed == total.ready
            and total.completed + 1 == total.processed
            and current.processed == current.ready
            and current.completed + 1 == current.processed
        )
        if not should_align:
            return False

        batch_progress.increment_completed()
        return True

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Run the parent hook with the progress counters temporarily aligned.

        :param trainer: Lightning trainer.
        :type trainer: Any
        :param pl_module: Lightning module being trained.
        :type pl_module: Any
        :param outputs: Outputs returned by ``training_step``.
        :type outputs: Any
        :param batch: Current batch.
        :type batch: Any
        :param batch_idx: Index of the current batch within the epoch.
        :type batch_idx: int
        """
        aligned = self._temporarily_align_completed_for_save(trainer)
        try:
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        finally:
            if aligned:
                epoch_loop = getattr(getattr(trainer, "fit_loop", None), "epoch_loop", None)
                batch_progress = getattr(epoch_loop, "batch_progress", None)
                if batch_progress is not None:
                    batch_progress.total.completed -= 1
                    batch_progress.current.completed -= 1
