"""
磁盘空间感知的 Checkpoint 回调
在保存 checkpoint 前检查磁盘空间，空间不足时强制清理旧 checkpoint
"""
import os
import shutil
from pathlib import Path
try:
    from lightning.pytorch.callbacks import ModelCheckpoint
except ImportError:
    from pytorch_lightning.callbacks import ModelCheckpoint


class DiskAwareCheckpoint(ModelCheckpoint):
    """带磁盘空间检测的 ModelCheckpoint"""

    def __init__(self, *args, min_free_gb=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_free_gb = min_free_gb

    def _check_disk_space(self):
        """检查磁盘剩余空间（GB）"""
        if self.dirpath:
            # 确保目录存在
            Path(self.dirpath).mkdir(parents=True, exist_ok=True)
            stat = shutil.disk_usage(self.dirpath)
            free_gb = stat.free / (1024**3)
            return free_gb
        return float('inf')

    def _cleanup_old_checkpoints(self):
        """强制清理旧 checkpoint，只保留 last.ckpt"""
        if not self.dirpath:
            return

        ckpt_dir = Path(self.dirpath)
        if not ckpt_dir.exists():
            return

        # 找到所有 checkpoint 文件（排除 last.ckpt）
        ckpts = [f for f in ckpt_dir.glob("*.ckpt") if f.name != "last.ckpt"]

        # 删除所有非 last.ckpt 的文件
        for ckpt in ckpts:
            try:
                ckpt.unlink()
                print(f"[DiskAware] 已删除旧 checkpoint: {ckpt.name}")
            except Exception as e:
                print(f"[DiskAware] 删除失败 {ckpt.name}: {e}")

    def _save_checkpoint(self, trainer, filepath):
        """保存前检查磁盘空间"""
        free_gb = self._check_disk_space()

        if free_gb < self.min_free_gb:
            print(f"[DiskAware] ⚠️  磁盘空间不足: {free_gb:.1f}GB < {self.min_free_gb}GB")
            print(f"[DiskAware] 🧹 开始清理旧 checkpoint...")
            self._cleanup_old_checkpoints()

            # 再次检查
            free_gb = self._check_disk_space()
            print(f"[DiskAware] 清理后剩余空间: {free_gb:.1f}GB")

        # 调用父类保存方法
        super()._save_checkpoint(trainer, filepath)
