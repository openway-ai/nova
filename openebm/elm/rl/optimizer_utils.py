"""Optimizer helpers for EBM RL training."""

from __future__ import annotations

import torch


def make_skip_missing_grad_muon_adamw(base_cls):
    """Wrap MuonAdamW so missing-gradient Muon groups are skipped, not zero-filled."""

    class SkipMissingGradMuonAdamW(base_cls):
        @torch.no_grad()
        def step(self, closure=None):
            if closure is not None:
                with torch.enable_grad():
                    closure()
            for group in self.param_groups:
                if group["kind"] == "adamw":
                    self._step_adamw(group)
                elif group["kind"] == "muon":
                    if any(p.grad is None for p in group["params"]):
                        continue
                    self._step_muon(group)
                else:
                    raise ValueError(f"Unknown optimizer kind: {group['kind']}")

    return SkipMissingGradMuonAdamW
