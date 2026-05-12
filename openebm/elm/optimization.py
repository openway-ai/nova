"""Optimizers and learning-rate schedulers used by EBT training.

Includes:

- :class:`LARS` — Layer-wise Adaptive Rate Scaling optimizer.
- :class:`WarmUpLinearWarmdownLR` — NanoChat-style linear warmup / constant /
  linear warmdown schedule with optional resume-warmup and dynamic weight decay.
- :class:`WarmUpCosineAnnealingLR` — linear warmup then cosine annealing, with
  optional dynamic weight decay.
- :class:`StableAdamW` and :class:`StableAdamWUnfused` — StableAdamW
  implementations (`arXiv:1804.04235`_).

.. _arXiv:1804.04235: https://arxiv.org/abs/1804.04235
"""

from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import (Optimizer, _get_value,
                        _fused_doc, _maximize_doc, _default_to_fused_or_foreach)


class LARS(optim.Optimizer):
    """Layer-wise Adaptive Rate Scaling optimizer.

    Applies LARS adaptation (scaling the update by
    ``eta * ||p|| / ||update||``) with configurable filters for weight decay
    and LARS adaptation.
    """

    def __init__(
        self,
        params: Any,
        lr: float,
        weight_decay: float = 0,
        momentum: float = 0.9,
        eta: float = 0.001,
        weight_decay_filter: Optional[Callable[[Tensor], bool]] = None,
        lars_adaptation_filter: Optional[Callable[[Tensor], bool]] = None,
        epoch: int = 0,
    ) -> None:
        """Initialize the optimizer.

        :param params: Iterable of parameters to optimize.
        :type params: Any
        :param lr: Learning rate.
        :type lr: float
        :param weight_decay: Weight decay coefficient.
        :type weight_decay: float
        :param momentum: SGD momentum.
        :type momentum: float
        :param eta: LARS trust coefficient.
        :type eta: float
        :param weight_decay_filter: Callable returning ``True`` for parameters
            that should skip weight decay (e.g. bias / norm).
        :type weight_decay_filter: Optional[Callable[[Tensor], bool]]
        :param lars_adaptation_filter: Callable returning ``True`` for
            parameters that should skip LARS adaptation.
        :type lars_adaptation_filter: Optional[Callable[[Tensor], bool]]
        :param epoch: Initial epoch counter.
        :type epoch: int
        """
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        self.epoch = epoch
        super().__init__(params, defaults)

    def update_epoch(self, epoch: int) -> None:
        """Update the tracked epoch counter.

        :param epoch: New epoch value.
        :type epoch: int
        """
        self.epoch = epoch

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Any]] = None) -> None:
        """Perform a single LARS optimization step.

        :param closure: Optional closure that re-evaluates the model.
        :type closure: Optional[Callable[[], Any]]
        """
        if closure is not None:
            with torch.enable_grad():
                closure()

        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if (g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p)):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def exclude_bias_and_norm(p: Tensor) -> bool:
    """Return ``True`` for bias / norm parameters (1-D tensors).

    Useful as a ``weight_decay_filter`` / ``lars_adaptation_filter``.

    :param p: Candidate parameter tensor.
    :type p: Tensor
    :return: ``True`` when ``p`` is 1-dimensional.
    :rtype: bool
    """
    return p.ndim == 1


class WarmUpLinearWarmdownLR(_LRScheduler):
    """NanoChat-style linear-warmup / constant / linear-warmdown scheduler.

    - **warmup**: linear from ``0`` (or ``lr / divider``) up to ``peak_lr``.
    - **constant**: holds ``peak_lr``.
    - **warmdown**: linear from ``peak_lr`` down to ``final_lr_frac * peak_lr``.

    Reference (``base_train.py``): ``warmup_ratio=0.0`` (no warmup),
    ``warmdown_ratio=0.5`` (decay over the last 50% of training),
    ``final_lr_frac=0.0`` (decay all the way to 0).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_ratio: float,
        warmdown_ratio: float,
        final_lr_frac: float,
        total_steps: int,
        warm_up_finished_func: Optional[Callable[[], None]] = None,
        enable_wd_decay: bool = False,
        resume_warmup_steps: int = 0,
    ) -> None:
        """Initialize the scheduler.

        :param optimizer: Optimizer whose LR this scheduler controls.
        :type optimizer: Optimizer
        :param warmup_ratio: Fraction of ``total_steps`` used for warmup.
        :type warmup_ratio: float
        :param warmdown_ratio: Fraction of ``total_steps`` used for warmdown.
        :type warmdown_ratio: float
        :param final_lr_frac: Final LR as a fraction of peak LR.
        :type final_lr_frac: float
        :param total_steps: Total scheduled steps.
        :type total_steps: int
        :param warm_up_finished_func: Optional callback fired when warmup ends.
        :type warm_up_finished_func: Optional[Callable[[], None]]
        :param enable_wd_decay: When ``True``, linearly decays weight decay
            toward zero over ``total_steps``.
        :type enable_wd_decay: bool
        :param resume_warmup_steps: Post-resume re-warmup length; the LR is
            linearly interpolated from the checkpoint LR back to the schedule
            LR over this many steps.
        :type resume_warmup_steps: int
        """
        self.warmup_steps = int(warmup_ratio * total_steps)
        self.warmdown_steps = int(warmdown_ratio * total_steps)
        self.constant_steps = total_steps - self.warmup_steps - self.warmdown_steps
        self.final_lr_frac = final_lr_frac
        self.total_steps = total_steps
        self.highest_lr = [group['lr'] for group in optimizer.param_groups]
        self.last_step = 0
        self.finished_warming_up = False
        self.warm_up_finished_func = warm_up_finished_func

        # Resume warmup: after loading a checkpoint, linearly ramp the LR from
        # the checkpoint value back up to the schedule value over
        # ``resume_warmup_steps`` steps.
        self.resume_warmup_steps = resume_warmup_steps
        self.resume_start_step = None
        self.resume_base_lr = None

        # Dynamic weight decay (paired with the LR schedule).
        self.enable_wd_decay = enable_wd_decay
        self.initial_weight_decays = [group.get('weight_decay', 0) for group in optimizer.param_groups]

        self._last_lr = [0.0 for _ in self.highest_lr] if self.warmup_steps > 0 else self.highest_lr.copy()

        print(f"[WarmUpLinearWarmdownLR] enabled:")
        print(f"  - Warmup steps: {self.warmup_steps} ({warmup_ratio*100:.1f}%)")
        print(f"  - Constant steps: {self.constant_steps}")
        print(f"  - Warmdown steps: {self.warmdown_steps} ({warmdown_ratio*100:.1f}%)")
        print(f"  - Final LR fraction: {final_lr_frac}")

        super(WarmUpLinearWarmdownLR, self).__init__(optimizer)

    def step(self) -> None:
        """Advance the scheduler by one step."""
        self.last_step += 1
        super().step()

    def _compute_schedule_lr(self) -> List[float]:
        """Compute the raw schedule LR (ignoring any resume-warmup blending).

        :return: LR per parameter group.
        :rtype: List[float]
        """
        if self.enable_wd_decay and self.total_steps > 0:
            wd_multiplier = max(0.0, 1.0 - self.last_step / self.total_steps)
            for i, group in enumerate(self.optimizer.param_groups):
                if self.initial_weight_decays[i] > 0:
                    group['weight_decay'] = self.initial_weight_decays[i] * wd_multiplier

        step = self.last_step

        if step <= self.warmup_steps:
            # Linear warmup from 0 up to peak_lr.
            progress = step / self.warmup_steps if self.warmup_steps > 0 else 1.0
            return [lr * progress for lr in self.highest_lr]
        elif step <= self.warmup_steps + self.constant_steps:
            if not self.finished_warming_up:
                self.finished_warming_up = True
                if self.warm_up_finished_func is not None:
                    self.warm_up_finished_func()
            return self.highest_lr.copy()
        else:
            # Linear warmdown from peak_lr down to final_lr_frac * peak_lr.
            if not self.finished_warming_up:
                self.finished_warming_up = True
                if self.warm_up_finished_func is not None:
                    self.warm_up_finished_func()
            warmdown_progress = (step - self.warmup_steps - self.constant_steps) / self.warmdown_steps if self.warmdown_steps > 0 else 1.0
            warmdown_progress = min(1.0, warmdown_progress)
            return [lr * (1.0 - warmdown_progress * (1.0 - self.final_lr_frac)) for lr in self.highest_lr]

    def get_lr(self) -> List[float]:
        """Return the LR for the current step.

        When resume-warmup is active, the returned LR is linearly interpolated
        between the checkpoint LR and the schedule LR.

        :return: LR per parameter group.
        :rtype: List[float]
        """
        target_lrs = self._compute_schedule_lr()

        if self.resume_start_step is not None and self.resume_warmup_steps > 0:
            steps_since_resume = self.last_step - self.resume_start_step
            if steps_since_resume < self.resume_warmup_steps:
                progress = steps_since_resume / self.resume_warmup_steps
                base_lrs = self.resume_base_lr if self.resume_base_lr is not None else [0.0] * len(target_lrs)
                self._last_lr = [base + progress * (target - base)
                                 for base, target in zip(base_lrs, target_lrs)]
                return self._last_lr

        self._last_lr = target_lrs
        return self._last_lr

    def get_last_lr(self) -> List[float]:
        """Return the most recently computed LR list.

        :return: LR per parameter group.
        :rtype: List[float]
        """
        return self._last_lr

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing.

        :return: State dict.
        :rtype: dict
        """
        return {
            'last_step': self.last_step,
            'last_lr': self._last_lr,
            'finished_warming_up': self.finished_warming_up,
            'resume_warmup_steps': self.resume_warmup_steps,
            'resume_start_step': self.resume_start_step,
            'resume_base_lr': self.resume_base_lr,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore scheduler state and optionally arm a resume-warmup phase.

        :param state_dict: State dict previously returned by :meth:`state_dict`.
        :type state_dict: dict
        """
        self.last_step = state_dict['last_step']
        self._last_lr = state_dict['last_lr']
        self.finished_warming_up = state_dict['finished_warming_up']

        if self.resume_warmup_steps > 0:
            self.resume_start_step = self.last_step
            self.resume_base_lr = list(self._last_lr)
            print(f"[Resume Warmup] starting at step={self.resume_start_step}, "
                  f"ramp LR from {self.resume_base_lr} to schedule LR over {self.resume_warmup_steps} steps")
        else:
            self.resume_start_step = state_dict.get('resume_start_step', None)
            self.resume_base_lr = state_dict.get('resume_base_lr', None)


class WarmUpCosineAnnealingLR(_LRScheduler):
    """Linear warmup followed by a user-provided cosine-annealing scheduler.

    Supports optional dynamic weight decay that linearly anneals to zero over
    ``total_steps``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warm_up_steps: int,
        warm_up_base_lr_divider: float,
        cosine_scheduler: _LRScheduler,
        warm_up_finished_func: Optional[Callable[[], None]] = None,
        total_steps: Optional[int] = None,
        enable_wd_decay: bool = False,
    ) -> None:
        """Initialize the scheduler.

        :param optimizer: Optimizer whose LR this scheduler controls.
        :type optimizer: Optimizer
        :param warm_up_steps: Number of linear warmup steps.
        :type warm_up_steps: int
        :param warm_up_base_lr_divider: ``-1`` warms up from ``0``; otherwise
            warms up from ``lr / divider``.
        :type warm_up_base_lr_divider: float
        :param cosine_scheduler: Cosine scheduler invoked once warmup ends.
        :type cosine_scheduler: _LRScheduler
        :param warm_up_finished_func: Optional callback fired when warmup ends.
        :type warm_up_finished_func: Optional[Callable[[], None]]
        :param total_steps: Total scheduled steps (only used for dynamic WD).
        :type total_steps: Optional[int]
        :param enable_wd_decay: When ``True``, linearly anneal WD toward 0.
        :type enable_wd_decay: bool
        """
        self.warm_up_steps = warm_up_steps
        self.cosine_scheduler = cosine_scheduler
        self.last_step = 0
        self.highest_lr = [group['lr'] for group in optimizer.param_groups]
        self.finished_warming_up = False
        self.warm_up_finished_func = warm_up_finished_func

        # Dynamic weight decay, paired with the LR schedule.
        self.total_steps = total_steps
        self.enable_wd_decay = enable_wd_decay
        self.initial_weight_decays = [group.get('weight_decay', 0) for group in optimizer.param_groups]
        if enable_wd_decay:
            print(f"[Dynamic WD] enabled: linearly decays to 0")
            print(f"  - initial WD values: {self.initial_weight_decays}")

        self.base_lr_divider = warm_up_base_lr_divider
        if self.base_lr_divider == -1:
            self._last_lr = [0.0 for _ in self.highest_lr]
        else:
            self._last_lr = [lr / self.base_lr_divider for lr in self.highest_lr]

        super(WarmUpCosineAnnealingLR, self).__init__(optimizer)

    def step(self) -> None:
        """Advance the scheduler by one step."""
        self.last_step += 1
        super().step()

    def get_lr(self) -> List[float]:
        """Return the LR for the current step.

        :return: LR per parameter group.
        :rtype: List[float]
        """
        if self.enable_wd_decay and self.total_steps and self.total_steps > 0:
            wd_multiplier = max(0.0, 1.0 - self.last_step / self.total_steps)
            for i, group in enumerate(self.optimizer.param_groups):
                if self.initial_weight_decays[i] > 0:
                    group['weight_decay'] = self.initial_weight_decays[i] * wd_multiplier

        if self.warm_up_steps != 0 and self.last_step <= self.warm_up_steps:
            if self.base_lr_divider == -1:
                # Warm-up from 0 to highest_lr.
                warmup_lr = [
                    lr * (self.last_step / self.warm_up_steps)
                    for lr in self.highest_lr
                ]
            else:
                # Warm-up from lr / base_lr_divider to highest_lr.
                warmup_lr = [
                    (lr / self.base_lr_divider) +
                    (lr - (lr / self.base_lr_divider)) * (self.last_step / self.warm_up_steps)
                    for lr in self.highest_lr
                ]
            self._last_lr = warmup_lr
            return warmup_lr
        else:
            if not self.finished_warming_up:
                self.finished_warming_up = True
                if self.warm_up_finished_func is not None:
                    self.warm_up_finished_func()
            # Proceed with the cosine scheduler once warmup is done.
            self.cosine_scheduler.step()
            self._last_lr = self.cosine_scheduler.get_last_lr()
            return self._last_lr

    def get_last_lr(self) -> List[float]:
        """Return the most recently computed LR list.

        :return: LR per parameter group.
        :rtype: List[float]
        """
        return self._last_lr

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing.

        :return: State dict.
        :rtype: dict
        """
        return {
            'warm_up_steps': self.warm_up_steps,
            'last_lr': self._last_lr,
            'last_step': self.last_step,
            'finished_warming_up': self.finished_warming_up,
            'cosine_scheduler_state_dict': self.cosine_scheduler.state_dict()
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore scheduler state from ``state_dict``.

        :param state_dict: State dict previously returned by :meth:`state_dict`.
        :type state_dict: dict
        """
        self.warm_up_steps = state_dict['warm_up_steps']
        self._last_lr = state_dict['last_lr']
        self.last_step = state_dict['last_step']
        self.finished_warming_up = state_dict['finished_warming_up']
        self.cosine_scheduler.load_state_dict(state_dict['cosine_scheduler_state_dict'])


class StableAdamW(torch.optim.Optimizer):
    """StableAdamW optimizer (``arXiv:1804.04235``).

    Applies tensor-wise update-RMS clipping on top of AdamW.
    """

    def __init__(
        self,
        params: Any,
        lr: Optional[float] = None,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:
        """Initialize the optimizer.

        :param params: Iterable of parameters to optimize.
        :type params: Any
        :param lr: Learning rate.
        :type lr: Optional[float]
        :param betas: Adam momentum coefficients.
        :type betas: Tuple[float, float]
        :param eps: Numerical stability term added to the RMS denominator.
        :type eps: float
        :param weight_decay: Weight decay coefficient.
        :type weight_decay: float
        :raises ValueError: For invalid ``eps`` or ``betas``.
        """
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(StableAdamW, self).__init__(params, defaults)

    def update_epoch(self, epoch: int) -> None:
        """Update the tracked epoch counter.

        :param epoch: New epoch value.
        :type epoch: int
        """
        self.epoch = epoch

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Any]] = None) -> None:
        """Perform a single StableAdamW step.

        :param closure: Optional closure that re-evaluates the model.
        :type closure: Optional[Callable[[], Any]]
        """
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                RMSt = corrected_exp_avg_sq.sqrt().add_(group['eps'])
                # Dynamically rescale the LR by the RMS of the update.
                eta_t = group['lr'] / max(1, RMSt.mean())

                p.data.mul_(1 - eta_t * group['weight_decay'])
                p.data.addcdiv_(corrected_exp_avg, RMSt, value=-eta_t)


class StableAdamWUnfused(torch.optim.Optimizer):
    """Unfused StableAdamW reference implementation.

    Based on Mitchell Wortsman's gist
    (``https://gist.github.com/mitchellnw/d42e22a0b9ec02ceaf4f7b4457f51423``).
    Supports an optional ``custom_fp16`` precision mode with a fixed loss
    scaler ``custom_scalar``; call ``(custom_scalar * loss).backward()`` when
    that mode is enabled.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 0.002,
        weight_decay: float = 0.2,
        betas: Tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-6,
        clip_thresh: float = 1.,
        custom_scalar: int = 65536,
    ) -> None:
        """Initialize the optimizer.

        :param params: Iterable of parameters to optimize.
        :type params: Any
        :param lr: Learning rate.
        :type lr: float
        :param weight_decay: Weight decay coefficient.
        :type weight_decay: float
        :param betas: Adam momentum coefficients ``(beta1, beta2)``.
        :type betas: Tuple[float, float]
        :param eps: Numerical stability term.
        :type eps: float
        :param clip_thresh: RMS clipping threshold (``d`` in the paper).
        :type clip_thresh: float
        :param custom_scalar: Fixed loss scaler used in ``custom_fp16`` mode.
        :type custom_scalar: int
        """
        beta1, beta2 = betas[0], betas[1]
        defaults = dict(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2)
        super(StableAdamWUnfused, self).__init__(params, defaults)

        self.eps=eps
        self.d = clip_thresh

        # Switch ``precision`` to ``"custom_fp16"`` to use the fixed loss
        # scaler ``custom_scalar``, which is divided out during the update.
        self.precision = None
        self.custom_scaler = custom_scalar

        for group in self.param_groups:
            group['step'] = 1.

        print('Using StableAdamWUnfused-v1')

    def update_epoch(self, epoch: int) -> None:
        """Placeholder kept for API compatibility with other optimizers.

        :param epoch: Ignored.
        :type epoch: int
        """
        pass

    def __setstate__(self, state: dict) -> None:
        """Restore optimizer state from a pickled dict.

        :param state: Optimizer state.
        :type state: dict
        """
        super(StableAdamWUnfused, self).__setstate__(state)

    def step(self, closure: Optional[Callable[[], Any]] = None) -> Any:
        """Perform a single StableAdamW (unfused) step.

        :param closure: Optional closure that re-evaluates the model and
            returns the loss.
        :type closure: Optional[Callable[[], Any]]
        :return: The loss returned by ``closure`` (or ``None``).
        :rtype: Any
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1 = group['beta1']
            beta2 = group['beta2']
            step = group['step']

            for p in group['params']:
                if p.grad is None:
                    continue
                theta=p.data
                param_state = self.state[p]

                if self.precision == 'custom_fp16':
                    g = p.grad.data / self.custom_scaler
                    if torch.any(torch.isnan(g) | torch.isinf(g)):
                        continue
                else:
                    g = p.grad.data

                if 'exp_avg' not in param_state:
                    v = param_state['exp_avg'] = torch.zeros_like(theta)
                    u = param_state['exp_avg_sq'] = torch.zeros_like(theta)
                else:
                    v = param_state['exp_avg']
                    u = param_state['exp_avg_sq']

                beta1hat = beta1 * (1 - beta1**(step - 1)) / (1 - beta1**step)
                beta2hat = beta2 * (1 - beta2**(step - 1)) / (1 - beta2**step)

                v = v.mul_(beta1hat).add_(g, alpha=1.0-beta1hat)
                u = u.mul_(beta2hat).addcmul_(g,g,value=1.0-beta2hat)

                denominator = u.sqrt().add_(self.eps)

                # StableAdamW = AdamW + tensor-wise update clipping
                # (https://arxiv.org/abs/1804.04235).
                rms = torch.div(
                    g.pow(2),
                    torch.maximum(u, (self.eps ** 2) * torch.ones_like(u))
                ).mean().sqrt().item()

                new_lr = lr * (1. / max(1., rms / self.d ))

                theta = theta.mul_(1.0-new_lr*weight_decay).addcdiv_(
                    v,
                    denominator,
                    value=-new_lr
                )

                param_state['exp_avg'] = v
                param_state['exp_avg_sq'] = u

            group['step'] = step + 1
