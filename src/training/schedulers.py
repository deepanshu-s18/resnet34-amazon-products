"""Learning rate scheduler factory for ResNet-34 experiments.

Ablation A4 compares four schedule types on identical architecture and data:

- ``"step"``: StepLR — decays by ``gamma`` every ``step_size`` epochs.
  Simple, well-understood, used in the original He et al. (2016) paper.
  Decay at epochs [100, 150] for 200-epoch CIFAR-10 training.

- ``"cosine"``: CosineAnnealingLR — smooth monotone decay from ``lr`` to
  ``eta_min`` over ``T_max`` epochs.  No sudden jumps; better for problems
  where the optimal LR changes continuously.

- ``"onecycle"``: OneCycleLR — linear warmup then cosine annealing.
  [CRITICAL] Must be stepped after **every batch**, not every epoch.
  The factory sets the ``steps_per_epoch`` argument and the caller must
  call ``scheduler.step()`` inside the batch loop, not after the epoch.

- ``"reduce_on_plateau"``: ReduceLROnPlateau — decays when validation
  loss stops improving.  [CRITICAL] Requires the validation loss as an
  argument: ``scheduler.step(val_loss)``.  The factory marks this
  type so the trainer dispatches correctly.

Warmup
------
All schedule types support an optional linear warmup phase.  During warmup
(first ``warmup_epochs`` epochs), the LR increases linearly from
``lr / warmup_steps`` to ``lr``.  After warmup, the main schedule takes
over.  This is implemented via :class:`WarmupThenScheduler`.

Interview note
--------------
A common interview question is: "What happens if you apply OneCycleLR
stepping per epoch instead of per batch?"  The answer: the LR completes
its full cycle in ``num_epochs`` LR steps instead of
``num_epochs × steps_per_epoch`` steps, so it effectively sees the schedule
compressed by a factor of ``steps_per_epoch`` (~391 for batch_size=128 on
CIFAR-10).  The LR drops to ``eta_min`` in the first few epochs and stays
there — the cycle collapses.

Example::

    from src.training.schedulers import create_scheduler, is_per_batch_scheduler

    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_name="cosine",
        num_epochs=200,
        steps_per_epoch=391,
        lr=0.1,
        warmup_epochs=5,
    )

    # In the training loop:
    for epoch in range(num_epochs):
        for batch in loader:
            ...
            if is_per_batch_scheduler(scheduler):
                scheduler.step()   # OneCycleLR only
        if not is_per_batch_scheduler(scheduler):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)

# Compatibility shim for the LRScheduler base class rename in PyTorch 2.0.
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # type: ignore[attr-defined]

__all__ = [
    "create_scheduler",
    "is_per_batch_scheduler",
    "get_current_lr",
    "WarmupThenScheduler",
]

logger = logging.getLogger(__name__)

# Scheduler types that must be stepped after every training batch.
_PER_BATCH_SCHEDULERS = (OneCycleLR,)
# Scheduler types that require val_loss as argument to .step().
_PLATEAU_SCHEDULERS = (ReduceLROnPlateau,)


# ---------------------------------------------------------------------------
# Warmup wrapper
# ---------------------------------------------------------------------------


class WarmupThenScheduler:
    """Wrapper that applies linear warmup then delegates to a main scheduler.

    During the first ``warmup_epochs`` epochs, the learning rate increases
    linearly from ``start_factor × base_lr`` to ``base_lr``.  After warmup,
    the main scheduler takes control.

    This wrapper exposes the same ``step()`` interface as a standard
    scheduler so the training loop does not need to know about warmup.

    Args:
        optimizer: The optimizer whose LR is being scheduled.
        warmup_epochs: Number of epochs for the linear warmup phase.
        main_scheduler: The post-warmup scheduler (StepLR, CosineAnnealingLR,
            etc.).
        start_factor: The LR at epoch 0 is ``base_lr × start_factor``.
            Defaults to ``1 / warmup_epochs`` (starts near zero).

    Attributes:
        _epoch: Internal epoch counter incremented on each ``step()`` call.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        main_scheduler: LRScheduler,
        start_factor: float = 0.0,
    ) -> None:
        self._optimizer      = optimizer
        self._warmup_epochs  = warmup_epochs
        self._main_scheduler = main_scheduler
        self._epoch          = 0
        self._base_lrs       = [g["lr"] for g in optimizer.param_groups]

        # Start from a very small LR.
        self._set_warmup_lr(0)

    def _set_warmup_lr(self, epoch: int) -> None:
        """Set LR during warmup phase: linearly interpolate 0 → base_lr."""
        if self._warmup_epochs == 0:
            return
        factor = (epoch + 1) / self._warmup_epochs
        for param_group, base_lr in zip(
            self._optimizer.param_groups, self._base_lrs
        ):
            param_group["lr"] = base_lr * factor

    def step(self, val_loss: Optional[float] = None) -> None:
        """Advance the schedule by one step (epoch or batch)."""
        self._epoch += 1
        if self._epoch <= self._warmup_epochs:
            self._set_warmup_lr(self._epoch)
        else:
            # Restore base LR before letting main scheduler modify it.
            if self._epoch == self._warmup_epochs + 1:
                for param_group, base_lr in zip(
                    self._optimizer.param_groups, self._base_lrs
                ):
                    param_group["lr"] = base_lr

            if isinstance(self._main_scheduler, ReduceLROnPlateau):
                if val_loss is None:
                    raise ValueError(
                        "ReduceLROnPlateau requires val_loss argument to step()."
                    )
                self._main_scheduler.step(val_loss)
            else:
                self._main_scheduler.step()

    def state_dict(self) -> dict:
        return {
            "_epoch": self._epoch,
            "main_scheduler": self._main_scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["_epoch"]
        self._main_scheduler.load_state_dict(state["main_scheduler"])

    def get_last_lr(self) -> list[float]:
        return [g["lr"] for g in self._optimizer.param_groups]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_scheduler(
    optimizer: Optimizer,
    scheduler_name: str,
    num_epochs: int,
    steps_per_epoch: int,
    lr: float,
    warmup_epochs: int = 0,
    step_size: int = 50,
    gamma: float = 0.1,
    eta_min: float = 1e-6,
    pct_start: float = 0.3,
) -> LRScheduler | WarmupThenScheduler | ReduceLROnPlateau:
    """Create a learning rate scheduler from configuration.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_name: One of ``"step"``, ``"cosine"``, ``"onecycle"``,
            or ``"reduce_on_plateau"``.  Case-insensitive.
        num_epochs: Total training epochs (used by CosineAnnealingLR and
            OneCycleLR).
        steps_per_epoch: Number of optimizer steps per epoch (DataLoader
            length).  Required for OneCycleLR.
        lr: Base (maximum) learning rate.  Used by OneCycleLR.
        warmup_epochs: Number of linear warmup epochs.  0 disables warmup.
            Has no effect for OneCycleLR (which has built-in warmup).
        step_size: Epoch period for StepLR decay.  Decay occurs at epochs
            ``step_size``, ``2×step_size``, etc.
        gamma: Multiplicative decay factor for StepLR.
        eta_min: Minimum LR for CosineAnnealingLR (default ``1e-6``).
        pct_start: Fraction of training for the OneCycleLR warmup phase.

    Returns:
        A configured scheduler.  May be a :class:`WarmupThenScheduler`
        wrapping the main scheduler when ``warmup_epochs > 0``.

    Raises:
        ValueError: If ``scheduler_name`` is not a supported choice.
    """
    name = scheduler_name.lower().strip()

    if name == "step":
        main = StepLR(optimizer, step_size=step_size, gamma=gamma)
        logger.info(
            "Scheduler: StepLR(step_size=%d, gamma=%.2f) + warmup=%d",
            step_size, gamma, warmup_epochs,
        )

    elif name == "cosine":
        # T_max = effective training epochs after warmup.
        t_max = max(num_epochs - warmup_epochs, 1)
        main = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        logger.info(
            "Scheduler: CosineAnnealingLR(T_max=%d, eta_min=%.1e) + warmup=%d",
            t_max, eta_min, warmup_epochs,
        )

    elif name == "onecycle":
        # [CRITICAL] OneCycleLR steps after EVERY BATCH, not every epoch.
        # total_steps = steps per epoch × number of epochs.
        # pct_start controls the fraction of total_steps used for warmup.
        total_steps = steps_per_epoch * num_epochs
        main = OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
        )
        logger.info(
            "Scheduler: OneCycleLR(max_lr=%.4f, total_steps=%d, "
            "pct_start=%.2f) — step after EVERY BATCH",
            lr, total_steps, pct_start,
        )
        # OneCycleLR has built-in warmup; skip WarmupThenScheduler.
        return main  # type: ignore[return-value]

    elif name == "reduce_on_plateau":
        main = ReduceLROnPlateau(
            optimizer,
            mode="min",       # monitor validation loss
            factor=gamma,
            patience=10,
            min_lr=eta_min,
        )
        logger.info(
            "Scheduler: ReduceLROnPlateau(factor=%.2f, patience=10) + warmup=%d "
            "— pass val_loss to scheduler.step(val_loss)",
            gamma, warmup_epochs,
        )

    else:
        raise ValueError(
            f"Unknown scheduler {scheduler_name!r}. "
            "Choose from: 'step', 'cosine', 'onecycle', 'reduce_on_plateau'."
        )

    if warmup_epochs > 0:
        return WarmupThenScheduler(optimizer, warmup_epochs, main)  # type: ignore[return-value]
    return main  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def is_per_batch_scheduler(
    scheduler: LRScheduler | WarmupThenScheduler | ReduceLROnPlateau,
) -> bool:
    """Return True if the scheduler must be stepped after each batch.

    The training loop must call this to decide whether to step after the
    batch (OneCycleLR) or after the epoch (all others).

    Args:
        scheduler: Any scheduler returned by :func:`create_scheduler`.

    Returns:
        ``True`` only for ``OneCycleLR`` instances.
    """
    return isinstance(scheduler, OneCycleLR)


def get_current_lr(
    optimizer: Optimizer,
    group_idx: int = 0,
) -> float:
    """Return the current learning rate from the specified parameter group.

    Args:
        optimizer: The optimizer.
        group_idx: Index of the parameter group to read (default 0 = the
            weight-decay group, which contains the main weights).

    Returns:
        Current learning rate as a float.
    """
    return float(optimizer.param_groups[group_idx]["lr"])
