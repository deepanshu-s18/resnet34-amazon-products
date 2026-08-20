"""Full PyTorch training pipeline for image classification.

This module provides :class:`Trainer`, the central orchestrator for all
training experiments in this project.  It owns the epoch loop, gradient
computation, metric accumulation, checkpointing, and early stopping.

Architecture
------------
The :class:`Trainer` is deliberately **not** a configuration object — it
receives already-constructed components (model, optimiser, dataloaders) so
that each component can be independently tested.  Configuration is captured
in :class:`TrainerConfig` and stored in every checkpoint for reproducibility.

Mixed Precision (AMP)
---------------------
When ``TrainerConfig.use_amp=True`` and a CUDA device is available,
:class:`torch.cuda.amp.GradScaler` is used to scale the loss before
``backward()`` and unscale before the optimiser step.  If gradients are
infinite (common early in training with aggressive LR), the scaler skips
the optimiser step and reduces the scale factor — no special handling is
needed in the outer loop.

Gradient clipping must be applied *after* ``scaler.unscale_()`` so that the
clip operates on the true gradient magnitudes, not the scaled ones.

AMP is silently disabled on CPU (``GradScaler(enabled=False)`` makes every
call a no-op), so the code is hardware-agnostic.

Scheduler conventions
---------------------
- :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`: stepped after each
  epoch with the validation loss.
- :class:`~torch.optim.lr_scheduler.OneCycleLR`: stepped after each
  training batch.
- All other schedulers: stepped once after each epoch.

The scheduler type is detected via ``isinstance`` checks so that training
scripts do not need to communicate this to the Trainer.

Example::

    from pathlib import Path
    import torch, torch.nn as nn
    from torch.optim import SGD
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from src.training.trainer import Trainer, TrainerConfig

    config = TrainerConfig(num_epochs=200, use_amp=True)
    trainer = Trainer(
        model=resnet34,
        optimizer=SGD(resnet34.parameters(), lr=0.1, momentum=0.9),
        criterion=nn.CrossEntropyLoss(),
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=CosineAnnealingLR(optimizer, T_max=200),
        config=config,
        device=torch.device("cuda"),
        experiment_dir=Path("results/run_001"),
    )
    metrics = trainer.train()
    print(f"Best val accuracy: {metrics['best_val_acc']:.4f}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.utils.checkpoint import CheckpointManager
from src.utils.logger import ExperimentLogger
from src.utils.seed import get_rng_state, restore_rng_state

# Compatibility shim for the LRScheduler base class rename in PyTorch 2.0.
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # type: ignore

from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; fall back to a no-op wrapper.
    def tqdm(iterable, **_):  # type: ignore[misc]
        return iterable

__all__ = ["TrainerConfig", "EarlyStopping", "Trainer"]

_module_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    """Hyperparameters and toggles that govern the training loop.

    All fields are stored in every checkpoint so that an experiment is fully
    reproducible from the checkpoint alone.

    Attributes:
        num_epochs: Total number of training epochs.
        grad_clip_norm: Maximum L2 norm for gradient clipping.  ``None``
            disables clipping.  Applied after ``scaler.unscale_()`` when AMP
            is active so that clipping acts on true gradient magnitudes.
        use_amp: Enable Automatic Mixed Precision (FP16 forward / FP32
            parameter updates).  Silently ignored on CPU.
        log_every_n_steps: Frequency of step-level TensorBoard writes
            (loss, LR, gradient norms).
        checkpoint_every_n_epochs: Save a periodic checkpoint every N epochs
            in addition to the best-model checkpoint.
        early_stopping_patience: Epochs to wait for improvement before
            halting.  Set to a large value (e.g. ``9999``) to disable.
        early_stopping_min_delta: Minimum absolute improvement in the
            monitored metric to count as an improvement.
        experiment_name: Human-readable identifier embedded in checkpoint
            filenames and log messages.
        seed: Experiment seed (stored for provenance; :func:`set_global_seed`
            must be called separately before training).
    """

    num_epochs: int = 200
    grad_clip_norm: Optional[float] = None
    use_amp: bool = False
    log_every_n_steps: int = 50
    checkpoint_every_n_epochs: int = 10
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    experiment_name: str = "experiment"
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for checkpoint storage."""
        return {
            "num_epochs": self.num_epochs,
            "grad_clip_norm": self.grad_clip_norm,
            "use_amp": self.use_amp,
            "log_every_n_steps": self.log_every_n_steps,
            "checkpoint_every_n_epochs": self.checkpoint_every_n_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "experiment_name": self.experiment_name,
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Stop training when a monitored metric has stopped improving.

    Uses a patience counter: if the metric does not improve by at least
    ``min_delta`` for ``patience`` consecutive epochs, :meth:`__call__`
    returns ``True``.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum absolute change to qualify as an improvement.
        mode: ``"max"`` to monitor a metric that should increase (e.g.
            accuracy); ``"min"`` to monitor a metric that should decrease
            (e.g. loss).
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        mode: str = "max",
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}.")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self._best: float = float("-inf") if mode == "max" else float("inf")
        self._counter: int = 0
        self._best_epoch: int = 0

    def __call__(self, value: float, epoch: int = 0) -> bool:
        """Evaluate the metric and update internal state.

        Args:
            value: The metric value for the current epoch.
            epoch: Current epoch number (for logging only).

        Returns:
            ``True`` if training should stop; ``False`` otherwise.
        """
        improved = (
            value > self._best + self.min_delta
            if self.mode == "max"
            else value < self._best - self.min_delta
        )

        if improved:
            self._best = value
            self._counter = 0
            self._best_epoch = epoch
        else:
            self._counter += 1

        return self._counter >= self.patience

    @property
    def best_value(self) -> float:
        """The best metric value seen so far."""
        return self._best

    @property
    def epochs_since_improvement(self) -> int:
        """Number of epochs since the last improvement."""
        return self._counter

    @property
    def best_epoch(self) -> int:
        """Epoch at which the best metric was achieved."""
        return self._best_epoch

    def reset(self) -> None:
        """Reset the counter (useful for fine-tuning phases)."""
        self._best = float("-inf") if self.mode == "max" else float("inf")
        self._counter = 0
        self._best_epoch = 0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Full training pipeline with AMP, early stopping, and checkpointing.

    The :class:`Trainer` owns the training loop.  Everything else — the
    model, optimiser, dataloaders, and scheduler — is injected so that each
    component can be unit-tested and swapped independently.

    Args:
        model: The model to train.  Must be on ``device`` before passing.
        optimizer: Configured optimiser.  The Trainer does not create or
            modify it.
        criterion: Loss function, e.g. :class:`~torch.nn.CrossEntropyLoss`.
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        config: Training hyperparameters.
        device: Target compute device.
        experiment_dir: Directory where logs and checkpoints are written.
            Created if it does not exist.
        scheduler: Optional LR scheduler.  The Trainer detects whether it
            should step per-batch (OneCycleLR) or per-epoch (all others),
            and whether it needs the validation loss (ReduceLROnPlateau).
        logger: :class:`~src.utils.logger.ExperimentLogger` instance.  If
            ``None``, a minimal logger writing only to the console is created
            automatically.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainerConfig,
        device: torch.device,
        experiment_dir: Path,
        scheduler: Optional[LRScheduler] = None,
        logger: Optional[ExperimentLogger] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.scheduler = scheduler

        # Logger — create a minimal one if not provided.
        if logger is not None:
            self._log = logger
        else:
            self._log = ExperimentLogger(
                experiment_name=config.experiment_name,
                log_dir=self.experiment_dir,
            )

        # Checkpoint manager.
        self._ckpt = CheckpointManager(
            checkpoint_dir=self.experiment_dir / "checkpoints",
            experiment_name=config.experiment_name,
        )

        # Early stopping (monitors validation accuracy).
        self._early_stop = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode="max",
        )

        # AMP GradScaler.  enabled=False is a complete no-op on CPU or when
        # the user has not requested AMP.
        _amp_enabled = config.use_amp and device.type == "cuda"
        # Use the canonical torch.amp API (PyTorch >= 2.0); fall back to the
        # cuda-specific path for older versions.
        if hasattr(torch.amp, "GradScaler"):
            self._scaler = torch.amp.GradScaler("cuda", enabled=_amp_enabled)
        else:
            self._scaler = torch.cuda.amp.GradScaler(enabled=_amp_enabled)  # type: ignore[attr-defined]
        if config.use_amp and not _amp_enabled:
            self._log.warning(
                "use_amp=True but device is CPU — AMP disabled."
            )

        # Training history.
        self._train_loss_history: list[float] = []
        self._val_loss_history: list[float] = []
        self._val_acc_history: list[float] = []
        self._best_val_acc: float = 0.0
        self._best_epoch: int = 0
        self._start_epoch: int = 1
        self._global_step: int = 0

        self._log.info(
            "Trainer ready | model=%s | device=%s | epochs=%d | amp=%s",
            type(model).__name__,
            device,
            config.num_epochs,
            _amp_enabled,
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def train(self) -> dict[str, Any]:
        """Execute the full training loop and return a final metrics summary.

        Runs from ``_start_epoch`` to ``config.num_epochs`` (inclusive),
        validating after every epoch.  Saves periodic and best-model
        checkpoints.  Stops early if the early-stopping criterion fires.

        Returns:
            A dict with keys:

            - ``"best_val_acc"`` — best validation accuracy achieved.
            - ``"best_epoch"``  — epoch at which the best was achieved.
            - ``"final_epoch"`` — last epoch executed.
            - ``"stopped_early"`` — whether early stopping triggered.
            - ``"train_loss_history"`` — per-epoch training losses.
            - ``"val_loss_history"``   — per-epoch validation losses.
            - ``"val_acc_history"``    — per-epoch validation accuracies.
            - ``"total_time_seconds"`` — wall-clock training duration.
        """
        self._log.info(
            "Starting training | epochs %d → %d | experiment=%s",
            self._start_epoch,
            self.config.num_epochs,
            self.config.experiment_name,
        )

        t_start = time.perf_counter()
        stopped_early = False

        for epoch in range(self._start_epoch, self.config.num_epochs + 1):
            epoch_start = time.perf_counter()

            # ── Training ────────────────────────────────────────────────────
            self.model.train()
            train_loss, train_acc = self._train_epoch(epoch)

            # ── Validation ──────────────────────────────────────────────────
            self.model.eval()
            val_loss, val_acc = self._validate(epoch)

            epoch_secs = time.perf_counter() - epoch_start

            # ── History ─────────────────────────────────────────────────────
            self._train_loss_history.append(train_loss)
            self._val_loss_history.append(val_loss)
            self._val_acc_history.append(val_acc)

            # ── LR scheduling (epoch-level) ──────────────────────────────────
            self._step_scheduler(val_loss)

            # ── TensorBoard epoch metrics ────────────────────────────────────
            current_lr = self._get_current_lr()
            self._log.log_scalars(
                {
                    "train/epoch_loss": train_loss,
                    "train/epoch_acc": train_acc,
                    "val/epoch_loss": val_loss,
                    "val/epoch_acc": val_acc,
                    "train/learning_rate": current_lr,
                },
                step=epoch,
            )
            self._log.flush()

            # ── Best model tracking ──────────────────────────────────────────
            is_best = val_acc > self._best_val_acc
            if is_best:
                self._best_val_acc = val_acc
                self._best_epoch = epoch

            # ── Checkpointing ────────────────────────────────────────────────
            save_periodic = (epoch % self.config.checkpoint_every_n_epochs == 0)
            if is_best or save_periodic:
                self._ckpt.save(
                    state=self._build_checkpoint_state(epoch, val_acc),
                    is_best=is_best,
                    epoch=epoch,
                    val_acc=val_acc,
                )

            # ── Console summary ──────────────────────────────────────────────
            self._log.info(
                "Epoch %4d/%d | "
                "train loss=%.4f acc=%.3f | "
                "val loss=%.4f acc=%.3f | "
                "lr=%.2e | "
                "time=%.1fs%s",
                epoch,
                self.config.num_epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                current_lr,
                epoch_secs,
                "  ★ best" if is_best else "",
            )

            # ── Early stopping ───────────────────────────────────────────────
            if self._early_stop(val_acc, epoch):
                self._log.info(
                    "Early stopping triggered at epoch %d "
                    "(no improvement for %d epochs). "
                    "Best val_acc=%.4f at epoch %d.",
                    epoch,
                    self.config.early_stopping_patience,
                    self._best_val_acc,
                    self._best_epoch,
                )
                stopped_early = True
                break

        total_time = time.perf_counter() - t_start
        self._log.info(
            "Training complete | best_val_acc=%.4f at epoch %d | "
            "total time=%.1fs",
            self._best_val_acc,
            self._best_epoch,
            total_time,
        )

        return {
            "best_val_acc": self._best_val_acc,
            "best_epoch": self._best_epoch,
            "final_epoch": epoch,
            "stopped_early": stopped_early,
            "train_loss_history": self._train_loss_history,
            "val_loss_history": self._val_loss_history,
            "val_acc_history": self._val_acc_history,
            "total_time_seconds": total_time,
        }

    def resume(self, checkpoint_path: Path) -> None:
        """Load a checkpoint and restore all training state.

        After calling this method, :meth:`train` will continue from
        ``checkpoint["epoch"] + 1`` with the same model weights, optimiser
        state, scheduler state, and training history as the interrupted run.

        Args:
            checkpoint_path: Path to the ``.pt`` checkpoint file.
        """
        state = self._ckpt.load(checkpoint_path, map_location=str(self.device))

        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])

        if self.scheduler is not None and state.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        if state.get("rng_states"):
            restore_rng_state(state["rng_states"])

        self._start_epoch = state["epoch"] + 1
        self._best_val_acc = state.get("best_val_acc", 0.0)
        self._best_epoch = state.get("metrics", {}).get("epoch", 0)
        self._train_loss_history = state.get("train_loss_history", [])
        self._val_loss_history = state.get("val_loss_history", [])
        self._val_acc_history = state.get("val_acc_history", [])

        # Restore the early-stopping counter so patience is correctly
        # preserved across the interruption.
        if self._val_acc_history:
            for i, acc in enumerate(self._val_acc_history):
                self._early_stop(acc, epoch=i + 1)

        self._log.info(
            "Resumed from %s | continuing from epoch %d | best_val_acc=%.4f",
            checkpoint_path,
            self._start_epoch,
            self._best_val_acc,
        )

    # -----------------------------------------------------------------------
    # Training epoch
    # -----------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        """Execute one training epoch.

        Args:
            epoch: Current epoch number (1-indexed).

        Returns:
            ``(mean_loss, accuracy)`` over the training split.
        """
        running_loss: float = 0.0
        correct: int = 0
        total: int = 0
        steps_per_epoch = len(self.train_loader)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:4d}/{self.config.num_epochs} [Train]",
            leave=False,
            dynamic_ncols=True,
        )

        for batch_idx, (images, targets) in enumerate(pbar):
            images  = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Zero gradients.  set_to_none=True avoids storing zero tensors
            # and is slightly more memory-efficient than the default.
            self.optimizer.zero_grad(set_to_none=True)

            # ── Forward pass (optionally in AMP autocast) ────────────────────
            _device_type = self.device.type
            with torch.amp.autocast(device_type=_device_type,
                                    enabled=self._scaler.is_enabled()):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            # ── Backward pass ────────────────────────────────────────────────
            self._scaler.scale(loss).backward()

            # ── Gradient clipping ────────────────────────────────────────────
            grad_norm: Optional[float] = None
            if self.config.grad_clip_norm is not None:
                # Must unscale before clipping so the clip threshold refers
                # to the true gradient magnitudes, not the AMP-scaled ones.
                self._scaler.unscale_(self.optimizer)
                raw_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.grad_clip_norm,
                )
                grad_norm = raw_norm.item()
            else:
                # Compute norm without clipping for monitoring.
                grad_norm = self._compute_grad_norm()

            # ── Optimiser step ───────────────────────────────────────────────
            # scaler.step skips the update (and decreases the scale) when
            # gradients are inf/nan — no special handling required.
            self._scaler.step(self.optimizer)
            self._scaler.update()

            # ── Per-batch scheduler (OneCycleLR) ─────────────────────────────
            if self.scheduler is not None and isinstance(
                self.scheduler, OneCycleLR
            ):
                self.scheduler.step()

            # ── Metrics ──────────────────────────────────────────────────────
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            preds = outputs.detach().argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += batch_size

            # ── Step-level logging ───────────────────────────────────────────
            self._global_step += 1
            if self._global_step % self.config.log_every_n_steps == 0:
                self._log.log_scalars(
                    {
                        "train/step_loss": loss.item(),
                        "train/learning_rate": self._get_current_lr(),
                        "train/grad_norm": grad_norm or 0.0,
                        "amp/scale": self._scaler.get_scale(),
                    },
                    step=self._global_step,
                )
                self._log_gradient_norms(self._global_step)

            # ── Progress bar ─────────────────────────────────────────────────
            running_mean = running_loss / total
            pbar.set_postfix(
                loss=f"{running_mean:.4f}",
                acc=f"{correct / total:.3f}",
                lr=f"{self._get_current_lr():.2e}",
            )

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    # -----------------------------------------------------------------------
    # Validation epoch
    # -----------------------------------------------------------------------

    def _validate(self, epoch: int) -> tuple[float, float]:
        """Execute one validation pass.

        AMP autocast is intentionally **not** used here — validation is not
        performance-critical and running in FP32 avoids any subtle numerical
        differences that could affect reported metrics.

        Args:
            epoch: Current epoch number (for the progress bar only).

        Returns:
            ``(mean_loss, accuracy)`` over the validation split.
        """
        running_loss: float = 0.0
        correct: int = 0
        total: int = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch:4d}/{self.config.num_epochs} [Val]  ",
            leave=False,
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for images, targets in pbar:
                images  = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                batch_size = targets.size(0)
                running_loss += loss.item() * batch_size
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += batch_size

                pbar.set_postfix(
                    loss=f"{running_loss / total:.4f}",
                    acc=f"{correct / total:.3f}",
                )

        return running_loss / total, correct / total

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _get_current_lr(self) -> float:
        """Return the current learning rate from the first parameter group."""
        return self.optimizer.param_groups[0]["lr"]

    def _compute_grad_norm(self) -> float:
        """Compute the global L2 gradient norm without clipping."""
        total_sq: float = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_sq += p.grad.detach().norm(2).item() ** 2
        return total_sq ** 0.5

    def _log_gradient_norms(self, global_step: int) -> None:
        """Log per-layer-group gradient norms to TensorBoard.

        Groups correspond to the named layer groups in ResNet-34:
        ``stem``, ``layer1``–``layer4``, and ``fc``.  Monitoring these
        separately reveals vanishing gradient patterns in early layers —
        the primary diagnostic for Ablation A1 (skip connections).
        """
        layer_groups = ["stem", "layer1", "layer2", "layer3", "layer4", "fc"]

        for group_name in layer_groups:
            sq_sum: float = 0.0
            found = False
            for name, param in self.model.named_parameters():
                if group_name in name and param.grad is not None:
                    sq_sum += param.grad.detach().norm(2).item() ** 2
                    found = True
            if found:
                self._log.log_scalar(
                    f"gradients/{group_name}_norm",
                    sq_sum ** 0.5,
                    global_step,
                )

    def _step_scheduler(self, val_loss: float) -> None:
        """Step the LR scheduler according to its type.

        - :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`: requires
          the validation loss as an argument.
        - :class:`~torch.optim.lr_scheduler.OneCycleLR`: already stepped
          per-batch inside :meth:`_train_epoch`; skip here.
        - All others: step once per epoch with no arguments.
        """
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        elif isinstance(self.scheduler, OneCycleLR):
            pass  # Already stepped per-batch.
        else:
            self.scheduler.step()

    def _build_checkpoint_state(
        self, epoch: int, val_acc: float
    ) -> dict[str, Any]:
        """Assemble the complete checkpoint dictionary.

        Args:
            epoch: Current epoch number.
            val_acc: Validation accuracy at this epoch.

        Returns:
            A dictionary with all state required to resume training
            bit-for-bit (given deterministic mode was active).
        """
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
            "metrics": {
                "val_acc": val_acc,
                "val_loss": (
                    self._val_loss_history[-1]
                    if self._val_loss_history
                    else float("nan")
                ),
                "train_loss": (
                    self._train_loss_history[-1]
                    if self._train_loss_history
                    else float("nan")
                ),
                "epoch": epoch,
            },
            "config": self.config.to_dict(),
            "rng_states": get_rng_state(),
            "best_val_acc": self._best_val_acc,
            "train_loss_history": list(self._train_loss_history),
            "val_loss_history": list(self._val_loss_history),
            "val_acc_history": list(self._val_acc_history),
        }
