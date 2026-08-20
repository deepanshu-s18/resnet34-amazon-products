"""Training checkpoint management.

This module provides :class:`CheckpointManager`, which handles the complete
checkpoint lifecycle:

- **Saving** model weights, optimiser state, scheduler state, RNG states,
  training history, and a config snapshot to a structured directory.
- **Loading** any checkpoint for resumption or evaluation.
- **Best-model tracking**: the checkpoint with the highest validation
  accuracy is always accessible at ``checkpoints/<name>/best.pt``.
- **Pruning**: only the ``keep_n_best`` highest-accuracy checkpoints are
  retained; older ones are deleted automatically.

Checkpoint format
-----------------
Every checkpoint is a single ``.pt`` file containing::

    {
        "epoch":                  int,
        "model_state_dict":       dict,
        "optimizer_state_dict":   dict,
        "scheduler_state_dict":   dict | None,
        "metrics":                {"val_acc": float, "val_loss": float, ...},
        "config":                 dict,
        "rng_states":             {"python": ..., "numpy": ..., "torch": ..., "cuda": ...},
        "best_val_acc":           float,
        "train_loss_history":     list[float],
        "val_loss_history":       list[float],
        "val_acc_history":        list[float],
    }

The config snapshot is also serialised as a human-readable YAML file in the
same directory so that checkpoint directories are self-documenting.

Atomic writes
-------------
Checkpoints are written to a temporary file first, then renamed.  This
prevents a corrupted checkpoint from being left on disk if the process is
interrupted mid-write.

Example::

    from pathlib import Path
    from src.utils.checkpoint import CheckpointManager

    manager = CheckpointManager(
        checkpoint_dir=Path("checkpoints"),
        experiment_name="cifar10_resnet34_seed42",
    )

    # During training:
    path = manager.save(state_dict, is_best=True, epoch=50, val_acc=0.932)

    # To resume:
    state = manager.load(manager.get_best_path())
    model.load_state_dict(state["model_state_dict"])
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

__all__ = ["CheckpointManager"]

logger = logging.getLogger(__name__)

# Filename for the best-model soft reference (copy, not symlink, for
# cross-platform compatibility).
_BEST_FILENAME = "best.pt"


class CheckpointManager:
    """Save, load, and prune training checkpoints.

    Checkpoints are stored under ``checkpoint_dir / experiment_name /``.
    Each epoch's checkpoint is named::

        epoch_{epoch:04d}_val{val_acc:.4f}.pt

    e.g. ``epoch_0050_val0.9320.pt``.

    After saving, the manager:

    1. Copies the file to ``best.pt`` if ``is_best=True``.
    2. Prunes the directory to keep only the ``keep_n_best`` checkpoints
       with the highest validation accuracy, deleting the rest.

    Args:
        checkpoint_dir: Parent directory under which per-experiment
            subdirectories are created.
        experiment_name: Unique experiment identifier.  Forms the
            subdirectory name.
        keep_n_best: Number of best checkpoints to retain.  The best
            checkpoint (``best.pt``) is never pruned regardless of this
            value.  Defaults to ``3``.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        experiment_name: str,
        keep_n_best: int = 3,
    ) -> None:
        self._dir = Path(checkpoint_dir) / experiment_name
        self._dir.mkdir(parents=True, exist_ok=True)
        self._keep_n_best = keep_n_best
        # Registry: list of (val_acc, path) sorted descending.
        self._registry: list[tuple[float, Path]] = []

        logger.info(
            "CheckpointManager | dir=%s | keep_n_best=%d",
            self._dir,
            keep_n_best,
        )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    def save(
        self,
        state: dict[str, Any],
        is_best: bool,
        epoch: int,
        val_acc: float,
    ) -> Path:
        """Atomically save a checkpoint to disk.

        Args:
            state: Complete checkpoint dictionary (see module docstring for
                the expected keys).  Must include at least
                ``"model_state_dict"`` and ``"optimizer_state_dict"``.
            is_best: If ``True``, the checkpoint is also copied to
                ``best.pt`` in the experiment directory.
            epoch: Current training epoch (used in the filename).
            val_acc: Validation accuracy achieved at this epoch (used in
                the filename and for pruning decisions).

        Returns:
            The absolute :class:`~pathlib.Path` of the saved checkpoint.
        """
        filename = f"epoch_{epoch:04d}_val{val_acc:.4f}.pt"
        target = self._dir / filename

        # Atomic write: save to a temp file in the same directory, then
        # rename.  rename() is atomic on POSIX and near-atomic on Windows.
        with tempfile.NamedTemporaryFile(
            dir=self._dir, suffix=".tmp", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)

        try:
            torch.save(state, tmp_path)
            tmp_path.rename(target)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        logger.info("Checkpoint saved → %s", target)

        if is_best:
            best_path = self._dir / _BEST_FILENAME
            shutil.copy2(target, best_path)
            logger.info("New best checkpoint → %s  (val_acc=%.4f)", best_path, val_acc)

        # Update registry and prune.
        self._registry.append((val_acc, target))
        self._registry.sort(key=lambda x: x[0], reverse=True)
        self._prune()

        # Persist the config snapshot as YAML alongside the checkpoints.
        if "config" in state and state["config"]:
            self._save_config_snapshot(state["config"])

        return target

    # -----------------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------------

    def load(
        self,
        checkpoint_path: Path,
        map_location: Optional[str] = None,
    ) -> dict[str, Any]:
        """Load a checkpoint from disk.

        Args:
            checkpoint_path: Absolute or relative path to the ``.pt`` file.
            map_location: Forwarded to :func:`torch.load`.  Set to
                ``"cpu"`` to load a GPU checkpoint on a CPU-only machine.
                Defaults to ``None`` (preserves original device).

        Returns:
            The checkpoint dictionary.

        Raises:
            FileNotFoundError: If ``checkpoint_path`` does not exist.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}.\n"
                f"Available checkpoints in {self._dir}:\n"
                + "\n".join(f"  {p.name}" for p in self._dir.glob("*.pt"))
            )

        state = torch.load(path, map_location=map_location, weights_only=False)
        logger.info(
            "Checkpoint loaded ← %s  (epoch=%s, val_acc=%s)",
            path,
            state.get("epoch", "?"),
            state.get("metrics", {}).get("val_acc", "?"),
        )
        return state

    def load_best(self, map_location: Optional[str] = None) -> dict[str, Any]:
        """Load the best checkpoint (``best.pt``).

        Args:
            map_location: Forwarded to :func:`torch.load`.

        Returns:
            The best checkpoint dictionary.

        Raises:
            FileNotFoundError: If no best checkpoint exists yet.
        """
        return self.load(self.get_best_path(), map_location=map_location)

    # -----------------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------------

    def get_best_path(self) -> Path:
        """Return the path to the best checkpoint.

        Raises:
            FileNotFoundError: If no best checkpoint has been saved yet.
        """
        best = self._dir / _BEST_FILENAME
        if not best.exists():
            raise FileNotFoundError(
                f"No best checkpoint found at {best}. "
                "Has training produced at least one checkpoint?"
            )
        return best

    def has_best(self) -> bool:
        """Return ``True`` if a best checkpoint exists on disk."""
        return (self._dir / _BEST_FILENAME).exists()

    def list_checkpoints(self) -> list[Path]:
        """Return all epoch checkpoint paths sorted by val_acc descending."""
        return [p for _, p in self._registry]

    @property
    def checkpoint_dir(self) -> Path:
        """The directory where checkpoints for this experiment are stored."""
        return self._dir

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _prune(self) -> None:
        """Delete epoch checkpoints beyond the ``keep_n_best`` threshold."""
        if len(self._registry) <= self._keep_n_best:
            return
        # Keep the top keep_n_best; delete the rest.
        to_keep = {p for _, p in self._registry[: self._keep_n_best]}
        for _, path in self._registry[self._keep_n_best :]:
            if path.exists() and path not in to_keep:
                path.unlink()
                logger.debug("Pruned checkpoint %s", path)
        self._registry = self._registry[: self._keep_n_best]

    def _save_config_snapshot(self, config: dict[str, Any]) -> None:
        """Serialise ``config`` to YAML in the checkpoint directory."""
        snapshot_path = self._dir / "config_snapshot.yaml"
        try:
            import yaml  # type: ignore

            with open(snapshot_path, "w") as fh:
                yaml.safe_dump(config, fh, default_flow_style=False, sort_keys=True)
            logger.debug("Config snapshot → %s", snapshot_path)
        except ImportError:
            # PyYAML not installed — fall back to JSON.
            import json

            snapshot_path = self._dir / "config_snapshot.json"
            with open(snapshot_path, "w") as fh:
                json.dump(config, fh, indent=2, default=str)
            logger.debug("Config snapshot (JSON fallback) → %s", snapshot_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save config snapshot: %s", exc)
