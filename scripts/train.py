#!/usr/bin/env python3
"""CLI entry point for launching ResNet-34 training experiments.

Usage
-----
::

    # E-002: ResNet-34 on CIFAR-10
    python scripts/train.py --config configs/cifar10_resnet34.yaml

    # ABL-A1: Skip connection ablation
    python scripts/train.py --config configs/ablation_no_skip.yaml

    # Resume from checkpoint
    python scripts/train.py --config configs/cifar10_resnet34.yaml \\
        --resume checkpoints/cifar10_resnet34_E002_seed42/best.pt

    # Dry run (2 batches — verifies setup without full training)
    python scripts/train.py --config configs/cifar10_resnet34.yaml --dry-run

    # Override seed for multi-seed ablation
    python scripts/train.py --config configs/ablation_no_skip.yaml --seed 123

    # Force CPU (for debugging)
    python scripts/train.py --config configs/cifar10_resnet34.yaml --device cpu

Design principle: thin script, fat modules
------------------------------------------
All logic lives in ``src/``.  This script is intentionally short — it does
only:

1. Parse arguments
2. Load YAML config
3. Set global seed
4. Create experiment directory + save config snapshot
5. Setup logging
6. Assemble components (model, optimizer, scheduler, loss, dataloaders)
7. Instantiate Trainer
8. Optionally resume from checkpoint
9. Call trainer.train()
10. Save final experiment summary JSON

No training logic, no metric computation, no augmentation decisions live
here.  Adding a new dataset, architecture, or scheduler requires no changes
to this script — only a new config YAML and (if needed) a new module.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import sys
import time
from pathlib import Path

# Make the repository root importable regardless of where the script is run.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import yaml

from src.models.model_factory import ModelFactory
from src.training.losses import LossFactory
from src.training.optimizers import create_optimizer
from src.training.schedulers import create_scheduler
from src.training.trainer import Trainer, TrainerConfig
from src.utils.logger import ExperimentLogger, setup_root_logging
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Config YAML values serve as defaults; CLI flags override them when
    provided, enabling quick one-off overrides without editing the YAML.
    """
    parser = argparse.ArgumentParser(
        description="Train ResNet-34 (or variant) from scratch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", required=True, metavar="PATH",
        help="Path to experiment YAML config file (e.g. configs/cifar10_resnet34.yaml).",
    )
    parser.add_argument(
        "--base-config", default=None, metavar="PATH",
        help="Optional base config whose values are overridden by --config.",
    )
    parser.add_argument(
        "--resume", default=None, metavar="CHECKPOINT",
        help="Path to checkpoint .pt file to resume training from.",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Compute device. 'auto' selects CUDA if available (default).",
    )
    parser.add_argument(
        "--results-root", default="results/", metavar="DIR",
        help="Root directory for experiment outputs (default: results/).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run for 2 batches only — validates setup without full training.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the seed in the config file.",
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable Automatic Mixed Precision even if config requests it.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level (default: INFO).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading and merging
# ---------------------------------------------------------------------------


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file into a plain dict."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base.  Override values win.

    Does not modify either input dict — returns a new dict.
    """
    result = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(args: argparse.Namespace) -> dict:
    """Load the experiment config, optionally merging with a base config."""
    if args.base_config:
        base = load_yaml(args.base_config)
    else:
        base_path = Path(args.config).parent / "base_config.yaml"
        if base_path.exists():
            base = load_yaml(base_path)
            logger.debug("Auto-loaded base config from %s", base_path)
        else:
            base = {}

    override = load_yaml(args.config)
    cfg = deep_merge(base, override)

    # Apply CLI overrides.
    if args.seed is not None:
        cfg.setdefault("training", {})["seed"] = args.seed
        cfg.setdefault("data", {})["seed"]     = args.seed

    if args.no_amp:
        cfg.setdefault("training", {})["use_amp"] = False

    if args.dry_run:
        # Minimise epochs for dry run — still exercises the full loop.
        cfg.setdefault("training", {})["num_epochs"] = 2
        cfg.setdefault("training", {})["early_stopping_patience"] = 9999
        cfg.setdefault("training", {})["checkpoint_every_n_epochs"] = 1

    return cfg


# ---------------------------------------------------------------------------
# DataLoader creation (thin wrapper around existing dataloaders.py)
# ---------------------------------------------------------------------------


def create_dataloaders(cfg: dict, device: torch.device, dry_run: bool):
    """Create train/val DataLoaders from config.

    Returns:
        ``(train_loader, val_loader)``
    """
    # Import here to avoid circular deps at module level.
    from src.data.dataloaders import DataLoaderConfig, create_dataloaders as _create
    from src.data.transforms import (
        AugmentationConfig,
        abo_augmentation,
        cifar10_augmentation,
    )

    data_cfg = cfg.get("data", {})
    aug_cfg  = cfg.get("augmentation", {})
    dataset_name = data_cfg.get("dataset_name", "cifar10")

    # Build augmentation config from YAML fields.
    aug = AugmentationConfig(
        random_crop                = aug_cfg.get("random_crop", True),
        crop_padding               = aug_cfg.get("crop_padding", 4),
        horizontal_flip            = aug_cfg.get("random_horizontal_flip", True),
        horizontal_flip_prob       = aug_cfg.get("horizontal_flip_prob", 0.5),
        color_jitter               = any(
            aug_cfg.get(f"color_jitter_{k}", 0.0) > 0
            for k in ["brightness", "contrast", "saturation", "hue"]
        ),
        color_jitter_brightness    = aug_cfg.get("color_jitter_brightness", 0.0),
        color_jitter_contrast      = aug_cfg.get("color_jitter_contrast",   0.0),
        color_jitter_saturation    = aug_cfg.get("color_jitter_saturation", 0.0),
        color_jitter_hue           = aug_cfg.get("color_jitter_hue",        0.0),
        random_grayscale           = aug_cfg.get("random_grayscale_prob", 0.0) > 0,
        random_grayscale_prob      = aug_cfg.get("random_grayscale_prob", 0.0),
        random_erasing             = aug_cfg.get("random_erasing_prob", 0.0) > 0,
        random_erasing_prob        = aug_cfg.get("random_erasing_prob", 0.0),
    )

    dl_config = DataLoaderConfig(
        batch_size          = data_cfg.get("batch_size", 128),
        val_batch_size      = data_cfg.get("batch_size", 128) * 2,
        num_workers         = 0 if dry_run else data_cfg.get("num_workers", 4),
        pin_memory          = data_cfg.get("pin_memory", True),
        persistent_workers  = not dry_run and data_cfg.get("num_workers", 4) > 0,
    )

    pipeline = _create(
        dataset_name = dataset_name,
        data_root    = data_cfg.get("data_root", "data/"),
        dl_config    = dl_config,
        aug_config   = aug,
        image_size   = data_cfg.get("image_size", 32),
        val_fraction = data_cfg.get("val_fraction", 0.1),
        seed         = data_cfg.get("seed", 42),
    )
    return pipeline.train_loader, pipeline.val_loader


# ---------------------------------------------------------------------------
# Hardware info logging
# ---------------------------------------------------------------------------


def log_hardware_info() -> None:
    """Log hardware and software environment at experiment start."""
    logger.info("=" * 60)
    logger.info("Platform:    %s", platform.platform())
    logger.info("Python:      %s", sys.version.split()[0])
    logger.info("PyTorch:     %s", torch.__version__)
    if torch.cuda.is_available():
        logger.info("CUDA:        %s", torch.version.cuda)
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            logger.info(
                "GPU[%d]:      %s (%.1f GB)",
                i, prop.name, prop.total_memory / 1024**3,
            )
    else:
        logger.info("CUDA:        not available — running on CPU")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Main entry point.  Returns exit code (0 = success, 1 = failure)."""
    args = parse_args()
    setup_root_logging(args.log_level)

    # ── 1. Load config ────────────────────────────────────────────────────
    cfg = load_config(args)
    train_cfg  = cfg.get("training", {})
    model_cfg  = cfg.get("model",    {})
    data_cfg   = cfg.get("data",     {})

    experiment_name = train_cfg.get("experiment_name", "experiment")
    seed            = train_cfg.get("seed", 42)

    # ── 2. Set global seed (MUST happen before any random operation) ───────
    set_global_seed(seed, deterministic=True, warn_only=True)
    logger.info("Global seed set: %d", seed)

    # ── 3. Select device ──────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    # ── 4. Create experiment directory ────────────────────────────────────
    results_root = Path(args.results_root)
    experiment_dir = results_root / data_cfg.get("dataset_name", "cifar10") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot to experiment directory.
    snapshot_path = experiment_dir / "config_snapshot.yaml"
    with open(snapshot_path, "w") as fh:
        yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=True)
    shutil.copy(args.config, experiment_dir / "original_config.yaml")
    logger.info("Config snapshot → %s", snapshot_path)

    # ── 5. Setup experiment logger ─────────────────────────────────────────
    exp_logger = ExperimentLogger(
        experiment_name=experiment_name,
        log_dir=experiment_dir,
        level=args.log_level,
        use_tensorboard=True,
    )
    exp_logger.log_config(cfg)
    log_hardware_info()

    exp_logger.info("=" * 60)
    exp_logger.info("EXPERIMENT START: %s", experiment_name)
    exp_logger.info("Config: %s", args.config)
    exp_logger.info("Dry run: %s", args.dry_run)
    exp_logger.info("=" * 60)

    try:
        # ── 6. Create DataLoaders ─────────────────────────────────────────
        exp_logger.info("Creating DataLoaders...")
        train_loader, val_loader = create_dataloaders(cfg, device, args.dry_run)
        steps_per_epoch = len(train_loader)
        exp_logger.info(
            "DataLoaders ready: train=%d batches/epoch, val=%d batches",
            steps_per_epoch, len(val_loader),
        )

        # ── 7. Create model ───────────────────────────────────────────────
        exp_logger.info("Creating model: %s", model_cfg.get("architecture"))
        model = ModelFactory.create(
            architecture  = model_cfg.get("architecture", "resnet34"),
            num_classes   = model_cfg.get("num_classes", 10),
            device        = device,
            dataset       = model_cfg.get("dataset", "cifar10"),
            dropout_rate  = model_cfg.get("dropout_rate", 0.0),
        )
        n_params = sum(p.numel() for p in model.parameters())
        exp_logger.info("Model params: %.2fM", n_params / 1e6)

        # ── 8. Create loss ────────────────────────────────────────────────
        criterion = LossFactory.create(
            label_smoothing = train_cfg.get("label_smoothing", 0.0),
            num_classes     = model_cfg.get("num_classes", 10),
        )

        # ── 9. Create optimizer ───────────────────────────────────────────
        optimizer = create_optimizer(
            model          = model,
            optimizer_name = train_cfg.get("optimizer", "sgd"),
            lr             = train_cfg.get("lr", 0.1),
            weight_decay   = train_cfg.get("weight_decay", 1e-4),
            momentum       = train_cfg.get("momentum", 0.9),
            nesterov       = train_cfg.get("nesterov", True),
        )

        # ── 10. Create scheduler ──────────────────────────────────────────
        num_epochs = train_cfg.get("num_epochs", 200)
        scheduler = create_scheduler(
            optimizer        = optimizer,
            scheduler_name   = train_cfg.get("scheduler", "step"),
            num_epochs       = num_epochs,
            steps_per_epoch  = steps_per_epoch,
            lr               = train_cfg.get("lr", 0.1),
            warmup_epochs    = train_cfg.get("warmup_epochs", 0),
            step_size        = train_cfg.get("step_size", 50),
            gamma            = train_cfg.get("gamma", 0.1),
        )

        # ── 11. Build TrainerConfig ───────────────────────────────────────
        trainer_config = TrainerConfig(
            num_epochs               = num_epochs,
            grad_clip_norm           = train_cfg.get("grad_clip_norm"),
            use_amp                  = train_cfg.get("use_amp", False),
            log_every_n_steps        = train_cfg.get("log_every_n_steps", 50),
            checkpoint_every_n_epochs= train_cfg.get("checkpoint_every_n_epochs", 10),
            early_stopping_patience  = train_cfg.get("early_stopping_patience", 9999),
            early_stopping_min_delta = train_cfg.get("early_stopping_min_delta", 1e-4),
            experiment_name          = experiment_name,
            seed                     = seed,
        )

        # ── 12. Create Trainer ────────────────────────────────────────────
        trainer = Trainer(
            model          = model,
            optimizer      = optimizer,
            criterion      = criterion,
            train_loader   = train_loader,
            val_loader     = val_loader,
            config         = trainer_config,
            device         = device,
            experiment_dir = experiment_dir,
            scheduler      = scheduler,
            logger         = exp_logger,
        )

        # ── 13. Resume from checkpoint if requested ───────────────────────
        if args.resume:
            exp_logger.info("Resuming from checkpoint: %s", args.resume)
            trainer.resume(Path(args.resume))

        # ── 14. Train ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        final_metrics = trainer.train()
        elapsed = time.perf_counter() - t0

        # ── 15. Save experiment summary ───────────────────────────────────
        summary = {
            "experiment_name":   experiment_name,
            "config":            args.config,
            "architecture":      model_cfg.get("architecture"),
            "dataset":           data_cfg.get("dataset_name"),
            "num_classes":       model_cfg.get("num_classes"),
            "seed":              seed,
            "device":            str(device),
            "dry_run":           args.dry_run,
            "total_params":      n_params,
            **final_metrics,
            "wall_time_seconds": elapsed,
        }
        summary_path = experiment_dir / "experiment_summary.json"
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)

        exp_logger.info("=" * 60)
        exp_logger.info("EXPERIMENT COMPLETE: %s", experiment_name)
        exp_logger.info("Best val accuracy: %.4f at epoch %d",
                        final_metrics.get("best_val_acc", 0.0),
                        final_metrics.get("best_epoch", 0))
        exp_logger.info("Total training time: %.1f s (%.1f min)",
                        elapsed, elapsed / 60)
        exp_logger.info("Results → %s", experiment_dir)
        exp_logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        exp_logger.warning("Training interrupted by user.")
        return 1

    except Exception as exc:  # noqa: BLE001
        exp_logger.error("Training failed: %s", exc, exc_info=True)
        return 1

    finally:
        exp_logger.close()


if __name__ == "__main__":
    sys.exit(main())
