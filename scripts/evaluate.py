#!/usr/bin/env python3
"""CLI entry point for model evaluation and Grad-CAM generation.

Usage
-----
::

    # Full evaluation on test set
    python scripts/evaluate.py \\
        --checkpoint results/cifar10/cifar10_resnet34_E002_seed42/checkpoints/best.pt \\
        --config configs/cifar10_resnet34.yaml

    # Evaluation + Grad-CAM visualizations
    python scripts/evaluate.py \\
        --checkpoint results/cifar10/cifar10_resnet34_E002_seed42/checkpoints/best.pt \\
        --config configs/cifar10_resnet34.yaml \\
        --compute-grad-cam \\
        --grad-cam-samples 20

    # Evaluate on val split (during development)
    python scripts/evaluate.py \\
        --checkpoint checkpoints/best.pt \\
        --config configs/cifar10_resnet34.yaml \\
        --split val

    # Load checkpoint and print summary only (no file output)
    python scripts/evaluate.py \\
        --checkpoint checkpoints/best.pt \\
        --config configs/cifar10_resnet34.yaml \\
        --print-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import yaml

from src.models.model_factory import ModelFactory
from src.utils.logger import setup_root_logging
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a ResNet-34 checkpoint and optionally generate Grad-CAM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", required=True, metavar="PATH",
        help="Path to .pt checkpoint file.",
    )
    parser.add_argument(
        "--config", required=True, metavar="PATH",
        help="Path to experiment YAML config (must match training config).",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--output-dir", default=None, metavar="DIR",
        help="Directory to save evaluation outputs. Default: checkpoint_dir/evaluation/",
    )
    parser.add_argument(
        "--compute-grad-cam", action="store_true",
        help="Generate Grad-CAM visualizations for correct and failed predictions.",
    )
    parser.add_argument(
        "--grad-cam-samples", type=int, default=20, metavar="N",
        help="Number of Grad-CAM samples to generate (default: 20 each for correct and failures).",
    )
    parser.add_argument(
        "--grad-cam-layer", default="layer4", metavar="LAYER",
        help="Target layer for Grad-CAM (default: layer4).",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Compute device (default: auto).",
    )
    parser.add_argument(
        "--print-only", action="store_true",
        help="Print metrics to console only; do not save files.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """Load YAML config, merging with base_config.yaml if present."""
    path = Path(config_path)
    base_path = path.parent / "base_config.yaml"

    cfg: dict = {}
    if base_path.exists():
        with open(base_path) as fh:
            cfg = yaml.safe_load(fh) or {}

    with open(path) as fh:
        override = yaml.safe_load(fh) or {}

    # Simple recursive merge.
    def merge(base: dict, over: dict) -> dict:
        r = dict(base)
        for k, v in over.items():
            r[k] = merge(r[k], v) if isinstance(v, dict) and isinstance(r.get(k), dict) else v
        return r

    return merge(cfg, override)


# ---------------------------------------------------------------------------
# DataLoader helper
# ---------------------------------------------------------------------------


def create_eval_dataloader(cfg: dict, split: str, device: torch.device):
    """Create a DataLoader for the specified evaluation split."""
    from src.data.dataloaders import DataLoaderConfig, create_dataloaders
    from src.data.transforms import AugmentationConfig

    data_cfg = cfg.get("data", {})

    dl_config = DataLoaderConfig(
        batch_size        = data_cfg.get("batch_size", 128) * 2,
        val_batch_size    = data_cfg.get("batch_size", 128) * 2,
        num_workers       = data_cfg.get("num_workers", 4),
        pin_memory        = data_cfg.get("pin_memory", True),
        persistent_workers= data_cfg.get("num_workers", 4) > 0,
        use_weighted_sampler=False,
    )

    # No augmentation at evaluation.
    pipeline = create_dataloaders(
        dataset_name = data_cfg.get("dataset_name", "cifar10"),
        data_root    = data_cfg.get("data_root", "data/"),
        dl_config    = dl_config,
        aug_config   = AugmentationConfig(),   # all augmentation OFF
        image_size   = data_cfg.get("image_size", 32),
        val_fraction = data_cfg.get("val_fraction", 0.1),
        seed         = data_cfg.get("seed", 42),
    )

    return pipeline.loaders[split], pipeline


# ---------------------------------------------------------------------------
# Grad-CAM generation
# ---------------------------------------------------------------------------


def generate_grad_cam_visualizations(
    model: torch.nn.Module,
    dataloader,
    class_names: list[str],
    device: torch.device,
    output_dir: Path,
    layer_name: str = "layer4",
    n_samples: int = 20,
    mean: tuple = (0.4914, 0.4822, 0.4465),
    std: tuple  = (0.2470, 0.2435, 0.2616),
) -> None:
    """Generate Grad-CAM for correct and failed predictions."""
    from src.explainability.gradcam import (
        GradCAM, get_target_layer, visualize_gradcam,
        visualize_class_comparison,
    )

    correct_dir = output_dir / "gradcam" / "correct"
    failure_dir = output_dir / "gradcam" / "failures"
    correct_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    target_layer = get_target_layer(model, layer_name)

    correct_batches: list = []
    failure_batches: list = []
    n_correct = n_failure = 0

    model.eval()
    with GradCAM(model, target_layer) as gcam:
        for images, targets in dataloader:
            if n_correct >= n_samples and n_failure >= n_samples:
                break

            images_d = images.to(device)
            with torch.no_grad():
                logits = model(images_d)
                preds  = logits.argmax(dim=1).cpu()

            mask_correct = preds == targets
            mask_failure = ~mask_correct

            # Collect correct samples
            if n_correct < n_samples and mask_correct.any():
                idx = mask_correct.nonzero(as_tuple=False).squeeze(1)[:n_samples - n_correct]
                batch_imgs = images[idx]
                batch_tgts = targets[idx]
                batch_preds = preds[idx]

                cams = gcam.generate(batch_imgs.to(device), target_class=None)
                correct_batches.append((batch_imgs.cpu(), cams.cpu(), batch_preds, batch_tgts))
                n_correct += len(idx)

            # Collect failure samples
            if n_failure < n_samples and mask_failure.any():
                idx = mask_failure.nonzero(as_tuple=False).squeeze(1)[:n_samples - n_failure]
                batch_imgs = images[idx]
                batch_tgts = targets[idx]
                batch_preds = preds[idx]

                cams = gcam.generate(batch_imgs.to(device), target_class=None)
                failure_batches.append((batch_imgs.cpu(), cams.cpu(), batch_preds, batch_tgts))
                n_failure += len(idx)

    # Save correct predictions grid
    if correct_batches:
        import torch as th
        all_imgs  = th.cat([b[0] for b in correct_batches])[:n_samples]
        all_cams  = th.cat([b[1] for b in correct_batches])[:n_samples]
        all_preds = th.cat([b[2] for b in correct_batches])[:n_samples]
        all_tgts  = th.cat([b[3] for b in correct_batches])[:n_samples]

        fig = visualize_gradcam(
            images=all_imgs, cams=all_cams,
            predictions=all_preds, targets=all_tgts,
            class_names=class_names,
            mean=mean, std=std,
            n_samples=min(n_samples, len(all_imgs)),
            title="Grad-CAM — Correctly Classified",
            save_path=correct_dir / "gradcam_correct_grid.png",
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
        logger.info("Correct predictions Grad-CAM → %s", correct_dir)

    # Save failure predictions grid
    if failure_batches:
        all_imgs  = th.cat([b[0] for b in failure_batches])[:n_samples]
        all_cams  = th.cat([b[1] for b in failure_batches])[:n_samples]
        all_preds = th.cat([b[2] for b in failure_batches])[:n_samples]
        all_tgts  = th.cat([b[3] for b in failure_batches])[:n_samples]

        fig = visualize_gradcam(
            images=all_imgs, cams=all_cams,
            predictions=all_preds, targets=all_tgts,
            class_names=class_names,
            mean=mean, std=std,
            n_samples=min(n_samples, len(all_imgs)),
            title="Grad-CAM — Misclassified Samples",
            save_path=failure_dir / "gradcam_failures_grid.png",
        )
        plt.close(fig)
        logger.info("Failure predictions Grad-CAM → %s", failure_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    setup_root_logging(args.log_level)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Config
    cfg       = load_config(args.config)
    model_cfg = cfg.get("model", {})
    data_cfg  = cfg.get("data", {})

    set_global_seed(data_cfg.get("seed", 42))

    # Output directory
    ckpt_path = Path(args.checkpoint)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ckpt_path.parent.parent / "evaluation"

    if not args.print_only:
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Evaluating: %s", ckpt_path)
    logger.info("Split: %s", args.split)
    logger.info("Device: %s", device)

    # Load model
    model, checkpoint = ModelFactory.load_from_checkpoint(
        checkpoint_path = ckpt_path,
        architecture    = model_cfg.get("architecture", "resnet34"),
        num_classes     = model_cfg.get("num_classes", 10),
        device          = device,
        dataset         = model_cfg.get("dataset", "cifar10"),
    )

    # Create DataLoader
    try:
        eval_loader, pipeline = create_eval_dataloader(cfg, args.split, device)
    except Exception as exc:
        logger.error("Failed to create DataLoader: %s", exc)
        return 1

    class_names = pipeline.class_names

    # Run evaluation
    try:
        from src.evaluation.evaluator import Evaluator

        evaluator = Evaluator(
            model        = model,
            dataloader   = eval_loader,
            device       = device,
            class_names  = class_names,
            topk         = (1, 5) if model_cfg.get("num_classes", 10) >= 5 else (1,),
            extract_features = False,
        )
        results = evaluator.evaluate()

    except Exception as exc:
        logger.error("Evaluation failed: %s", exc, exc_info=True)
        return 1

    # Print summary
    print(results)

    if args.print_only:
        return 0

    # Save results
    evaluator.generate_report(results, save_dir=output_dir)
    logger.info("Evaluation report → %s", output_dir)

    # Generate Grad-CAM
    if args.compute_grad_cam:
        if not hasattr(model, "layer4"):
            logger.warning(
                "Model does not have '%s' — Grad-CAM skipped.", args.grad_cam_layer
            )
        else:
            logger.info("Generating Grad-CAM on %d samples...", args.grad_cam_samples)
            dataset_name = data_cfg.get("dataset_name", "cifar10")
            from src.data.transforms import NORMALIZATION_STATS as _NSTATS
            try:
                nstats = _NSTATS.get(dataset_name, _NSTATS.get("imagenet"))
            except AttributeError:
                nstats = None

            mean = nstats.mean if nstats else (0.4914, 0.4822, 0.4465)
            std  = nstats.std  if nstats else (0.2470, 0.2435, 0.2616)

            try:
                generate_grad_cam_visualizations(
                    model        = model,
                    dataloader   = eval_loader,
                    class_names  = class_names,
                    device       = device,
                    output_dir   = output_dir,
                    layer_name   = args.grad_cam_layer,
                    n_samples    = args.grad_cam_samples,
                    mean         = mean,
                    std          = std,
                )
            except Exception as exc:
                logger.warning("Grad-CAM generation failed: %s — skipping.", exc)

    # Final summary
    summary = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "epoch": checkpoint.get("epoch"),
        **results.scalar_metrics(),
    }
    with open(output_dir / "evaluation_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("=" * 50)
    logger.info("Evaluation complete.")
    logger.info("Top-1 Accuracy: %.4f", results.top1_accuracy)
    logger.info("F1 Macro:       %.4f", results.f1_macro)
    logger.info("ECE:            %.4f", results.ece)
    logger.info("Results →       %s", output_dir)
    logger.info("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
