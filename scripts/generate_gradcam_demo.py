#!/usr/bin/env python3
"""Generate Grad-CAM demonstration images and run the Adebayo sanity check.

This script demonstrates that the GradCAM implementation in
``src/explainability/gradcam.py`` is correct without requiring:
- A trained checkpoint (uses random-weight model)
- The ABO dataset (uses CIFAR-10 or synthetic images)

All output is saved to ``results/gradcam/`` and a summary is printed.

Usage::

    # Full demo on CIFAR-10 (auto-downloads ~170MB)
    python scripts/generate_gradcam_demo.py

    # Synthetic data only (no downloads)
    python scripts/generate_gradcam_demo.py --synthetic

    # Use a trained checkpoint for meaningful CAMs
    python scripts/generate_gradcam_demo.py --checkpoint checkpoints/best.pt

Note on demo images
-------------------
Without a trained checkpoint, CAMs are generated from a randomly initialised
model.  These are labelled "Demo (random weights)" in all figure titles and
README embeds.  They prove the implementation is correct (hooks fire, backward
pass runs, CAMs are generated) but do NOT show meaningful explanations.

Trained-model CAMs that show meaningful localisation will be committed after
the ABO training run completes.  See EXPERIMENT_LOG.md entry E-003.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── Repo root on sys.path ─────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

from src.models.resnet34 import ResNet34
from src.explainability.gradcam import (
    GradCAM,
    get_target_layer,
    run_sanity_check,
    denormalize_image,
    cam_to_heatmap,
)

# ── Constants ─────────────────────────────────────────────────────────────────

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

OUTPUT_DIR = _REPO_ROOT / "results" / "gradcam"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None, metavar="PATH",
                   help="Optional path to a trained .pt checkpoint. "
                        "Without this, random-weight model is used.")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic noise images instead of CIFAR-10. "
                        "Faster (no download) but less visually meaningful.")
    p.add_argument("--n-images", type=int, default=8,
                   help="Number of images in the demo grid (default 8).")
    p.add_argument("--layer", default="layer4",
                   help="Target layer for Grad-CAM (default: 'layer4').")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR), metavar="DIR",
                   help="Directory to save output images.")
    p.add_argument("--device", default=None,
                   help="Device: 'cpu', 'cuda', 'mps'. Auto-detected if omitted.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_cifar10_batch(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Load the first n images from CIFAR-10 test set."""
    try:
        import torchvision
        import torchvision.transforms as T
    except ImportError:
        print("torchvision not available. Using synthetic data instead.")
        return make_synthetic_batch(n, device)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    try:
        ds = torchvision.datasets.CIFAR10(
            root="/tmp/cifar10_gradcam", train=False, transform=transform, download=True
        )
    except Exception as e:
        print(f"Could not load CIFAR-10: {e}. Using synthetic data.")
        return make_synthetic_batch(n, device)

    images, labels = [], []
    for i in range(n):
        img, lbl = ds[i]
        images.append(img)
        labels.append(lbl)

    images_t = torch.stack(images).to(device)
    labels_t = torch.tensor(labels).to(device)
    class_names = [CIFAR10_CLASSES[l] for l in labels]
    return images_t, labels_t, class_names


def make_synthetic_batch(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Create random synthetic images in normalized CIFAR-10 space."""
    torch.manual_seed(42)
    images = torch.randn(n, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (n,), device=device)
    class_names = [CIFAR10_CLASSES[l.item()] for l in labels]
    return images, labels, class_names


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: str | None, device: torch.device) -> ResNet34:
    """Load ResNet-34. Uses random weights if no checkpoint is given."""
    model = ResNet34(num_classes=10, dataset="cifar10")
    is_random = True

    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            print(f"[WARNING] Checkpoint not found: {ckpt_path}. Using random weights.")
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            model.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Loaded checkpoint: {ckpt_path}")
            is_random = False

    if is_random:
        print("[INFO] Using RANDOM weights — CAMs are for implementation demo only.")

    model.eval().to(device)
    return model, is_random


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def save_gradcam_grid(
    images: torch.Tensor,         # (N, 3, H, W) normalised
    cams: torch.Tensor,           # (N, 1, H, W) in [0, 1]
    class_names: list[str],       # length N
    title: str,
    save_path: Path,
    mean: tuple = CIFAR10_MEAN,
    std: tuple  = CIFAR10_STD,
) -> None:
    """Save a grid of original images and their CAM overlays."""
    n = len(images)
    fig = plt.figure(figsize=(n * 2.4, 5.5), facecolor="#0f0f0f")
    fig.suptitle(title, fontsize=11, fontweight="bold", color="white", y=1.01)

    gs = gridspec.GridSpec(2, n, figure=fig, hspace=0.04, wspace=0.04)

    for i in range(n):
        # ── Original image ────────────────────────────────────────────────
        img_np = denormalize_image(images[i].cpu(), mean=mean, std=std)

        ax_orig = fig.add_subplot(gs[0, i])
        ax_orig.imshow(img_np)
        ax_orig.axis("off")
        ax_orig.set_title(class_names[i], fontsize=7, color="#aaaaaa", pad=2)

        # ── CAM overlay ───────────────────────────────────────────────────
        cam_np = cams[i, 0].detach().cpu().numpy()

        # Resize CAM to image size for overlay
        cam_resized = F.interpolate(
            cams[i].unsqueeze(0), size=img_np.shape[:2], mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()

        heatmap = cam_to_heatmap(cam_resized)   # (H, W, 3) in [0,1]
        overlay = 0.55 * img_np + 0.45 * heatmap
        overlay = np.clip(overlay, 0, 1)

        ax_cam = fig.add_subplot(gs[1, i])
        ax_cam.imshow(overlay)
        ax_cam.axis("off")

    # Row labels
    fig.text(0.01, 0.75, "Original", va="center", rotation=90, color="#777777", fontsize=8)
    fig.text(0.01, 0.25, "Grad-CAM", va="center", rotation=90, color="#777777", fontsize=8)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {save_path.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Device: {device}")

    # Load model
    model, is_random = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model: ResNet-34, {n_params:,} params")

    # Load images
    if args.synthetic:
        images, labels, class_names = make_synthetic_batch(args.n_images, device)
        source = "synthetic"
    else:
        images, labels, class_names = load_cifar10_batch(args.n_images, device)
        source = "CIFAR-10"
    print(f"[INFO] Images: {len(images)} from {source}")

    # Target layer
    target_layer = get_target_layer(model, args.layer)
    print(f"[INFO] Target layer: {args.layer} ({type(target_layer).__name__})")

    weight_label = "random-weight model [DEMO]" if is_random else "trained model"
    title_prefix = f"Grad-CAM — layer4 — {weight_label}"

    # ── Generate CAMs ─────────────────────────────────────────────────────
    print("\n[1/3] Generating Grad-CAM maps...")
    with GradCAM(model, target_layer) as gcam:
        cams = gcam.generate(images, target_class=None)   # top-1 class

    save_gradcam_grid(
        images=images,
        cams=cams,
        class_names=class_names,
        title=f"{title_prefix} — All predictions",
        save_path=output_dir / "gradcam_demo_all.png",
    )

    # ── Multi-layer comparison ─────────────────────────────────────────────
    print("\n[2/3] Generating multi-layer comparison...")
    layer_names = ["layer1", "layer2", "layer3", "layer4"]
    n_layers = len(layer_names)
    n_show = min(4, len(images))

    fig, axes = plt.subplots(
        n_layers, n_show + 1,
        figsize=((n_show + 1) * 2.2, n_layers * 2.2),
        facecolor="#0f0f0f"
    )
    fig.suptitle(
        f"Grad-CAM across layers — {weight_label}",
        fontsize=10, fontweight="bold", color="white"
    )

    for row, layer_name in enumerate(layer_names):
        layer = get_target_layer(model, layer_name)
        with GradCAM(model, layer) as gcam:
            layer_cams = gcam.generate(images[:n_show])

        # Row label
        axes[row, 0].text(
            0.5, 0.5, layer_name,
            ha="center", va="center", color="#dddddd",
            fontsize=9, fontweight="bold",
            transform=axes[row, 0].transAxes
        )
        axes[row, 0].axis("off")

        for col in range(n_show):
            img_np = denormalize_image(images[col].cpu(), mean=CIFAR10_MEAN, std=CIFAR10_STD)
            cam_np = layer_cams[col, 0].detach().cpu().numpy()
            cam_resized = F.interpolate(
                layer_cams[col].unsqueeze(0), size=img_np.shape[:2],
                mode="bilinear", align_corners=False
            )[0, 0].cpu().numpy()
            heatmap = cam_to_heatmap(cam_resized)
            overlay = np.clip(0.55 * img_np + 0.45 * heatmap, 0, 1)

            axes[row, col + 1].imshow(overlay)
            if row == 0:
                axes[row, col + 1].set_title(class_names[col], fontsize=7, color="#aaaaaa")
            axes[row, col + 1].axis("off")

    plt.tight_layout()
    multi_path = output_dir / "gradcam_multilayer.png"
    plt.savefig(multi_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {multi_path.relative_to(_REPO_ROOT)}")

    # ── Adebayo sanity check ──────────────────────────────────────────────
    print("\n[3/3] Running Adebayo et al. (2018) sanity check...")
    result = run_sanity_check(
        model=model,
        input_tensor=images,
        layer_name=args.layer,
        target_class=None,
        n_samples=min(4, len(images)),
    )

    print(f"  {result['interpretation']}")

    # Save sanity check output to text file
    sanity_path = output_dir / "gradcam_sanity_check.txt"
    with open(sanity_path, "w") as f:
        f.write("Grad-CAM Adebayo et al. (2018) Sanity Check\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Weight source:     {'random init [DEMO]' if is_random else 'trained checkpoint'}\n")
        f.write(f"Image source:      {source}\n")
        f.write(f"Target layer:      {args.layer}\n")
        f.write(f"n_samples:         {result['mean_correlation']}\n\n")
        f.write(f"Per-sample |ρ|:    {result['correlations']}\n")
        f.write(f"Mean |ρ|:          {result['mean_correlation']:.4f}\n")
        f.write(f"Threshold:         0.5\n")
        f.write(f"Result:            {'PASS' if result['passes'] else 'FAIL'}\n\n")
        f.write(result["interpretation"] + "\n\n")
        f.write("Interpretation:\n")
        f.write("  A low |ρ| (< 0.5) means Grad-CAM maps from the trained model\n")
        f.write("  look different from maps from a randomly-initialised model.\n")
        f.write("  This is the expected behaviour for a faithful explanation method.\n")
        f.write("  High |ρ| (> 0.5) would indicate the maps reflect data structure\n")
        f.write("  rather than learned model behaviour — a red flag.\n")
    print(f"  Saved → {sanity_path.relative_to(_REPO_ROOT)}")

    # ── Write per-results README ───────────────────────────────────────────
    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(f"""# results/gradcam/

Grad-CAM visualizations generated by `scripts/generate_gradcam_demo.py`.

## Contents

| File | Description |
|------|-------------|
| `gradcam_demo_all.png` | {args.n_images} images with layer4 Grad-CAM overlays |
| `gradcam_multilayer.png` | CAMs at layer1 → layer4 for 4 images (shows semantic progression) |
| `gradcam_sanity_check.txt` | Adebayo et al. (2018) sanity check results |

## ⚠️ Demo vs Trained-Model CAMs

> **These images were generated with a {'randomly-initialised' if is_random else 'trained'} model.**

{"Random-weight CAMs do NOT show meaningful localisation — they prove the implementation is correct (hooks fire, backward runs, CAMs are normalized) but the heatmaps are arbitrary." if is_random else "Trained-model CAMs show meaningful localisation of the predicted class."}

Trained-model CAMs with semantic meaning will be committed after ABO training (E-003) completes.
See [EXPERIMENT_LOG.md](../../EXPERIMENT_LOG.md) for status.

## Sanity Check Result

```
{result['interpretation']}
```

Mean |ρ| = **{result['mean_correlation']:.4f}** (threshold 0.5) — {'✅ PASS' if result['passes'] else '❌ FAIL'}

## Regenerate

```bash
# Demo with random weights (fastest)
python scripts/generate_gradcam_demo.py --synthetic

# Demo with CIFAR-10 images
python scripts/generate_gradcam_demo.py

# With a trained checkpoint
python scripts/generate_gradcam_demo.py --checkpoint checkpoints/best.pt
```
""")
    print(f"  Saved → {readme_path.relative_to(_REPO_ROOT)}")

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"""
{'=' * 60}
  GRAD-CAM DEMO COMPLETE
{'=' * 60}
  Output:           results/gradcam/
  Files generated:  3 images + 1 sanity check + README
  Sanity check:     Mean |ρ| = {result['mean_correlation']:.4f} — {'PASS ✓' if result['passes'] else 'FAIL ✗'}
  Weight source:    {'Random init [DEMO — replace with trained checkpoint]' if is_random else 'Trained model'}
{'=' * 60}
""")


if __name__ == "__main__":
    main()
