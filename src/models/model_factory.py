"""Baseline CNN and model factory for ResNet-34 ablation experiments.

This module provides two components:

1. :class:`BasicCNN` — A plain convolutional network without skip connections.
   Used as the ablation A1 baseline to demonstrate the empirical advantage
   of residual learning.  Designed to be a fair comparison:

   - Comparable parameter count (within ±25% of ResNet-34)
   - Same channel widths (64→128→256→512)
   - Same BatchNorm placement
   - Key difference: NO skip connections, NO residual formulation

2. :class:`ModelFactory` — Single entry point for all model creation.
   Training scripts call ``ModelFactory.create()`` — they never import
   ResNet34 or BasicCNN directly.  This decouples the training script
   from architecture implementation details.

Architecture rationale for BasicCNN
--------------------------------------
The BasicCNN must be a *fair* comparison to ResNet-34.  If BasicCNN uses a
different width, depth, or training regime, the ablation conflates skip
connection effects with those differences.

Design decisions:

- Same number of channel width stages: 64, 128, 256, 512
- Each stage uses MaxPool for spatial downsampling (same effect as stride=2)
- BatchNorm is kept in BasicCNN — we isolate skip connections, not BN
- AdaptiveAvgPool2d before the FC layer — same as ResNet-34

This means any performance difference between BasicCNN and ResNet-34 is
attributable specifically to the skip connection mechanism.

Example::

    from src.models.model_factory import ModelFactory
    import torch

    model = ModelFactory.create("resnet34", num_classes=10, dataset="cifar10")
    model = ModelFactory.create("plain34", num_classes=10, dataset="cifar10")
    model = ModelFactory.create("basic_cnn", num_classes=10)

    x = torch.randn(4, 3, 32, 32)
    logits = model(x)   # [4, 10]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

__all__ = ["BasicCNN", "ModelFactory"]

logger = logging.getLogger(__name__)

_SUPPORTED_ARCHITECTURES = frozenset(
    ["resnet34", "plain34", "resnet34_no_bn", "basic_cnn"]
)


# ---------------------------------------------------------------------------
# BasicCNN (Ablation A1 baseline)
# ---------------------------------------------------------------------------


class BasicCNN(nn.Module):
    """Plain convolutional network — ablation A1 control condition.

    Five convolutional stages with BatchNorm and ReLU, using MaxPool for
    spatial downsampling.  No skip connections.  Comparable in parameter
    count and width to ResNet-34 for a controlled ablation comparison.

    Architecture for 32×32 CIFAR-10 inputs::

        Conv(3→64, 3×3) → BN → ReLU → MaxPool(2×2)    → [64, 16, 16]
        Conv(64→128,3×3) → BN → ReLU → MaxPool(2×2)   → [128, 8, 8]
        Conv(128→256,3×3) → BN → ReLU → MaxPool(2×2)  → [256, 4, 4]
        Conv(256→512,3×3) → BN → ReLU                 → [512, 4, 4]
        Conv(512→512,3×3) → BN → ReLU                 → [512, 4, 4]
        AdaptiveAvgPool2d(1, 1)                        → [512, 1, 1]
        Flatten                                        → [512]
        Linear(512 → num_classes)                      → [K]

    Args:
        num_classes: Number of output classes.
        input_channels: Number of input image channels (3 for RGB).
        dropout_rate: Optional dropout probability applied before the FC
            layer.  ``0.0`` disables dropout.
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        def _block(in_c: int, out_c: int, pool: bool = True) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            _block(input_channels,  64,  pool=True),   # [64, H/2, W/2]
            _block(64,             128,  pool=True),   # [128, H/4, W/4]
            _block(128,            256,  pool=True),   # [256, H/8, W/8]
            _block(256,            512,  pool=False),  # [512, H/8, W/8]
            _block(512,            512,  pool=False),  # [512, H/8, W/8]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(512, num_classes)

        # Weight initialisation (same as ResNet-34 for fair comparison).
        self._initialize_weights()

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "BasicCNN: num_classes=%d, params=%.2fM (no skip connections)",
            num_classes, n_params / 1e6,
        )

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images ``[B, 3, H, W]``.

        Returns:
            Raw logits ``[B, num_classes]``.  NOT softmax — the loss
            function handles the log-softmax internally.
        """
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.dropout(out)
        return self.fc(out)

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts broken down by trainable status.

        Returns:
            ``{"total": N, "trainable": M}``
        """
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
# ModelFactory
# ---------------------------------------------------------------------------


class ModelFactory:
    """Single entry point for all model creation and checkpoint loading.

    Training scripts do not import ``ResNet34`` or ``BasicCNN`` directly —
    they call ``ModelFactory.create(arch_name, ...)`` instead.  This pattern:

    - Makes architecture choice config-driven (change a YAML string, not code)
    - Makes adding new architectures backward-compatible
    - Makes the loaded architecture auditable from the checkpoint

    Supported architectures
    -----------------------
    ``"resnet34"``
        Full ResNet-34 with skip connections and BatchNorm.  The primary
        research model.

    ``"plain34"``
        ResNet-34 with all skip connections disabled
        (``use_skip_connection=False``).  Ablation A1 variant.

    ``"resnet34_no_bn"``
        ResNet-34 with BatchNorm removed (``use_batch_norm=False``).
        Ablation A5 variant.  Requires reduced LR for stable training.

    ``"basic_cnn"``
        Plain CNN baseline without skip connections.  Ablation A1 baseline.
    """

    @staticmethod
    def create(
        architecture: str,
        num_classes: int,
        device: Optional[torch.device] = None,
        dataset: str = "cifar10",
        dropout_rate: float = 0.0,
    ) -> nn.Module:
        """Instantiate a model and move it to ``device``.

        Args:
            architecture: Architecture name (case-insensitive).  One of
                ``"resnet34"``, ``"plain34"``, ``"resnet34_no_bn"``,
                ``"basic_cnn"``.
            num_classes: Number of output classes.
            device: Target device.  If ``None``, the model stays on CPU.
            dataset: ``"cifar10"`` or ``"imagenet"`` — controls the stem
                architecture for ResNet variants (no effect on BasicCNN).
            dropout_rate: Dropout probability before the FC layer.

        Returns:
            ``nn.Module`` on ``device``.

        Raises:
            ValueError: If ``architecture`` is not a supported choice.
        """
        # Late import to avoid circular dependency (model_factory ← resnet34
        # ← residual_block).  Only triggered at call time, not import time.
        from src.models.resnet34 import ResNet34  # noqa: PLC0415

        arch = architecture.lower().strip()

        if arch == "resnet34":
            model: nn.Module = ResNet34(
                num_classes=num_classes,
                dataset=dataset,
                use_skip_connection=True,
                use_batch_norm=True,
                dropout_rate=dropout_rate,
            )

        elif arch == "plain34":
            model = ResNet34(
                num_classes=num_classes,
                dataset=dataset,
                use_skip_connection=False,  # ← the only change from resnet34
                use_batch_norm=True,
                dropout_rate=dropout_rate,
            )

        elif arch == "resnet34_no_bn":
            model = ResNet34(
                num_classes=num_classes,
                dataset=dataset,
                use_skip_connection=True,
                use_batch_norm=False,  # ← removes all BatchNorm layers
                dropout_rate=dropout_rate,
            )

        elif arch == "basic_cnn":
            model = BasicCNN(
                num_classes=num_classes,
                dropout_rate=dropout_rate,
            )

        else:
            raise ValueError(
                f"Unknown architecture {architecture!r}. "
                f"Supported: {sorted(_SUPPORTED_ARCHITECTURES)}"
            )

        if device is not None:
            model = model.to(device)

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "ModelFactory: created %s — %d classes, %.2fM params, device=%s",
            arch, num_classes, n_params / 1e6, device or "cpu",
        )
        return model

    @staticmethod
    def load_from_checkpoint(
        checkpoint_path: str | Path,
        architecture: str,
        num_classes: int,
        device: Optional[torch.device] = None,
        dataset: str = "cifar10",
        strict: bool = True,
    ) -> tuple[nn.Module, dict]:
        """Create a model and load weights from a checkpoint file.

        Validates that the checkpoint's saved ``num_classes`` matches the
        requested ``num_classes`` before loading weights — prevents the
        silent FC dimension mismatch that causes uninformative RuntimeErrors.

        Args:
            checkpoint_path: Path to the ``.pt`` checkpoint file.
            architecture: Architecture name (must match the architecture
                used when the checkpoint was saved).
            num_classes: Expected number of output classes.
            device: Target device.
            dataset: Dataset mode for ResNet stem architecture.
            strict: Forwarded to ``model.load_state_dict(strict=strict)``.
                Set to ``False`` when loading a superset of the checkpoint's
                weights (e.g. fine-tuning a subset of layers).

        Returns:
            ``(model, checkpoint_dict)`` — the loaded model and the full
            checkpoint dictionary (includes epoch, metrics, config, etc.).

        Raises:
            FileNotFoundError: If ``checkpoint_path`` does not exist.
            ValueError: If the checkpoint's num_classes does not match.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)

        # Validate class count from saved config (if available).
        saved_config = ckpt.get("config", {})
        saved_classes = saved_config.get("num_classes") or saved_config.get(
            "training", {}
        ).get("num_classes")

        if saved_classes is not None and saved_classes != num_classes:
            raise ValueError(
                f"Checkpoint was trained for num_classes={saved_classes} "
                f"but ModelFactory.load_from_checkpoint was called with "
                f"num_classes={num_classes}.  The FC layer dimensions will "
                f"not match.  Either adjust num_classes or set strict=False."
            )

        model = ModelFactory.create(
            architecture, num_classes, device=device, dataset=dataset
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=strict)
        model.eval()

        logger.info(
            "ModelFactory: loaded %s from %s (epoch=%s, val_acc=%s)",
            architecture,
            path.name,
            ckpt.get("epoch", "?"),
            ckpt.get("metrics", {}).get("val_acc", "?"),
        )
        return model, ckpt
