"""Loss function implementations and factory for ResNet-34 experiments.

All experiments use this module as the single source of loss function creation.
Training scripts call ``LossFactory.create()`` rather than instantiating loss
functions directly, which ensures the choice is config-driven and auditable.

Scientific rationale for label smoothing
-----------------------------------------
Standard cross-entropy drives the model toward logit → ±∞, producing
overconfident predictions.  Label smoothing (Szegedy et al., 2016; Müller
et al., 2019) replaces the one-hot target with a soft distribution::

    y_smooth[k] = (1 − ε) · 1[k == y] + ε / K

This bounds the maximum achievable logit gap to ``log((K−1)(1−ε) / ε)``,
which has two effects:

1. **Better calibration**: softmax outputs are less extreme, so the model's
   confidence scores are better aligned with empirical accuracy (lower ECE).
2. **Implicit regularisation**: the model cannot overfit the noise in
   individual hard labels; it learns more distributed representations.

Ablation A3 tests ε=0.0 (standard CE) vs ε=0.1 (label smoothing).

Common failure mode
-------------------
Calling ``nn.CrossEntropyLoss`` on *already-softmaxed* probabilities instead
of raw logits.  ``CrossEntropyLoss`` applies log-softmax internally — passing
softmax outputs causes ``log(softmax(softmax(logits)))`` which is numerically
unstable and always produces the wrong loss value.  All loss functions here
expect **raw logits**, never probabilities.

Example::

    from src.training.losses import LossFactory

    criterion = LossFactory.create(label_smoothing=0.1, num_classes=10)
    loss = criterion(logits, targets)   # logits: [B, K], targets: [B]
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LabelSmoothingCrossEntropy", "LossFactory"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LabelSmoothingCrossEntropy
# ---------------------------------------------------------------------------


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing (Szegedy et al., 2016).

    Converts the one-hot target into a soft distribution::

        y_smooth[k] = (1 − ε) · 𝟙[k == y] + ε / K

    and computes the standard cross-entropy against this soft target::

        L = −∑_k y_smooth[k] · log p(k)

    where ``p(k)`` is obtained via log-softmax for numerical stability.

    When ``smoothing=0.0`` this is numerically identical to
    ``nn.CrossEntropyLoss`` — useful for asserting this in unit tests.

    Args:
        num_classes: Number of output classes K.
        smoothing: Label smoothing coefficient ε ∈ [0, 1).  Typical value
            is 0.1.  ``smoothing=0.0`` recovers standard cross-entropy.
        reduction: One of ``"mean"`` (default), ``"sum"``, or ``"none"``.
        weight: Optional per-class weight tensor of shape ``(K,)``.  When
            supplied, each sample's loss is multiplied by
            ``weight[target_class]``.  Used for class imbalance correction.

    Raises:
        ValueError: If ``smoothing`` is not in ``[0, 1)``.
    """

    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        reduction: str = "mean",
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        if not (0.0 <= smoothing < 1.0):
            raise ValueError(
                f"smoothing must be in [0, 1), got {smoothing}."
            )
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}."
            )

        self.num_classes = num_classes
        self.smoothing   = smoothing
        self.reduction   = reduction

        # Register weight as a buffer so it moves to the correct device
        # automatically when model.to(device) is called.
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight: Optional[torch.Tensor] = None  # type: ignore[assignment]

        logger.debug(
            "LabelSmoothingCrossEntropy: K=%d, ε=%.3f, reduction=%s",
            num_classes, smoothing, reduction,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the label-smoothed cross-entropy loss.

        Args:
            logits: Raw model output of shape ``(B, K)``.  Must **not** have
                softmax applied — this function applies log-softmax internally
                for numerical stability.
            targets: Integer class labels of shape ``(B,)`` with values in
                ``[0, K)``.

        Returns:
            Scalar loss tensor when ``reduction`` is ``"mean"`` or ``"sum"``;
            per-sample losses of shape ``(B,)`` when ``"none"``.
        """
        # log-softmax for numerical stability: avoids log(exp(x)/sum(exp(x)))
        log_probs = F.log_softmax(logits, dim=1)           # [B, K]

        # -- Smooth target distribution -----------------------------------
        # Fill every class with ε/K, then add (1-ε) for the true class.
        # This is equivalent to the soft-target CE derived in the paper.
        B = logits.size(0)
        smooth_val   = self.smoothing / self.num_classes
        target_val   = 1.0 - self.smoothing + smooth_val   # (1-ε) + ε/K

        # Allocate soft targets: [B, K] filled with ε/K
        y_soft = torch.full(
            (B, self.num_classes),
            smooth_val,
            dtype=log_probs.dtype,
            device=log_probs.device,
        )
        # Scatter the (1-ε + ε/K) mass onto the true class position.
        y_soft.scatter_(1, targets.unsqueeze(1), target_val)

        # -- Cross-entropy against soft targets --------------------------
        # L_i = −∑_k y_soft[i,k] · log_probs[i,k]
        per_sample_loss = -(y_soft * log_probs).sum(dim=1)  # [B]

        # -- Optional per-class weighting --------------------------------
        if self.weight is not None:
            w = self.weight[targets]  # [B]
            per_sample_loss = per_sample_loss * w

        if self.reduction == "mean":
            if self.weight is not None:
                # Normalise by the sum of weights, not batch size.
                return per_sample_loss.sum() / self.weight[targets].sum()
            return per_sample_loss.mean()
        elif self.reduction == "sum":
            return per_sample_loss.sum()
        else:
            return per_sample_loss

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"smoothing={self.smoothing}, "
            f"reduction={self.reduction!r}"
        )


# ---------------------------------------------------------------------------
# LossFactory
# ---------------------------------------------------------------------------


class LossFactory:
    """Factory for creating loss functions from experiment configuration.

    All training scripts obtain their loss function through this factory.
    This guarantees that the loss choice is config-driven and that every
    experiment's loss function is explicitly logged.

    Usage::

        # Standard cross-entropy (Ablation A3 baseline)
        criterion = LossFactory.create(label_smoothing=0.0, num_classes=10)

        # Label smoothing (Ablation A3 variant)
        criterion = LossFactory.create(label_smoothing=0.1, num_classes=10)

        # Weighted CE for class-imbalanced ABO
        criterion = LossFactory.create(
            label_smoothing=0.0,
            num_classes=50,
            class_weights=torch.tensor([...]),
        )
    """

    @staticmethod
    def create(
        label_smoothing: float = 0.0,
        num_classes: int = 10,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> nn.Module:
        """Create the appropriate loss function.

        Decision logic:

        - ``label_smoothing > 0``: returns
          :class:`LabelSmoothingCrossEntropy` which correctly handles both
          smoothing and optional class weights.
        - ``label_smoothing == 0`` and ``class_weights is None``: returns
          ``nn.CrossEntropyLoss()`` — the standard choice, most efficient.
        - ``label_smoothing == 0`` and ``class_weights is not None``: returns
          ``nn.CrossEntropyLoss(weight=class_weights)`` — weighted CE without
          smoothing.

        Args:
            label_smoothing: ε ∈ [0, 1).  ``0.0`` disables smoothing.
            num_classes: Total number of output classes.  Required when
                ``label_smoothing > 0``.
            class_weights: Optional per-class weight tensor ``(K,)``.
            reduction: ``"mean"``, ``"sum"``, or ``"none"``.

        Returns:
            A configured ``nn.Module`` that accepts ``(logits, targets)``
            where ``logits`` are raw (pre-softmax) model outputs.

        Raises:
            ValueError: If ``label_smoothing`` is outside ``[0, 1)``.
        """
        if label_smoothing > 0.0:
            criterion: nn.Module = LabelSmoothingCrossEntropy(
                num_classes=num_classes,
                smoothing=label_smoothing,
                reduction=reduction,
                weight=class_weights,
            )
            logger.info(
                "Loss: LabelSmoothingCrossEntropy(ε=%.3f, K=%d, weighted=%s)",
                label_smoothing, num_classes, class_weights is not None,
            )
        else:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                reduction=reduction,
            )
            logger.info(
                "Loss: CrossEntropyLoss(weighted=%s)",
                class_weights is not None,
            )

        return criterion
