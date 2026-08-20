"""Optimizer factory with BatchNorm parameter group separation.

The single most important engineering decision in this module is separating
model parameters into two groups before passing them to the optimizer:

Group 1 — weight decay applied:
    All weight tensors with ndim ≥ 2.  These are the Conv2d weight matrices
    and the FC weight matrix — the high-dimensional parameters that can
    over-fit and benefit from L2 regularisation.

Group 2 — NO weight decay:
    All 1-D parameters: BatchNorm γ (weight) and β (bias), Conv2d bias
    (absent in our architecture since bias=False), and FC bias.

Why exclude BN parameters from weight decay
--------------------------------------------
BN's γ and β are normalisation scale and shift factors:

    y_norm = γ · (x − μ_B) / σ_B + β

They are scalar per-channel parameters.  Applying weight decay penalises
``‖γ‖²`` and ``‖β‖²``, which:

1. **Pushes γ toward 0**, partially defeating batch normalisation's ability
   to rescale activations to the optimal magnitude for the next layer.
2. **Has no regularisation benefit**: γ and β are single scalars per
   channel — they have essentially zero over-fitting capacity.

The consequence of incorrectly including them is a subtle but measurable
performance degradation (typically 0.3–1.5% accuracy) that is difficult
to diagnose because no error is raised.

Reference implementations
--------------------------
This is the approach used in the official torchvision ResNet training recipe
and in most SOTA papers (He et al. 2016, DeiT, ConvNeXt).

Example::

    from src.training.optimizers import create_optimizer
    from src.models.resnet34 import ResNet34

    model = ResNet34(num_classes=10)
    optimizer = create_optimizer(
        model,
        optimizer_name="sgd",
        lr=0.1,
        weight_decay=1e-4,
        momentum=0.9,
        nesterov=True,
    )
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW, Optimizer

__all__ = [
    "create_optimizer",
    "get_parameter_groups",
    "log_parameter_group_summary",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter group construction
# ---------------------------------------------------------------------------


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float,
) -> list[dict]:
    """Split model parameters into decay and no-decay groups.

    Criterion: parameters with ``ndim >= 2`` receive weight decay; all
    others (BN γ/β, biases) do not.

    Args:
        model: The model whose parameters are being split.
        weight_decay: L2 regularisation coefficient applied to the decay
            group.  The no-decay group always has ``weight_decay=0.0``.

    Returns:
        A list of two parameter group dicts suitable for passing directly
        to an ``Optimizer`` constructor::

            [
                {"params": [...], "weight_decay": weight_decay},
                {"params": [...], "weight_decay": 0.0},
            ]
    """
    decay_params:    list[torch.Tensor] = []
    no_decay_params: list[torch.Tensor] = []
    decay_names:     list[str] = []
    no_decay_names:  list[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            # Conv weights [out, in, kH, kW], FC weights [out, in]
            decay_params.append(param)
            decay_names.append(name)
        else:
            # BN weight (γ) ndim=1, BN bias (β) ndim=1, FC bias ndim=1
            no_decay_params.append(param)
            no_decay_names.append(name)

    logger.debug(
        "Parameter groups: %d params with weight_decay=%.1e, "
        "%d params with weight_decay=0.0",
        len(decay_params), weight_decay, len(no_decay_params),
    )
    logger.debug("Decay group names (first 5): %s", decay_names[:5])
    logger.debug("No-decay group names (first 5): %s", no_decay_names[:5])

    return [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def log_parameter_group_summary(model: nn.Module, weight_decay: float) -> None:
    """Log a human-readable summary of parameter group assignment.

    Call once after ``create_optimizer`` to verify the split is correct.

    Args:
        model: Model whose parameter groups are being summarised.
        weight_decay: The weight decay value used for the decay group.
    """
    n_decay = n_no_decay = 0
    p_decay = p_no_decay = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        if param.ndim >= 2:
            n_decay   += 1
            p_decay   += n
        else:
            n_no_decay += 1
            p_no_decay += n

    logger.info(
        "Optimizer param groups: "
        "decay(wd=%.1e)=%d params (%.2fM) | "
        "no-decay(wd=0.0)=%d params (%.2fM)",
        weight_decay,
        n_decay,   p_decay   / 1e6,
        n_no_decay, p_no_decay / 1e6,
    )


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "sgd",
    lr: float = 0.1,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    nesterov: bool = True,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> Optimizer:
    """Create an optimizer with correct BN parameter group separation.

    Supports SGD (with optional Nesterov momentum), Adam, and AdamW.
    SGD is the recommended optimizer for ResNet-34 on CIFAR-10 / ABO
    following the original He et al. (2016) training recipe.

    Args:
        model: The model to optimise.  Must already be on the target device.
        optimizer_name: One of ``"sgd"``, ``"adam"``, ``"adamw"``.
            Case-insensitive.
        lr: Initial learning rate.  The scheduler will modify this during
            training; this is only the starting value.
        weight_decay: L2 regularisation coefficient.  Applied to weight
            matrices only — BN parameters and biases are excluded.
        momentum: Momentum coefficient for SGD (ignored for Adam/AdamW).
        nesterov: Use Nesterov momentum for SGD.  Improves convergence
            speed on most vision tasks.  Requires ``momentum > 0``.
        betas: Exponential decay rates for Adam's first and second moment
            estimates.  Standard defaults ``(0.9, 0.999)`` work for most
            tasks.
        eps: Numerical stability term for Adam.  Default ``1e-8`` is
            standard; some practitioners use ``1e-6`` for half-precision.

    Returns:
        Configured ``torch.optim.Optimizer`` instance.

    Raises:
        ValueError: If ``optimizer_name`` is not a supported choice.
    """
    param_groups = get_parameter_groups(model, weight_decay)
    name = optimizer_name.lower().strip()

    if name == "sgd":
        optimizer: Optimizer = SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov and momentum > 0,
        )
        logger.info(
            "Optimizer: SGD(lr=%.4f, momentum=%.2f, nesterov=%s, "
            "weight_decay=%.1e on 2D+ params)",
            lr, momentum, nesterov and momentum > 0, weight_decay,
        )

    elif name == "adam":
        optimizer = Adam(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
        )
        logger.info(
            "Optimizer: Adam(lr=%.4f, betas=%s, eps=%.1e, "
            "weight_decay=%.1e on 2D+ params)",
            lr, betas, eps, weight_decay,
        )

    elif name == "adamw":
        # AdamW decouples weight decay from the gradient update, which is
        # the correct implementation of L2 regularisation for adaptive
        # optimisers (Loshchilov & Hutter, 2019).
        optimizer = AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
        )
        logger.info(
            "Optimizer: AdamW(lr=%.4f, betas=%s, eps=%.1e, "
            "weight_decay=%.1e on 2D+ params)",
            lr, betas, eps, weight_decay,
        )

    else:
        raise ValueError(
            f"Unsupported optimizer {optimizer_name!r}. "
            "Choose from: 'sgd', 'adam', 'adamw'."
        )

    log_parameter_group_summary(model, weight_decay)
    return optimizer
