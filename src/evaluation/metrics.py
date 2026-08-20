"""Classification metric computation and visualization utilities.

This module provides two layers:

1. **Pure computation functions** — stateless, side-effect-free functions
   that accept predictions and targets and return metric values.  Every
   function accepts both :class:`torch.Tensor` and :class:`numpy.ndarray`
   inputs and converts internally so that callers can pass model outputs
   directly.

2. **Visualization functions** — matplotlib/seaborn functions that produce
   :class:`~matplotlib.figure.Figure` objects.  Every function accepts an
   optional ``save_path`` argument; when provided the figure is saved to
   disk *and* returned so that callers can embed it in notebooks.

Design principles
-----------------
- Functions are unit-testable: given known inputs, they return deterministic
  outputs that can be verified against manually computed values.
- Visualizations are decoupled from computation: plotting functions receive
  pre-computed arrays, not raw model outputs.
- Edge cases are handled explicitly: missing classes, single-class batches,
  and binary problems all produce valid (though possibly degenerate) results
  rather than raising unhandled exceptions.

Example::

    import torch
    from src.evaluation.metrics import compute_topk_accuracy, plot_confusion_matrix

    logits = torch.randn(100, 10)
    targets = torch.randint(0, 10, (100,))
    acc = compute_topk_accuracy(logits, targets, topk=(1, 5))
    print(acc)  # {"top1": 0.12, "top5": 0.47}
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

__all__ = [
    # Computation
    "compute_topk_accuracy",
    "compute_per_class_accuracy",
    "compute_precision_recall_f1",
    "compute_confusion_matrix",
    "compute_normalized_confusion_matrix",
    "compute_classification_report",
    "compute_roc_auc",
    "compute_roc_curves_per_class",
    "compute_ece",
    "compute_balanced_accuracy",
    "compute_all_metrics",
    # Visualization
    "plot_confusion_matrix",
    "plot_per_class_metrics",
    "plot_roc_curves",
    "plot_reliability_diagram",
    "plot_metric_summary",
]

logger = logging.getLogger(__name__)

# Annotation threshold: show numbers in heatmap cells only for small matrices.
_ANNOT_THRESHOLD = 25
# Maximum individual ROC curves plotted before switching to aggregate-only.
_MAX_ROC_CURVES = 12

# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------

_ArrayLike = Union[torch.Tensor, np.ndarray]


def _to_numpy(x: _ArrayLike) -> np.ndarray:
    """Convert a Tensor or ndarray to a CPU float64 ndarray."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_int_numpy(x: _ArrayLike) -> np.ndarray:
    """Convert a Tensor or ndarray to a CPU int64 ndarray."""
    arr = _to_numpy(x)
    return arr.astype(np.int64)


# ---------------------------------------------------------------------------
# 1. Top-k accuracy
# ---------------------------------------------------------------------------


def compute_topk_accuracy(
    logits: _ArrayLike,
    targets: _ArrayLike,
    topk: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Compute Top-k accuracy for one or more values of k.

    For each k, a prediction is correct if the true label appears within
    the k highest-scoring classes.  When k exceeds the number of classes,
    it is silently clipped so that the result is always well-defined.

    Args:
        logits: Unnormalised model output of shape ``(N, C)`` where N is the
            number of samples and C the number of classes.  Softmax
            probabilities also work — any monotone-increasing transformation
            of the score preserves the ranking.
        targets: Integer class labels of shape ``(N,)`` with values in
            ``[0, C)``.
        topk: Tuple of k values to evaluate.  Defaults to ``(1, 5)``
            following the ImageNet convention.

    Returns:
        Dictionary mapping ``"top{k}"`` → accuracy in ``[0, 1]``.

    Example::

        acc = compute_topk_accuracy(logits, targets, topk=(1, 3, 5))
        print(acc["top1"], acc["top5"])
    """
    logits_t = torch.as_tensor(_to_numpy(logits), dtype=torch.float32)
    targets_t = torch.as_tensor(_to_int_numpy(targets), dtype=torch.long)

    num_classes = logits_t.shape[1]
    # Clip k values so that topk ≤ num_classes.
    clipped_topk = tuple(min(k, num_classes) for k in topk)
    maxk = max(clipped_topk)
    n = targets_t.size(0)

    _, pred = logits_t.topk(maxk, dim=1, largest=True, sorted=True)  # (N, maxk)
    pred = pred.t()                                                    # (maxk, N)
    correct = pred.eq(targets_t.view(1, -1).expand_as(pred))         # (maxk, N)

    results: dict[str, float] = {}
    for k_orig, k_clip in zip(topk, clipped_topk):
        correct_k = correct[:k_clip].reshape(-1).float().sum().item()
        results[f"top{k_orig}"] = correct_k / n

    return results


# ---------------------------------------------------------------------------
# 2. Per-class accuracy
# ---------------------------------------------------------------------------


def compute_per_class_accuracy(
    predictions: _ArrayLike,
    targets: _ArrayLike,
    num_classes: int,
) -> np.ndarray:
    """Compute the accuracy (recall) for every class individually.

    A class with zero test samples receives accuracy ``0.0``.  Classes that
    appear in ``targets`` but not in ``predictions`` also receive ``0.0``.

    Args:
        predictions: Predicted class indices of shape ``(N,)``.
        targets: True class indices of shape ``(N,)``.
        num_classes: Total number of classes ``C``.

    Returns:
        Float64 array of shape ``(C,)`` where element ``c`` is the fraction
        of samples from class ``c`` that were predicted correctly.
    """
    preds = _to_int_numpy(predictions)
    tgts  = _to_int_numpy(targets)

    per_class = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        mask = tgts == c
        if mask.sum() == 0:
            per_class[c] = 0.0
        else:
            per_class[c] = (preds[mask] == c).mean()
    return per_class


# ---------------------------------------------------------------------------
# 3. Precision, Recall, F1
# ---------------------------------------------------------------------------


def compute_precision_recall_f1(
    predictions: _ArrayLike,
    targets: _ArrayLike,
    num_classes: int,
    class_names: Optional[list[str]] = None,
) -> dict[str, object]:
    """Compute precision, recall, and F1 with macro, weighted, and per-class breakdowns.

    Uses scikit-learn with ``zero_division=0`` to handle classes that have
    no predicted samples without raising warnings.

    Args:
        predictions: Predicted class indices ``(N,)``.
        targets: True class indices ``(N,)``.
        num_classes: Total number of classes.
        class_names: Optional list of class name strings for per-class keys.
            Defaults to ``["class_0", "class_1", ...]``.

    Returns:
        Dictionary with the following keys:

        - ``"precision_macro"``  — float
        - ``"recall_macro"``     — float
        - ``"f1_macro"``         — float
        - ``"precision_weighted"`` — float
        - ``"recall_weighted"``  — float
        - ``"f1_weighted"``      — float
        - ``"per_class_precision"`` — ndarray ``(C,)``
        - ``"per_class_recall"``    — ndarray ``(C,)``
        - ``"per_class_f1"``        — ndarray ``(C,)``
    """
    preds = _to_int_numpy(predictions)
    tgts  = _to_int_numpy(targets)

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    labels = list(range(num_classes))
    kwargs = dict(labels=labels, zero_division=0)

    return {
        "precision_macro":    float(precision_score(tgts, preds, average="macro",    **kwargs)),
        "recall_macro":       float(recall_score(   tgts, preds, average="macro",    **kwargs)),
        "f1_macro":           float(f1_score(       tgts, preds, average="macro",    **kwargs)),
        "precision_weighted": float(precision_score(tgts, preds, average="weighted", **kwargs)),
        "recall_weighted":    float(recall_score(   tgts, preds, average="weighted", **kwargs)),
        "f1_weighted":        float(f1_score(       tgts, preds, average="weighted", **kwargs)),
        "per_class_precision": precision_score(tgts, preds, average=None, **kwargs).astype(np.float64),
        "per_class_recall":    recall_score(   tgts, preds, average=None, **kwargs).astype(np.float64),
        "per_class_f1":        f1_score(       tgts, preds, average=None, **kwargs).astype(np.float64),
    }


# ---------------------------------------------------------------------------
# 4. Confusion matrix
# ---------------------------------------------------------------------------


def compute_confusion_matrix(
    predictions: _ArrayLike,
    targets: _ArrayLike,
    num_classes: int,
) -> np.ndarray:
    """Return the ``(C, C)`` integer confusion matrix.

    ``CM[i, j]`` is the number of samples from true class ``i`` that were
    predicted as class ``j``.  The diagonal contains correct predictions.

    Args:
        predictions: Predicted class indices ``(N,)``.
        targets: True class indices ``(N,)``.
        num_classes: Total number of classes.

    Returns:
        Integer ndarray of shape ``(C, C)``.
    """
    preds = _to_int_numpy(predictions)
    tgts  = _to_int_numpy(targets)
    return confusion_matrix(tgts, preds, labels=list(range(num_classes)))


def compute_normalized_confusion_matrix(
    cm: np.ndarray,
    mode: str = "true",
) -> np.ndarray:
    """Normalise a raw confusion matrix.

    Args:
        cm: Raw integer confusion matrix of shape ``(C, C)``.
        mode: Normalisation mode:

            - ``"true"``: Each row sums to 1 (recall / true-class rates).
            - ``"pred"``: Each column sums to 1 (precision / predicted-class rates).
            - ``"all"``:  Matrix sums to 1 (global frequency).

    Returns:
        Float64 ndarray of shape ``(C, C)`` with values in ``[0, 1]``.

    Raises:
        ValueError: If ``mode`` is not one of the supported strings.
    """
    cm = cm.astype(np.float64)
    if mode == "true":
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid ÷0.
        return cm / row_sums
    elif mode == "pred":
        col_sums = cm.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1, col_sums)
        return cm / col_sums
    elif mode == "all":
        total = cm.sum()
        return cm / (total if total > 0 else 1)
    else:
        raise ValueError(f"mode must be 'true', 'pred', or 'all', got {mode!r}.")


# ---------------------------------------------------------------------------
# 5. Classification report
# ---------------------------------------------------------------------------


def compute_classification_report(
    predictions: _ArrayLike,
    targets: _ArrayLike,
    class_names: list[str],
) -> dict[str, object]:
    """Generate a full sklearn classification report.

    Args:
        predictions: Predicted class indices ``(N,)``.
        targets: True class indices ``(N,)``.
        class_names: Human-readable name for each class.

    Returns:
        Dictionary with keys ``"text"`` (human-readable report string) and
        ``"dict"`` (structured dictionary returned by
        :func:`sklearn.metrics.classification_report` with
        ``output_dict=True``).
    """
    preds = _to_int_numpy(predictions)
    tgts  = _to_int_numpy(targets)

    common = dict(target_names=class_names, zero_division=0)
    return {
        "text": classification_report(tgts, preds, **common),
        "dict": classification_report(tgts, preds, output_dict=True, **common),
    }


# ---------------------------------------------------------------------------
# 6. ROC-AUC
# ---------------------------------------------------------------------------


def compute_roc_auc(
    probabilities: _ArrayLike,
    targets: _ArrayLike,
    num_classes: int,
) -> Optional[float]:
    """Compute macro-averaged one-vs-rest ROC-AUC.

    Returns ``None`` when computation is not possible (e.g. only one class
    present in ``targets``, or the input has fewer than two classes).

    Args:
        probabilities: Softmax probability matrix of shape ``(N, C)``.
            Must contain valid probabilities (non-negative, rows sum to 1),
            *not* raw logits.
        targets: True integer labels ``(N,)``.
        num_classes: Total number of classes.

    Returns:
        Macro-averaged OvR AUC in ``[0, 1]``, or ``None`` on failure.
    """
    probs = _to_numpy(probabilities).astype(np.float64)
    tgts  = _to_int_numpy(targets)

    unique_classes = np.unique(tgts)
    if len(unique_classes) < 2:
        logger.warning(
            "ROC-AUC undefined: only %d class(es) present in targets.",
            len(unique_classes),
        )
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(
                roc_auc_score(
                    tgts, probs,
                    multi_class="ovr",
                    average="macro",
                    labels=list(range(num_classes)),
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("ROC-AUC computation failed: %s", exc)
        return None


def compute_roc_curves_per_class(
    probabilities: _ArrayLike,
    targets: _ArrayLike,
    num_classes: int,
) -> dict[str, object]:
    """Compute per-class ROC curves and AUC values using OvR strategy.

    Args:
        probabilities: Softmax probability matrix ``(N, C)``.
        targets: True labels ``(N,)``.
        num_classes: Total number of classes.

    Returns:
        Dictionary with keys:

        - ``"fpr"``: dict mapping class index → FPR array.
        - ``"tpr"``: dict mapping class index → TPR array.
        - ``"auc"``: dict mapping class index → AUC float.
        - ``"classes_computed"``: list of class indices that had valid curves.
    """
    from sklearn.metrics import auc as sklearn_auc

    probs = _to_numpy(probabilities).astype(np.float64)
    tgts  = _to_int_numpy(targets)

    labels = list(range(num_classes))
    tgts_binary = label_binarize(tgts, classes=labels)
    if num_classes == 2:
        # label_binarize with 2 classes returns (N,1); fix to (N,2).
        tgts_binary = np.hstack([1 - tgts_binary, tgts_binary])

    fpr: dict[int, np.ndarray] = {}
    tpr: dict[int, np.ndarray] = {}
    auc_vals: dict[int, float] = {}
    classes_computed: list[int] = []

    for c in range(num_classes):
        if tgts_binary[:, c].sum() == 0:
            continue  # Class not present — skip.
        try:
            fpr[c], tpr[c], _ = roc_curve(tgts_binary[:, c], probs[:, c])
            auc_vals[c] = float(sklearn_auc(fpr[c], tpr[c]))
            classes_computed.append(c)
        except Exception as exc:  # noqa: BLE001
            logger.debug("ROC curve failed for class %d: %s", c, exc)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc_vals,
        "classes_computed": classes_computed,
    }


# ---------------------------------------------------------------------------
# 7. Expected Calibration Error
# ---------------------------------------------------------------------------


def compute_ece(
    probabilities: _ArrayLike,
    targets: _ArrayLike,
    n_bins: int = 15,
) -> float:
    """Compute the Expected Calibration Error (ECE).

    ECE measures the weighted average of the absolute gap between confidence
    (mean predicted probability of the top class) and accuracy within equal-
    width confidence bins::

        ECE = Σ_b (|B_b| / N) × |acc(B_b) − conf(B_b)|

    A perfectly calibrated model has ECE = 0.

    Args:
        probabilities: Softmax probability matrix ``(N, C)``.
        targets: True labels ``(N,)``.
        n_bins: Number of equal-width confidence bins.  Defaults to ``15``.

    Returns:
        ECE value in ``[0, 1]``.  Lower is better.
    """
    probs = _to_numpy(probabilities).astype(np.float64)
    tgts  = _to_int_numpy(targets)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = (predictions == tgts).astype(np.float64)
    n = len(tgts)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if not mask.any():
            continue
        bin_acc  = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_weight = mask.sum() / n
        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def compute_reliability_diagram_data(
    probabilities: _ArrayLike,
    targets: _ArrayLike,
    n_bins: int = 15,
) -> dict[str, np.ndarray]:
    """Compute data for a calibration reliability diagram.

    Args:
        probabilities: Softmax probability matrix ``(N, C)``.
        targets: True labels ``(N,)``.
        n_bins: Number of confidence bins.

    Returns:
        Dictionary with keys ``"bin_centers"``, ``"accuracies"``,
        ``"confidences"``, and ``"counts"``, each a 1-D float array of
        length ``n_bins``.  Bins with no samples have accuracy and
        confidence set to 0.
    """
    probs = _to_numpy(probabilities).astype(np.float64)
    tgts  = _to_int_numpy(targets)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = (predictions == tgts).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers   = (bin_edges[:-1] + bin_edges[1:]) / 2

    accs   = np.zeros(n_bins)
    confs  = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.any():
            accs[i]   = correctness[mask].mean()
            confs[i]  = confidences[mask].mean()
            counts[i] = mask.sum()

    return {
        "bin_centers": centers,
        "accuracies":  accs,
        "confidences": confs,
        "counts":      counts,
    }


# ---------------------------------------------------------------------------
# 8. Balanced accuracy
# ---------------------------------------------------------------------------


def compute_balanced_accuracy(
    predictions: _ArrayLike,
    targets: _ArrayLike,
) -> float:
    """Compute balanced accuracy (mean per-class recall).

    Unlike standard accuracy, balanced accuracy gives equal weight to each
    class regardless of frequency.  It equals standard accuracy when the
    dataset is balanced.

    Args:
        predictions: Predicted class indices ``(N,)``.
        targets: True class indices ``(N,)``.

    Returns:
        Balanced accuracy in ``[0, 1]``.
    """
    return float(balanced_accuracy_score(_to_int_numpy(targets), _to_int_numpy(predictions)))


# ---------------------------------------------------------------------------
# 9. Convenience aggregator
# ---------------------------------------------------------------------------


def compute_all_metrics(
    logits: _ArrayLike,
    targets: _ArrayLike,
    class_names: list[str],
    topk: tuple[int, ...] = (1, 5),
    n_ece_bins: int = 15,
) -> dict[str, object]:
    """Compute every metric in one call.

    This is the single entry point for :class:`~src.evaluation.evaluator.Evaluator`.
    Calling each function individually gives identical results.

    Args:
        logits: Raw model output ``(N, C)``.  **Not** softmax probabilities —
            this function applies softmax internally.
        targets: True labels ``(N,)``.
        class_names: Human-readable class names (length C).
        topk: k values for top-k accuracy.
        n_ece_bins: Number of bins for ECE computation.

    Returns:
        Flat dictionary containing all scalar and array metrics.
    """
    num_classes = len(class_names)
    logits_np = _to_numpy(logits).astype(np.float64)
    tgts_np   = _to_int_numpy(targets)

    # Softmax probabilities for calibration and ROC metrics.
    logits_t = torch.as_tensor(logits_np, dtype=torch.float32)
    probs_np  = torch.softmax(logits_t, dim=1).numpy().astype(np.float64)
    preds_np  = probs_np.argmax(axis=1).astype(np.int64)

    topk_acc = compute_topk_accuracy(logits_t, torch.as_tensor(tgts_np), topk=topk)
    prf      = compute_precision_recall_f1(preds_np, tgts_np, num_classes, class_names)
    cm       = compute_confusion_matrix(preds_np, tgts_np, num_classes)
    pca      = compute_per_class_accuracy(preds_np, tgts_np, num_classes)
    report   = compute_classification_report(preds_np, tgts_np, class_names)
    roc_auc  = compute_roc_auc(probs_np, tgts_np, num_classes)
    ece      = compute_ece(probs_np, tgts_np, n_ece_bins)
    bal_acc  = compute_balanced_accuracy(preds_np, tgts_np)

    return {
        # Scalar metrics
        "accuracy":             float(accuracy_score(tgts_np, preds_np)),
        "balanced_accuracy":    bal_acc,
        "ece":                  ece,
        "roc_auc_macro":        roc_auc,
        # Top-k
        **topk_acc,
        # Precision / Recall / F1
        **{k: v for k, v in prf.items() if not k.startswith("per_class")},
        # Per-class arrays
        "per_class_accuracy":   pca,
        "per_class_precision":  prf["per_class_precision"],
        "per_class_recall":     prf["per_class_recall"],
        "per_class_f1":         prf["per_class_f1"],
        # Confusion matrix
        "confusion_matrix":            cm,
        "confusion_matrix_normalized": compute_normalized_confusion_matrix(cm, "true"),
        # Reports
        "classification_report_text": report["text"],
        "classification_report_dict": report["dict"],
        # Raw predictions for downstream analysis
        "predictions":   preds_np,
        "probabilities": probs_np,
    }


# ---------------------------------------------------------------------------
# Visualization utilities
# ---------------------------------------------------------------------------


def _save_figure(fig: matplotlib.figure.Figure, save_path: Optional[Path]) -> None:
    """Save a figure to disk if ``save_path`` is provided."""
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.debug("Figure saved → %s", path)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    normalize: bool = True,
    title: str = "Confusion Matrix",
    figsize: Optional[tuple[float, float]] = None,
    save_path: Optional[Path] = None,
) -> matplotlib.figure.Figure:
    """Plot a confusion matrix heatmap using seaborn.

    For matrices with more than :data:`_ANNOT_THRESHOLD` classes, cell
    annotations are suppressed to prevent illegible overlap.

    Args:
        cm: Raw integer confusion matrix ``(C, C)``.
        class_names: Class label strings (length C).
        normalize: If ``True``, normalise by true-class totals (row-wise)
            so that each cell shows the recall rate for that class.
        title: Figure title.
        figsize: Figure size ``(width, height)`` in inches.  Auto-computed
            from the number of classes when ``None``.
        save_path: Optional path to save the figure.

    Returns:
        :class:`~matplotlib.figure.Figure` object.
    """
    num_classes = cm.shape[0]
    display_cm  = compute_normalized_confusion_matrix(cm, "true") if normalize else cm.astype(float)

    # Auto-scale figure.
    if figsize is None:
        side = max(8.0, num_classes * 0.45)
        figsize = (side + 2.0, side)

    annotate = num_classes <= _ANNOT_THRESHOLD
    fmt      = ".2f" if normalize else "d"
    annot_fs = max(5, 12 - num_classes // 5)
    tick_fs  = max(5, 11 - num_classes // 8)

    with plt.style.context("seaborn-v0_8-white"):
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            display_cm,
            annot=annotate,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            square=True,
            linewidths=0.3 if num_classes <= 20 else 0.0,
            linecolor="lightgray",
            ax=ax,
            annot_kws={"size": annot_fs} if annotate else {},
            vmin=0.0,
            vmax=1.0 if normalize else None,
        )

        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xlabel("Predicted Class", fontsize=11)
        ax.set_ylabel("True Class", fontsize=11)
        ax.tick_params(axis="x", rotation=90, labelsize=tick_fs)
        ax.tick_params(axis="y", rotation=0,  labelsize=tick_fs)

        norm_label = " (row-normalised, recall)" if normalize else " (raw counts)"
        ax.figure.text(
            0.5, -0.02,
            norm_label,
            ha="center",
            fontsize=9,
            color="gray",
            transform=ax.transAxes,
        )

        fig.tight_layout()

    _save_figure(fig, save_path)
    return fig


def plot_per_class_metrics(
    per_class_precision: np.ndarray,
    per_class_recall: np.ndarray,
    per_class_f1: np.ndarray,
    class_names: list[str],
    title: str = "Per-Class Metrics",
    sort_by: str = "f1",
    save_path: Optional[Path] = None,
) -> matplotlib.figure.Figure:
    """Plot per-class precision, recall, and F1 as a grouped bar chart.

    For datasets with more than 20 classes the chart uses horizontal bars
    and sorts classes by the ``sort_by`` metric to highlight the best and
    worst performers.

    Args:
        per_class_precision: Precision per class, shape ``(C,)``.
        per_class_recall:    Recall per class, shape ``(C,)``.
        per_class_f1:        F1 score per class, shape ``(C,)``.
        class_names: Class label strings (length C).
        title: Figure title.
        sort_by: Metric to sort by — ``"precision"``, ``"recall"``,
            or ``"f1"``.  Defaults to ``"f1"``.
        save_path: Optional path to save the figure.

    Returns:
        :class:`~matplotlib.figure.Figure` object.
    """
    num_classes = len(class_names)
    horizontal  = num_classes > 20

    sort_key = {"f1": per_class_f1, "precision": per_class_precision,
                "recall": per_class_recall}.get(sort_by, per_class_f1)
    order = np.argsort(sort_key)  # Ascending — worst performers at bottom.

    names_sorted = [class_names[i] for i in order]
    prec_sorted  = per_class_precision[order]
    rec_sorted   = per_class_recall[order]
    f1_sorted    = per_class_f1[order]

    x      = np.arange(num_classes)
    width  = 0.25
    colors = ["#4878CF", "#6ACC65", "#D65F5F"]  # Blue, Green, Red

    if horizontal:
        fig_h = max(6.0, num_classes * 0.22)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        ax.barh(x - width, prec_sorted, width, label="Precision", color=colors[0], alpha=0.85)
        ax.barh(x,         rec_sorted,  width, label="Recall",    color=colors[1], alpha=0.85)
        ax.barh(x + width, f1_sorted,   width, label="F1",        color=colors[2], alpha=0.85)
        ax.set_yticks(x)
        ax.set_yticklabels(names_sorted, fontsize=max(5, 9 - num_classes // 15))
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Score", fontsize=11)
        ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    else:
        fig_w = max(10.0, num_classes * 0.55)
        fig, ax = plt.subplots(figsize=(fig_w, 5))
        ax.bar(x - width, prec_sorted, width, label="Precision", color=colors[0], alpha=0.85)
        ax.bar(x,         rec_sorted,  width, label="Recall",    color=colors[1], alpha=0.85)
        ax.bar(x + width, f1_sorted,   width, label="F1",        color=colors[2], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names_sorted, rotation=45, ha="right",
                           fontsize=max(6, 10 - num_classes // 8))
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score", fontsize=11)
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))

    ax.set_title(f"{title} (sorted by {sort_by})", fontsize=14)
    ax.legend(loc="lower right" if horizontal else "upper right", fontsize=10)
    ax.grid(axis="x" if horizontal else "y", alpha=0.35, linestyle=":")
    fig.tight_layout()

    _save_figure(fig, save_path)
    return fig


def plot_roc_curves(
    roc_data: dict[str, object],
    class_names: list[str],
    title: str = "ROC Curves (One-vs-Rest)",
    max_curves: int = _MAX_ROC_CURVES,
    save_path: Optional[Path] = None,
) -> matplotlib.figure.Figure:
    """Plot multi-class ROC curves.

    When the number of classes exceeds ``max_curves``, only the
    ``max_curves // 2`` best and worst classes by AUC are plotted
    individually; the macro average is always shown.

    Args:
        roc_data: Dictionary returned by :func:`compute_roc_curves_per_class`.
        class_names: Class label strings.
        title: Figure title.
        max_curves: Maximum number of individual class curves to draw.
        save_path: Optional path to save the figure.

    Returns:
        :class:`~matplotlib.figure.Figure` object.
    """
    fpr: dict[int, np.ndarray] = roc_data["fpr"]  # type: ignore[assignment]
    tpr: dict[int, np.ndarray] = roc_data["tpr"]  # type: ignore[assignment]
    auc_vals: dict[int, float] = roc_data["auc"]  # type: ignore[assignment]
    computed: list[int]        = roc_data["classes_computed"]  # type: ignore[assignment]

    if not computed:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "No valid ROC curves computed.",
                ha="center", va="center", transform=ax.transAxes)
        _save_figure(fig, save_path)
        return fig

    # Select which classes to plot individually.
    sorted_by_auc = sorted(computed, key=lambda c: auc_vals[c])
    if len(computed) > max_curves:
        half   = max_curves // 2
        show   = sorted_by_auc[:half] + sorted_by_auc[-half:]  # Worst + best.
        truncated = True
    else:
        show   = sorted_by_auc
        truncated = False

    # Macro-average curve via interpolation.
    all_fpr  = np.unique(np.concatenate([fpr[c] for c in computed]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in computed:
        mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])
    mean_tpr /= len(computed)
    # np.trapz was removed in NumPy 2.0; use np.trapezoid when available.
    _trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    macro_auc = float(_trapezoid(mean_tpr, all_fpr))

    cmap   = plt.get_cmap("tab20", len(show))
    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, c in enumerate(show):
        label = f"{class_names[c]} (AUC={auc_vals[c]:.3f})"
        ax.plot(fpr[c], tpr[c], color=cmap(idx), lw=1.4, alpha=0.8, label=label)

    ax.plot(all_fpr, mean_tpr,
            color="navy", lw=2.5, linestyle="--",
            label=f"Macro avg (AUC={macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1.0, alpha=0.5, label="Random (AUC=0.500)")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    suffix = f"\n(showing {len(show)}/{len(computed)} classes)" if truncated else ""
    ax.set_title(f"{title}{suffix}", fontsize=13)
    ax.legend(loc="lower right", fontsize=8, ncol=1 + len(show) // 15)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    _save_figure(fig, save_path)
    return fig


def plot_reliability_diagram(
    probabilities: _ArrayLike,
    targets: _ArrayLike,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    save_path: Optional[Path] = None,
) -> matplotlib.figure.Figure:
    """Plot a calibration reliability diagram.

    Shows whether predicted confidences match empirical accuracies.  A
    perfectly calibrated model's curve follows the diagonal.  Points above
    the diagonal indicate under-confidence; below indicates over-confidence.

    Args:
        probabilities: Softmax probability matrix ``(N, C)``.
        targets: True labels ``(N,)``.
        n_bins: Number of confidence bins.
        title: Figure title.
        save_path: Optional path to save the figure.

    Returns:
        :class:`~matplotlib.figure.Figure` object.
    """
    data  = compute_reliability_diagram_data(probabilities, targets, n_bins)
    ece   = compute_ece(probabilities, targets, n_bins)
    bins  = data["bin_centers"]
    accs  = data["accuracies"]
    confs = data["confidences"]
    cnts  = data["counts"]
    width = bins[1] - bins[0] if len(bins) > 1 else 1.0 / n_bins

    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1, figsize=(6, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # Main calibration plot.
    ax_main.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    # Gap between diagonal and actual.
    for b_center, acc, conf in zip(bins, accs, confs):
        if cnts[list(bins).index(b_center)] == 0:
            continue
        color = "#D65F5F" if conf > acc else "#4878CF"
        ax_main.fill_between([b_center - width / 2, b_center + width / 2],
                              [acc, acc], [conf, conf],
                              alpha=0.25, color=color)
    ax_main.plot(confs, accs, "o-", color="#D65F5F", lw=1.5,
                 ms=5, label=f"Model (ECE={ece:.4f})")
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1.05)
    ax_main.set_ylabel("Accuracy", fontsize=11)
    ax_main.set_title(title, fontsize=13)
    ax_main.legend(fontsize=10)
    ax_main.grid(alpha=0.3)

    # Histogram of confidence distribution.
    ax_hist.bar(bins, cnts / cnts.sum(), width=width * 0.9,
                color="#6ACC65", alpha=0.8, edgecolor="white")
    ax_hist.set_xlabel("Confidence", fontsize=11)
    ax_hist.set_ylabel("Fraction", fontsize=10)
    ax_hist.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax_hist.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_metric_summary(
    metrics: dict[str, float],
    title: str = "Evaluation Summary",
    save_path: Optional[Path] = None,
) -> matplotlib.figure.Figure:
    """Plot a horizontal bar chart of key scalar evaluation metrics.

    Args:
        metrics: Flat dictionary of ``{metric_name: value}`` for scalar
            metrics only (float values in ``[0, 1]``).  Array-valued
            entries are ignored.
        title: Figure title.
        save_path: Optional path to save the figure.

    Returns:
        :class:`~matplotlib.figure.Figure` object.
    """
    # Filter to scalar metrics in [0, 1].
    scalar_keys = [
        "top1", "top5", "accuracy", "balanced_accuracy",
        "precision_macro", "recall_macro", "f1_macro",
        "f1_weighted", "roc_auc_macro",
    ]
    names, values = [], []
    for k in scalar_keys:
        if k in metrics and metrics[k] is not None:
            v = metrics[k]
            if isinstance(v, (int, float)):
                names.append(k.replace("_", "\n"))
                values.append(float(v))

    colors = ["#4878CF" if v >= 0.8 else "#D65F5F" if v < 0.6 else "#E8A838"
              for v in values]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.55)))
    bars = ax.barh(names, values, color=colors, alpha=0.85, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(
            min(val + 0.01, 0.99),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left", fontsize=10, fontweight="bold",
        )

    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.axvline(x=1.0, color="gray", linestyle="--", lw=0.8, alpha=0.6)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()  # Highest metric at top.
    fig.tight_layout()

    _save_figure(fig, save_path)
    return fig
