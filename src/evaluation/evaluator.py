"""Model evaluation pipeline for image classification.

This module provides :class:`Evaluator`, the single entry point for all
post-training analysis.  It performs batch inference, aggregates every
classification metric supplied by :mod:`src.evaluation.metrics`, generates
publication-quality visualisations, and writes a self-contained results
directory that can be inspected without re-running inference.

Architecture
------------
The pipeline has three stages:

1. **Inference** — the model runs in ``eval`` mode under
   ``torch.no_grad()``.  Outputs (logits, probabilities, targets, and
   optionally GAP feature vectors) are accumulated on CPU so that a model
   too large to hold its activations on the GPU still evaluates correctly.

2. **Metric computation** — all numeric metrics are delegated to pure
   functions in :mod:`src.evaluation.metrics`.  The evaluator owns only
   the orchestration.

3. **Report generation** — plots are saved to ``<save_dir>/plots/`` and
   all scalar/array data is saved under ``<save_dir>/``.  Every
   visualisation is wrapped in an independent try/except so a single
   failure (e.g. ROC-AUC undefined for a single-class batch) does not
   abort the rest of the report.

Results directory layout::

    <save_dir>/
    ├── metrics.json                     ← scalar metrics + per-class lists
    ├── classification_report.txt        ← full sklearn text report
    ├── confusion_matrix.npy             ← raw integer  (C, C)
    ├── confusion_matrix_normalized.npy  ← row-normalised float  (C, C)
    ├── predictions.npy                  ← argmax predictions  (N,)
    ├── targets.npy                      ← true labels  (N,)
    ├── probabilities.npy                ← softmax probabilities  (N, C)
    ├── features.npy                     ← GAP features  (N, D)  if extracted
    └── plots/
        ├── confusion_matrix_normalized.png
        ├── confusion_matrix_counts.png
        ├── per_class_metrics.png
        ├── roc_curves.png
        ├── reliability_diagram.png
        └── metric_summary.png

GPU support
-----------
Pass ``device=torch.device("cuda")`` and ensure the model is already on that
device.  Images and targets are moved to the device inside :meth:`_run_inference`
with ``non_blocking=True`` so that pinned-memory DataLoaders overlap H<->D
transfer with the previous batch's compute.

Example::

    from pathlib import Path
    import torch
    from src.evaluation.evaluator import Evaluator

    evaluator = Evaluator(
        model=resnet34,
        dataloader=test_loader,
        device=torch.device("cuda"),
        class_names=cifar10_class_names,
        topk=(1, 5),
        extract_features=True,
    )
    results = evaluator.evaluate()
    evaluator.generate_report(results, save_dir=Path("results/run_01/eval"))
    print(results)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_):
        return iterable

from src.evaluation.metrics import (
    compute_all_metrics,
    compute_roc_curves_per_class,
    plot_confusion_matrix,
    plot_metric_summary,
    plot_per_class_metrics,
    plot_reliability_diagram,
    plot_roc_curves,
)

__all__ = ["EvaluationResults", "Evaluator"]

logger = logging.getLogger(__name__)

_ANNOT_THRESHOLD: int = 25
_PLOTS_DIR: str = "plots"


# ---------------------------------------------------------------------------
# EvaluationResults
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResults:
    """Complete evaluation snapshot for one model / dataset pair.

    Scalar metrics are JSON-serialisable; array-valued fields are
    :class:`numpy.ndarray` and are saved as ``.npy`` files.

    Attributes:
        accuracy: Standard top-1 accuracy across all test samples.
        top1_accuracy: Identical to ``accuracy``; present for explicit labelling.
        top5_accuracy: Top-5 accuracy.
        balanced_accuracy: Mean per-class recall, unweighted by class size.
        precision_macro: Macro-averaged precision.
        recall_macro: Macro-averaged recall.
        f1_macro: Macro-averaged F1 score.
        f1_weighted: Class-frequency-weighted F1 score.
        ece: Expected Calibration Error.  0.0 is perfect; lower is better.
        roc_auc_macro: Macro one-vs-rest AUC, or ``None`` when unavailable.
        per_class_accuracy: Recall for every class, shape ``(C,)``.
        per_class_precision: Precision for every class, shape ``(C,)``.
        per_class_recall: Recall for every class, shape ``(C,)``.
        per_class_f1: F1 for every class, shape ``(C,)``.
        confusion_matrix: Raw integer confusion matrix, shape ``(C, C)``.
        confusion_matrix_normalized: Row-normalised confusion matrix ``(C, C)``.
        classification_report_text: Full sklearn classification report string.
        classification_report_dict: Structured dict form of the report.
        all_predictions: Predicted class indices for every sample ``(N,)``.
        all_targets: True class indices for every sample ``(N,)``.
        all_probabilities: Softmax probabilities ``(N, C)``.
        all_features: GAP feature vectors ``(N, D)`` or empty ``(0, 0)``.
        num_samples: Total samples evaluated (N).
        num_classes: Total number of classes (C).
        class_names: Human-readable label for each class index.
        evaluation_time_seconds: Wall-clock duration of the inference pass.
    """

    # ── Core scalars
    accuracy: float = 0.0
    top1_accuracy: float = 0.0
    top5_accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    ece: float = 0.0
    roc_auc_macro: Optional[float] = None

    # ── Per-class arrays
    per_class_accuracy: np.ndarray  = field(default_factory=lambda: np.empty(0))
    per_class_precision: np.ndarray = field(default_factory=lambda: np.empty(0))
    per_class_recall: np.ndarray    = field(default_factory=lambda: np.empty(0))
    per_class_f1: np.ndarray        = field(default_factory=lambda: np.empty(0))

    # ── Confusion matrices
    confusion_matrix: np.ndarray            = field(default_factory=lambda: np.empty(0))
    confusion_matrix_normalized: np.ndarray = field(default_factory=lambda: np.empty(0))

    # ── Classification report
    classification_report_text: str  = ""
    classification_report_dict: dict = field(default_factory=dict)

    # ── Raw prediction arrays
    all_predictions: np.ndarray   = field(default_factory=lambda: np.empty(0))
    all_targets: np.ndarray       = field(default_factory=lambda: np.empty(0))
    all_probabilities: np.ndarray = field(default_factory=lambda: np.empty(0))
    all_features: np.ndarray      = field(default_factory=lambda: np.empty((0, 0)))

    # ── Metadata
    num_samples: int = 0
    num_classes: int = 0
    class_names: list = field(default_factory=list)
    evaluation_time_seconds: float = 0.0

    # ------------------------------------------------------------------
    # Scalar view
    # ------------------------------------------------------------------

    def scalar_metrics(self) -> dict:
        """Return a JSON-serialisable dict of every scalar metric.

        Returns:
            Flat ``{name: value}`` mapping.  ``None`` values are preserved
            so consumers can distinguish "not computed" from zero.
        """
        return {
            "accuracy":                self.accuracy,
            "top1_accuracy":           self.top1_accuracy,
            "top5_accuracy":           self.top5_accuracy,
            "balanced_accuracy":       self.balanced_accuracy,
            "precision_macro":         self.precision_macro,
            "recall_macro":            self.recall_macro,
            "f1_macro":                self.f1_macro,
            "f1_weighted":             self.f1_weighted,
            "ece":                     self.ece,
            "roc_auc_macro":           self.roc_auc_macro,
            "num_samples":             self.num_samples,
            "num_classes":             self.num_classes,
            "evaluation_time_seconds": self.evaluation_time_seconds,
        }

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, save_dir) -> None:
        """Write all results to ``save_dir``.

        Creates metrics.json, classification_report.txt, confusion
        matrix arrays, predictions/targets/probabilities arrays,
        and optionally features.npy.

        Args:
            save_dir: Directory to write into.  Created if absent.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            **self.scalar_metrics(),
            "class_names":        self.class_names,
            "per_class_accuracy":  self.per_class_accuracy.tolist(),
            "per_class_precision": self.per_class_precision.tolist(),
            "per_class_recall":    self.per_class_recall.tolist(),
            "per_class_f1":        self.per_class_f1.tolist(),
            "classification_report": self.classification_report_dict,
        }
        with open(save_dir / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        (save_dir / "classification_report.txt").write_text(
            self.classification_report_text, encoding="utf-8"
        )

        np.save(save_dir / "confusion_matrix.npy",            self.confusion_matrix)
        np.save(save_dir / "confusion_matrix_normalized.npy", self.confusion_matrix_normalized)
        np.save(save_dir / "predictions.npy",                 self.all_predictions)
        np.save(save_dir / "targets.npy",                     self.all_targets)
        np.save(save_dir / "probabilities.npy",               self.all_probabilities)

        if self.all_features.ndim == 2 and self.all_features.shape[0] > 0:
            np.save(save_dir / "features.npy", self.all_features)

        logger.info("EvaluationResults saved -> %s", save_dir)

    @classmethod
    def load(cls, save_dir) -> "EvaluationResults":
        """Reload results previously written by :meth:`save`.

        Args:
            save_dir: Directory written by :meth:`save`.

        Returns:
            Fully populated :class:`EvaluationResults`.

        Raises:
            FileNotFoundError: If ``metrics.json`` is absent.
        """
        save_dir = Path(save_dir)
        mpath = save_dir / "metrics.json"
        if not mpath.exists():
            raise FileNotFoundError(
                f"metrics.json not found in {save_dir}. "
                "Has EvaluationResults.save() been called?"
            )

        with open(mpath, encoding="utf-8") as fh:
            data = json.load(fh)

        def _npy(name):
            p = save_dir / name
            return np.load(p) if p.exists() else np.empty(0)

        report_path = save_dir / "classification_report.txt"
        features = _npy("features.npy")

        return cls(
            accuracy           = float(data.get("accuracy", 0.0)),
            top1_accuracy      = float(data.get("top1_accuracy", 0.0)),
            top5_accuracy      = float(data.get("top5_accuracy", 0.0)),
            balanced_accuracy  = float(data.get("balanced_accuracy", 0.0)),
            precision_macro    = float(data.get("precision_macro", 0.0)),
            recall_macro       = float(data.get("recall_macro", 0.0)),
            f1_macro           = float(data.get("f1_macro", 0.0)),
            f1_weighted        = float(data.get("f1_weighted", 0.0)),
            ece                = float(data.get("ece", 0.0)),
            roc_auc_macro      = data.get("roc_auc_macro"),
            per_class_accuracy  = np.array(data.get("per_class_accuracy",  []), dtype=np.float64),
            per_class_precision = np.array(data.get("per_class_precision", []), dtype=np.float64),
            per_class_recall    = np.array(data.get("per_class_recall",    []), dtype=np.float64),
            per_class_f1        = np.array(data.get("per_class_f1",        []), dtype=np.float64),
            confusion_matrix            = _npy("confusion_matrix.npy"),
            confusion_matrix_normalized = _npy("confusion_matrix_normalized.npy"),
            classification_report_text  = (
                report_path.read_text(encoding="utf-8")
                if report_path.exists() else ""
            ),
            classification_report_dict = data.get("classification_report", {}),
            all_predictions   = _npy("predictions.npy"),
            all_targets       = _npy("targets.npy"),
            all_probabilities = _npy("probabilities.npy"),
            all_features      = features if features.ndim == 2 else np.empty((0, 0)),
            num_samples             = int(data.get("num_samples", 0)),
            num_classes             = int(data.get("num_classes", 0)),
            class_names             = data.get("class_names", []),
            evaluation_time_seconds = float(data.get("evaluation_time_seconds", 0.0)),
        )

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        roc_str = (
            f"{self.roc_auc_macro:.4f}" if self.roc_auc_macro is not None else "N/A"
        )
        d = "-" * 45
        return "\n".join([
            d,
            "  Evaluation Results",
            d,
            f"  Samples:              {self.num_samples:>10,}",
            f"  Classes:              {self.num_classes:>10,}",
            d,
            f"  Top-1  Accuracy:      {self.top1_accuracy:>10.4f}",
            f"  Top-5  Accuracy:      {self.top5_accuracy:>10.4f}",
            f"  Balanced Accuracy:    {self.balanced_accuracy:>10.4f}",
            d,
            f"  Precision (macro):    {self.precision_macro:>10.4f}",
            f"  Recall    (macro):    {self.recall_macro:>10.4f}",
            f"  F1        (macro):    {self.f1_macro:>10.4f}",
            f"  F1        (weighted): {self.f1_weighted:>10.4f}",
            d,
            f"  ECE:                  {self.ece:>10.4f}",
            f"  ROC-AUC (macro):      {roc_str:>10}",
            d,
            f"  Eval time:            {self.evaluation_time_seconds:>9.1f}s",
            d,
        ])


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """End-to-end evaluation pipeline for a trained classification model.

    Runs inference over a DataLoader, computes the full suite of
    classification metrics, and saves plots and numeric results to disk.

    The model is never modified: no gradient computation is performed and
    the training/eval mode is restored after inference.

    Args:
        model: Trained :class:`~torch.nn.Module` on ``device``.  If the
            model exposes ``get_features`` and ``extract_features=True``,
            512-d GAP representations are collected.
        dataloader: Evaluation-split DataLoader (``shuffle=False``).
        device: Compute device.  Both CPU and CUDA are supported.
        class_names: Human-readable label per class index.
        topk: k-values for top-k accuracy.  Defaults to ``(1, 5)``.
        extract_features: Collect GAP features when ``True`` and the model
            exposes ``get_features``.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        class_names: list,
        topk: tuple = (1, 5),
        extract_features: bool = False,
    ) -> None:
        self.model       = model
        self.dataloader  = dataloader
        self.device      = device
        self.class_names = list(class_names)
        self.num_classes = len(class_names)
        self.topk        = topk

        self._extract_features = extract_features and hasattr(model, "get_features")
        if extract_features and not hasattr(model, "get_features"):
            logger.warning(
                "extract_features=True but model has no get_features() — disabled."
            )

        for k in topk:
            if k > self.num_classes:
                logger.warning(
                    "topk=%d > num_classes=%d; value clipped — top-%d accuracy "
                    "may appear artificially high.",
                    k, self.num_classes, self.num_classes,
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self) -> EvaluationResults:
        """Run inference and compute all metrics.

        Sets the model to ``eval`` mode and uses ``torch.no_grad()``
        throughout.  The original training mode is restored on exit.

        Returns:
            A fully populated :class:`EvaluationResults` instance.
        """
        was_training = self.model.training
        self.model.eval()

        try:
            logger.info(
                "Evaluator | model=%s | device=%s | classes=%d",
                type(self.model).__name__, self.device, self.num_classes,
            )

            t0 = time.perf_counter()
            logits_np, probs_np, targets_np, features_np = self._run_inference()
            elapsed = time.perf_counter() - t0

            logger.info(
                "Inference complete | %d samples | %.1fs | %.0f samp/s",
                len(targets_np), elapsed,
                len(targets_np) / max(elapsed, 1e-9),
            )

            metrics = self._compute_metrics(logits_np, probs_np, targets_np)
            results = self._build_results(
                metrics, probs_np, targets_np, features_np, elapsed
            )

        finally:
            if was_training:
                self.model.train()

        logger.info("\n%s", results)
        return results

    def generate_report(
        self,
        results: EvaluationResults,
        save_dir,
        show: bool = False,
    ) -> None:
        """Save all numeric results and visualisations to ``save_dir``.

        Each plot is generated in an independent try/except block so that
        a failure in one visualisation does not prevent others from saving.

        Files created (see module docstring for the complete layout).

        Args:
            results: :class:`EvaluationResults` from :meth:`evaluate`.
            save_dir: Root output directory; created if absent.
            show: Call ``plt.show()`` after each figure when ``True``.
                Leave ``False`` in non-interactive scripts.
        """
        save_dir  = Path(save_dir)
        plots_dir = save_dir / _PLOTS_DIR
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Numeric data first.
        results.save(save_dir)

        # ── Confusion matrix — normalised (float, always safe) ─────────
        self._safe_plot(
            fn=plot_confusion_matrix,
            label="confusion_matrix_normalized",
            save_path=plots_dir / "confusion_matrix_normalized.png",
            show=show,
            cm=results.confusion_matrix,
            class_names=results.class_names,
            normalize=True,
            title="Confusion Matrix  (row-normalised — recall per class)",
        )

        # ── Confusion matrix — raw integer counts (local impl, no fmt bug) ─
        self._safe_plot(
            fn=self._plot_raw_counts,
            label="confusion_matrix_counts",
            save_path=plots_dir / "confusion_matrix_counts.png",
            show=show,
            cm=results.confusion_matrix,
            class_names=results.class_names,
        )

        # ── Per-class Precision / Recall / F1 bar chart ─────────────────
        if results.per_class_f1.size > 0:
            self._safe_plot(
                fn=plot_per_class_metrics,
                label="per_class_metrics",
                save_path=plots_dir / "per_class_metrics.png",
                show=show,
                per_class_precision=results.per_class_precision,
                per_class_recall=results.per_class_recall,
                per_class_f1=results.per_class_f1,
                class_names=results.class_names,
                title="Per-Class Precision / Recall / F1",
            )

        # ── Multi-class ROC curves ───────────────────────────────────────
        if results.all_probabilities.size > 0 and results.num_classes >= 2:
            roc_data = compute_roc_curves_per_class(
                results.all_probabilities,
                results.all_targets,
                results.num_classes,
            )
            if roc_data.get("classes_computed"):
                self._safe_plot(
                    fn=plot_roc_curves,
                    label="roc_curves",
                    save_path=plots_dir / "roc_curves.png",
                    show=show,
                    roc_data=roc_data,
                    class_names=results.class_names,
                    title=f"ROC Curves  ({results.num_classes} classes, OvR)",
                )

        # ── Calibration reliability diagram ─────────────────────────────
        if results.all_probabilities.size > 0:
            self._safe_plot(
                fn=plot_reliability_diagram,
                label="reliability_diagram",
                save_path=plots_dir / "reliability_diagram.png",
                show=show,
                probabilities=results.all_probabilities,
                targets=results.all_targets,
                title=f"Reliability Diagram  (ECE = {results.ece:.4f})",
            )

        # ── Scalar metric summary ────────────────────────────────────────
        self._safe_plot(
            fn=plot_metric_summary,
            label="metric_summary",
            save_path=plots_dir / "metric_summary.png",
            show=show,
            metrics=results.scalar_metrics(),
            title="Evaluation Metric Summary",
        )

        plt.close("all")
        logger.info("Report written -> %s", save_dir)

    def get_most_confused_pairs(
        self,
        results: EvaluationResults,
        top_n: int = 10,
    ) -> list:
        """Return the ``top_n`` most-confused off-diagonal class pairs.

        Args:
            results: :class:`EvaluationResults` from :meth:`evaluate`.
            top_n: Maximum number of pairs to return.

        Returns:
            List of dicts sorted by ``"count"`` descending, each with keys
            ``"true_class"``, ``"pred_class"``, ``"count"``, ``"rate"``,
            and ``"symmetric_count"`` (CM[i,j] + CM[j,i]).
        """
        cm    = results.confusion_matrix.astype(float)
        names = results.class_names
        pairs = []

        for i in range(results.num_classes):
            row_total = float(cm[i].sum())
            for j in range(results.num_classes):
                if i == j:
                    continue
                pairs.append({
                    "true_class":      names[i],
                    "pred_class":      names[j],
                    "count":           int(cm[i, j]),
                    "rate":            float(cm[i, j] / max(row_total, 1.0)),
                    "symmetric_count": int(cm[i, j] + cm[j, i]),
                })

        return sorted(pairs, key=lambda p: p["count"], reverse=True)[:top_n]

    def format_confused_pairs(self, pairs: list) -> str:
        """Return a formatted ASCII table of confused pairs.

        Args:
            pairs: Output of :meth:`get_most_confused_pairs`.

        Returns:
            Multi-line string ready for logging or printing.
        """
        if not pairs:
            return "No confused pairs."

        w = max(
            max(len(p["true_class"]) for p in pairs),
            max(len(p["pred_class"]) for p in pairs),
            12,
        )
        sep    = "  " + "-" * (w * 2 + 34)
        header = (
            f"  {'True Class':<{w}}  {'Predicted As':<{w}}  "
            f"{'Count':>7}  {'Rate':>7}  {'Sym.':>7}"
        )
        rows = [sep, header, sep]
        for p in pairs:
            rows.append(
                f"  {p['true_class']:<{w}}  {p['pred_class']:<{w}}  "
                f"{p['count']:>7}  {p['rate']:>7.3f}  "
                f"{p['symmetric_count']:>7}"
            )
        rows.append(sep)
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Private — inference
    # ------------------------------------------------------------------

    def _run_inference(self):
        """Batch inference over the DataLoader.

        Tensors accumulate on CPU.  Images and targets move to
        :attr:`device` with ``non_blocking=True`` for faster H<->D transfer
        when the DataLoader uses pinned memory.

        Returns:
            4-tuple ``(logits, probabilities, targets, features)`` as
            float32 / int64 NumPy arrays:

            - ``logits``       — raw output ``(N, C)``.
            - ``probabilities`` — softmax output ``(N, C)``.
            - ``targets``      — true labels ``(N,)`` int64.
            - ``features``     — GAP vectors ``(N, D)`` or ``(0, 0)``.
        """
        logit_chunks:   list = []
        target_chunks:  list = []
        feature_chunks: list = []

        pbar = tqdm(
            self.dataloader,
            desc="Evaluating",
            leave=False,
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for images, targets in pbar:
                images  = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                logits = self.model(images)
                logit_chunks.append(logits.cpu())
                target_chunks.append(targets.cpu())

                if self._extract_features:
                    feats = self.model.get_features(images)
                    feature_chunks.append(feats.cpu())

        all_logits  = torch.cat(logit_chunks,  dim=0).float()
        all_targets = torch.cat(target_chunks, dim=0).long()
        all_probs   = torch.softmax(all_logits, dim=1)

        logits_np  = all_logits.numpy().astype(np.float32)
        probs_np   = all_probs.numpy().astype(np.float32)
        targets_np = all_targets.numpy().astype(np.int64)

        if feature_chunks:
            features_np = (
                torch.cat(feature_chunks, dim=0).float().numpy().astype(np.float32)
            )
        else:
            features_np = np.empty((0, 0), dtype=np.float32)

        return logits_np, probs_np, targets_np, features_np

    # ------------------------------------------------------------------
    # Private — metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self, logits_np, probs_np, targets_np):
        """Delegate all metric computation to :mod:`src.evaluation.metrics`.

        Args:
            logits_np: Raw logits ``(N, C)`` float32.
            probs_np:  Softmax probabilities ``(N, C)`` float32.
            targets_np: True labels ``(N,)`` int64.

        Returns:
            Flat dict of metric name -> scalar or ndarray.
        """
        return compute_all_metrics(
            logits=logits_np,
            targets=targets_np,
            class_names=self.class_names,
            topk=self.topk,
        )

    # ------------------------------------------------------------------
    # Private — result assembly
    # ------------------------------------------------------------------

    def _build_results(
        self,
        metrics,
        probs_np,
        targets_np,
        features_np,
        elapsed: float,
    ) -> EvaluationResults:
        """Pack a :class:`EvaluationResults` from computed metrics.

        Args:
            metrics: Output of :meth:`_compute_metrics`.
            probs_np: Softmax probabilities ``(N, C)``.
            targets_np: True labels ``(N,)``.
            features_np: GAP features or empty array.
            elapsed: Inference duration in seconds.

        Returns:
            Populated :class:`EvaluationResults`.
        """
        top1 = float(metrics.get("top1", metrics.get("accuracy", 0.0)))
        top5 = float(metrics.get("top5", 0.0))

        return EvaluationResults(
            accuracy           = float(metrics["accuracy"]),
            top1_accuracy      = top1,
            top5_accuracy      = top5,
            balanced_accuracy  = float(metrics["balanced_accuracy"]),
            precision_macro    = float(metrics["precision_macro"]),
            recall_macro       = float(metrics["recall_macro"]),
            f1_macro           = float(metrics["f1_macro"]),
            f1_weighted        = float(metrics["f1_weighted"]),
            ece                = float(metrics["ece"]),
            roc_auc_macro      = metrics.get("roc_auc_macro"),
            per_class_accuracy  = np.asarray(metrics["per_class_accuracy"],  dtype=np.float64),
            per_class_precision = np.asarray(metrics["per_class_precision"], dtype=np.float64),
            per_class_recall    = np.asarray(metrics["per_class_recall"],    dtype=np.float64),
            per_class_f1        = np.asarray(metrics["per_class_f1"],        dtype=np.float64),
            confusion_matrix            = metrics["confusion_matrix"].astype(np.int64),
            confusion_matrix_normalized = metrics["confusion_matrix_normalized"].astype(np.float64),
            classification_report_text = metrics["classification_report_text"],
            classification_report_dict = metrics["classification_report_dict"],
            all_predictions   = metrics["predictions"].astype(np.int64),
            all_targets       = targets_np.astype(np.int64),
            all_probabilities = probs_np.astype(np.float32),
            all_features      = features_np,
            num_samples             = int(len(targets_np)),
            num_classes             = self.num_classes,
            class_names             = list(self.class_names),
            evaluation_time_seconds = elapsed,
        )

    # ------------------------------------------------------------------
    # Private — plot helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_plot(fn, label: str, save_path, show: bool, **kwargs) -> None:
        """Call a plotting function; log a warning on failure instead of crashing.

        Args:
            fn: Callable that accepts ``save_path`` and other ``kwargs``.
            label: Human-readable name for log messages.
            save_path: Destination PNG path.
            show: Whether to call ``plt.show()``.
            **kwargs: Forwarded to ``fn``.
        """
        try:
            fig = fn(**kwargs, save_path=save_path)
            if show and fig is not None:
                plt.show()
            logger.debug("Plot saved -> %s", save_path)
        except Exception as exc:
            logger.warning(
                "Could not generate '%s' plot: %s — skipping.", label, exc
            )

    @staticmethod
    def _plot_raw_counts(
        cm: np.ndarray,
        class_names: list,
        save_path=None,
    ) -> matplotlib.figure.Figure:
        """Plot the raw integer confusion matrix with correct int annotation.

        This local implementation keeps the confusion matrix as ``int64``
        throughout so that seaborn's ``fmt="d"`` annotation never encounters
        a float value (which caused a ValueError in the shared
        :func:`~src.evaluation.metrics.plot_confusion_matrix` when called
        with ``normalize=False``).

        Args:
            cm: Integer confusion matrix ``(C, C)``.
            class_names: Class label strings.
            save_path: Optional file path to save.

        Returns:
            :class:`~matplotlib.figure.Figure`.
        """
        num_classes = cm.shape[0]
        annotate = num_classes <= _ANNOT_THRESHOLD
        annot_fs = max(5, 12 - num_classes // 5)
        tick_fs  = max(5, 11 - num_classes // 8)
        side     = max(8.0, num_classes * 0.45)

        fig, ax = plt.subplots(figsize=(side + 2.0, side))

        # Explicitly keep integer dtype so fmt="d" is valid.
        cm_int = cm.astype(np.int64)

        sns.heatmap(
            cm_int,
            annot=annotate,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            square=True,
            linewidths=0.3 if num_classes <= 20 else 0.0,
            linecolor="lightgray",
            ax=ax,
            annot_kws={"size": annot_fs} if annotate else {},
        )
        ax.set_title("Confusion Matrix  (raw sample counts)", fontsize=14, pad=12)
        ax.set_xlabel("Predicted Class", fontsize=11)
        ax.set_ylabel("True Class",      fontsize=11)
        ax.tick_params(axis="x", rotation=90, labelsize=tick_fs)
        ax.tick_params(axis="y", rotation=0,  labelsize=tick_fs)
        fig.tight_layout()

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight")

        return fig
