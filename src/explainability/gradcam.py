"""Grad-CAM: Gradient-weighted Class Activation Mapping.

Complete from-scratch implementation following the original paper::

    Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D.
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization."
    ICCV 2017.  https://arxiv.org/abs/1610.02391

Mathematical formulation
------------------------
Given a trained CNN, an input image, and a target class *c*:

1. **Forward pass** — run the image through the network; register a
   forward hook on the target convolutional layer to capture its output
   feature maps **A** of shape ``(B, K, h, w)``.

2. **Backward pass** — backpropagate the class score ``y^c`` and register
   a backward hook to capture the gradients
   ``∂y^c / ∂A^k`` for every feature-map channel *k*.

3. **Importance weights** — global-average-pool the gradients spatially
   to obtain a scalar weight per channel::

       α^c_k = (1 / h·w)  Σ_{i,j}  ∂y^c / ∂A^k_{ij}

4. **Localisation map** — weighted sum of feature maps followed by ReLU::

       L^c = ReLU( Σ_k  α^c_k · A^k )

   ReLU keeps only activations that positively influence class *c*.

5. **Resize** — bilinearly upsample ``L^c`` to the input image resolution.

Hook design
-----------
Two hooks are attached to the same target layer:

- **Forward hook** (``register_forward_hook``) — captures ``output`` (the
  feature maps) on every forward pass.  The tensor is detached so that
  retaining a reference to it does not prevent the computation graph from
  being freed after ``backward()``.

- **Backward hook** (``register_full_backward_hook``) — captures
  ``grad_output[0]`` (the gradient of the loss w.r.t. the layer output)
  on every backward pass.  ``register_full_backward_hook`` is used instead
  of the deprecated ``register_backward_hook`` because it correctly
  handles non-tensor inputs and multi-output modules.

Both hooks store their values in private attributes that :meth:`GradCAM.generate`
reads after calling ``backward()``.

Important constraints
---------------------
- ``torch.no_grad()`` must **not** be active during :meth:`GradCAM.generate`;
  Grad-CAM requires gradient computation.
- The model should be in ``eval`` mode so that BatchNorm uses running
  statistics rather than batch statistics.
- Always call :meth:`GradCAM.remove_hooks` (or use the context manager) when
  done; unharvested hooks memory-leak and interfere with subsequent calls.

Sanity check (Adebayo et al., 2018)
------------------------------------
:func:`run_sanity_check` compares Grad-CAM maps from the trained model
against maps from a model with randomly re-initialised weights.  A faithful
explanation of the model's *learned* function should look qualitatively
different from random-weight maps.  High Pearson correlation between the
two indicates the maps capture data structure rather than model learning.

Example::

    import torch
    from src.explainability.gradcam import GradCAM, get_target_layer
    from src.explainability.gradcam import visualize_gradcam

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    target_layer = get_target_layer(model, "layer4")          # ResNet34 last stage

    with GradCAM(model, target_layer) as gcam:
        images = images.to(device)                            # [B, 3, H, W]
        cams   = gcam.generate(images, target_class=None)     # [B, 1, H, W]

    fig = visualize_gradcam(
        images=images.cpu(),
        cams=cams.cpu(),
        predictions=logits.argmax(1).cpu(),
        targets=labels.cpu(),
        class_names=class_names,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        save_path=Path("results/gradcam_batch.png"),
    )
"""

from __future__ import annotations

import copy
import logging
import warnings
from pathlib import Path
from typing import Optional, Union

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    # Core
    "GradCAM",
    # Image utilities
    "denormalize_image",
    "cam_to_heatmap",
    "overlay_heatmap",
    # Visualization
    "visualize_gradcam",
    "visualize_class_comparison",
    "visualize_multilayer_gradcam",
    # Helpers
    "get_target_layer",
    "run_sanity_check",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalization presets (dataset → (mean, std))
# ---------------------------------------------------------------------------

NORMALIZATION_STATS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "cifar10":  ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "imagenet": ((0.485,  0.456,  0.406),  (0.229,  0.224,  0.225)),
    "abo":      ((0.485,  0.456,  0.406),  (0.229,  0.224,  0.225)),
}

# ---------------------------------------------------------------------------
# Core: GradCAM
# ---------------------------------------------------------------------------


class GradCAM:
    """Grad-CAM explainer using forward and backward hooks.

    Generates a spatial importance map highlighting which image regions most
    influenced the model's score for a chosen target class.

    This class is intended to be used as a context manager::

        with GradCAM(model, target_layer) as gcam:
            cams = gcam.generate(images, target_class=3)

    When used as a context manager, hooks are registered on entry and
    automatically removed on exit — even if an exception is raised.  This
    prevents the memory leaks that result from abandoned hooks.

    Args:
        model: A trained :class:`~torch.nn.Module` in ``eval`` mode.
            The model must **not** be wrapped in ``torch.no_grad()``
            during :meth:`generate`.
        target_layer: The convolutional layer whose feature maps are used
            for the localisation map.  For ResNet-34, the last residual
            stage ``model.layer4`` (or its final block ``model.layer4[-1]``)
            is the standard choice.  Earlier layers produce higher-resolution
            but less semantically abstract maps.

    Attributes:
        model: The model passed at construction.
        target_layer: The hooked layer.

    Raises:
        RuntimeError: From :meth:`generate` if called inside
            ``torch.no_grad()``.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model        = model
        self.target_layer = target_layer

        # Storage for hook-captured tensors.
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        # Hook handles kept so we can remove them later.
        self._fwd_handle = None
        self._bwd_handle = None

        self._register_hooks()

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_) -> None:
        self.remove_hooks()

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """Attach forward and backward hooks to ``target_layer``.

        **Forward hook** captures the layer's output feature maps
        (``output``) on each forward pass.  The tensor is immediately
        detached so that the reference does not pin the computation graph.

        **Backward hook** captures ``grad_output[0]`` — the gradient of
        the loss with respect to the layer's output — on each backward
        pass.  ``register_full_backward_hook`` is used (PyTorch ≥ 1.8)
        because it correctly handles the gradient tuples for all module
        types including those with multiple outputs.
        """

        def _save_activation(
            module: nn.Module,
            inp: tuple,
            output: torch.Tensor,
        ) -> None:
            # Detach so the hook does not prevent graph cleanup.
            self._activations = output.detach()

        def _save_gradient(
            module: nn.Module,
            grad_input: tuple,
            grad_output: tuple,
        ) -> None:
            # grad_output[0] = ∂L/∂(layer output), shape [B, K, h, w].
            self._gradients = grad_output[0].detach()

        self._fwd_handle = self.target_layer.register_forward_hook(_save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(_save_gradient)

        logger.debug(
            "GradCAM hooks registered on layer: %s",
            type(self.target_layer).__name__,
        )

    def remove_hooks(self) -> None:
        """Deregister both hooks and release stored tensors.

        Must be called when Grad-CAM is no longer needed.  Failure to
        remove hooks causes a memory leak because each hook holds a
        reference to ``self``, which in turn holds references to
        ``_activations`` and ``_gradients``.
        """
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
            self._fwd_handle = None
        if self._bwd_handle is not None:
            self._bwd_handle.remove()
            self._bwd_handle = None
        self._activations = None
        self._gradients   = None
        logger.debug("GradCAM hooks removed.")

    # ------------------------------------------------------------------
    # CAM generation
    # ------------------------------------------------------------------

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[Union[int, list, torch.Tensor]] = None,
        retain_graph: bool = False,
    ) -> torch.Tensor:
        """Compute a Grad-CAM localisation map for ``target_class``.

        Executes one forward pass and one backward pass.  The model's
        current mode (``train`` / ``eval``) is preserved; it is the
        caller's responsibility to call ``model.eval()`` before generating
        explanations so that BatchNorm uses running statistics.

        **Do not wrap this call in** ``torch.no_grad()``; Grad-CAM requires
        active gradient computation.

        Steps (paper §3.1):

        1. Forward pass → capture activations **A** via forward hook.
        2. Compute class scores; backward pass → capture gradients
           ∂y^c/∂A^k via backward hook.
        3. Importance weights: ``α^c_k = mean_{i,j}(∂y^c/∂A^k_{ij})``.
        4. Weighted combination + ReLU: ``L = ReLU(Σ_k α^c_k A^k)``.
        5. Normalise each map to ``[0, 1]`` per sample.
        6. Bilinearly upsample to input spatial resolution.

        Args:
            input_tensor: Batch of normalised images ``(B, C, H, W)``
                on the same device as the model.  Must **not** have been
                created inside a ``torch.no_grad()`` context.
            target_class: Class index (or indices) for which to compute
                the CAM.

                - ``None`` (default): use the model's top-1 predicted class
                  for each sample.  Different samples may get different
                  target classes.
                - ``int``: same target class for every sample in the batch.
                - ``list[int]`` or ``torch.Tensor`` of shape ``(B,)``:
                  per-sample target classes.

            retain_graph: Passed to ``loss.backward()``.  Set ``True`` when
                :meth:`generate` will be called again on the same
                computation graph (e.g. :meth:`generate_class_comparison`).

        Returns:
            Float32 tensor of shape ``(B, 1, H, W)`` with values in
            ``[0, 1]``.  The spatial resolution matches ``input_tensor``.

        Raises:
            RuntimeError: If called inside ``torch.no_grad()``.
            RuntimeError: If the hooks have been removed (the context
                manager has exited).
        """
        if not torch.is_grad_enabled():
            raise RuntimeError(
                "GradCAM.generate() requires gradient computation but was "
                "called inside torch.no_grad().  Remove the no_grad context."
            )
        if self._fwd_handle is None:
            raise RuntimeError(
                "GradCAM hooks are not registered.  Either re-instantiate "
                "GradCAM or use it as a context manager."
            )

        B, C, H, W = input_tensor.shape
        device      = input_tensor.device

        # ── 1. Forward pass ───────────────────────────────────────────────
        # Zero all existing gradients in the model before backward.
        self.model.zero_grad()
        logits = self.model(input_tensor)   # (B, num_classes)

        # ── 2. Resolve target classes ─────────────────────────────────────
        if target_class is None:
            # Top-1 predicted class per sample.
            target_t = logits.detach().argmax(dim=1)                # (B,)
        elif isinstance(target_class, int):
            target_t = torch.full((B,), target_class,
                                  dtype=torch.long, device=device)  # (B,)
        elif isinstance(target_class, (list, np.ndarray)):
            target_t = torch.tensor(target_class,
                                    dtype=torch.long, device=device)
        else:
            target_t = target_class.to(device=device, dtype=torch.long)

        # ── 3. Backward pass ───────────────────────────────────────────────
        # Index logits with per-sample target classes.
        # Summing the scalar scores is valid because sample i's activation
        # does not participate in sample j's score — the gradients are
        # independent in a standard feed-forward network.
        scores = logits[torch.arange(B, device=device), target_t]  # (B,)
        scores.sum().backward(retain_graph=retain_graph)

        # ── 4. Read captured tensors ──────────────────────────────────────
        assert self._activations is not None, (
            "Forward hook did not fire.  The target layer may not be part "
            "of the model's computation path for this input."
        )
        assert self._gradients is not None, (
            "Backward hook did not fire.  Check that the target layer's "
            "output participates in the computation graph."
        )

        activations = self._activations.float()   # (B, K, h, w)
        gradients   = self._gradients.float()     # (B, K, h, w)

        # ── 5. Importance weights: α^c_k = GAP(∂y^c / ∂A^k) ─────────────
        # Global Average Pooling over spatial dimensions (h, w).
        weights = gradients.mean(dim=[2, 3], keepdim=True)   # (B, K, 1, 1)

        # ── 6. Weighted sum of feature maps + ReLU ────────────────────────
        # L^c = ReLU( Σ_k α^c_k · A^k )
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (B, 1, h, w)
        cam = F.relu(cam)

        # ── 7. Normalise each sample to [0, 1] ───────────────────────────
        cam = self._normalize(cam)

        # ── 8. Upsample to input resolution ──────────────────────────────
        cam = F.interpolate(
            cam,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (B, 1, H, W)

        return cam

    def generate_class_comparison(
        self,
        input_tensor: torch.Tensor,
        class_a: int,
        class_b: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate Grad-CAM maps for two classes from the same image.

        Useful for error analysis: given a misclassified sample, this
        method shows *what the model attends to* when it predicts
        ``class_a`` versus *what it should attend to* for ``class_b``.

        Each call to :meth:`generate` triggers an independent forward and
        backward pass, so the two maps are computed separately with no
        shared graph state.

        Args:
            input_tensor: Single normalised image ``(1, C, H, W)`` or a
                batch ``(B, C, H, W)`` where the same class pair is applied
                to every sample.
            class_a: First target class (e.g. the predicted class).
            class_b: Second target class (e.g. the true class).

        Returns:
            A 2-tuple ``(cam_a, cam_b)``, each ``(B, 1, H, W)`` float32
            in ``[0, 1]``.
        """
        cam_a = self.generate(input_tensor, target_class=class_a,
                              retain_graph=True)
        cam_b = self.generate(input_tensor, target_class=class_b,
                              retain_graph=False)
        return cam_a, cam_b

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(cam: torch.Tensor) -> torch.Tensor:
        """Normalise each sample's CAM to ``[0, 1]`` independently.

        Per-sample normalisation ensures that both high-confidence and
        low-confidence predictions produce visible heatmaps.  Without it,
        weak activations would be invisible when plotted next to strong ones.

        The degenerate case (all-zero map, which occurs when the target
        class has no influence on the chosen layer) is handled by clamping
        the divisor to a small positive value, returning an all-zero map.

        Args:
            cam: Float tensor of shape ``(B, 1, h, w)``.

        Returns:
            Normalised tensor of shape ``(B, 1, h, w)`` in ``[0, 1]``.
        """
        B = cam.shape[0]
        flat    = cam.view(B, -1)
        cam_min = flat.min(dim=1)[0].view(B, 1, 1, 1)
        cam_max = flat.max(dim=1)[0].view(B, 1, 1, 1)
        divisor = (cam_max - cam_min).clamp(min=1e-8)
        return (cam - cam_min) / divisor

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GradCAM("
            f"model={type(self.model).__name__}, "
            f"target_layer={type(self.target_layer).__name__}, "
            f"hooks_active={self._fwd_handle is not None})"
        )


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def denormalize_image(
    tensor: torch.Tensor,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std:  tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Reverse channel-wise normalisation and convert to uint8 ndarray.

    Applies the inverse transform: ``pixel = tensor * std + mean``, clamps
    to ``[0, 1]``, then scales to ``[0, 255]``.

    Args:
        tensor: Normalised image(s) of shape ``(C, H, W)`` or
            ``(B, C, H, W)`` as a float tensor.
        mean: Per-channel means used in the original normalisation.
        std: Per-channel standard deviations used in the original
            normalisation.

    Returns:
        - Single image ``(C, H, W)`` input → ``(H, W, 3)`` uint8.
        - Batch ``(B, C, H, W)`` input  → ``(B, H, W, 3)`` uint8.
    """
    t = tensor.detach().cpu().float()
    single = (t.ndim == 3)
    if single:
        t = t.unsqueeze(0)                                          # (1, C, H, W)

    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32).view(1, 3, 1, 1)

    t = (t * std_t + mean_t).clamp(0.0, 1.0)                       # reverse norm
    arr = (t.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)   # (B, H, W, 3)

    return arr[0] if single else arr


def cam_to_heatmap(
    cam: np.ndarray,
    colormap: str = "inferno",
) -> np.ndarray:
    """Apply a matplotlib colormap to a normalised CAM array.

    Args:
        cam: Grayscale float32 array of shape ``(H, W)`` with values in
            ``[0, 1]``.  Higher values should correspond to higher
            importance.
        colormap: A matplotlib colormap name.  Defaults to ``"inferno"``,
            which is perceptually uniform and prints well in greyscale.
            ``"jet"`` is traditional but perceptually non-linear.

    Returns:
        ``(H, W, 3)`` uint8 array with the colormap applied.
    """
    try:
        cmap = matplotlib.colormaps[colormap]
    except (AttributeError, KeyError):
        # Fallback for matplotlib < 3.5.
        cmap = plt.get_cmap(colormap)   # type: ignore[attr-defined]

    rgba = cmap(cam.astype(np.float32))         # (H, W, 4) in [0, 1]
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def overlay_heatmap(
    image:   np.ndarray,
    heatmap: np.ndarray,
    alpha:   float = 0.5,
) -> np.ndarray:
    """Alpha-blend a colourmap heatmap onto an original image.

    The blend formula is: ``output = alpha * image + (1 - alpha) * heatmap``.
    ``alpha = 0.5`` gives equal weight to both; higher alpha shows more of
    the original image, lower alpha emphasises the heatmap.

    Args:
        image:   Original image as ``(H, W, 3)`` uint8.
        heatmap: Coloured heatmap as ``(H, W, 3)`` uint8.  Must have the
            same spatial dimensions as ``image``.
        alpha: Weight of the original image in the blend.  Must be in
            ``[0, 1]``.  Defaults to ``0.5``.

    Returns:
        Blended ``(H, W, 3)`` uint8 array.
    """
    blended = (
        alpha       * image.astype(np.float32)
        + (1 - alpha) * heatmap.astype(np.float32)
    )
    return blended.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------


def visualize_gradcam(
    images:      torch.Tensor,
    cams:        torch.Tensor,
    predictions: torch.Tensor,
    targets:     torch.Tensor,
    class_names: list[str],
    mean:        tuple[float, ...],
    std:         tuple[float, ...],
    n_samples:   int = 8,
    colormap:    str = "inferno",
    alpha:       float = 0.5,
    title:       str = "Grad-CAM Visualisation",
    save_path:   Optional[Path] = None,
) -> matplotlib.figure.Figure:
    """Display a grid of Grad-CAM overlays for a batch of images.

    Each row in the figure shows one sample with three panels:

    1. **Original** — the denormalised input image.
    2. **Heatmap**  — the Grad-CAM importance map with the chosen colormap.
    3. **Overlay**  — the heatmap blended onto the original image.

    The row title shows the true and predicted class names and the model's
    maximum confidence (softmax probability is not computed here; confidence
    is inferred by the panel title being colour-coded green/red).

    Args:
        images: Normalised image batch ``(N, 3, H, W)``.
        cams: Grad-CAM maps ``(N, 1, H, W)`` in ``[0, 1]``.
        predictions: Predicted class indices ``(N,)`` (argmax of logits).
        targets: True class indices ``(N,)``.
        class_names: Human-readable name for each class index.
        mean: Per-channel mean used in image normalisation.
        std: Per-channel std used in image normalisation.
        n_samples: Maximum number of samples to show.  Clipped to
            ``len(images)`` when fewer samples are available.
        colormap: Matplotlib colormap for the heatmap panels.
        alpha: Blend weight for the overlay panel.
        title: Figure suptitle.
        save_path: If provided, save the figure to this path (PNG, 150 dpi).

    Returns:
        :class:`~matplotlib.figure.Figure` containing the grid.
    """
    n = min(n_samples, images.shape[0])
    preds_np = predictions.cpu().numpy()
    tgts_np  = targets.cpu().numpy()

    col_labels = ["Original", "Grad-CAM Heatmap", "Overlay"]
    fig, axes  = plt.subplots(
        nrows=n, ncols=3,
        figsize=(11, 3.0 * n),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.002)

    for row in range(n):
        img_np  = denormalize_image(images[row], mean, std)   # (H, W, 3) uint8
        cam_np  = cams[row, 0].cpu().numpy()                  # (H, W)    float
        heat_np = cam_to_heatmap(cam_np, colormap)            # (H, W, 3) uint8
        over_np = overlay_heatmap(img_np, heat_np, alpha)     # (H, W, 3) uint8

        pred  = int(preds_np[row])
        true_ = int(tgts_np[row])
        correct = pred == true_

        row_title = (
            f"True: {class_names[true_]}  |  "
            f"Pred: {class_names[pred]}"
        )
        title_color = "#2ca02c" if correct else "#d62728"  # green / red

        for col, panel in enumerate([img_np, heat_np, over_np]):
            ax = axes[row][col]
            ax.imshow(panel)
            ax.axis("off")
            if col == 0:
                ax.set_title(row_title, fontsize=9,
                             color=title_color, fontweight="bold", pad=3)
            elif row == 0:
                ax.set_title(col_labels[col], fontsize=9,
                             color="gray", pad=3)

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def visualize_class_comparison(
    image:           torch.Tensor,
    cam_predicted:   torch.Tensor,
    cam_correct:     torch.Tensor,
    predicted_class: int,
    correct_class:   int,
    confidence:      float,
    class_names:     list[str],
    mean:            tuple[float, ...],
    std:             tuple[float, ...],
    colormap:        str = "inferno",
    alpha:           float = 0.5,
    save_path:       Optional[Path] = None,
) -> matplotlib.figure.Figure:
    """Compare Grad-CAM for the predicted class versus the correct class.

    This five-panel figure is the primary tool for error analysis.  For a
    misclassified sample it reveals whether:

    - The model attends to a *wrong region* (background or irrelevant object
      part) for the predicted class.
    - The model attends to the *right region* when forced to consider the
      correct class (suggesting a representation vs. decision problem).

    Layout (one row)::

        [Original] | [CAM(predicted)] | [Overlay(pred)] |
        [CAM(correct)] | [Overlay(correct)]

    Args:
        image: Single normalised image ``(3, H, W)``.
        cam_predicted: Grad-CAM for the predicted class ``(1, H, W)``
            in ``[0, 1]``.
        cam_correct: Grad-CAM for the correct class ``(1, H, W)``
            in ``[0, 1]``.
        predicted_class: Index of the predicted class.
        correct_class: Index of the true class.
        confidence: Model's softmax confidence for the predicted class.
        class_names: Human-readable class names.
        mean: Normalisation mean.
        std: Normalisation std.
        colormap: Colormap for heatmap panels.
        alpha: Overlay blend weight.
        save_path: Optional path to save the figure.

    Returns:
        :class:`~matplotlib.figure.Figure` with the five-panel comparison.
    """
    img_np  = denormalize_image(image, mean, std)       # (H, W, 3)

    cam_pred_np = cam_predicted[0].cpu().numpy()        # (H, W)
    cam_corr_np = cam_correct[0].cpu().numpy()          # (H, W)

    heat_pred = cam_to_heatmap(cam_pred_np, colormap)
    heat_corr = cam_to_heatmap(cam_corr_np, colormap)
    over_pred = overlay_heatmap(img_np, heat_pred, alpha)
    over_corr = overlay_heatmap(img_np, heat_corr, alpha)

    pred_name = class_names[predicted_class]
    corr_name = class_names[correct_class]

    panels = [img_np, heat_pred, over_pred, heat_corr, over_corr]
    col_titles = [
        "Original",
        f"CAM → \"{pred_name}\"",
        f"Overlay → \"{pred_name}\"",
        f"CAM → \"{corr_name}\" (true)",
        f"Overlay → \"{corr_name}\" (true)",
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))
    fig.suptitle(
        f"Misclassification: True = \"{corr_name}\"  |  "
        f"Predicted = \"{pred_name}\"  (conf {confidence:.2%})",
        fontsize=12, fontweight="bold", color="#d62728",
    )

    for ax, panel, col_title in zip(axes, panels, col_titles):
        ax.imshow(panel)
        ax.axis("off")
        ax.set_title(col_title, fontsize=9, pad=4)

    fig.tight_layout()
    _save_figure(fig, save_path)
    return fig


def visualize_multilayer_gradcam(
    image:       torch.Tensor,
    model:       nn.Module,
    layer_names: list[str],
    target_class: Optional[int],
    class_names: list[str],
    mean:        tuple[float, ...],
    std:         tuple[float, ...],
    colormap:    str = "inferno",
    alpha:       float = 0.5,
    save_path:   Optional[Path] = None,
) -> matplotlib.figure.Figure:
    """Compare Grad-CAM maps from multiple convolutional layers.

    Shows how the model's attention evolves from early, texture-sensitive
    layers to late, semantically abstract layers — the "manifold unfolding"
    visualisation described in the project's t-SNE analysis section.

    Args:
        image: Single normalised image ``(3, H, W)`` on the correct device.
        model: Trained model in ``eval`` mode.
        layer_names: List of layer identifiers accepted by
            :func:`get_target_layer`, e.g.
            ``["layer1", "layer2", "layer3", "layer4"]``.
        target_class: Class index for the CAM, or ``None`` for top-1.
        class_names: Human-readable class names.
        mean: Normalisation mean.
        std: Normalisation std.
        colormap: Colormap for heatmap panels.
        alpha: Blend weight for overlay panels.
        save_path: Optional save path.

    Returns:
        :class:`~matplotlib.figure.Figure` with ``len(layer_names) + 1``
        columns and 2 rows (heatmap / overlay).
    """
    img_np  = denormalize_image(image, mean, std)   # (H, W, 3)
    x       = image.unsqueeze(0) if image.ndim == 3 else image

    n_layers = len(layer_names)
    fig = plt.figure(figsize=(3.5 * (n_layers + 1), 7))
    gs  = gridspec.GridSpec(2, n_layers + 1, figure=fig,
                            hspace=0.05, wspace=0.08)

    # Column 0 — original image, spanning both rows.
    ax_orig = fig.add_subplot(gs[:, 0])
    ax_orig.imshow(img_np)
    ax_orig.axis("off")
    ax_orig.set_title("Original", fontsize=10, pad=4)

    for col, layer_name in enumerate(layer_names, start=1):
        layer  = get_target_layer(model, layer_name)

        with GradCAM(model, layer) as gcam:
            cam_t = gcam.generate(x, target_class=target_class)  # (1, 1, H, W)

        cam_np  = cam_t[0, 0].cpu().numpy()
        heat_np = cam_to_heatmap(cam_np, colormap)
        over_np = overlay_heatmap(img_np, heat_np, alpha)

        ax_heat = fig.add_subplot(gs[0, col])
        ax_heat.imshow(heat_np)
        ax_heat.axis("off")
        ax_heat.set_title(layer_name, fontsize=9, pad=4)

        ax_over = fig.add_subplot(gs[1, col])
        ax_over.imshow(over_np)
        ax_over.axis("off")

    # Row labels.
    fig.text(0.01, 0.75, "Heatmap", va="center", rotation="vertical",
             fontsize=10, color="gray")
    fig.text(0.01, 0.25, "Overlay", va="center", rotation="vertical",
             fontsize=10, color="gray")

    pred_label = (
        class_names[target_class]
        if target_class is not None
        else "top-1"
    )
    fig.suptitle(
        f"Grad-CAM across layers — target: \"{pred_label}\"",
        fontsize=12, fontweight="bold",
    )

    _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def _save_figure(
    fig: matplotlib.figure.Figure,
    save_path: Optional[Path],
    dpi: int = 150,
) -> None:
    """Save ``fig`` to ``save_path`` at ``dpi`` dots per inch.

    Args:
        fig: Figure to save.
        save_path: Destination path.  Parent directories are created.
            ``None`` is a no-op.
        dpi: Output resolution.
    """
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info("Grad-CAM figure saved → %s", path)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def get_target_layer(
    model: nn.Module,
    layer_name: str = "layer4",
) -> nn.Module:
    """Retrieve a sub-module from a ResNet-34 model by name.

    Supports the dot-separated path convention used throughout this project::

        get_target_layer(model, "layer4")         # nn.Sequential
        get_target_layer(model, "layer4.2")       # last BasicBlock
        get_target_layer(model, "layer4.2.conv2") # specific Conv2d

    Delegates to ``model.get_layer_by_name`` when available (ResNet34
    exposes this method), and falls back to manual attribute traversal
    for any other module.

    The recommended default for ResNet-34 is ``"layer4"`` (or its last
    block ``"layer4.2"``): this layer produces the most semantically
    abstract feature maps and yields the most interpretable Grad-CAM maps.
    Earlier layers (``"layer1"``–``"layer3"``) produce higher-resolution
    but less semantically focused maps — useful for the multi-layer
    comparison in :func:`visualize_multilayer_gradcam`.

    Args:
        model: A :class:`~torch.nn.Module`, typically a ResNet-34 instance.
        layer_name: Dot-separated path to the target sub-module.
            Numeric path segments index into :class:`~torch.nn.Sequential`
            containers.

    Returns:
        The target :class:`~torch.nn.Module`.

    Raises:
        AttributeError: If any segment of ``layer_name`` does not correspond
            to a valid attribute or index.
    """
    # Fast path: the model exposes the canonical lookup method.
    if hasattr(model, "get_layer_by_name"):
        return model.get_layer_by_name(layer_name)

    # Manual traversal for arbitrary modules.
    parts  = layer_name.split(".")
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]          # type: ignore[index]
        else:
            if not hasattr(module, part):
                raise AttributeError(
                    f"Module '{type(module).__name__}' has no attribute "
                    f"'{part}' (from path '{layer_name}')."
                )
            module = getattr(module, part)
    return module


# ---------------------------------------------------------------------------
# Sanity check (Adebayo et al., 2018)
# ---------------------------------------------------------------------------


def run_sanity_check(
    model:       nn.Module,
    input_tensor: torch.Tensor,
    layer_name:  str = "layer4",
    target_class: Optional[int] = None,
    n_samples:   int = 4,
) -> dict:
    """Run the Adebayo et al. (2018) sanity check for Grad-CAM.

    A faithful saliency map should look qualitatively *different* from the
    map produced by a model with randomly re-initialised weights.  If both
    maps look similar, the explanation is capturing properties of the *data
    distribution* rather than the model's *learned function*.

    Procedure:

    1. Generate Grad-CAM maps with the trained model.
    2. Deep-copy the model and re-initialise all weights with Kaiming Normal.
    3. Generate Grad-CAM maps with the randomised model.
    4. Compute the Pearson correlation between each pair of maps.
       Low correlation (|ρ| < 0.5) supports faithfulness.

    Args:
        model: Trained model in ``eval`` mode.
        input_tensor: Sample images ``(N, C, H, W)`` on the model's device.
        layer_name: Target layer for Grad-CAM (default ``"layer4"``).
        target_class: Fixed target class, or ``None`` for top-1.
        n_samples: Number of samples from ``input_tensor`` to check.

    Returns:
        Dictionary with keys:

        - ``"cams_trained"``  — Grad-CAM maps from the trained model ``(n, H, W)``.
        - ``"cams_random"``   — Grad-CAM maps from the randomised model ``(n, H, W)``.
        - ``"correlations"``  — Per-sample Pearson ρ, list of length ``n``.
        - ``"mean_correlation"`` — Mean |ρ| across samples.
        - ``"passes"``        — ``True`` when mean |ρ| < 0.5 (maps differ).
        - ``"interpretation"`` — Human-readable summary string.
    """
    n = min(n_samples, input_tensor.shape[0])
    x = input_tensor[:n]

    # ── Trained model CAMs ────────────────────────────────────────────────
    target_layer = get_target_layer(model, layer_name)
    with GradCAM(model, target_layer) as gcam:
        cams_trained = gcam.generate(x, target_class=target_class)   # (n, 1, H, W)

    # ── Randomised model CAMs ─────────────────────────────────────────────
    random_model = copy.deepcopy(model)
    _randomize_weights(random_model)
    random_model.eval()

    rand_layer = get_target_layer(random_model, layer_name)
    with GradCAM(random_model, rand_layer) as gcam_r:
        cams_random = gcam_r.generate(x, target_class=target_class)   # (n, 1, H, W)

    del random_model  # Free memory.

    # ── Correlation analysis ──────────────────────────────────────────────
    correlations: list[float] = []
    trained_np = cams_trained[:, 0].detach().cpu().numpy()  # (n, H, W)
    random_np  = cams_random[:, 0].detach().cpu().numpy()   # (n, H, W)

    for i in range(n):
        t_flat = trained_np[i].ravel()
        r_flat = random_np[i].ravel()
        # Pearson r: numerically stable with clipping.
        if t_flat.std() < 1e-8 or r_flat.std() < 1e-8:
            correlations.append(0.0)
        else:
            rho = float(np.corrcoef(t_flat, r_flat)[0, 1])
            correlations.append(rho if not np.isnan(rho) else 0.0)

    mean_corr = float(np.mean(np.abs(correlations)))
    passes    = mean_corr < 0.5

    interp = (
        f"Mean |ρ| = {mean_corr:.4f} (threshold 0.5). "
        + ("PASS — maps differ between trained and random model; "
           "explanation reflects learned function."
           if passes
           else "FAIL — maps are similar to random model; "
                "explanation may capture data structure, not model learning.")
    )

    logger.info("Sanity check: %s", interp)

    return {
        "cams_trained":      trained_np,
        "cams_random":       random_np,
        "correlations":      correlations,
        "mean_correlation":  mean_corr,
        "passes":            passes,
        "interpretation":    interp,
    }


def _randomize_weights(model: nn.Module) -> None:
    """Re-initialise all Conv2d and Linear weights with Kaiming Normal.

    Used internally by :func:`run_sanity_check` to create a random-weight
    baseline model.

    Args:
        model: Module whose weights are re-initialised in-place.
    """
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                m.running_mean.zero_()
                m.running_var.fill_(1.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
