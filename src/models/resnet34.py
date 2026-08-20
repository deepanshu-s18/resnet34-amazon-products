"""ResNet-34 architecture implementation for image classification.

This module provides ``ResNet34``, a full implementation of the 34-layer
Residual Network from He et al. (2016):

    "Deep Residual Learning for Image Recognition"
    He, K., Zhang, X., Ren, S., & Sun, J. (CVPR 2016).
    https://arxiv.org/abs/1512.03385

The architecture assembles :class:`~src.models.residual_block.BasicBlock`
instances into four residual layer groups following Table 1 of the paper::

    Stem  → Layer1 (×3) → Layer2 (×4) → Layer3 (×6) → Layer4 (×3)
           [64]           [64]           [128]          [256]          [512]

Two stem configurations are provided to accommodate different input sizes:

- **ImageNet mode** (``dataset="imagenet"``): 7×7 convolution with stride 2
  followed by 3×3 max-pooling with stride 2.  Designed for 224×224 inputs;
  reduces spatial resolution from 224→112→56 before the residual groups.

- **CIFAR-10 mode** (``dataset="cifar10"``): 3×3 convolution with stride 1,
  no max-pooling.  Designed for 32×32 inputs; preserves the full spatial
  resolution into the residual groups.

Ablation study flags (``use_skip_connection``, ``use_batch_norm``) are
propagated uniformly to every :class:`BasicBlock`, enabling Plain-34
(Ablation A1) and no-BN (Ablation A5) variants without architecture changes.

Example::

    >>> import torch
    >>> from resnet34 import ResNet34
    >>>
    >>> # Standard ResNet-34 for ABO (224×224, 50 classes)
    >>> model = ResNet34(num_classes=50, dataset="imagenet")
    >>> x = torch.randn(2, 3, 224, 224)
    >>> model(x).shape
    torch.Size([2, 50])
    >>>
    >>> # CIFAR-10 adapted ResNet-34 (32×32, 10 classes)
    >>> model = ResNet34(num_classes=10, dataset="cifar10")
    >>> x = torch.randn(4, 3, 32, 32)
    >>> model(x).shape
    torch.Size([4, 10])
    >>>
    >>> # Parameter count correctness gate
    >>> model_1k = ResNet34(num_classes=1000, dataset="imagenet")
    >>> model_1k.count_parameters()["total"]
    21797672
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.init as init

from .residual_block import BasicBlock

__all__ = ["ResNet34"]

# ---------------------------------------------------------------------------
# Type alias for the dataset mode parameter.
# ---------------------------------------------------------------------------
_DatasetMode = Literal["imagenet", "cifar10"]


class ResNet34(nn.Module):
    """ResNet-34 image classifier built from first principles.

    Implements the complete 34-layer residual network architecture from
    He et al. (2016), assembled from :class:`BasicBlock` units.  The model
    returns raw logits — softmax is never applied internally.

    The class exposes ``get_features`` for 512-d representation extraction
    (used by Grad-CAM and t-SNE analysis) and ``get_layer_by_name`` for
    hook registration without hard-coded module traversal.

    Attributes:
        num_classes: Number of output logits.
        dataset: Stem configuration — ``"imagenet"`` or ``"cifar10"``.
        use_skip_connection: Whether residual additions are active in every
            ``BasicBlock``.  ``False`` produces Plain-34 (Ablation A1).
        use_batch_norm: Whether ``BatchNorm2d`` is active in every
            ``BasicBlock`` and in the stem.  ``False`` substitutes
            ``nn.Identity`` everywhere (Ablation A5).
        dropout_rate: Dropout probability applied between the global average
            pooling output and the classification head.
        zero_init_residual: Whether the final BN scale (γ) in each
            ``BasicBlock`` was initialised to zero.
        stem: Stem sub-network (Conv → BN → ReLU [→ MaxPool for imagenet]).
        layer1: First residual group — 3 ``BasicBlock`` units, 64 channels.
        layer2: Second residual group — 4 ``BasicBlock`` units, 128 channels.
        layer3: Third residual group — 6 ``BasicBlock`` units, 256 channels.
        layer4: Fourth residual group — 3 ``BasicBlock`` units, 512 channels.
        avgpool: Global Average Pooling — reduces any ``H×W`` to ``1×1``.
        dropout: Dropout layer (no-op when ``dropout_rate=0.0``).
        fc: Fully-connected classification head (512 → ``num_classes``).
    """

    def __init__(
        self,
        num_classes: int = 1000,
        input_channels: int = 3,
        dataset: _DatasetMode = "imagenet",
        use_skip_connection: bool = True,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        zero_init_residual: bool = False,
    ) -> None:
        """Build all sub-modules and apply weight initialisation.

        Args:
            num_classes: Number of output units in the classification head.
                Positive integer.  Changes only the final ``nn.Linear`` layer;
                all other layer dimensions are fixed by the ResNet-34 spec.
            input_channels: Number of channels in the input tensor — ``3``
                for RGB, ``1`` for greyscale.  Affects the stem convolution
                only; all downstream layers operate on 64-channel feature maps.
                Defaults to ``3``.
            dataset: Selects the stem architecture.

                - ``"imagenet"``: 7×7 conv (stride 2, padding 3) followed by
                  3×3 max-pool (stride 2, padding 1).  For 224×224 inputs,
                  produces a 56×56 feature map entering ``layer1``.
                - ``"cifar10"``: 3×3 conv (stride 1, padding 1), no max-pool.
                  For 32×32 inputs, the full spatial resolution is preserved
                  into ``layer1``.

                Any other value raises ``ValueError``.  Defaults to
                ``"imagenet"``.
            use_skip_connection: If ``True`` (default), every ``BasicBlock``
                applies the residual addition ``F(x) + shortcut(x)``.
                If ``False``, skip connections are suppressed in all blocks,
                producing a depth-equivalent plain network (Ablation A1).
            use_batch_norm: If ``True`` (default), ``BatchNorm2d`` is placed
                after every convolution in the stem and in all blocks.
                If ``False``, all ``BatchNorm2d`` layers — including projection
                shortcuts — are replaced with ``nn.Identity`` (Ablation A5).
            dropout_rate: Probability of zeroing elements in the 512-d
                representation before the FC layer.  ``0.0`` (default) disables
                dropout.  The ``nn.Dropout`` module is always constructed;
                ``p=0.0`` makes it a no-op in both train and eval modes.
            zero_init_residual: If ``True``, initialises the scale parameter
                (γ) of the final ``BatchNorm2d`` in every ``BasicBlock`` to
                zero after the main Kaiming initialisation pass.  This makes
                every residual branch an identity at the start of training,
                improving optimisation stability (He et al., 2019).  Only
                effective when ``use_batch_norm=True``.  Defaults to ``False``.

        Raises:
            ValueError: If ``dataset`` is not ``"imagenet"`` or ``"cifar10"``.
            ValueError: If ``num_classes`` is less than 1.
            ValueError: If ``dropout_rate`` is not in ``[0.0, 1.0]``.
        """
        super().__init__()

        if dataset not in ("imagenet", "cifar10"):
            raise ValueError(
                f"dataset must be 'imagenet' or 'cifar10', got '{dataset}'."
            )
        if num_classes < 1:
            raise ValueError(
                f"num_classes must be >= 1, got {num_classes}."
            )
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(
                f"dropout_rate must be in [0.0, 1.0], got {dropout_rate}."
            )

        # Store configuration as instance attributes for inspection and
        # serialisation.  ``get_layer_by_name`` and ``count_parameters``
        # rely on these; the testing suite also inspects them directly.
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dataset = dataset
        self.use_skip_connection = use_skip_connection
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.zero_init_residual = zero_init_residual

        # ------------------------------------------------------------------
        # Stem
        # ------------------------------------------------------------------
        self.stem: nn.Sequential = self._build_stem()

        # ------------------------------------------------------------------
        # Residual layer groups
        # Block counts [3, 4, 6, 3] are fixed by the ResNet-34 specification.
        # Channel progression: 64 → 128 → 256 → 512.
        # Only the *first* block of each group (layer2–layer4) uses stride=2
        # to halve the spatial dimensions; _make_layer handles this.
        # ------------------------------------------------------------------
        self.layer1 = self._make_layer(
            in_channels=64,
            out_channels=64,
            num_blocks=3,
            stride=1,
        )
        self.layer2 = self._make_layer(
            in_channels=64,
            out_channels=128,
            num_blocks=4,
            stride=2,
        )
        self.layer3 = self._make_layer(
            in_channels=128,
            out_channels=256,
            num_blocks=6,
            stride=2,
        )
        self.layer4 = self._make_layer(
            in_channels=256,
            out_channels=512,
            num_blocks=3,
            stride=2,
        )

        # ------------------------------------------------------------------
        # Classification head
        # ------------------------------------------------------------------
        # AdaptiveAvgPool2d(1, 1) computes the per-channel spatial mean over
        # any H×W, making the head agnostic to the stem's spatial output.
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Dropout is always instantiated; p=0.0 makes it a functional no-op.
        # This keeps the forward path unconditional across all configurations.
        self.dropout = nn.Dropout(p=dropout_rate)

        # FC layer uses bias=True — there is no BatchNorm after the linear
        # layer, so the bias is not redundant.
        # 512 * BasicBlock.expansion == 512 * 1 == 512.
        self.fc = nn.Linear(
            in_features=512 * BasicBlock.expansion,
            out_features=num_classes,
            bias=True,
        )

        # ------------------------------------------------------------------
        # Weight initialisation
        # ------------------------------------------------------------------
        self._initialize_weights()

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _build_stem(self) -> nn.Sequential:
        """Construct the dataset-appropriate stem sub-network.

        The stem is the first stage of the network and conditions the spatial
        resolution entering the residual groups.

        - **ImageNet** (224×224 input): 7×7 conv (stride 2, padding 3) brings
          the spatial dimension from 224 to 112; a 3×3 max-pool (stride 2,
          padding 1) further halves it to 56.  Output: ``[B, 64, 56, 56]``.
        - **CIFAR-10** (32×32 input): 3×3 conv (stride 1, padding 1) preserves
          the full spatial resolution.  No max-pool.
          Output: ``[B, 64, 32, 32]``.

        In both modes, ``bias=False`` is used in the convolution because a
        ``BatchNorm2d`` (or its ``nn.Identity`` substitute) immediately follows.

        Returns:
            An ``nn.Sequential`` containing the stem layers in forward order.
        """
        stem_bn: nn.Module = (
            nn.BatchNorm2d(64) if self.use_batch_norm else nn.Identity()
        )

        if self.dataset == "imagenet":
            return nn.Sequential(
                nn.Conv2d(
                    self.input_channels,
                    64,
                    kernel_size=7,
                    stride=2,
                    padding=3,      # Preserves: floor((224 + 6 - 7) / 2) + 1 = 112
                    bias=False,
                ),
                stem_bn,
                nn.ReLU(inplace=True),
                nn.MaxPool2d(
                    kernel_size=3,
                    stride=2,
                    padding=1,      # Preserves: floor((112 + 2 - 3) / 2) + 1 = 56
                ),
            )

        # dataset == "cifar10"
        return nn.Sequential(
            nn.Conv2d(
                self.input_channels,
                64,
                kernel_size=3,
                stride=1,
                padding=1,          # Preserves: floor((32 + 2 - 3) / 1) + 1 = 32
                bias=False,
            ),
            stem_bn,
            nn.ReLU(inplace=True),
            # No max-pool: a 32×32 input cannot afford the 4× spatial
            # reduction that the ImageNet stem applies.
        )

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Build one residual layer group as a sequential stack of BasicBlocks.

        The first block in the group receives ``stride`` and is responsible
        for any spatial downsampling and channel dimension change.  All
        subsequent blocks use ``stride=1`` and ``in_channels=out_channels``,
        making them identity-shortcut blocks.

        Args:
            in_channels: Channel depth of the tensor entering this group.
                Must match the output channel count of the preceding group
                (or of the stem for ``layer1``).
            out_channels: Channel depth produced by every block in this group
                and returned by the ``nn.Sequential``.
            num_blocks: Number of ``BasicBlock`` instances to stack.
                ResNet-34 uses ``[3, 4, 6, 3]`` for layers 1–4 respectively.
            stride: Stride for the *first* block's ``conv1``.  Use ``1`` for
                ``layer1`` (no spatial change) and ``2`` for layers 2–4
                (halve spatial dimensions).  All subsequent blocks use
                ``stride=1`` unconditionally.

        Returns:
            An ``nn.Sequential`` of ``num_blocks`` ``BasicBlock`` instances.
        """
        blocks: list[nn.Module] = []

        # First block: may downsample spatially (stride=2) and always handles
        # the channel transition (in_channels → out_channels).
        blocks.append(
            BasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_skip_connection=self.use_skip_connection,
                use_batch_norm=self.use_batch_norm,
            )
        )

        # Remaining blocks: same channel width, stride=1, identity shortcut.
        for _ in range(1, num_blocks):
            blocks.append(
                BasicBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    use_skip_connection=self.use_skip_connection,
                    use_batch_norm=self.use_batch_norm,
                )
            )

        return nn.Sequential(*blocks)

    def _initialize_weights(self) -> None:
        """Apply Kaiming Normal initialisation to all weight tensors.

        Iterates over ``self.modules()`` and applies type-specific rules:

        - ``nn.Conv2d``: Kaiming Normal with ``mode='fan_out'`` and
          ``nonlinearity='relu'``.  The ``fan_out`` mode prioritises gradient
          variance stability in the backward pass, which is the critical
          constraint for deep networks.
        - ``nn.BatchNorm2d``: Scale (γ) initialised to 1, shift (β) to 0.
          These are the identity transform values; gradients learn from there.
        - ``nn.Linear``: Weights drawn from ``N(0, 0.01)``; bias set to 0.

        After the main loop, if ``zero_init_residual=True``, the final BN
        scale (``bn2.weight``) in every ``BasicBlock`` is set to zero.  This
        initialises every residual branch as a zero function, making the
        whole network a chain of ``ReLU(x)`` mappings at the start of
        training — a more stable optimisation starting point (He et al., 2019,
        "Bag of Tricks for Image Classification").

        Note:
            This method is called once at the end of ``__init__``.  It must
            not be called again after checkpoint loading — ``load_state_dict``
            overwrites all parameter tensors, making re-initialisation both
            unnecessary and destructive.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                # Conv layers that precede BatchNorm always have bias=False,
                # so module.bias is None in those cases.
                if module.bias is not None:
                    init.constant_(module.bias, 0.0)

            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1.0)   # γ = 1 (identity scale)
                init.constant_(module.bias, 0.0)     # β = 0 (no shift)

            elif isinstance(module, nn.Linear):
                init.normal_(module.weight, mean=0.0, std=0.01)
                init.constant_(module.bias, 0.0)

        # Optional zero-initialisation of the last BN in each residual branch.
        # Must run after the main loop so that it overrides the γ=1 default.
        if self.zero_init_residual:
            for module in self.modules():
                if isinstance(module, BasicBlock) and isinstance(
                    module.bn2, nn.BatchNorm2d
                ):
                    init.constant_(module.bn2.weight, 0.0)

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of images.

        Applies the full ResNet-34 forward pass in order::

            Stem → Layer1 → Layer2 → Layer3 → Layer4
            → GlobalAvgPool → Flatten → Dropout → FC

        The output is a tensor of **raw logits**.  Softmax is not applied.
        The caller or the loss function (e.g. ``nn.CrossEntropyLoss``, which
        applies ``log_softmax`` internally) is responsible for converting
        logits to probabilities.  Never apply softmax in this method — doing
        so and then passing to ``CrossEntropyLoss`` applies the operation
        twice, producing incorrect, numerically unstable loss values.

        Args:
            x: Input image batch of shape ``(B, C, H, W)`` where ``C``
                equals ``input_channels`` (typically ``3``), and ``H``, ``W``
                match the configured dataset mode:
                224×224 for ``"imagenet"``, 32×32 for ``"cifar10"``.

        Returns:
            Raw logit tensor of shape ``(B, num_classes)``.  Values are
            unbounded real numbers; negative values are expected and correct.
        """
        # Stem: reduces spatial resolution before the residual groups.
        x = self.stem(x)

        # Residual groups: progressively deepen the representation while
        # halving spatial dimensions at the start of groups 2, 3, and 4.
        x = self.layer1(x)   # [B, 64,  56, 56] imagenet / [B, 64,  32, 32] cifar10
        x = self.layer2(x)   # [B, 128, 28, 28]           / [B, 128, 16, 16]
        x = self.layer3(x)   # [B, 256, 14, 14]           / [B, 256,  8,  8]
        x = self.layer4(x)   # [B, 512,  7,  7]           / [B, 512,  4,  4]

        # Global Average Pooling: collapse H×W to 1×1 per channel.
        x = self.avgpool(x)  # [B, 512, 1, 1]

        # Flatten: remove the trailing 1×1 spatial dimensions.
        # start_dim=1 preserves the batch dimension at index 0.
        x = torch.flatten(x, start_dim=1)   # [B, 512]

        # Optional dropout between the representation and the classifier.
        x = self.dropout(x)  # [B, 512]

        # Classification head: linear projection to class logit space.
        x = self.fc(x)       # [B, num_classes]

        return x

    # -----------------------------------------------------------------------
    # Public analysis interface
    # -----------------------------------------------------------------------

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the 512-dimensional representation before the classifier.

        Runs the forward pass through all layers up to and including the
        global average pooling and flatten steps, then stops before the FC
        layer.  The returned vector is the representation used by:

        - **Grad-CAM**: feature maps at ``layer4`` are intermediate outputs on
          this path; see ``get_layer_by_name`` for hook registration.
        - **t-SNE visualisation**: the 512-d vector is passed to
          ``sklearn.manifold.TSNE`` for 2-D projection.
        - **Nearest-neighbour analysis**: cosine or Euclidean distances in
          this space measure representational similarity.

        The model should be in ``eval`` mode and called within a
        ``torch.no_grad()`` context when used for analysis::

            model.eval()
            with torch.no_grad():
                features = model.get_features(x)   # [B, 512]

        Args:
            x: Input image batch of shape ``(B, C, H, W)``.

        Returns:
            Feature tensor of shape ``(B, 512)`` — the pre-classifier,
            post-GAP representation.
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        # Dropout is included for path consistency with forward().
        # In eval mode (the expected context for get_features), nn.Dropout
        # is disabled automatically regardless of dropout_rate.
        x = self.dropout(x)
        return x

    def count_parameters(self) -> dict[str, int]:
        """Count learnable parameters grouped by architectural component.

        Only ``nn.Parameter`` tensors (returned by ``model.parameters()``)
        are counted.  BatchNorm running statistics (``running_mean``,
        ``running_var``) are buffers, not parameters, and are excluded.

        The **correctness gate** for a ``num_classes=1000`` model::

            assert model.count_parameters()["total"] == 21_797_672

        Any deviation from this value indicates a structural bug in the
        layer configuration or block counts.

        Returns:
            A dictionary with the following keys:

            - ``"total"``: total parameter count across the entire model.
            - ``"trainable"``: parameters with ``requires_grad=True``.
            - ``"stem"``: parameters in the stem sub-network.
            - ``"layer1"`` through ``"layer4"``: parameters in each residual
              group, including all blocks and projection shortcuts.
            - ``"fc"``: parameters in the classification head (weight + bias).
        """

        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "stem": _count(self.stem),
            "layer1": _count(self.layer1),
            "layer2": _count(self.layer2),
            "layer3": _count(self.layer3),
            "layer4": _count(self.layer4),
            "fc": _count(self.fc),
        }

    def get_layer_by_name(self, name: str) -> nn.Module:
        """Retrieve a sub-module by dot-separated name string.

        Supports the naming convention used by ``torch.nn`` module trees,
        including integer indices for ``nn.Sequential`` children::

            model.get_layer_by_name("layer4")         # nn.Sequential
            model.get_layer_by_name("layer4.2")       # last BasicBlock
            model.get_layer_by_name("layer4.2.conv2") # specific Conv2d
            model.get_layer_by_name("stem")            # stem Sequential
            model.get_layer_by_name("fc")              # Linear head

        This method is the recommended way for ``GradCAM`` and other hook-
        based tools to obtain a reference to a target layer without hard-coding
        attribute traversal logic in the caller.

        Args:
            name: Dot-separated path to the target module.  Top-level names
                must be direct attributes of ``ResNet34`` (e.g. ``"stem"``,
                ``"layer1"``–``"layer4"``, ``"avgpool"``, ``"dropout"``,
                ``"fc"``).  Numeric path segments index into ``nn.Sequential``
                containers.

        Returns:
            The ``nn.Module`` at the specified path.

        Raises:
            AttributeError: If any segment of ``name`` does not correspond to
                a valid attribute or index, with a message listing the valid
                top-level names.
        """
        _valid_roots = (
            "stem", "layer1", "layer2", "layer3", "layer4",
            "avgpool", "dropout", "fc",
        )

        parts = name.split(".")
        try:
            module: nn.Module = getattr(self, parts[0])
            for part in parts[1:]:
                if part.isdigit():
                    module = module[int(part)]  # type: ignore[index]
                else:
                    module = getattr(module, part)
        except (AttributeError, IndexError, TypeError) as exc:
            raise AttributeError(
                f"No sub-module named '{name}' found in ResNet34. "
                f"Valid top-level names: {_valid_roots}."
            ) from exc

        return module

    # -----------------------------------------------------------------------
    # Dunder helpers
    # -----------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Return a compact configuration summary for ``print(model)`` output.

        Returns:
            A formatted string displaying the key constructor arguments.
        """
        return (
            f"num_classes={self.num_classes}, "
            f"dataset='{self.dataset}', "
            f"skip={self.use_skip_connection}, "
            f"bn={self.use_batch_norm}, "
            f"dropout={self.dropout_rate}, "
            f"zero_init_residual={self.zero_init_residual}"
        )
