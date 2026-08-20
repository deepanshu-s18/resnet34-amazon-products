"""Residual block implementation for ResNet architectures.

This module provides ``BasicBlock``, the fundamental computational unit of
ResNet-18 and ResNet-34.  The block implements the residual learning
formulation from He et al. (2016):

    "Deep Residual Learning for Image Recognition"
    He, K., Zhang, X., Ren, S., & Sun, J. (CVPR 2016).
    https://arxiv.org/abs/1512.03385

Mathematical formulation::

    output = ReLU(F(x, {Wᵢ}) + shortcut(x))

    where:
        F(x) = BN₂(Conv₂(ReLU(BN₁(Conv₁(x)))))   # residual path
        shortcut(x) = x                              # identity (same dims)
        shortcut(x) = BN_s(Conv_s(x))               # projection (dim change)

The ``use_skip_connection`` and ``use_batch_norm`` constructor flags enable
ablation studies without modifying the surrounding architecture:

- ``use_skip_connection=False`` produces a Plain block (Ablation A1).
- ``use_batch_norm=False`` removes all normalisation layers (Ablation A5).

Example::

    >>> import torch
    >>> from residual_block import BasicBlock
    >>>
    >>> # Identity block (same spatial size and channel count)
    >>> block = BasicBlock(in_channels=64, out_channels=64, stride=1)
    >>> x = torch.randn(2, 64, 56, 56)
    >>> block(x).shape
    torch.Size([2, 64, 56, 56])
    >>>
    >>> # Projection block (halve spatial dims, double channels)
    >>> block = BasicBlock(in_channels=64, out_channels=128, stride=2)
    >>> x = torch.randn(2, 64, 56, 56)
    >>> block(x).shape
    torch.Size([2, 128, 28, 28])
"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn as nn

__all__ = ["BasicBlock"]


class BasicBlock(nn.Module):
    """A residual building block for ResNet-18 and ResNet-34.

    Each block stacks two 3×3 convolutional layers separated by Batch
    Normalisation and ReLU, then adds the original input via a skip
    connection before applying a final ReLU.  When spatial resolution or
    channel depth changes (``stride != 1`` or ``in_channels != out_channels``),
    the skip path uses a learned 1×1 projection convolution to match
    dimensions before the element-wise addition.

    The ``expansion`` class attribute is always ``1`` for ``BasicBlock``,
    meaning the output channel count equals ``out_channels`` with no internal
    expansion.  ``ResNet34`` reads this attribute when computing the input
    channel count for successive layer groups.

    Attributes:
        expansion: Channel expansion factor.  ``1`` for BasicBlock;
            ``4`` for BottleneckBlock (not used in ResNet-34).
        in_channels: Number of channels in the block input.
        out_channels: Number of channels in the block output.
        stride: Stride applied to the first convolution.
        use_skip_connection: Whether the residual addition is active.
        use_batch_norm: Whether Batch Normalisation layers are active.
        conv1: First 3×3 convolutional layer.
        bn1: Batch normalisation after ``conv1`` (or ``nn.Identity``).
        relu: Shared ReLU activation (stateless; safely reused twice).
        conv2: Second 3×3 convolutional layer (stride always 1).
        bn2: Batch normalisation after ``conv2`` (or ``nn.Identity``).
        shortcut: Skip connection — ``nn.Identity`` or a 1×1 projection.
    """

    expansion: ClassVar[int] = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_skip_connection: bool = True,
        use_batch_norm: bool = True,
    ) -> None:
        """Initialise a BasicBlock and build all sub-modules.

        Args:
            in_channels: Number of channels in the input feature map ``x``.
                Must be positive.  In ResNet-34, valid values are
                ``{64, 128, 256, 512}``.
            out_channels: Number of channels produced by both convolutional
                layers and returned by the block.  In ResNet-34, valid values
                are ``{64, 128, 256, 512}``.
            stride: Stride for ``conv1``.  ``conv2`` always uses ``stride=1``.
                Use ``stride=2`` for the first block of each layer group in
                ResNet-34 to halve the spatial dimensions.  Defaults to ``1``.
            use_skip_connection: If ``True`` (default), the block applies the
                residual addition ``out = F(x) + shortcut(x)`` before the
                final ReLU, implementing the He et al. residual formulation.
                If ``False``, the skip connection is suppressed entirely,
                producing a plain stacked-convolution block (Ablation A1).
            use_batch_norm: If ``True`` (default), ``nn.BatchNorm2d`` layers
                are placed after each convolution.  If ``False``, all
                BatchNorm layers — including those inside the projection
                shortcut — are replaced with ``nn.Identity`` (Ablation A5).

        Note:
            Weight initialisation is intentionally excluded from this class.
            ``ResNet34._initialize_weights`` iterates over all modules after
            construction and applies Kaiming Normal initialisation to every
            ``nn.Conv2d`` and constant initialisation to every
            ``nn.BatchNorm2d``.  This keeps initialisation logic centralised
            and prevents it from running redundantly during checkpoint loading.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_skip_connection = use_skip_connection
        self.use_batch_norm = use_batch_norm

        # ------------------------------------------------------------------
        # Residual path: Conv → BN → ReLU → Conv → BN
        # ------------------------------------------------------------------
        # bias=False on both convolutions: the BatchNorm layer that follows
        # each conv subtracts the per-channel mean, which would cancel any
        # additive bias.  Keeping bias=False avoids wasted parameters.
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,   # Spatial downsampling happens here only.
            padding=1,       # padding=1 preserves H,W at stride=1.
            bias=False,
        )
        self.bn1: nn.Module = (
            nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        )

        # A single ReLU module is created and called twice in forward().
        # nn.ReLU is stateless, so reuse is safe for sequential operations.
        # inplace=True reduces peak memory by modifying tensors in place;
        # both uses operate on freshly created tensors with no other
        # references in the computation graph.
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,   # Input to conv2 is the conv1 output channel count.
            out_channels,
            kernel_size=3,
            stride=1,       # stride is ALWAYS 1 for conv2.
            padding=1,
            bias=False,
        )
        self.bn2: nn.Module = (
            nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        )

        # ------------------------------------------------------------------
        # Skip connection path
        # ------------------------------------------------------------------
        self.shortcut: nn.Module = self._build_shortcut()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _needs_projection(self) -> bool:
        """Return whether a projection shortcut is required.

        A projection is needed whenever the skip connection cannot be a
        plain identity — i.e. when the spatial dimensions change
        (``stride != 1``) or the channel count changes
        (``in_channels != out_channels``).

        Returns:
            ``True`` if a 1×1 projection convolution must be used for the
            shortcut path; ``False`` if an identity shortcut suffices.
        """
        return self.stride != 1 or self.in_channels != self.out_channels

    def _build_shortcut(self) -> nn.Module:
        """Construct the appropriate skip connection module.

        Three cases determine the returned module:

        1. ``use_skip_connection=False``:  returns ``nn.Identity()``.
           The shortcut is never used in ``forward``, but the attribute is
           always present for consistent module-tree inspection and
           checkpoint serialisation.

        2. ``use_skip_connection=True`` and no projection needed:
           returns ``nn.Identity()``.  The input tensor is passed through
           unchanged.

        3. ``use_skip_connection=True`` and projection needed:
           returns ``nn.Sequential(Conv1×1, BN-or-Identity)``.  The 1×1
           convolution adjusts channel depth and spatial resolution to match
           the residual path output before the element-wise addition.

        Returns:
            An ``nn.Module`` that transforms the block input for the skip
            path.  Always produces the same shape as the residual path output
            when called on the block input.

        Note:
            No activation is applied in the projection shortcut.  The skip
            connection is intended as a linear path; adding a ReLU would clip
            negative values, reducing gradient flow and corrupting the
            residual learning formulation.
        """
        if not self.use_skip_connection or not self._needs_projection():
            return nn.Identity()

        projection_bn: nn.Module = (
            nn.BatchNorm2d(self.out_channels)
            if self.use_batch_norm
            else nn.Identity()
        )

        return nn.Sequential(
            # 1×1 conv: changes channel depth; stride matches conv1 to
            # produce the same spatial output size as the residual path.
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                stride=self.stride,
                bias=False,       # Followed by BN; bias would be cancelled.
            ),
            projection_bn,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the residual block output for an input feature map.

        Applies the following sequence::

            identity = x
            out      = conv1(x)   → bn1 → relu
                     = conv2(out) → bn2
            if use_skip_connection:
                identity = shortcut(identity)
                out      = out + identity        # element-wise addition
            out = relu(out)                      # final activation
            return out

        The final ReLU is applied **after** the skip-connection addition,
        not before.  This is the post-activation formulation defined in
        He et al. (2016).

        Args:
            x: Input feature map of shape ``(B, in_channels, H, W)``.

        Returns:
            Output feature map of shape ``(B, out_channels, H_out, W_out)``
            where ``H_out = H // stride`` and ``W_out = W // stride``.
        """
        # Save the input before any transformation.  This reference is used
        # as the skip connection value regardless of what happens to `out`.
        identity = x

        # ------------------------------------------------------------------
        # Residual path
        # ------------------------------------------------------------------
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)   # First use of the shared ReLU module.

        out = self.conv2(out)
        out = self.bn2(out)    # No ReLU here — addition comes first.

        # ------------------------------------------------------------------
        # Skip connection addition
        # ------------------------------------------------------------------
        if self.use_skip_connection:
            # self.shortcut is either nn.Identity (no-op) or a 1×1
            # projection Sequential.  Both produce the same shape as `out`.
            identity = self.shortcut(identity)

            # Use `out + identity`, NOT `out += identity`.
            # The in-place variant modifies `out` before autograd has
            # finished recording its use in the residual path, which raises
            # a RuntimeError during backward.
            out = out + identity

        # ------------------------------------------------------------------
        # Final activation
        # ------------------------------------------------------------------
        # Second use of the shared relu module.  Operates on the freshly
        # created tensor from `out + identity` (or directly on bn2 output
        # when use_skip_connection=False); both are safe for inplace ops.
        out = self.relu(out)

        return out

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Return a concise string summary for ``print(model)`` output.

        Returns:
            A formatted string showing the key configuration parameters.
        """
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"stride={self.stride}, "
            f"skip={self.use_skip_connection}, "
            f"bn={self.use_batch_norm}"
        )
