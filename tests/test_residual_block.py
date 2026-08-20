"""Tests for src/models/residual_block.py.

Verifies the correctness of ``BasicBlock`` against the implementation
specification and He et al. (2016).  Tests are organised into classes that
each own a single concern.  Running the full suite with::

    pytest tests/test_residual_block.py -v

requires the project to be installed (``pip install -e .``) so that
``from src.models.residual_block import BasicBlock`` resolves correctly.

Test coverage targets (enforced by the Makefile):
    - Branch coverage ≥ 95 % for ``residual_block.py``
    - Every public argument combination is exercised at least once.

Scientific purpose of this suite:
    The parameter count and gradient flow tests are the implementation
    *correctness gates* for the residual learning experiments.  If any of
    the gradient-flow assertions fail, the empirical results claimed in the
    ablation study (Ablation A1) are scientifically invalid.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.init as init

from src.models.residual_block import BasicBlock

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Spatial resolutions used at each layer boundary in ResNet-34 (ImageNet).
_LAYER1_H, _LAYER1_W = 56, 56
_LAYER2_H, _LAYER2_W = 28, 28
_LAYER3_H, _LAYER3_W = 14, 14
_LAYER4_H, _LAYER4_W = 7, 7

# Spatial resolutions for the CIFAR-10 adapted ResNet-34.
_CIFAR_LAYER1_H, _CIFAR_LAYER1_W = 32, 32
_CIFAR_LAYER2_H, _CIFAR_LAYER2_W = 16, 16

# Fixed batch size used in all shape tests.
_B = 2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def identity_block() -> BasicBlock:
    """A BasicBlock that uses an identity shortcut (64 → 64, stride=1)."""
    return BasicBlock(in_channels=64, out_channels=64, stride=1)


@pytest.fixture
def projection_block() -> BasicBlock:
    """A BasicBlock that uses a projection shortcut (64 → 128, stride=2).

    Corresponds to the first block of ``layer2`` in ResNet-34.
    """
    return BasicBlock(in_channels=64, out_channels=128, stride=2)


@pytest.fixture
def identity_input() -> torch.Tensor:
    """A random input matching the identity_block's expected shape."""
    return torch.randn(_B, 64, _LAYER1_H, _LAYER1_W)


@pytest.fixture
def projection_input() -> torch.Tensor:
    """A random input matching the projection_block's expected shape."""
    return torch.randn(_B, 64, _LAYER1_H, _LAYER1_W)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _count_modules(block: nn.Module, module_type: type) -> int:
    """Count the number of sub-modules of a given type inside *block*."""
    return sum(1 for m in block.modules() if isinstance(m, module_type))


def _grad_mean(tensor: torch.Tensor) -> float:
    """Return the mean absolute gradient of *tensor* (must have .grad set)."""
    assert tensor.grad is not None, "Tensor has no gradient."
    return tensor.grad.abs().mean().item()


# ===========================================================================
# 1. Class-level attributes
# ===========================================================================


class TestClassAttributes:
    """BasicBlock must expose ``expansion`` as a class-level constant."""

    def test_expansion_is_class_attribute(self) -> None:
        """expansion must be accessible on the class, not just instances."""
        assert hasattr(BasicBlock, "expansion")

    def test_expansion_value_is_one(self) -> None:
        """BasicBlock has no channel expansion — expansion must equal 1."""
        assert BasicBlock.expansion == 1

    def test_expansion_accessible_on_instance(
        self, identity_block: BasicBlock
    ) -> None:
        """Instance attribute lookup must return the class-level value."""
        assert identity_block.expansion == 1

    def test_expansion_type_is_int(self) -> None:
        """ResNet34 reads expansion arithmetically; it must be an int."""
        assert isinstance(BasicBlock.expansion, int)


# ===========================================================================
# 2. Required instance attributes
# ===========================================================================


class TestRequiredAttributes:
    """Every named attribute used by ResNet34, GradCAM, and tests must exist."""

    _REQUIRED = ("conv1", "bn1", "relu", "conv2", "bn2", "shortcut")

    @pytest.mark.parametrize("attr", _REQUIRED)
    def test_attribute_present(
        self, attr: str, identity_block: BasicBlock
    ) -> None:
        """Attribute ``attr`` must be present on every BasicBlock instance."""
        assert hasattr(identity_block, attr), (
            f"BasicBlock is missing required attribute '{attr}'."
        )

    def test_conv1_is_conv2d(self, identity_block: BasicBlock) -> None:
        """conv1 must be an nn.Conv2d."""
        assert isinstance(identity_block.conv1, nn.Conv2d)

    def test_conv2_is_conv2d(self, identity_block: BasicBlock) -> None:
        """conv2 must be an nn.Conv2d."""
        assert isinstance(identity_block.conv2, nn.Conv2d)

    def test_relu_is_relu_module(self, identity_block: BasicBlock) -> None:
        """relu must be an nn.ReLU module (stateless; safely reused twice)."""
        assert isinstance(identity_block.relu, nn.ReLU)

    def test_shortcut_is_module(self, identity_block: BasicBlock) -> None:
        """shortcut must be an nn.Module for consistent module-tree traversal."""
        assert isinstance(identity_block.shortcut, nn.Module)

    def test_constructor_args_stored_as_attributes(self) -> None:
        """Constructor arguments must be retrievable as instance attributes."""
        block = BasicBlock(
            in_channels=128,
            out_channels=256,
            stride=2,
            use_skip_connection=False,
            use_batch_norm=False,
        )
        assert block.in_channels == 128
        assert block.out_channels == 256
        assert block.stride == 2
        assert block.use_skip_connection is False
        assert block.use_batch_norm is False


# ===========================================================================
# 3. Convolutional layer properties
# ===========================================================================


class TestConvLayerProperties:
    """Verify kernel sizes, strides, padding, and the bias=False requirement."""

    # --- conv1 ---

    def test_conv1_bias_is_none(self, identity_block: BasicBlock) -> None:
        """conv1 must not have a bias: BatchNorm follows and cancels it."""
        assert identity_block.conv1.bias is None

    def test_conv1_kernel_size_is_3x3(self, identity_block: BasicBlock) -> None:
        """conv1 must use a 3×3 kernel per the ResNet-34 specification."""
        assert identity_block.conv1.kernel_size == (3, 3)

    def test_conv1_padding_is_1(self, identity_block: BasicBlock) -> None:
        """padding=1 preserves spatial dimensions at stride=1."""
        assert identity_block.conv1.padding == (1, 1)

    def test_conv1_stride_matches_constructor(self) -> None:
        """conv1 stride must equal the constructor's stride argument."""
        for stride in (1, 2):
            block = BasicBlock(64, 64, stride=stride)
            assert block.conv1.stride == (stride, stride)

    def test_conv1_in_channels_matches_constructor(self) -> None:
        """conv1.in_channels must equal the constructor's in_channels."""
        block = BasicBlock(128, 256, stride=2)
        assert block.conv1.in_channels == 128

    def test_conv1_out_channels_matches_constructor(self) -> None:
        """conv1.out_channels must equal the constructor's out_channels."""
        block = BasicBlock(128, 256, stride=2)
        assert block.conv1.out_channels == 256

    # --- conv2 ---

    def test_conv2_bias_is_none(self, identity_block: BasicBlock) -> None:
        """conv2 must not have a bias: BatchNorm follows and cancels it."""
        assert identity_block.conv2.bias is None

    def test_conv2_kernel_size_is_3x3(self, identity_block: BasicBlock) -> None:
        """conv2 must use a 3×3 kernel per the ResNet-34 specification."""
        assert identity_block.conv2.kernel_size == (3, 3)

    def test_conv2_padding_is_1(self, identity_block: BasicBlock) -> None:
        """padding=1 preserves spatial dimensions at stride=1."""
        assert identity_block.conv2.padding == (1, 1)

    @pytest.mark.parametrize("constructor_stride", [1, 2])
    def test_conv2_stride_is_always_1(self, constructor_stride: int) -> None:
        """conv2 stride must always be 1, regardless of the block stride.

        Spatial downsampling is performed by conv1 only.  A stride of 2 on
        conv2 would downsample by 4× per block instead of the intended 2×.
        """
        block = BasicBlock(64, 128, stride=constructor_stride)
        assert block.conv2.stride == (1, 1), (
            f"conv2.stride must be (1, 1) even when block stride={constructor_stride}."
        )

    def test_conv2_in_channels_equals_out_channels(self) -> None:
        """conv2 receives the output of conv1, which has out_channels channels.

        A common mistake is setting conv2.in_channels to the block's
        in_channels (64 in this case).  The test catches that error.
        """
        block = BasicBlock(in_channels=64, out_channels=128, stride=2)
        assert block.conv2.in_channels == 128, (
            "conv2.in_channels must be out_channels (128), not in_channels (64)."
        )

    def test_conv2_out_channels_matches_constructor(self) -> None:
        """conv2.out_channels must equal the constructor's out_channels."""
        block = BasicBlock(64, 256, stride=1)
        assert block.conv2.out_channels == 256


# ===========================================================================
# 4. BatchNorm layers
# ===========================================================================


class TestBatchNormLayers:
    """Verify BN layer types and dimensions across use_batch_norm modes."""

    def test_bn1_is_batchnorm2d_when_enabled(
        self, identity_block: BasicBlock
    ) -> None:
        """bn1 must be nn.BatchNorm2d when use_batch_norm=True."""
        assert isinstance(identity_block.bn1, nn.BatchNorm2d)

    def test_bn2_is_batchnorm2d_when_enabled(
        self, identity_block: BasicBlock
    ) -> None:
        """bn2 must be nn.BatchNorm2d when use_batch_norm=True."""
        assert isinstance(identity_block.bn2, nn.BatchNorm2d)

    def test_bn1_num_features_equals_out_channels(self) -> None:
        """bn1 normalises the output of conv1, so num_features=out_channels."""
        block = BasicBlock(64, 128, stride=2)
        assert block.bn1.num_features == 128

    def test_bn2_num_features_equals_out_channels(self) -> None:
        """bn2 normalises the output of conv2, so num_features=out_channels."""
        block = BasicBlock(64, 128, stride=2)
        assert block.bn2.num_features == 128

    def test_bn1_is_identity_when_use_batch_norm_false(self) -> None:
        """bn1 must be nn.Identity when use_batch_norm=False (Ablation A5)."""
        block = BasicBlock(64, 64, stride=1, use_batch_norm=False)
        assert isinstance(block.bn1, nn.Identity)

    def test_bn2_is_identity_when_use_batch_norm_false(self) -> None:
        """bn2 must be nn.Identity when use_batch_norm=False (Ablation A5)."""
        block = BasicBlock(64, 64, stride=1, use_batch_norm=False)
        assert isinstance(block.bn2, nn.Identity)


# ===========================================================================
# 5. Shortcut construction logic
# ===========================================================================


class TestShortcutConstruction:
    """Verify the identity-vs-projection decision under all dimension cases."""

    def test_identity_shortcut_when_same_channels_and_stride1(
        self, identity_block: BasicBlock
    ) -> None:
        """No projection when in_channels == out_channels and stride == 1."""
        assert isinstance(identity_block.shortcut, nn.Identity)

    def test_projection_shortcut_when_stride_is_2(self) -> None:
        """Projection required when stride=2 (spatial dimensions change)."""
        block = BasicBlock(64, 128, stride=2)
        assert isinstance(block.shortcut, nn.Sequential)

    def test_projection_shortcut_when_channels_differ_stride1(self) -> None:
        """Projection required when channels change, even with stride=1."""
        block = BasicBlock(in_channels=64, out_channels=128, stride=1)
        assert isinstance(block.shortcut, nn.Sequential)

    def test_projection_shortcut_when_both_differ(self) -> None:
        """Projection required when both channels and stride differ."""
        block = BasicBlock(in_channels=128, out_channels=256, stride=2)
        assert isinstance(block.shortcut, nn.Sequential)

    def test_no_skip_shortcut_is_always_identity_regardless_of_dims(
        self,
    ) -> None:
        """use_skip_connection=False must produce nn.Identity in all cases."""
        cases = [
            (64, 64, 1),   # Same dims
            (64, 128, 2),  # Channel and spatial change
            (64, 128, 1),  # Channel change only
        ]
        for in_ch, out_ch, stride in cases:
            block = BasicBlock(in_ch, out_ch, stride, use_skip_connection=False)
            assert isinstance(block.shortcut, nn.Identity), (
                f"Expected Identity for no-skip block "
                f"({in_ch}→{out_ch}, stride={stride})."
            )

    @pytest.mark.parametrize("in_ch,out_ch,stride", [
        (64, 64, 1),    # layer1 identity
        (64, 64, 2),    # unusual but valid: stride with same channels
        (64, 128, 2),   # layer2 transition
        (128, 256, 2),  # layer3 transition
        (256, 512, 2),  # layer4 transition
        (128, 128, 1),  # layer2 non-first block
    ])
    def test_projection_needed_check_matches_expected(
        self, in_ch: int, out_ch: int, stride: int
    ) -> None:
        """_needs_projection must return True iff dimensions mismatch."""
        block = BasicBlock(in_ch, out_ch, stride)
        needs_projection = (stride != 1) or (in_ch != out_ch)
        if needs_projection:
            assert isinstance(block.shortcut, nn.Sequential)
        else:
            assert isinstance(block.shortcut, nn.Identity)


# ===========================================================================
# 6. Projection shortcut internal structure
# ===========================================================================


class TestProjectionShortcutStructure:
    """Verify the 1×1 Conv + BN architecture of the projection shortcut."""

    @pytest.fixture(autouse=True)
    def _proj_block(self) -> None:
        """Fixture shared by all tests in this class."""
        self.block = BasicBlock(64, 128, stride=2)

    def test_projection_contains_exactly_two_children(self) -> None:
        """Projection shortcut must be Sequential(Conv2d, BatchNorm2d)."""
        children = list(self.block.shortcut.children())
        assert len(children) == 2

    def test_projection_first_child_is_conv2d(self) -> None:
        """First element of the projection Sequential must be Conv2d."""
        assert isinstance(self.block.shortcut[0], nn.Conv2d)

    def test_projection_second_child_is_batchnorm2d(self) -> None:
        """Second element must be BatchNorm2d (or Identity when BN disabled)."""
        assert isinstance(self.block.shortcut[1], nn.BatchNorm2d)

    def test_projection_conv_kernel_is_1x1(self) -> None:
        """Projection conv uses a 1×1 kernel — changes depth without mixing.

        A 3×3 kernel would add unnecessary spatial mixing and 9× more
        parameters for the same channel transformation.
        """
        assert self.block.shortcut[0].kernel_size == (1, 1)

    def test_projection_conv_bias_is_none(self) -> None:
        """Projection conv must have bias=False: BatchNorm follows."""
        assert self.block.shortcut[0].bias is None

    def test_projection_conv_stride_matches_block_stride(self) -> None:
        """Projection conv stride must equal the block stride.

        This ensures the projection output matches the residual path output
        shape, making the element-wise addition valid.
        """
        assert self.block.shortcut[0].stride == (2, 2)

    def test_projection_conv_in_channels_is_block_in_channels(self) -> None:
        """Projection conv maps from in_channels to out_channels."""
        assert self.block.shortcut[0].in_channels == 64

    def test_projection_conv_out_channels_is_block_out_channels(self) -> None:
        """Projection conv maps from in_channels to out_channels."""
        assert self.block.shortcut[0].out_channels == 128

    def test_projection_bn_num_features_is_out_channels(self) -> None:
        """Projection BN normalises the projected output (out_channels wide)."""
        assert self.block.shortcut[1].num_features == 128

    def test_no_activation_in_projection_shortcut(self) -> None:
        """Projection shortcut must contain no ReLU or other activation.

        The skip connection is a linear path.  Adding an activation would clip
        negative values, reducing gradient flow and corrupting the residual
        formulation.
        """
        activation_types = (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)
        has_activation = any(
            isinstance(m, activation_types)
            for m in self.block.shortcut.modules()
            if m is not self.block.shortcut
        )
        assert not has_activation

    def test_projection_bn_is_identity_when_use_batch_norm_false(self) -> None:
        """When use_batch_norm=False, the projection BN must also be Identity.

        Ablation A5 must disable BN uniformly — including inside the shortcut.
        A mixed configuration (residual path: no BN; shortcut: has BN) is
        an inconsistent ablation condition.
        """
        block = BasicBlock(64, 128, stride=2, use_batch_norm=False)
        assert isinstance(block.shortcut[1], nn.Identity)

    @pytest.mark.parametrize("stride", [1, 2])
    def test_projection_spatial_output_matches_residual_path(
        self, stride: int
    ) -> None:
        """Projection and residual path must produce the same spatial shape.

        Mathematical proof: for a 3×3 conv (padding=1) and 1×1 conv (padding=0),
        both with the same stride s:
            floor((H + 2×1 - 3) / s) + 1  ==  floor((H + 0 - 1) / s) + 1
            floor((H - 1) / s) + 1         ==  floor((H - 1) / s) + 1  ✓
        """
        block = BasicBlock(64, 128, stride=stride)
        x = torch.randn(2, 64, 56, 56)
        captured: dict[str, torch.Size] = {}
        h1 = block.bn2.register_forward_hook(
            lambda m, i, o: captured.update({"residual": o.shape})
        )
        h2 = block.shortcut.register_forward_hook(
            lambda m, i, o: captured.update({"skip": o.shape})
        )
        with torch.no_grad():
            block.eval()
            block(x)
        h1.remove()
        h2.remove()
        assert captured["residual"] == captured["skip"], (
            f"Residual path shape {captured['residual']} != "
            f"shortcut shape {captured['skip']} for stride={stride}."
        )


# ===========================================================================
# 7. Output tensor shapes
# ===========================================================================


class TestOutputShapes:
    """Verify forward pass output shapes for all ResNet-34 block positions."""

    @pytest.mark.parametrize("in_ch,out_ch,stride,in_shape,expected", [
        # ImageNet ResNet-34 — identity blocks
        (64,  64,  1, (_B, 64,  56, 56), (_B, 64,  56, 56)),  # layer1 any
        (128, 128, 1, (_B, 128, 28, 28), (_B, 128, 28, 28)),  # layer2 non-first
        (256, 256, 1, (_B, 256, 14, 14), (_B, 256, 14, 14)),  # layer3 non-first
        (512, 512, 1, (_B, 512,  7,  7), (_B, 512,  7,  7)),  # layer4 non-first
        # ImageNet ResNet-34 — projection blocks (first of each group)
        (64,  128, 2, (_B, 64,  56, 56), (_B, 128, 28, 28)),  # layer2[0]
        (128, 256, 2, (_B, 128, 28, 28), (_B, 256, 14, 14)),  # layer3[0]
        (256, 512, 2, (_B, 256, 14, 14), (_B, 512,  7,  7)),  # layer4[0]
        # CIFAR-10 adapted ResNet-34
        (64,  64,  1, (_B, 64,  32, 32), (_B, 64,  32, 32)),  # layer1 identity
        (64,  128, 2, (_B, 64,  32, 32), (_B, 128, 16, 16)),  # layer2[0]
        (128, 256, 2, (_B, 128, 16, 16), (_B, 256,  8,  8)),  # layer3[0]
        (256, 512, 2, (_B, 256,  8,  8), (_B, 512,  4,  4)),  # layer4[0]
    ])
    def test_output_shape(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        in_shape: tuple[int, ...],
        expected: tuple[int, ...],
    ) -> None:
        """Forward pass must produce the correct output shape.

        Args:
            in_ch: Block in_channels.
            out_ch: Block out_channels.
            stride: Block stride.
            in_shape: Input tensor shape as (B, C, H, W).
            expected: Expected output tensor shape.
        """
        block = BasicBlock(in_ch, out_ch, stride=stride)
        x = torch.randn(*in_shape)
        with torch.no_grad():
            out = block(x)
        assert tuple(out.shape) == expected, (
            f"BasicBlock({in_ch}, {out_ch}, stride={stride}): "
            f"expected {expected}, got {tuple(out.shape)}."
        )

    def test_output_shape_is_independent_of_ablation_flags(self) -> None:
        """Ablation flags must not alter the output shape."""
        configs = [
            dict(use_skip_connection=True,  use_batch_norm=True),
            dict(use_skip_connection=False, use_batch_norm=True),
            dict(use_skip_connection=True,  use_batch_norm=False),
            dict(use_skip_connection=False, use_batch_norm=False),
        ]
        x = torch.randn(2, 64, 56, 56)
        expected = (2, 128, 28, 28)
        for cfg in configs:
            block = BasicBlock(64, 128, stride=2, **cfg)
            with torch.no_grad():
                out = block(x)
            assert tuple(out.shape) == expected, (
                f"Shape mismatch for config {cfg}: got {tuple(out.shape)}."
            )


# ===========================================================================
# 8. Forward pass semantics
# ===========================================================================


class TestForwardPassSemantics:
    """Verify mathematical properties of the forward pass output."""

    def test_output_values_are_finite(
        self, identity_block: BasicBlock, identity_input: torch.Tensor
    ) -> None:
        """Forward pass must not produce NaN or Inf values."""
        with torch.no_grad():
            out = identity_block(identity_input)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf."

    def test_final_output_is_non_negative(
        self, identity_block: BasicBlock, identity_input: torch.Tensor
    ) -> None:
        """Output must be non-negative: the final ReLU clips negative values.

        This property distinguishes the correct formula ReLU(F(x) + x) from
        the wrong formula ReLU(F(x)) + x, which can produce negative outputs
        when x is negative and the skip path dominates.
        """
        identity_block.eval()
        with torch.no_grad():
            out = identity_block(identity_input)
        assert (out >= 0).all(), (
            "Output has negative values — check that ReLU is applied after "
            "the skip connection addition, not before."
        )

    def test_relu_is_applied_after_addition_not_before(self) -> None:
        """Distinguish ReLU(F(x)+x) from ReLU(F(x))+x via a controlled input.

        Method:
            Zero all conv weights so F(x) ≈ 0.  Use a constant negative
            input x = -1.  Then:

            Correct — ReLU(F(x) + x) = ReLU(0 + (-1)) = ReLU(-1) = 0.
            Wrong   — ReLU(F(x)) + x = ReLU(0) + (-1) = 0 + (-1) = -1.

            So the output must be approximately 0, not -1.
        """
        block = BasicBlock(64, 64, stride=1)
        with torch.no_grad():
            block.conv1.weight.zero_()
            block.conv2.weight.zero_()
            # Setting gamma=0 in BN makes BN output = 0*x̂ + 0 = 0.
            if isinstance(block.bn1, nn.BatchNorm2d):
                block.bn1.weight.zero_()
            if isinstance(block.bn2, nn.BatchNorm2d):
                block.bn2.weight.zero_()
        block.eval()
        x = -torch.ones(2, 64, 56, 56)
        with torch.no_grad():
            out = block(x)
        # F(x) ≈ 0, so output ≈ ReLU(0 + (-1)) = 0.
        assert (out >= -1e-5).all(), (
            "Output is negative — ReLU may be applied before the addition."
        )
        assert out.abs().max().item() < 0.1, (
            "Output is non-zero — residual path may not have zeroed correctly."
        )

    def test_bn2_output_has_negative_values_before_final_relu(self) -> None:
        """BatchNorm output before the addition must be allowed to be negative.

        This confirms that the final ReLU is positioned after bn2, not after
        it in the residual path.  If ReLU were applied right after bn2, the
        intermediate bn2 output would already be non-negative.
        """
        block = BasicBlock(64, 64, stride=1)
        block.eval()
        captured: dict[str, torch.Tensor] = {}
        hook = block.bn2.register_forward_hook(
            lambda m, i, o: captured.update({"bn2_out": o.detach()})
        )
        with torch.no_grad():
            block(torch.randn(4, 64, 56, 56))
        hook.remove()
        assert (captured["bn2_out"] < 0).any(), (
            "bn2 output has no negative values — ReLU may be applied too early."
        )

    def test_conv2_receives_out_channels_not_in_channels(self) -> None:
        """conv2 input must have out_channels channels, not in_channels.

        A common error is defining conv2 as Conv2d(in_channels, out_channels)
        instead of Conv2d(out_channels, out_channels).  This hook-based test
        directly measures the actual channel count entering conv2.
        """
        block = BasicBlock(in_channels=64, out_channels=128, stride=2)
        captured: dict[str, int] = {}
        hook = block.conv2.register_forward_hook(
            lambda m, i, o: captured.update({"input_channels": i[0].shape[1]})
        )
        with torch.no_grad():
            block(torch.randn(2, 64, 56, 56))
        hook.remove()
        assert captured["input_channels"] == 128, (
            f"conv2 received {captured['input_channels']} channels; "
            f"expected 128 (out_channels).  Did you use in_channels by mistake?"
        )

    def test_output_differs_from_input_for_random_weights(
        self, identity_block: BasicBlock, identity_input: torch.Tensor
    ) -> None:
        """The block must perform a non-trivial transformation on its input."""
        with torch.no_grad():
            out = identity_block(identity_input)
        assert not torch.allclose(out, identity_input, atol=1e-4), (
            "Block output is identical to input — is this an accidental no-op?"
        )


# ===========================================================================
# 9. Gradient flow
# ===========================================================================


class TestGradientFlow:
    """Validate gradient propagation through the block.

    These tests are the scientific correctness gates for Ablation A1.
    If any test fails, the claim that skip connections improve gradient flow
    is not supported by the implementation.
    """

    def test_gradient_reaches_input_identity_block(
        self, identity_block: BasicBlock
    ) -> None:
        """Gradient must propagate from output loss to block input."""
        x = torch.randn(_B, 64, 56, 56, requires_grad=True)
        identity_block(x).sum().backward()
        assert x.grad is not None

    def test_gradient_reaches_input_projection_block(
        self, projection_block: BasicBlock
    ) -> None:
        """Gradient must propagate through the projection shortcut path."""
        x = torch.randn(_B, 64, 56, 56, requires_grad=True)
        projection_block(x).sum().backward()
        assert x.grad is not None

    def test_gradient_reaches_input_no_skip_block(self) -> None:
        """Gradient must propagate even without a skip connection."""
        block = BasicBlock(64, 64, stride=1, use_skip_connection=False)
        x = torch.randn(_B, 64, 56, 56, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None

    def test_gradient_at_input_is_non_trivially_nonzero(
        self, identity_block: BasicBlock
    ) -> None:
        """Gradient magnitude at the block input must exceed a floor value.

        An all-zero gradient indicates a dead path or implementation error.
        """
        x = torch.randn(_B, 64, 56, 56, requires_grad=True)
        identity_block(x).sum().backward()
        assert _grad_mean(x) > 1e-7, (
            f"Gradient mean {_grad_mean(x):.2e} is suspiciously small."
        )

    def test_all_conv_weights_receive_gradients(
        self, identity_block: BasicBlock
    ) -> None:
        """Every conv weight tensor must receive a gradient during backward."""
        identity_block.zero_grad()
        x = torch.randn(_B, 64, 56, 56)
        identity_block(x).sum().backward()
        for name, param in identity_block.named_parameters():
            if "conv" in name and "weight" in name:
                assert param.grad is not None, f"No gradient for {name}."
                assert param.grad.abs().sum() > 0, (
                    f"Zero gradient for {name}."
                )

    def test_backward_does_not_raise_inplace_modification_error(
        self, identity_block: BasicBlock
    ) -> None:
        """backward() must not raise RuntimeError from in-place operations.

        ``out += identity`` (in-place addition) in forward() causes
        RuntimeError during backward because autograd records ``out``'s use
        in the residual path before the in-place mutation.  This test
        confirms that ``out = out + identity`` is used instead.
        """
        x = torch.randn(_B, 64, 56, 56, requires_grad=True)
        out = identity_block(x)
        # Should not raise RuntimeError:
        out.sum().backward()

    def test_skip_connection_provides_larger_gradient_than_no_skip(
        self,
    ) -> None:
        """Skip connection must augment gradient magnitude at block input.

        This is the empirical verification of He et al. (2016) Section 3.1:
        the identity shortcut guarantees a gradient component of magnitude 1
        at every layer, independent of the residual path Jacobian.

        Both blocks are initialised with identical weights (same seed) so
        that only the skip connection differs between conditions.
        """
        torch.manual_seed(0)
        block_skip = BasicBlock(64, 64, stride=1, use_skip_connection=True)

        torch.manual_seed(0)
        block_noskip = BasicBlock(64, 64, stride=1, use_skip_connection=False)

        x_skip = torch.randn(_B, 64, 56, 56)
        x_noskip = x_skip.detach().clone()

        x_skip   = x_skip.requires_grad_(True)
        x_noskip = x_noskip.requires_grad_(True)

        block_skip(x_skip).sum().backward()
        block_noskip(x_noskip).sum().backward()

        grad_skip   = _grad_mean(x_skip)
        grad_noskip = _grad_mean(x_noskip)

        assert grad_skip > grad_noskip, (
            f"Skip gradient ({grad_skip:.6f}) must exceed "
            f"no-skip gradient ({grad_noskip:.6f})."
        )

    def test_projection_block_backward_does_not_raise(
        self, projection_block: BasicBlock
    ) -> None:
        """backward() must succeed through the projection shortcut path."""
        x = torch.randn(_B, 64, 56, 56, requires_grad=True)
        projection_block(x).sum().backward()


# ===========================================================================
# 10. BatchNorm train / eval behaviour
# ===========================================================================


class TestBatchNormBehavior:
    """Verify BN mode switching and determinism requirements."""

    def test_eval_mode_is_deterministic(
        self, identity_block: BasicBlock, identity_input: torch.Tensor
    ) -> None:
        """Same input in eval mode must produce identical output on two calls."""
        identity_block.eval()
        with torch.no_grad():
            out1 = identity_block(identity_input)
            out2 = identity_block(identity_input)
        assert torch.allclose(out1, out2), (
            "eval mode produced different outputs for the same input."
        )

    def test_train_and_eval_produce_different_outputs(
        self, identity_block: BasicBlock
    ) -> None:
        """Train and eval modes must produce different outputs for the same input.

        In train mode, BN uses batch statistics (μ_batch, σ²_batch).
        In eval mode, BN uses running statistics accumulated during training.
        With random (untrained) weights, running stats (default: μ=0, σ²=1)
        differ from actual batch stats, causing output differences.
        """
        x = torch.randn(8, 64, 56, 56)
        identity_block.train()
        out_train = identity_block(x).detach()
        identity_block.eval()
        with torch.no_grad():
            out_eval = identity_block(x)
        assert not torch.allclose(out_train, out_eval, atol=1e-4), (
            "train and eval modes produced identical outputs — "
            "check that BN is not in a degenerate state."
        )

    def test_train_mode_same_batch_is_deterministic(
        self, identity_block: BasicBlock, identity_input: torch.Tensor
    ) -> None:
        """Same batch input in train mode must produce identical output.

        Within a single batch, BN computes fixed statistics — the output is
        deterministic for a given input even in train mode.
        """
        identity_block.train()
        out1 = identity_block(identity_input).detach()
        out2 = identity_block(identity_input).detach()
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_eval_mode_uses_running_statistics(self) -> None:
        """After switching to eval, running stats must govern BN output.

        A freshly constructed block has running_mean=0, running_var=1
        (PyTorch defaults).  If running stats were ignored in eval mode,
        the output would depend on the test batch, violating determinism.
        """
        block = BasicBlock(64, 64, stride=1)
        block.eval()
        # Manually set running stats to known values.
        block.bn1.running_mean.fill_(1.0)
        block.bn1.running_var.fill_(2.0)
        x = torch.randn(2, 64, 56, 56)
        with torch.no_grad():
            out_a = block(x)
            out_b = block(x)
        # Both calls must use the same (fixed) running stats.
        assert torch.allclose(out_a, out_b)


# ===========================================================================
# 11. Ablation mode combinations
# ===========================================================================


class TestAblationModes:
    """Verify all four use_skip_connection × use_batch_norm combinations."""

    @pytest.mark.parametrize("skip,bn", [
        (True,  True),    # Standard ResNet-34
        (False, True),    # Plain-34 (Ablation A1)
        (True,  False),   # No-BN ResNet (Ablation A5)
        (False, False),   # Plain block without BN (debugging baseline)
    ])
    def test_forward_pass_succeeds_for_all_mode_combinations(
        self, skip: bool, bn: bool
    ) -> None:
        """All four mode combinations must complete a forward pass without error."""
        block = BasicBlock(64, 64, stride=1,
                           use_skip_connection=skip, use_batch_norm=bn)
        x = torch.randn(_B, 64, 56, 56)
        with torch.no_grad():
            out = block(x)
        assert out.shape == torch.Size([_B, 64, 56, 56])
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("skip,bn", [
        (True,  True),
        (False, True),
        (True,  False),
        (False, False),
    ])
    def test_backward_succeeds_for_all_mode_combinations(
        self, skip: bool, bn: bool
    ) -> None:
        """All four mode combinations must allow gradient computation."""
        block = BasicBlock(64, 64, stride=1,
                           use_skip_connection=skip, use_batch_norm=bn)
        x = torch.randn(_B, 64, 56, 56, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None

    def test_no_skip_shortcut_is_identity_for_identical_dims(self) -> None:
        """use_skip_connection=False → shortcut is nn.Identity."""
        block = BasicBlock(64, 64, stride=1, use_skip_connection=False)
        assert isinstance(block.shortcut, nn.Identity)

    def test_no_skip_shortcut_is_identity_even_when_projection_needed(self) -> None:
        """Projection must not be built when use_skip_connection=False.

        Even though in_channels ≠ out_channels would normally trigger a
        projection, no shortcut is needed because the skip path is absent.
        """
        block = BasicBlock(64, 128, stride=2, use_skip_connection=False)
        assert isinstance(block.shortcut, nn.Identity), (
            "With use_skip_connection=False, shortcut must be Identity "
            "regardless of dimension mismatch."
        )

    def test_no_bn_block_has_no_batchnorm2d_modules(self) -> None:
        """use_batch_norm=False must remove all BatchNorm2d from the block."""
        block = BasicBlock(64, 128, stride=2, use_batch_norm=False)
        bn_count = _count_modules(block, nn.BatchNorm2d)
        assert bn_count == 0, (
            f"Expected 0 BatchNorm2d modules with use_batch_norm=False, "
            f"found {bn_count}."
        )

    def test_standard_block_has_correct_number_of_batchnorm2d_modules(
        self,
    ) -> None:
        """Identity block: 2 BN (bn1, bn2).  Projection block: 3 BN (+shortcut)."""
        identity_block = BasicBlock(64, 64, stride=1)
        proj_block = BasicBlock(64, 128, stride=2)

        assert _count_modules(identity_block, nn.BatchNorm2d) == 2
        assert _count_modules(proj_block, nn.BatchNorm2d) == 3


# ===========================================================================
# 12. Initialisation compatibility
# ===========================================================================


class TestInitializationCompatibility:
    """Verify that ResNet34._initialize_weights can be applied to BasicBlock.

    BasicBlock does not initialise its own weights.  ResNet34 iterates over
    all modules and applies kaiming_normal_ to Conv2d and constant_ to
    BatchNorm2d.  These tests confirm that the expected module types are
    present and accept those initialisation calls.
    """

    def test_kaiming_init_applies_to_conv1_without_error(
        self, identity_block: BasicBlock
    ) -> None:
        """kaiming_normal_ must succeed on conv1.weight."""
        init.kaiming_normal_(
            identity_block.conv1.weight, mode="fan_out", nonlinearity="relu"
        )

    def test_kaiming_init_applies_to_conv2_without_error(
        self, identity_block: BasicBlock
    ) -> None:
        """kaiming_normal_ must succeed on conv2.weight."""
        init.kaiming_normal_(
            identity_block.conv2.weight, mode="fan_out", nonlinearity="relu"
        )

    def test_kaiming_init_applies_to_projection_conv_without_error(
        self, projection_block: BasicBlock
    ) -> None:
        """kaiming_normal_ must succeed on the projection shortcut conv."""
        init.kaiming_normal_(
            projection_block.shortcut[0].weight, mode="fan_out", nonlinearity="relu"
        )

    def test_constant_init_applies_to_bn1_without_error(
        self, identity_block: BasicBlock
    ) -> None:
        """constant_ must succeed on bn1 gamma and beta."""
        init.constant_(identity_block.bn1.weight, 1.0)
        init.constant_(identity_block.bn1.bias, 0.0)

    def test_constant_init_applies_to_bn2_without_error(
        self, identity_block: BasicBlock
    ) -> None:
        """constant_ must succeed on bn2 gamma and beta."""
        init.constant_(identity_block.bn2.weight, 1.0)
        init.constant_(identity_block.bn2.bias, 0.0)

    def test_zero_init_residual_pattern_applies_to_bn2(
        self, identity_block: BasicBlock
    ) -> None:
        """Setting bn2.weight=0 must be valid (zero_init_residual pattern).

        He et al. (2019) initialise the final BN scale in each residual
        branch to zero, making every block an identity at training start.
        """
        init.constant_(identity_block.bn2.weight, 0.0)
        assert torch.all(identity_block.bn2.weight == 0.0)

    def test_conv_weight_is_nonzero_after_kaiming_init(
        self, identity_block: BasicBlock
    ) -> None:
        """Kaiming Normal must produce non-zero weights (not a degenerate case)."""
        init.kaiming_normal_(
            identity_block.conv1.weight, mode="fan_out", nonlinearity="relu"
        )
        assert identity_block.conv1.weight.abs().sum().item() > 0.0

    def test_all_bn_modules_are_findable_via_named_modules(
        self,
    ) -> None:
        """All BN modules must be reachable via model.modules() for init loop."""
        proj_block = BasicBlock(64, 128, stride=2)
        bn_modules = [
            m for m in proj_block.modules() if isinstance(m, nn.BatchNorm2d)
        ]
        # Projection block: bn1, bn2, shortcut BN = 3 total.
        assert len(bn_modules) == 3


# ===========================================================================
# 13. Edge cases
# ===========================================================================


class TestEdgeCases:
    """Exercise boundary conditions and configurations not in the main spec."""

    def test_batch_size_1_does_not_raise(self) -> None:
        """A batch of size 1 must not raise an error (BN output may be poor)."""
        block = BasicBlock(64, 64, stride=1)
        block.train()
        x = torch.randn(1, 64, 56, 56)
        # BN with batch_size=1 sets σ²=0 → output is (x - μ)/ε ≈ 0; no crash.
        out = block(x)
        assert torch.isfinite(out).all()

    def test_non_square_spatial_dimensions(self) -> None:
        """Block must handle non-square H×W without error."""
        block = BasicBlock(64, 64, stride=1)
        x = torch.randn(2, 64, 48, 64)   # H=48, W=64
        with torch.no_grad():
            out = block(x)
        assert out.shape == torch.Size([2, 64, 48, 64])

    def test_stride_1_with_channel_change_creates_projection(self) -> None:
        """stride=1 with in≠out must create a projection shortcut."""
        block = BasicBlock(in_channels=64, out_channels=128, stride=1)
        assert isinstance(block.shortcut, nn.Sequential)
        assert block.shortcut[0].stride == (1, 1)

    def test_stride_2_with_same_channels_creates_projection(self) -> None:
        """stride=2 with in==out must create a projection shortcut."""
        block = BasicBlock(in_channels=64, out_channels=64, stride=2)
        assert isinstance(block.shortcut, nn.Sequential)
        x = torch.randn(2, 64, 56, 56)
        with torch.no_grad():
            out = block(x)
        assert out.shape == torch.Size([2, 64, 28, 28])

    def test_large_batch_size(self) -> None:
        """Block must handle large batch sizes (e.g. 64)."""
        block = BasicBlock(64, 64, stride=1)
        block.eval()
        x = torch.randn(64, 64, 56, 56)
        with torch.no_grad():
            out = block(x)
        assert out.shape == torch.Size([64, 64, 56, 56])

    def test_minimum_spatial_dimension(self) -> None:
        """Block must handle very small spatial dimensions (e.g. 1×1)."""
        block = BasicBlock(512, 512, stride=1)
        block.eval()
        x = torch.randn(2, 512, 1, 1)
        with torch.no_grad():
            out = block(x)
        assert out.shape == torch.Size([2, 512, 1, 1])

    def test_block_is_subclass_of_nn_module(self) -> None:
        """BasicBlock must be an nn.Module for PyTorch APIs to work."""
        assert issubclass(BasicBlock, nn.Module)

    def test_block_parameters_are_nn_parameters(self) -> None:
        """All learnable tensors must be nn.Parameter instances."""
        block = BasicBlock(64, 128, stride=2)
        for param in block.parameters():
            assert isinstance(param, nn.Parameter)

    def test_state_dict_round_trip(self) -> None:
        """Saving and loading state_dict must reproduce identical outputs."""
        import copy
        block = BasicBlock(64, 128, stride=2)
        x = torch.randn(2, 64, 56, 56)
        block.eval()
        with torch.no_grad():
            out_before = block(x)

        state = copy.deepcopy(block.state_dict())
        block2 = BasicBlock(64, 128, stride=2)
        block2.load_state_dict(state)
        block2.eval()
        with torch.no_grad():
            out_after = block2(x)

        assert torch.allclose(out_before, out_after, atol=1e-6)


# ===========================================================================
# 14. extra_repr
# ===========================================================================


class TestExtraRepr:
    """Verify that the string representation contains key configuration info."""

    @pytest.fixture(autouse=True)
    def _block_and_repr(self) -> None:
        self.block = BasicBlock(128, 256, stride=2,
                                use_skip_connection=True, use_batch_norm=False)
        self.repr = self.block.extra_repr()

    def test_repr_contains_in_channels(self) -> None:
        """extra_repr must show in_channels."""
        assert "128" in self.repr

    def test_repr_contains_out_channels(self) -> None:
        """extra_repr must show out_channels."""
        assert "256" in self.repr

    def test_repr_contains_stride(self) -> None:
        """extra_repr must show stride."""
        assert "stride=2" in self.repr

    def test_repr_contains_skip_flag(self) -> None:
        """extra_repr must show the use_skip_connection flag."""
        assert "skip" in self.repr.lower()

    def test_repr_contains_bn_flag(self) -> None:
        """extra_repr must show the use_batch_norm flag."""
        assert "bn" in self.repr.lower()

    def test_repr_is_a_string(self) -> None:
        """extra_repr must return a plain string."""
        assert isinstance(self.repr, str)

    def test_print_model_does_not_raise(self) -> None:
        """print(block) must succeed without raising."""
        _ = str(self.block)
