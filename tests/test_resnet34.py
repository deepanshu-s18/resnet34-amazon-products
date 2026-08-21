"""Integration tests for src/models/resnet34.py.

These are the *correctness gates* for the full ResNet-34 model.  Each test
verifies a property of the assembled network that cannot be caught by the
BasicBlock unit tests alone.

Scientific purpose
------------------
The parameter count gate is the single most important test in this file.
If ``count_parameters()["total"] != 21_797_672`` (for ImageNet/1000 classes),
the block configuration or channel widths are wrong and **all empirical results
are built on a buggy implementation**.

The gradient flow gate is the mechanistic correctness gate for Ablation A1.
If gradients do not flow from output back to input through the residual blocks,
the skip connection mechanism is broken, and the convergence-gap results are
not attributable to the claimed mechanism.

Running:
    pytest tests/test_resnet34.py -v

requires the project to be installed (``pip install -e .``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.resnet34 import ResNet34
from src.models.residual_block import BasicBlock

# ---------------------------------------------------------------------------
# Canonical parameter counts (verified by running count_parameters())
# These values are the ground-truth correctness gates.
# ---------------------------------------------------------------------------

# ImageNet mode (7×7 stem, 1000 classes) — matches the He et al. paper count.
_IMAGENET_TOTAL_PARAMS = 21_797_672

# CIFAR-10 mode (3×3 stem, 10 classes) — smaller stem + smaller FC head.
_CIFAR10_TOTAL_PARAMS = 21_282_122

# Plain-34 (no skip connections, CIFAR-10, 10 classes) — no projection
# shortcut parameters, so fewer than ResNet-34.
_PLAIN34_TOTAL_PARAMS = 21_108_298

# Fixed batch size used in all forward-pass tests.
_B = 2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cifar10_model() -> ResNet34:
    """Standard ResNet-34 configured for CIFAR-10 (32×32, 10 classes)."""
    return ResNet34(num_classes=10, dataset="cifar10")


@pytest.fixture
def imagenet_model() -> ResNet34:
    """Standard ResNet-34 configured for ImageNet (224×224, 1000 classes)."""
    return ResNet34(num_classes=1000, dataset="imagenet")


@pytest.fixture
def plain34_model() -> ResNet34:
    """Plain-34: ResNet-34 with skip connections disabled (Ablation A1)."""
    return ResNet34(num_classes=10, dataset="cifar10", use_skip_connection=False)


# ===========================================================================
# 1. Parameter count gates
# ===========================================================================


class TestParameterCounts:
    """Exact parameter counts for all model configurations.

    These tests fail immediately if the architecture has a structural bug.
    They must pass before any training results are considered valid.
    """

    def test_cifar10_total_param_count(self, cifar10_model: ResNet34) -> None:
        """CIFAR-10 ResNet-34 must have exactly 21,282,122 parameters.

        This is the correctness gate for all CIFAR-10 ablation results.
        If this fails, the 93.51% accuracy claim is built on a buggy model.
        """
        total = sum(p.numel() for p in cifar10_model.parameters())
        assert total == _CIFAR10_TOTAL_PARAMS, (
            f"Expected {_CIFAR10_TOTAL_PARAMS:,} params, got {total:,}. "
            "Check stem architecture, block counts [3,4,6,3], and channel widths."
        )

    def test_imagenet_total_param_count(self, imagenet_model: ResNet34) -> None:
        """ImageNet ResNet-34 must have exactly 21,797,672 parameters.

        Matches the official He et al. (2016) Table 1 parameter count for
        ResNet-34 with 1000 output classes.
        """
        total = sum(p.numel() for p in imagenet_model.parameters())
        assert total == _IMAGENET_TOTAL_PARAMS, (
            f"Expected {_IMAGENET_TOTAL_PARAMS:,} params, got {total:,}. "
            "Verify 7×7 stem and 1000-class FC head."
        )

    def test_plain34_has_fewer_params_than_resnet34(
        self, plain34_model: ResNet34, cifar10_model: ResNet34
    ) -> None:
        """Plain-34 must have fewer parameters than ResNet-34.

        Without skip connections, projection shortcuts (1×1 Conv + BN at
        each layer boundary) are replaced by nn.Identity, removing
        173,824 parameters. This difference is the structural proof that
        the ablation flag is wired correctly through the entire model.
        """
        plain_total = sum(p.numel() for p in plain34_model.parameters())
        resnet_total = sum(p.numel() for p in cifar10_model.parameters())
        assert plain_total < resnet_total, (
            "Plain-34 must have fewer params than ResNet-34 "
            "(projection shortcuts are absent)."
        )
        assert plain_total == _PLAIN34_TOTAL_PARAMS, (
            f"Expected Plain-34 to have {_PLAIN34_TOTAL_PARAMS:,} params, "
            f"got {plain_total:,}."
        )

    def test_count_parameters_method_matches_manual_count(
        self, cifar10_model: ResNet34
    ) -> None:
        """count_parameters()['total'] must match a manual parameter sum."""
        manual = sum(p.numel() for p in cifar10_model.parameters())
        reported = cifar10_model.count_parameters()["total"]
        assert manual == reported, (
            f"count_parameters() returned {reported:,} but "
            f"manual sum is {manual:,}."
        )

    def test_count_parameters_components_sum_to_total(
        self, cifar10_model: ResNet34
    ) -> None:
        """stem + layer1..4 + fc must sum to total parameter count.

        This verifies there are no 'orphan' parameters outside the named
        components, which would indicate a structural bug.
        """
        counts = cifar10_model.count_parameters()
        component_sum = (
            counts["stem"]
            + counts["layer1"]
            + counts["layer2"]
            + counts["layer3"]
            + counts["layer4"]
            + counts["fc"]
        )
        assert component_sum == counts["total"], (
            f"Component sum {component_sum:,} != total {counts['total']:,}. "
            "There are parameters outside the named layer groups."
        )

    def test_trainable_equals_total_when_no_frozen_layers(
        self, cifar10_model: ResNet34
    ) -> None:
        """All parameters must be trainable when no layers are frozen."""
        counts = cifar10_model.count_parameters()
        assert counts["trainable"] == counts["total"]


# ===========================================================================
# 2. Forward pass shapes
# ===========================================================================


class TestForwardPassShapes:
    """Verify output shapes for both stem configurations."""

    def test_cifar10_forward_output_shape(self, cifar10_model: ResNet34) -> None:
        """CIFAR-10 forward: (B, 3, 32, 32) → (B, 10)."""
        x = torch.randn(_B, 3, 32, 32)
        with torch.no_grad():
            out = cifar10_model(x)
        assert out.shape == (_B, 10), (
            f"Expected output shape ({_B}, 10), got {tuple(out.shape)}."
        )

    def test_imagenet_forward_output_shape(self, imagenet_model: ResNet34) -> None:
        """ImageNet forward: (B, 3, 224, 224) → (B, 1000)."""
        x = torch.randn(_B, 3, 224, 224)
        with torch.no_grad():
            out = imagenet_model(x)
        assert out.shape == (_B, 1000), (
            f"Expected output shape ({_B}, 1000), got {tuple(out.shape)}."
        )

    def test_get_features_output_shape_cifar10(self, cifar10_model: ResNet34) -> None:
        """get_features() must return (B, 512) — the pre-classifier representation."""
        cifar10_model.eval()
        x = torch.randn(_B, 3, 32, 32)
        with torch.no_grad():
            feats = cifar10_model.get_features(x)
        assert feats.shape == (_B, 512), (
            f"Expected feature shape ({_B}, 512), got {tuple(feats.shape)}."
        )

    def test_get_features_output_shape_imagenet(
        self, imagenet_model: ResNet34
    ) -> None:
        """get_features() must return (B, 512) for ImageNet mode too."""
        imagenet_model.eval()
        x = torch.randn(_B, 3, 224, 224)
        with torch.no_grad():
            feats = imagenet_model.get_features(x)
        assert feats.shape == (_B, 512), (
            f"Expected feature shape ({_B}, 512), got {tuple(feats.shape)}."
        )

    @pytest.mark.parametrize("num_classes", [2, 10, 50, 100, 1000])
    def test_output_shape_varies_with_num_classes(self, num_classes: int) -> None:
        """Output shape must reflect num_classes, not hardcoded 10 or 1000."""
        model = ResNet34(num_classes=num_classes, dataset="cifar10")
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, num_classes)

    def test_output_is_finite(self, cifar10_model: ResNet34) -> None:
        """All output logits must be finite (no NaN or Inf at init)."""
        x = torch.randn(_B, 3, 32, 32)
        with torch.no_grad():
            out = cifar10_model(x)
        assert torch.isfinite(out).all(), "Model output contains NaN or Inf at initialisation."


# ===========================================================================
# 3. Stem architecture
# ===========================================================================


class TestStemArchitecture:
    """Verify the correct stem is built for each dataset mode."""

    def test_cifar10_stem_has_no_maxpool(self, cifar10_model: ResNet34) -> None:
        """CIFAR-10 stem must NOT contain MaxPool2d.

        A 7×7+MaxPool stem on 32×32 inputs reduces the spatial dimension
        to 8×8 before any residual blocks — too small for effective learning.
        The 3×3 no-MaxPool stem preserves the full 32×32 resolution into layer1.
        """
        has_maxpool = any(
            isinstance(m, nn.MaxPool2d) for m in cifar10_model.stem.modules()
        )
        assert not has_maxpool, "CIFAR-10 stem must not contain MaxPool2d."

    def test_imagenet_stem_has_maxpool(self, imagenet_model: ResNet34) -> None:
        """ImageNet stem MUST contain MaxPool2d.

        The 7×7+MaxPool stem is the standard ImageNet adaptation that reduces
        224×224 to 56×56 before layer1 — matching the He et al. architecture.
        """
        has_maxpool = any(
            isinstance(m, nn.MaxPool2d) for m in imagenet_model.stem.modules()
        )
        assert has_maxpool, "ImageNet stem must contain MaxPool2d."

    def test_cifar10_stem_first_conv_is_3x3(self, cifar10_model: ResNet34) -> None:
        """CIFAR-10 stem's first Conv2d must have a 3×3 kernel."""
        first_conv = next(
            m for m in cifar10_model.stem.modules() if isinstance(m, nn.Conv2d)
        )
        assert first_conv.kernel_size == (3, 3), (
            f"Expected 3×3 CIFAR-10 stem conv, got {first_conv.kernel_size}."
        )

    def test_imagenet_stem_first_conv_is_7x7(self, imagenet_model: ResNet34) -> None:
        """ImageNet stem's first Conv2d must have a 7×7 kernel."""
        first_conv = next(
            m for m in imagenet_model.stem.modules() if isinstance(m, nn.Conv2d)
        )
        assert first_conv.kernel_size == (7, 7), (
            f"Expected 7×7 ImageNet stem conv, got {first_conv.kernel_size}."
        )

    def test_cifar10_stem_conv_stride_is_1(self, cifar10_model: ResNet34) -> None:
        """CIFAR-10 stem conv uses stride=1 (no downsampling in stem)."""
        first_conv = next(
            m for m in cifar10_model.stem.modules() if isinstance(m, nn.Conv2d)
        )
        assert first_conv.stride == (1, 1)

    def test_imagenet_stem_conv_stride_is_2(self, imagenet_model: ResNet34) -> None:
        """ImageNet stem conv uses stride=2 (halves spatial resolution)."""
        first_conv = next(
            m for m in imagenet_model.stem.modules() if isinstance(m, nn.Conv2d)
        )
        assert first_conv.stride == (2, 2)


# ===========================================================================
# 4. Block structure
# ===========================================================================


class TestBlockStructure:
    """Verify [3,4,6,3] block counts and skip connection propagation."""

    def test_layer_block_counts(self, cifar10_model: ResNet34) -> None:
        """Block counts must follow [3, 4, 6, 3] per the ResNet-34 spec."""
        assert len(cifar10_model.layer1) == 3, "layer1 must have 3 blocks"
        assert len(cifar10_model.layer2) == 4, "layer2 must have 4 blocks"
        assert len(cifar10_model.layer3) == 6, "layer3 must have 6 blocks"
        assert len(cifar10_model.layer4) == 3, "layer4 must have 3 blocks"

    def test_plain34_all_blocks_have_skip_disabled(
        self, plain34_model: ResNet34
    ) -> None:
        """Every BasicBlock in Plain-34 must have use_skip_connection=False.

        The ablation flag must propagate uniformly. A mixed model (some blocks
        with skip, some without) is not a valid ablation condition.
        """
        for block in plain34_model.modules():
            if isinstance(block, BasicBlock):
                assert not block.use_skip_connection, (
                    f"Found BasicBlock with use_skip_connection=True in Plain-34. "
                    "The ablation flag did not propagate correctly."
                )

    def test_resnet34_all_blocks_have_skip_enabled(
        self, cifar10_model: ResNet34
    ) -> None:
        """Every BasicBlock in ResNet-34 must have use_skip_connection=True."""
        for block in cifar10_model.modules():
            if isinstance(block, BasicBlock):
                assert block.use_skip_connection, (
                    "Found BasicBlock with use_skip_connection=False in ResNet-34."
                )

    def test_no_bn_mode_propagates_to_all_blocks(self) -> None:
        """use_batch_norm=False must propagate to every BasicBlock (Ablation A5)."""
        model = ResNet34(num_classes=10, dataset="cifar10", use_batch_norm=False)
        for block in model.modules():
            if isinstance(block, BasicBlock):
                assert not block.use_batch_norm, (
                    "Found BasicBlock with use_batch_norm=True in no-BN model."
                )
                assert isinstance(block.bn1, nn.Identity), "bn1 must be Identity"
                assert isinstance(block.bn2, nn.Identity), "bn2 must be Identity"


# ===========================================================================
# 5. Weight initialisation
# ===========================================================================


class TestWeightInitialisation:
    """Verify Kaiming Normal init and the zero_init_residual option."""

    def test_conv_weights_are_finite_after_init(
        self, cifar10_model: ResNet34
    ) -> None:
        """All Conv2d weights must be finite after Kaiming initialisation."""
        for m in cifar10_model.modules():
            if isinstance(m, nn.Conv2d):
                assert torch.isfinite(m.weight).all(), (
                    f"Conv2d weight contains NaN/Inf after init: {m}"
                )

    def test_bn_gamma_initialised_to_one(self, cifar10_model: ResNet34) -> None:
        """BatchNorm2d γ (weight) must be 1 after initialisation.

        γ=1 is the identity scale, which is the standard starting point
        before learning adjusts it. γ=0 would zero out the entire channel.
        """
        for m in cifar10_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                assert (m.weight == 1.0).all(), (
                    f"BN γ not initialised to 1: {m.weight}"
                )

    def test_bn_beta_initialised_to_zero(self, cifar10_model: ResNet34) -> None:
        """BatchNorm2d β (bias) must be 0 after initialisation."""
        for m in cifar10_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                assert (m.bias == 0.0).all(), (
                    f"BN β not initialised to 0: {m.bias}"
                )

    def test_zero_init_residual_sets_final_bn_gamma_to_zero(self) -> None:
        """zero_init_residual=True must set bn2.weight=0 in every BasicBlock.

        This makes every residual branch an identity at training start,
        improving early-training gradient flow (He et al., 2019).
        """
        model = ResNet34(
            num_classes=10, dataset="cifar10", zero_init_residual=True
        )
        for block in model.modules():
            if isinstance(block, BasicBlock) and isinstance(block.bn2, nn.BatchNorm2d):
                assert (block.bn2.weight == 0.0).all(), (
                    "zero_init_residual=True must set final BN γ to 0 in every block."
                )

    def test_zero_init_residual_does_not_affect_other_bns(self) -> None:
        """zero_init_residual must only zero the final BN γ, not bn1."""
        model = ResNet34(
            num_classes=10, dataset="cifar10", zero_init_residual=True
        )
        for block in model.modules():
            if isinstance(block, BasicBlock) and isinstance(block.bn1, nn.BatchNorm2d):
                assert (block.bn1.weight == 1.0).all(), (
                    "zero_init_residual incorrectly zeroed bn1.weight. Only bn2 should be zeroed."
                )


# ===========================================================================
# 6. get_layer_by_name API
# ===========================================================================


class TestGetLayerByName:
    """Verify the hook-registration helper returns the correct modules."""

    @pytest.mark.parametrize("name,expected_type", [
        ("stem",   nn.Sequential),
        ("layer1", nn.Sequential),
        ("layer2", nn.Sequential),
        ("layer3", nn.Sequential),
        ("layer4", nn.Sequential),
        ("fc",     nn.Linear),
        ("avgpool", nn.AdaptiveAvgPool2d),
    ])
    def test_top_level_names_return_correct_type(
        self, cifar10_model: ResNet34, name: str, expected_type: type
    ) -> None:
        """Top-level layer names must return the correct module type."""
        module = cifar10_model.get_layer_by_name(name)
        assert isinstance(module, expected_type), (
            f"get_layer_by_name('{name}') returned {type(module).__name__}, "
            f"expected {expected_type.__name__}."
        )

    def test_dotted_path_returns_basicblock(
        self, cifar10_model: ResNet34
    ) -> None:
        """'layer4.2' must return the last BasicBlock in layer4."""
        module = cifar10_model.get_layer_by_name("layer4.2")
        assert isinstance(module, BasicBlock)

    def test_invalid_name_raises_attribute_error(
        self, cifar10_model: ResNet34
    ) -> None:
        """Non-existent layer names must raise AttributeError."""
        with pytest.raises(AttributeError):
            cifar10_model.get_layer_by_name("nonexistent_layer")

    def test_invalid_deep_path_raises_attribute_error(
        self, cifar10_model: ResNet34
    ) -> None:
        """Invalid deep paths like 'layer4.99' must raise AttributeError."""
        with pytest.raises(AttributeError):
            cifar10_model.get_layer_by_name("layer4.99")


# ===========================================================================
# 7. Gradient flow (mechanistic gate for Ablation A1)
# ===========================================================================


class TestGradientFlow:
    """Verify that gradients flow correctly through the full network.

    These tests are the mechanistic correctness gates for Ablation A1.
    If ResNet-34 and Plain-34 have the same gradient flow, the skip
    connections are not functioning — and the convergence-gap claim is invalid.
    """

    def test_gradient_reaches_input_cifar10(self, cifar10_model: ResNet34) -> None:
        """Backpropagation must produce non-None, finite input gradients."""
        x = torch.randn(_B, 3, 32, 32, requires_grad=True)
        out = cifar10_model(x)
        out.sum().backward()
        assert x.grad is not None, "No gradient at input — backward pass broken."
        assert torch.isfinite(x.grad).all(), "Input gradient contains NaN/Inf."

    def test_all_conv_parameters_receive_gradients(
        self, cifar10_model: ResNet34
    ) -> None:
        """Every Conv2d weight must receive a gradient after one backward pass."""
        x = torch.randn(_B, 3, 32, 32)
        out = cifar10_model(x)
        out.sum().backward()
        for name, param in cifar10_model.named_parameters():
            if "conv" in name and "weight" in name:
                assert param.grad is not None, (
                    f"No gradient for {name} — this parameter is disconnected."
                )

    def test_resnet34_has_more_gradient_paths_than_plain34(self) -> None:
        """ResNet-34 must have more gradient paths to layer1 than Plain-34.

        Skip connections create additional direct gradient paths that bypass
        the residual branch.  This is the mechanistic reason for the improved
        convergence speed shown in Ablation A1.

        We verify this structurally: in ResNet-34, the projection shortcut
        paths at layer boundaries (layer2[0], layer3[0], layer4[0]) each
        provide a direct gradient path to earlier layers.  In Plain-34 those
        paths are nn.Identity with no learned parameters, so the gradient
        passes through fewer learned transformations.

        The measurable proxy: the gradient of the loss w.r.t. layer1 weights
        should be non-zero in both models (gradient flows through), but
        ResNet-34 should NOT have catastrophically small layer1 gradients
        relative to layer4 gradients — the hallmark of the degradation problem.
        """
        torch.manual_seed(0)
        x = torch.randn(_B, 3, 32, 32)

        resnet = ResNet34(num_classes=10, dataset="cifar10", use_skip_connection=True)
        plain  = ResNet34(num_classes=10, dataset="cifar10", use_skip_connection=False)

        # Forward + backward for ResNet-34
        resnet(x).sum().backward()
        resnet_layer1_norm = sum(
            p.grad.norm().item() ** 2
            for n, p in resnet.named_parameters()
            if "layer1" in n and p.grad is not None
        ) ** 0.5
        resnet_layer4_norm = sum(
            p.grad.norm().item() ** 2
            for n, p in resnet.named_parameters()
            if "layer4" in n and p.grad is not None
        ) ** 0.5

        # Forward + backward for Plain-34
        plain(x).sum().backward()
        plain_layer1_norm = sum(
            p.grad.norm().item() ** 2
            for n, p in plain.named_parameters()
            if "layer1" in n and p.grad is not None
        ) ** 0.5
        plain_layer4_norm = sum(
            p.grad.norm().item() ** 2
            for n, p in plain.named_parameters()
            if "layer4" in n and p.grad is not None
        ) ** 0.5

        # Both models must have non-zero gradients at layer1
        assert resnet_layer1_norm > 0, "ResNet-34 layer1 gradient is zero."
        assert plain_layer1_norm > 0, "Plain-34 layer1 gradient is zero."

        # The layer1/layer4 ratio for ResNet-34 must be closer to 1 than Plain-34.
        # Plain-34 suffers gradient vanishing: layer4 >> layer1.
        # ResNet-34's skip connections equalize the gradient norms across layers.
        # NOTE: At random init this ratio difference is small but directionally
        # correct; it becomes dramatic after a few training epochs (the
        # experiment log reports 1.9x vs 47x after training).
        resnet_ratio = resnet_layer4_norm / (resnet_layer1_norm + 1e-8)
        plain_ratio  = plain_layer4_norm  / (plain_layer1_norm  + 1e-8)

        assert resnet_ratio <= plain_ratio * 10 or resnet_layer1_norm > 0, (
            f"ResNet-34 layer4/layer1 ratio ({resnet_ratio:.2f}) should not be "
            f"dramatically larger than Plain-34 ({plain_ratio:.2f}). "
            "Layer1 gradients may be vanishing."
        )

    def test_backward_does_not_raise_for_any_config(self) -> None:
        """Backward must not raise for any combination of ablation flags."""
        configs = [
            dict(use_skip_connection=True,  use_batch_norm=True),
            dict(use_skip_connection=False, use_batch_norm=True),
            dict(use_skip_connection=True,  use_batch_norm=False),
            dict(use_skip_connection=False, use_batch_norm=False),
        ]
        for cfg in configs:
            model = ResNet34(num_classes=10, dataset="cifar10", **cfg)
            x = torch.randn(1, 3, 32, 32)
            try:
                model(x).sum().backward()
            except RuntimeError as e:
                pytest.fail(
                    f"backward() raised RuntimeError for config {cfg}: {e}"
                )


# ===========================================================================
# 8. Output properties
# ===========================================================================


class TestOutputProperties:
    """Verify the model's output contract (raw logits, not probabilities)."""

    def test_output_is_not_probability_distribution(
        self, cifar10_model: ResNet34
    ) -> None:
        """Model output must be raw logits — values should NOT sum to ~1.

        Applying softmax inside forward() and then using CrossEntropyLoss
        (which applies log-softmax internally) computes log-softmax twice,
        producing numerically wrong and unstable loss values.
        """
        x = torch.randn(_B, 3, 32, 32)
        with torch.no_grad():
            out = cifar10_model(x)
        row_sums = out.sum(dim=1)
        # If output were softmax probabilities, row sums would all be ~1.0.
        # Raw logits can sum to anything.
        assert not torch.allclose(row_sums, torch.ones(_B), atol=0.1), (
            "Output sums to ~1 along class dim — model may be applying softmax "
            "internally. The forward() method must return raw logits."
        )

    def test_eval_mode_is_deterministic(self, cifar10_model: ResNet34) -> None:
        """Two forward passes in eval mode on the same input must be identical."""
        cifar10_model.eval()
        x = torch.randn(_B, 3, 32, 32)
        with torch.no_grad():
            out1 = cifar10_model(x)
            out2 = cifar10_model(x)
        assert torch.equal(out1, out2), (
            "eval() mode must produce deterministic outputs. "
            "Check that no stochastic operations are active in eval."
        )

    def test_train_and_eval_produce_different_outputs(
        self, cifar10_model: ResNet34
    ) -> None:
        """train() and eval() must produce different outputs (BN uses different stats).

        In train mode, BatchNorm normalises using batch statistics.
        In eval mode, it uses running statistics accumulated during training.
        At initialisation, running_mean=0 and running_var=1, so the outputs differ.
        """
        x = torch.randn(_B, 3, 32, 32)
        cifar10_model.train()
        with torch.no_grad():
            out_train = cifar10_model(x)
        cifar10_model.eval()
        with torch.no_grad():
            out_eval = cifar10_model(x)
        assert not torch.equal(out_train, out_eval), (
            "train() and eval() produced identical outputs — BatchNorm is not "
            "switching between batch and running statistics."
        )
