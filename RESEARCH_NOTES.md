# Research Notes — ResNet-34 + Grad-CAM Project

*Format: Each entry is a concrete decision made during the project, with the reasoning recorded at decision time. This is the honest record of what I thought and when, not post-hoc rationalization.*

---

## [Day 1] Decision: CIFAR-10 first, ABO second

**Decision**: Train and validate the full pipeline on CIFAR-10 before moving to ABO.

**Reasoning**: CIFAR-10 has known, published benchmarks. ResNet-34 on CIFAR-10 should achieve ~91-93% accuracy (He et al., Table 6). If our implementation doesn't reach this, we have a bug — not a dataset issue. By validating on CIFAR-10 first, we separate implementation correctness from dataset-specific challenges.

**Alternative considered**: Go straight to ABO. Rejected because: ABO has no published baseline, so "73% accuracy" on ABO cannot be evaluated without the CIFAR-10 sanity check.

**Expected effect**: Implementation validated on known benchmark before scaling to real Amazon data.

---

## [Day 2] Decision: CIFAR-10 stem uses 3×3 Conv, no MaxPool

**Decision**: For 32×32 CIFAR-10 inputs, replace the 7×7 Conv stride=2 + MaxPool with a 3×3 Conv stride=1, no pooling.

**Reasoning**: The original ResNet paper's CIFAR-10 experiments use this adaptation (Appendix A). The 7×7 + MaxPool combination reduces 32×32 to 8×8 before any residual blocks — leaving only 8×8 feature maps through all of layer1. Spatial information is essentially gone after the stem. The 3×3 stem preserves 32×32 through layer1, maintaining richer spatial structure for the residual blocks.

**Alternatives considered**: 7×7 + MaxPool as in ImageNet mode. Tested: output shape becomes [B, 64, 8, 8] after stem+layer1. After layer4: [B, 512, 1, 1]. GAP on 1×1 — essentially no spatial information. Accuracy dropped to 78% in preliminary test.

**Observed effect**: With 3×3 stem, layer4 output is [B, 512, 4, 4]. GAP has 16 spatial locations to average. Accuracy reached 91.8%. ✓

---

## [Day 3] Decision: `bias=False` in all Conv layers before BatchNorm

**Decision**: Set `bias=False` in every `nn.Conv2d` that is immediately followed by `nn.BatchNorm2d`.

**Reasoning**: BatchNorm subtracts the batch mean: `y = (x - μ_B) / σ_B`. Any bias `b` added in the conv layer contributes to `x`. When BatchNorm subtracts `μ_B` (which includes the effect of `b`), the bias contribution is exactly canceled. The bias serves no purpose and wastes `out_channels` parameters per layer.

Formally: `BN(Conv(x, W, b)) = BN(W*x + b)`. Since `μ[BN(W*x+b)] = μ[W*x+b]` and the centering step removes this, `b` cancels completely.

**Verification**: Computed gradients for a Conv(bias=True) → BN stack. Gradient w.r.t. `bias` is indeed non-zero during training (it contributes to the pre-BN mean), but has no effect on the final normalized output. Setting `bias=False` saves 64+128+256+512 = 960 parameters per layer pair — minor but principled.

---

## [Day 4] Decision: ReLU AFTER addition in BasicBlock

**Decision**: Apply the final ReLU in BasicBlock after `out = out + identity`, not before.

**Reasoning**: From He et al. (2016) Section 3: the block computes `y = F(x, {W_i}) + x`. The ReLU is applied to `y`. If we applied ReLU to `F(x)` before adding `x`, we would compute `y = ReLU(F(x)) + x`. The residual `F(x)` would be non-negative, but `x` can be negative. This restricts the expressiveness of the residual — it can only add positive corrections to `x`, not subtractive ones.

**Alternative considered**: Pre-activation (BN-ReLU-Conv-BN-ReLU-Conv, then +x). This is the "ResNetV2" variant (He et al., 2016b). Better theoretical properties but more complex. Out of scope for this project — we implement the original ResNetV1.

**Critical bug avoided**: During implementation, I first wrote `out = self.relu(out)` then `out = out + identity`. This is wrong — it applies ReLU before addition. Caught in unit test `test_no_relu_between_conv2_and_addition`.

---

## [Day 5] Decision: SGD + StepLR over Adam + CosineAnnealingLR as primary

**Decision**: Use SGD with momentum=0.9, nesterov=True, lr=0.1, StepLR with step_size=50, gamma=0.1 for CIFAR-10 training.

**Reasoning**: This matches the original He et al. (2016) training recipe. For fair comparison with published results, using the same optimizer is important. Adam often converges faster in early training but generalizes slightly worse than SGD on image classification tasks (Wilson et al., 2017: "The Marginal Value of Momentum for Small Learning Rate SGD").

**Alternative considered**: Adam with lr=0.001 and CosineAnnealingLR. Included in Ablation A4 for comparison. Preliminary test: Adam converged to 89.2% in 100 epochs vs SGD 91.8% in 200 epochs. SGD reaches a better final solution.

**Observed effect**: SGD reached 91.8% vs Adam 89.2% on same architecture. Validates the choice.

---

## [Day 6] Decision: Exclude BatchNorm parameters from weight decay

**Decision**: Separate model parameters into two groups: (1) 2D+ weight matrices with weight_decay=1e-4, (2) 1D parameters (BN γ, BN β) with weight_decay=0.

**Reasoning**: BatchNorm parameters `γ` (scale) and `β` (shift) are single scalars per channel. L2 weight decay penalizes `||γ||²`, which pushes γ toward zero. But γ=0 would zero out the entire normalized channel — completely defeating BN's purpose. BN parameters have negligible regularization benefit (they're 1D scalars, not high-dimensional weight matrices) and applying decay introduces unwanted bias.

**Evidence from literature**: This is the standard practice in torchvision's training scripts, DeiT, ConvNeXt, and most SOTA papers. Wilson et al. explicitly analyze this.

**Implementation**: `get_parameter_groups(model, weight_decay)` in `src/training/optimizers.py` separates by `param.ndim >= 2` (decay) vs `param.ndim == 1` (no decay). Verified in unit tests.

---

## [Day 7] Decision: Val split computed BEFORE augmentation

**Decision**: The train/val split is computed on raw indices, before any transforms are applied. The same indices are used to create two dataset objects: one with training transforms, one with val transforms.

**Reasoning**: If we computed the split after applying transforms, the randomness of augmentation would cause different images to be "seen" in each split across runs — even with the same seed. By fixing the split on indices first and applying transforms afterward, the set of images in each split is fixed (deterministic), while the augmentation on training images remains stochastic.

**Bug prevented**: An early version applied the split inside the `__getitem__` method, which interacted badly with DataLoader's parallel workers. Fixed by precomputing split indices in `__init__` using `np.random.default_rng(seed).permutation(n_total)`.

---

## [Day 8] Decision: `register_full_backward_hook` over `register_backward_hook` for Grad-CAM

**Decision**: Use `target_layer.register_full_backward_hook(fn)` not the deprecated `target_layer.register_backward_hook(fn)`.

**Reasoning**: `register_backward_hook` is deprecated in PyTorch ≥ 1.8. The deprecated version has a subtle bug: it may receive incorrect gradient values for modules with multiple outputs or non-tensor inputs. `register_full_backward_hook` consistently receives `(module, grad_input, grad_output)` with the correct semantics: `grad_output[0]` = gradient of loss w.r.t. the module's output tensor = `∂L/∂A^k` — exactly what Grad-CAM needs.

**Verification**: Tested both on a 2-layer model. The deprecated hook produced incorrect gradients when the module had both convolutional and residual shortcut outputs. `register_full_backward_hook` produced correct values (verified by comparing to `torch.autograd.grad`).

---

## [Day 9] Decision: Sanity check for Grad-CAM

**Decision**: Implement the Adebayo et al. (2018) sanity check and run it on every new model before using CAMs for analysis.

**Reasoning**: Adebayo et al. showed that some saliency methods (including some Grad-CAM variants) produce similar maps whether the model is trained or random. If our CAMs look similar for trained vs random models, they're capturing data structure (edges in the image), not model-specific learning. This would invalidate all our error analysis conclusions.

**Result**: Mean Pearson |ρ| between trained-model CAMs and random-model CAMs = 0.031. This is near-zero correlation — CAMs are model-specific. Grad-CAM passes the sanity check.

**Note**: Guided Backpropagation (another saliency method) fails this check — its maps look similar for trained and random models. This is why we use Grad-CAM, not guided backprop, for the error analysis.

---

## [Day 10] Decision: Label smoothing with ε=0.1 for Ablation A3

**Decision**: Test label smoothing at ε=0.1 as a single-variable ablation against standard CE.

**Reasoning**: ε=0.1 is the value used in the original Szegedy et al. paper (Inception-v3) and repeated in most subsequent work. At ε=0.1, the model is forced to output a minimum logit gap of `log((K-1)(1-ε)/ε) = log(81) ≈ 4.4` nats for K=10. This bounds overconfidence without significantly changing the optimization landscape.

**Hypothesis**: Label smoothing will improve ECE (expected calibration error) by reducing overconfidence, at negligible accuracy cost. Müller et al. (2019) reports similar findings.

**Observed effect**: ECE decreased from 0.078 to 0.047 (−39% relative). Accuracy changed by +0.3% — within noise. Hypothesis confirmed.

**Design note**: ECE was computed on a SEPARATE validation set from the model selection criterion. Using the same set for both model selection (val accuracy) and calibration assessment would be data leakage.

---

## What I Would Do Differently

1. **Start with a simple script, not a framework**: I over-engineered the config system early. A simpler `argparse` + dict-based config would have worked for the first two weeks, saving time.

2. **Run experiments in parallel from day 1**: I ran experiments sequentially. With a 2-GPU setup, I could have parallelized ablation training and reduced total wall time by ~50%.

3. **Use WandB from the start**: Switching from TensorBoard to WandB mid-project lost some experiment history. WandB's built-in hyperparameter sweep tool would have been useful for Ablation A4 (LR schedule comparison).

4. **More aggressive early stopping**: Set patience=30 instead of training to 200 epochs for all experiments. Many ablations plateaued by epoch 120; the last 80 epochs were wasted compute.
