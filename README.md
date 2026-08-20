# ResNet-34 From Scratch + Grad-CAM for Amazon Product Understanding

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Implementation of ResNet-34 (He et al., 2016) and Grad-CAM (Selvaraju et al., 2017) from scratch using PyTorch primitives — no `torchvision.models` imports. Built as a research portfolio project targeting Amazon Berkeley Objects (ABO) product classification.

**Research question**: Do skip connections empirically solve the degradation problem, and where does a trained model fail on real Amazon product images?

---

## Experiment Status

| Experiment | Status | Result |
|---|---|---|
| ResNet-34 on CIFAR-10 | ✅ Complete | **93.51% top-1 accuracy** |
| Plain-34 on CIFAR-10 (no skip connections) | ✅ Complete | **92.13% top-1 accuracy** |
| BasicCNN baseline | 🔄 Running | — |
| ResNet-18 depth ablation | 🔄 Running | — |
| ResNet-34 no BatchNorm | 🔄 Running | — |
| ResNet-34 no augmentation | 🔄 Running | — |
| ResNet-34 on ABO (50 categories) | 🔄 Running | — |
| Grad-CAM on ABO product images | 🔄 Pending ABO training | — |

*Results table and Grad-CAM images will be updated as experiments complete.*

---

## Verified Results (CIFAR-10)

These numbers are from actual training runs. Seed 42. T4 GPU. 100 epochs. CosineAnnealingLR.

```
ResNet-34 (with skip connections):  93.51%  F1=0.9349  21.28M params  60 min
Plain-34  (no skip connections):    92.13%  F1=0.9212  21.11M params  57 min
```

### The Skip Connection Story

The final accuracy gap (1.4pp) understates what skip connections actually do. The real finding is **convergence speed**:

```
Epoch  1:  ResNet-34  34.4%  vs  Plain-34  27.8%  →  +6.6pp
Epoch 10:  ResNet-34  81.8%  vs  Plain-34  58.7%  →  +23.1pp  ← biggest gap
Epoch 20:  ResNet-34  87.5%  vs  Plain-34  80.3%  →  +7.2pp
Epoch 50:  ResNet-34  91.4%  vs  Plain-34  88.6%  →  +2.8pp
Epoch100:  ResNet-34  93.5%  vs  Plain-34  92.1%  →  +1.4pp
```

Plain-34 eventually closes most of the gap at 100 epochs — but only because CosineAnnealingLR gives it enough optimization budget. With fixed compute (e.g., 30 epochs), the gap remains large. This is exactly the mechanism He et al. describe: skip connections make the optimization landscape easier to navigate, not just the final accuracy higher.

This also confirms the paper's core claim: **the degradation problem is an optimization problem, not an overfitting problem**. Plain-34's training loss was also higher than ResNet-34 at every epoch.

---

## Architecture

### ResNet-34 — CIFAR-10 Mode (32×32 inputs)

```
Input [B, 3, 32, 32]
        │
        ▼
 Stem: Conv3×3, stride=1, BN, ReLU     [B, 64, 32, 32]
 (no MaxPool — preserves spatial info on small images)
        │
 Layer 1: 3× BasicBlock  64→64   s=1   [B,  64, 32, 32]
 Layer 2: 4× BasicBlock  64→128  s=2   [B, 128, 16, 16]
 Layer 3: 6× BasicBlock  128→256 s=2   [B, 256,  8,  8]
 Layer 4: 3× BasicBlock  256→512 s=2   [B, 512,  4,  4]
        │
 Global Average Pooling                [B, 512]
 Linear(512 → num_classes)             [B, K]
```

### BasicBlock

```
x ──────────────────────────────────┐
│                                   │ identity / projection shortcut
▼                                   │
Conv3×3 → BN → ReLU                 │
▼                                   │
Conv3×3 → BN                        │
│                                   │
└──────────── + ────────────────────┘
              │
            ReLU   ← applied AFTER addition, not before
              │
           output
```

**Why ReLU after addition?** The residual F(x) can be negative. Applying ReLU before addition clips negative residuals, preventing the network from learning subtractive corrections. Post-addition ReLU preserves the full residual expressiveness.

### Implementation decisions (with reasoning)

| Decision | Why |
|---|---|
| `bias=False` in Conv before BN | BN subtracts batch mean, which exactly cancels any conv bias — it adds parameters with zero effect |
| BN params excluded from weight decay | Weight decay on γ/β pushes them toward 0, partially defeating normalisation — no regularisation benefit |
| `register_full_backward_hook` for Grad-CAM | `register_backward_hook` is deprecated and mishandles non-tensor inputs |
| CIFAR-10 stem: 3×3 no MaxPool | 7×7+MaxPool on 32×32 leaves 8×8 feature maps after stem — too small |
| SGD + Nesterov over Adam | Adam converges faster early but SGD generalises better at convergence on CIFAR-10 |

---

## Grad-CAM Implementation

Implemented from scratch using PyTorch hooks — no Captum dependency.

```python
# Forward hook captures feature maps A^k from layer4
target_layer.register_forward_hook(
    lambda m, inp, out: save_activation(out.detach()))

# Backward hook captures gradients ∂y^c/∂A^k
target_layer.register_full_backward_hook(
    lambda m, grad_in, grad_out: save_gradient(grad_out[0].detach()))

# Importance weights: global average pool over spatial dims
alpha_k = gradients.mean(dim=[2, 3], keepdim=True)   # [B, K, 1, 1]

# Weighted combination + ReLU (keep only positive evidence)
cam = F.relu((alpha_k * activations).sum(dim=1))       # [B, h, w]
```

**Sanity check (Adebayo et al. 2018):** Mean Pearson |ρ| = 0.031 between trained-model and random-model CAMs. The explanations reflect learned model behaviour, not image structure.

*Grad-CAM visualizations on ABO product images will be added when ABO training completes.*

---

## Repository Structure

```
resnet34-amazon-products/
├── src/
│   ├── models/         residual_block.py  resnet34.py  model_factory.py
│   ├── training/       trainer.py  losses.py  optimizers.py  schedulers.py
│   ├── data/           dataset.py  transforms.py  dataloaders.py
│   ├── evaluation/     metrics.py  evaluator.py
│   ├── explainability/ gradcam.py
│   └── utils/          seed.py  logger.py  checkpoint.py
├── scripts/            train.py  evaluate.py
├── configs/            base_config.yaml + 6 experiment configs
├── tests/              test_residual_block.py  (62 tests, all passing)
├── paper_notes/        resnet_summary.md  gradcam_summary.md
├── results/
│   ├── ablation_table.csv         (updating as experiments run)
│   └── gradcam/                   (updating after ABO training)
├── error_analysis.md              (will update with real inference data)
├── RESEARCH_NOTES.md
├── EXPERIMENT_LOG.md
└── colab_train_and_verify.py      (self-contained training script)
```

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/resnet34-amazon-products
cd resnet34-amazon-products
pip install -e .
```

**Verify architecture (no data needed):**
```bash
python - << 'EOF'
from src.models.resnet34 import ResNet34
import torch
model = ResNet34(num_classes=10, dataset="cifar10")
x = torch.randn(2, 3, 32, 32)
print(f"Output: {model(x).shape}")
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
# Expected: [2, 10] and 21.28M
EOF
```

**Run the core ablation:**
```bash
python scripts/train.py --config configs/cifar10_resnet34.yaml   # ResNet-34
python scripts/train.py --config configs/ablation_no_skip.yaml   # Plain-34
```

**Run all tests:**
```bash
pytest tests/ -v
# 62 tests, all passing
```

---

## What is verified vs what is in progress

**Verified from code or actual runs:**
- 21.28M parameters — confirmed by `count_parameters()`
- [3,4,6,3] block structure — confirmed by `len(layer)`
- ResNet-34 CIFAR-10: 93.51% — from actual training run
- Plain-34 CIFAR-10: 92.13% — from actual training run
- 23pp convergence gap at epoch 10 — from training logs
- Grad-CAM output shape [B,1,H,W] ∈ [0,1] — from forward+backward pass
- BN params correctly excluded from weight decay — from unit test
- 62 unit tests passing — `pytest tests/`

**In progress (will update):**
- Remaining CIFAR-10 ablations (BasicCNN, ResNet-18, no-BN, augmentation variants)
- ABO training and test accuracy
- Real Grad-CAM heatmaps on Amazon product images
- Error analysis from actual ABO inference failures

---

## Citation

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={CVPR}, year={2016}}

@inproceedings{selvaraju2017grad,
  title={Grad-CAM: Visual explanations from deep networks via gradient-based localization},
  author={Selvaraju, Ravi Ramprasad and others},
  booktitle={ICCV}, year={2017}}
```

---

## License

MIT. ABO dataset is released under CC BY 4.0 by Amazon.com.
