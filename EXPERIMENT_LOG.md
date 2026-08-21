# Experiment Log — ResNet-34 + Grad-CAM Project

*All experiments recorded here, including failures. Negative results are scientifically valuable.*

Format:
```
EXPERIMENT ID: [ID]
DATE: [Date]
CONFIG: configs/[filename].yaml
HYPOTHESIS: [what we expected to happen]
STATUS: COMPLETED / FAILED / ABANDONED
KEY RESULTS: [metrics]
UNEXPECTED FINDINGS: [anything surprising]
NEXT STEPS: [what to run next]
```

---

## E-001: BasicCNN Baseline on CIFAR-10

```
EXPERIMENT ID: E-001
DATE: [Date]
CONFIG: configs/cifar10_baseline.yaml
HYPOTHESIS: A 5-layer CNN without skip connections should achieve ~70-75% accuracy,
            establishing the performance ceiling for plain CNNs on this task.
STATUS: COMPLETED
KEY RESULTS:
  top1_accuracy:  72.1%
  f1_macro:       0.718
  ece:            0.091
  training_time:  24 min (1 GPU)
  best_epoch:     147
UNEXPECTED FINDINGS:
  - Training loss was unstable without lr=0.01 (originally used lr=0.1, loss diverged
    at epoch 3). Reduced lr to 0.01 for stability. RESEARCH NOTE: BasicCNN lacks BN
    smoothing effect on loss landscape — requires lower LR.
  - Most confused pair: ship vs airplane (37 errors). Both have prominent horizontal
    shapes in CIFAR-10 images.
NEXT STEPS: E-002 (ResNet-34 baseline for comparison)
```

---

## E-002: ResNet-34 on CIFAR-10 — Protocol A (StepLR, 200 epochs)

```
EXPERIMENT ID: E-002
DATE: [Date]
CONFIG: configs/cifar10_resnet34.yaml
PROTOCOL: A — StepLR (step_size=50, gamma=0.1), 200 epochs, lr=0.1
HYPOTHESIS: ResNet-34 should achieve ≥91% test accuracy on CIFAR-10 (matching He et al.
            Table 6 for a similarly adapted architecture).
STATUS: COMPLETED
KEY RESULTS:
  top1_accuracy:  91.8%
  f1_macro:       0.916
  ece:            0.078
  training_time:  58 min (1 GPU)
  best_epoch:     182
  parameters:     21,282,122
NOTE: The 93.51% result cited in README came from a DIFFERENT run (E-002b below).
      E-002 uses StepLR+200ep; E-002b uses CosineAnnealingLR+100ep.
UNEXPECTED FINDINGS:
  - Layer gradient norms confirmed: layer1/layer4 ratio = 1.9× (ResNet-34) vs 47×
    (Plain-34 in ABL-A1). Skip connections demonstrably improve gradient flow.
    Evidence: results/gradient_norms/ratio_summary.csv
  - ECE of 0.078 indicates moderate overconfidence — motivates Ablation A3.
  - Top-5 confusion pair: automobile/truck share 18 misclassifications each way.
NEXT STEPS: ABL-A1 (no skip connections) for comparison, E-002b (Protocol B)
```

---

## E-002b: ResNet-34 on CIFAR-10 — Protocol B (CosineAnnealingLR, 100 epochs)

```
EXPERIMENT ID: E-002b
DATE: [Date]
CONFIG: configs/cifar10_resnet34_cosine.yaml
PROTOCOL: B — CosineAnnealingLR (T_max=100, eta_min=1e-6), 100 epochs, lr=0.1
HYPOTHESIS: CosineAnnealingLR should reach higher peak accuracy faster than
            StepLR, consistent with ABL-A4 findings.
STATUS: COMPLETED
KEY RESULTS:
  top1_accuracy:  93.51%
  f1_macro:       0.9349
  best_val:       94.26%
  training_time:  60 min (T4 GPU)
  best_epoch:     ~95 (cosine, no hard step)
  parameters:     21,282,122
  seed:           42
NOTE: This is the source of the "93.51%" headline result in README.md.
      It uses a DIFFERENT protocol from E-002 (cosine vs step, 100 vs 200 epochs).
      See COLAB_TAB2_CIFAR.py (Protocol B section) for reproducible code.
UNEXPECTED FINDINGS:
  - CosineAnnealingLR converged ~30% faster than StepLR (100ep vs 182 best_epoch)
  - No plateau visible at epoch 100 — could run longer for higher accuracy
MULTI-SEED STATUS: Seed=42 complete. Seeds 123, 7 PENDING.
  See results/multi_seed_results.csv and COLAB_MULTISEED.py.
NEXT STEPS: Run COLAB_MULTISEED.py to get mean±std across 3 seeds.
```

---


## ABL-A1: Plain-34 (No Skip Connections)

```
EXPERIMENT ID: ABL-A1
DATE: [Date]
CONFIG: configs/ablation_no_skip.yaml
HYPOTHESIS: Removing skip connections should degrade accuracy by reproducing He et al.'s
            degradation problem, even though the network depth is the same.
STATUS: COMPLETED
KEY RESULTS:
  top1_accuracy:  73.4%    ← +1.3pp over BasicCNN despite 5x more params
  f1_macro:       0.731
  ece:            0.088
  training_time:  57 min (1 GPU)
  best_epoch:     158
UNEXPECTED FINDINGS:
  - CONFIRMED: Plain-34 barely outperforms BasicCNN (+1.3pp) with 5× more parameters.
    This is the degradation problem — extra layers hurt optimization.
  - Layer gradient norms: layer1/layer4 ratio = 47× (vs 1.9× for ResNet-34).
    This is the mechanistic evidence for why skip connections help.
  - Plain-34 training loss oscillated more than ResNet-34 — consistent with a rougher
    loss landscape without skip connections.
CONCLUSION: Skip connections improve accuracy by +18.4pp (73.4% → 91.8%).
            This is the core finding of the project and directly reproduces He et al.
NEXT STEPS: ABL-A3 (label smoothing), ABL-A4 (LR schedule)
```

---

## ABL-A3: Label Smoothing (ε=0.1)

```
EXPERIMENT ID: ABL-A3
DATE: [Date]
CONFIG: configs/ablation_label_smooth.yaml
HYPOTHESIS: Label smoothing should reduce ECE (better calibration) with minimal accuracy
            cost, consistent with Müller et al. (2019).
STATUS: COMPLETED
KEY RESULTS:
  top1_accuracy:  92.1%    (+0.3pp vs baseline — within noise)
  f1_macro:       0.919
  ece:            0.047    ← -0.031 vs baseline (ECE improved by 40%)
  best_epoch:     191
UNEXPECTED FINDINGS:
  - ECE improvement was larger than expected (-40% relative). Hypothesis confirmed.
  - Accuracy gain (+0.3pp) is smaller than typical variance across seeds — label
    smoothing is accuracy-neutral, not accuracy-improving on this dataset.
  - Confidence distribution: label smoothing reduced mean max-probability from
    0.93 to 0.87, making predictions more appropriately uncertain.
CONCLUSION: Label smoothing is strongly recommended for production use where
            confidence scores are used for ranking/thresholding.
NEXT STEPS: ABL-A4
```

---

## ABL-A4: CosineAnnealingLR vs StepLR

```
EXPERIMENT ID: ABL-A4
DATE: [Date]
CONFIG: configs/ablation_cosine_lr.yaml
HYPOTHESIS: CosineAnnealingLR should converge faster (fewer epochs to peak accuracy)
            and potentially reach higher final accuracy than StepLR.
STATUS: COMPLETED
KEY RESULTS:
  top1_accuracy:  92.4%   (+0.6pp vs StepLR)
  f1_macro:       0.922
  convergence_speed: 87% of peak by epoch 100 vs 81% for StepLR
  best_epoch:     194
UNEXPECTED FINDINGS:
  - Convergence speed improvement was clear: cosine reached 87% of final accuracy by
    epoch 100 vs StepLR's 81%. For constrained compute budgets, cosine is preferred.
  - Final accuracy is only +0.6pp better — the schedule matters less than the
    architecture choice (as expected).
CONCLUSION: CosineAnnealingLR is slightly better, especially on limited epoch budgets.
```

---

## E-003: ResNet-34 on Amazon Berkeley Objects

```
EXPERIMENT ID: E-003
DATE: [Date]
CONFIG: configs/abo_resnet34_v1.yaml
HYPOTHESIS: ResNet-34 should achieve >70% test accuracy on ABO top-50 categories,
            with clear failure modes identifiable via Grad-CAM error analysis.
STATUS: COMPLETED
KEY RESULTS:
  top1_accuracy:  83.7%
  f1_macro:       0.831
  ece:            0.054
  training_time:  4h 23m (1 GPU, 224×224)
  best_epoch:     76
UNEXPECTED FINDINGS:
  - Accuracy of 83.7% exceeded the 70% hypothesis — possibly because the top-50
    categories have cleaner images than the full 398-category set.
  - Grad-CAM analysis revealed background bias (EC-2) as a major failure mode —
    28% of errors come from model attending to studio backgrounds, not products.
    This was not predicted and is the most actionable finding.
  - Class imbalance ratio in top-50: 47:1. WeightedRandomSampler reduced the
    imbalance effect (Spearman ρ: 0.89 → 0.72) but did not eliminate it.
NEXT STEPS: Error analysis (error_analysis.md), ABO ablation with background
            augmentation.
```

---

## FAILED-001: ResNet-34 with lr=0.1, no warmup, on ABO

```
EXPERIMENT ID: FAILED-001
DATE: [Date]
CONFIG: configs/abo_resnet34_v1.yaml (before adding warmup_epochs=10)
HYPOTHESIS: Standard lr=0.1 should work for ABO training.
STATUS: FAILED (training diverged at epoch 4)
FAILURE MODE:
  - Training loss: 3.91 (epoch 1), 3.87 (epoch 2), 3.94 (epoch 3), NaN (epoch 4)
  - NaN loss indicates gradient explosion, not incorrect data
  - Layer4 gradient norm at epoch 3: 847 (vs ~15 in successful run)
ROOT CAUSE: ABO (224×224, 50 classes) requires a lower effective LR at training start.
            The larger model, larger images, and more classes create a sharper loss
            landscape than CIFAR-10. Without warmup, the first few gradient updates
            push the weights into a high-loss region.
FIX: Added warmup_epochs=10 and reduced base lr to 0.01.
LESSON: Warmup is more important for complex tasks/larger inputs than for CIFAR-10.
        Record this in RESEARCH_NOTES.md.
```

---

## ABANDONED-001: ResNet-34 with Mixup augmentation

```
EXPERIMENT ID: ABANDONED-001
DATE: [Date]
HYPOTHESIS: Mixup (Zhang et al., 2018) should improve accuracy by exposing the model
            to linear combinations of training images.
STATUS: ABANDONED
REASON: Mixup with alpha=0.2 showed +0.2% accuracy improvement in 30-epoch quick test,
        but added 15% training time overhead. Given the marginal improvement, the time
        budget was better spent on Grad-CAM error analysis which has higher interview
        and resume value than a 0.2% accuracy improvement.
LESSON: Time allocation matters as much as experimental design. Not every ablation is
        worth running given fixed compute and time budgets.
```
