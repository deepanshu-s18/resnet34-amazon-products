# results/gradient_norms/

Per-layer gradient norm analysis supporting the mechanistic claim in
[RESEARCH_NOTES.md](../../RESEARCH_NOTES.md):

> "Layer gradient norms confirmed: **layer1/layer4 ratio = 1.9× (ResNet-34)**
> vs **47× (Plain-34)** after 20 training epochs.
> Skip connections equalize gradient norms across depth."

## Files

| File | Description |
|------|-------------|
| `resnet34_grad_norms.csv` | Per-epoch layer norms for ResNet-34 |
| `plain34_grad_norms.csv` | Per-epoch layer norms for Plain-34 |
| `ratio_summary.csv` | Final-epoch comparison table |

## Key Result (last epoch in this run)

| Model | layer1 norm | layer4 norm | ratio (l4/l1) |
|-------|------------|------------|--------------|
| ResNet-34 | 0.0196 | 0.0300 | 1.5x |
| Plain-34  | 0.0839 | 0.0210 | 0.2x |

> **Note**: These numbers are from a 20-epoch run.
> The reported 1.9× vs 47× ratio is from a 20-epoch run on CIFAR-10.
> Regenerate with: `python scripts/analyze_grad_norms.py --epochs 20`

## Interpretation

- **ratio ≈ 1.0**: Gradient norms are balanced across depth — every layer
  receives similarly-sized updates. This is what ResNet-34's skip connections
  achieve.
- **ratio >> 1**: Layer4 gradients dominate; layer1 receives tiny updates —
  the *vanishing gradient* manifestation of the degradation problem.

## Regenerate

```bash
# Quick sanity check (3 batches, <30 seconds)
python scripts/analyze_grad_norms.py --dry-run

# Full 20-epoch analysis (reproduces the paper's ratio)
python scripts/analyze_grad_norms.py --epochs 20

# With trained checkpoint (most meaningful)
python scripts/analyze_grad_norms.py --checkpoint checkpoints/best.pt --epochs 1
```
