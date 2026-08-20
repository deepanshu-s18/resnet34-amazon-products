# Paper Notes: Deep Residual Learning for Image Recognition

**Paper**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.  
**arXiv**: https://arxiv.org/abs/1512.03385  
**Read**: [Date]

---

## Five-Bullet Summary

1. **Problem**: Deeper plain networks perform *worse* than shallower ones — not because of overfitting (train error also increases) but because gradient vanishing makes 20-layer optimization harder than 10-layer, contradicting the theory that more layers should help.

2. **Solution**: Add an identity shortcut `F(x) + x` to every pair of layers, letting the network learn the residual `F(x) = H(x) − x` instead of the full mapping `H(x)`. If the identity is optimal, `F(x) → 0` is easier than `H(x) → x`.

3. **Key insight**: Residual formulation re-parameterizes the optimization. The hypothesis is not that skip connections "fix" gradients by adding gradient paths (though they do), but that the residual *function* is easier to optimize than the unreferenced function. A zero residual is a trivial solution; a zero mapping is not.

4. **Results**: ResNet-34 on ImageNet: 7.02% top-5 error vs 11.72% for VGG-19 with 2.5× fewer parameters. On CIFAR-10: ResNet-20 achieves 8.75% error where a 20-layer plain net achieves 8.82% — skip connections matter more at 56 and 110 layers where plain nets degrade significantly.

5. **Limitations**: (1) Skip connections require identical input/output dimensions or a projection shortcut — adds parameters and design complexity. (2) The theory is mostly empirical; a rigorous proof that residuals are easier to optimize remains an open question. (3) ImageNet training still takes days even on modern hardware.

---

## Why the Degradation Problem is NOT Overfitting

This is the most common misconception and the thing interviewers test.

A deeper model should be no worse than a shallower one: you can always set the extra layers to identity and recover the shallower model's behavior. The fact that a 56-layer plain network has *higher training error* than a 20-layer plain network means the optimizer cannot even find that identity solution. This is an **optimization problem**, not a generalization problem.

Evidence from the paper (Table 1): 56-layer plain network trains to 7.93% training error. 20-layer plain network trains to 7.02% training error. The deeper network is strictly worse on training data → this eliminates overfitting as the explanation.

---

## The Mathematics of Skip Connections

For a block with input `x` and layers `{W1, W2}`:

**Without skip connection** (plain network):
```
output = ReLU(W2 · ReLU(W1 · x))
```

**With skip connection** (residual block):
```
output = ReLU(W2 · ReLU(W1 · x) + x)
```

The gradient of the loss `L` w.r.t. the block input `x`:
```
∂L/∂x = ∂L/∂output · (∂output/∂(W2·W1·x) + 1)
```

The `+1` term is the skip connection's gradient contribution. Even if `∂output/∂(W2·W1·x) → 0` (vanishing gradient through the convolutional path), the gradient through the skip path remains intact. This is why gradients flow to early layers even in very deep networks.

**In a 34-layer network**, the gradient product through the skip connection path is essentially an uninterrupted signal from output to input. The convolutional path gradient may vanish, but the skip path doesn't.

---

## Architecture Details (ResNet-34 Specifically)

| Layer Group | Blocks | Channels | Stride | Output Size (224×224 input) |
|---|---|---|---|---|
| Stem | Conv7×7 + BN + ReLU + MaxPool | 64 | 2, 2 | 56×56 |
| Layer 1 | 3× BasicBlock | 64→64 | 1 | 56×56 |
| Layer 2 | 4× BasicBlock | 64→128 | 2 (first block) | 28×28 |
| Layer 3 | 6× BasicBlock | 128→256 | 2 (first block) | 14×14 |
| Layer 4 | 3× BasicBlock | 256→512 | 2 (first block) | 7×7 |
| Head | GAP + FC | 512→K | — | K |

**Parameter count**: ~21.8M for K=1000. Changes only in the FC layer for different K.

**Projection shortcut**: Required when `stride ≠ 1` OR `in_channels ≠ out_channels`. Uses `Conv1×1 + BN` — no non-linearity. Not `Conv3×3` (that would add too many parameters and change the semantic meaning of the shortcut).

**CIFAR-10 adaptation**: Replace `Conv7×7 stride=2 + MaxPool` with `Conv3×3 stride=1` (no pooling). On 32×32 inputs, the 7×7 stem halves resolution twice (to 8×8), making later feature maps too small. The 3×3 stem preserves 32×32 through all of layer1.

---

## Implementation Decisions and Why

**Q: Why `bias=False` in conv layers before BatchNorm?**  
A: BatchNorm subtracts the batch mean, which would exactly cancel the conv bias during training. The bias adds a learned constant that gets immediately removed. After training, the bias contributes zero signal (it's absorbed by BN's `β` parameter). `bias=False` saves parameters with no loss of expressiveness.

**Q: Why ReLU after addition, not before?**  
A: If ReLU were applied before addition: `output = ReLU(F(x)) + x`. The shortcut input `x` can be negative; adding a non-negative `ReLU(F(x))` to a potentially negative `x` prevents learning subtractive residuals. The formula `F(x) + x` allows any real value, not just non-negative ones.

**Q: Why Global Average Pooling instead of fully connected flattening?**  
A: GAP replaces a massive dense layer. Without GAP, flattening `[B, 512, 7, 7]` = `[B, 25088]` → FC(25088, K) adds 25M parameters for K=1000. GAP reduces to `[B, 512]` → FC(512, K) = 512K parameters. Also makes the network input-size independent.

**Q: Why Kaiming Normal initialization for conv weights?**  
A: Xavier initialization assumes linear activations. ReLU approximately halves the effective input (zeros out negatives), halving the effective fan-in. Kaiming (He) accounts for this: `std = sqrt(2 / fan_out)` for `mode='fan_out'`. The `2` factor corrects for ReLU's halving effect.

---

## Comparison: BasicBlock vs Bottleneck

| | BasicBlock (ResNet-34) | Bottleneck (ResNet-50+) |
|---|---|---|
| Layers per block | 2 (3×3, 3×3) | 3 (1×1, 3×3, 1×1) |
| Purpose of 1×1 convs | — | Reduce then expand channels (bottleneck) |
| Parameters | ~73k per block (64-channel) | ~70k per block (256-channel) |
| Why used | Simpler, efficient at 34 layers | Needed at 50+ layers to control param count |

ResNet-34 uses BasicBlock because at 34 layers, the parameter budget is manageable without the bottleneck. At 50 layers with the same channel widths, the parameter count would be prohibitive.

---

## Key Experimental Results to Know

**From Table 6 (CIFAR-10)**:
- Plain-20: 8.82% error | Plain-56: 7.93% → degradation is real even at 56 layers
- ResNet-20: 8.75% error | ResNet-56: 6.97% → residuals fix degradation

Note: 56-layer plain (7.93%) is *worse* than 20-layer plain (8.82%) despite being trained longer. This is the smoking gun.

**From Table 1 (ImageNet)**:
- VGGNet-19: 7.32% top-5 error, 19.6B FLOPs
- ResNet-34: 7.02% top-5 error, 3.6B FLOPs → better accuracy, 5× fewer FLOPs

---

## Interview Questions and Answers

**Q: "What is the degradation problem?"**
A: The observation that adding more layers to a plain CNN increases *training* error, not just validation error. This rules out overfitting. It's an optimization problem: SGD cannot find the identity mapping for extra layers.

**Q: "How does the skip connection help mathematically?"**
A: The residual formulation adds 1 to the gradient signal: `∂L/∂x = ∂L/∂output · (∂F/∂x + 1)`. Even when the convolutional gradient `∂F/∂x → 0`, the total gradient remains `∂L/∂output`. Gradients propagate without vanishing.

**Q: "Could you achieve the same result by initializing the extra layers to identity?"**
A: In theory, yes. In practice, SGD starting from random initialization cannot find that solution reliably for 34+ layers. The skip connection makes the identity the *default* behavior at initialization (when `F(x) ≈ 0`), so the model starts near the right answer and refines from there.

**Q: "Why not just use 1×1 projections for all shortcuts?"**
A: The paper found that identity shortcuts (free — no parameters) work as well as 1×1 projections for same-dimension connections. Only when dimensions change (stride=2 or channel change) is a projection needed. The paper's ablation Table 3 shows minimal difference between options A (zero-pad), B (project only when needed), and C (project always). Option B is used as it balances cost and accuracy.
