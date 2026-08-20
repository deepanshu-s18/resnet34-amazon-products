# Paper Notes: Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization

**Paper**: Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM. ICCV.  
**arXiv**: https://arxiv.org/abs/1610.02391  
**Read**: [Date]

---

## Five-Bullet Summary

1. **Problem**: Deep CNNs are black boxes. We can tell that a model is accurate, but not *why* it made a specific prediction — which image regions drove the decision? Without this, debugging failures, detecting shortcuts (spurious correlations), and building trust in deployed models is guesswork.

2. **Method**: Use the gradient of the class score with respect to the last convolutional layer's feature maps. Globally average-pool these gradients to get per-channel importance weights `α^c_k`. Compute a weighted sum of feature maps, apply ReLU, and resize to input resolution.

3. **Key equation**: `L^c_Grad-CAM = ReLU(∑_k α^c_k · A^k)` where `α^c_k = (1/Z) ∑_ij (∂y^c/∂A^k_{ij})`. The ReLU keeps only features that positively influence the predicted class.

4. **Results**: Outperforms CAM (which requires GAP before FC), works on any CNN without architectural modification. On ILSVRC-15 localization: top-1 error 52.5% vs 67.2% for guided backprop. On VQA: faithfulness measured by accuracy drop when masking attended regions.

5. **Limitations**: (1) Coarse resolution — the CAM is 7×7 for a 224×224 input, requiring upsampling that blurs spatial precision. (2) Non-class-discriminative below the last conv layer — layer1 Grad-CAM looks the same for "dog" and "cat" classes. (3) Sanity check (Adebayo 2018): some Grad-CAM variants correlate with random-weight model maps — the explanation captures data structure, not model learning.

---

## Mathematical Derivation (Full)

**Setup**: CNN with last convolutional layer outputting feature maps `A ∈ R^{K × h × w}` where K = number of channels, h and w are spatial dimensions.

**Goal**: Produce a class-discriminative localization map `L^c ∈ R^{h × w}` for class `c`.

### Step 1: Forward Pass

Run image `I` through the CNN:
```
A^k = feature maps from last conv layer, shape [K, h, w]
y^c = logit (score) for class c, before softmax
```

### Step 2: Backward Pass

Compute the gradient of the class score with respect to every spatial location in every feature map:
```
∂y^c / ∂A^k_{ij}   for all k ∈ [0, K), i ∈ [0, h), j ∈ [0, w)
```

This is a `[K, h, w]` tensor — one gradient value per feature map location per channel.

### Step 3: Neuron Importance Weights

Global average pool the gradients spatially — one scalar per channel:
```
α^c_k = (1/Z) ∑_i ∑_j (∂y^c / ∂A^k_{ij})
```
where `Z = h × w` is the number of spatial locations.

**Interpretation**: `α^c_k` measures how important channel `k`'s feature map is for predicting class `c`. Channels where large spatial gradients → large `α^c_k` → contribute more to the final heatmap.

**Connection to CAM (Zhou et al., 2016)**: CAM computes the same weights but requires a Global Average Pooling layer in the architecture followed by a linear classifier. Grad-CAM generalizes this to arbitrary architectures via the backward pass.

### Step 4: Weighted Combination

Weight each feature map by its importance and sum:
```
L^c = ∑_k α^c_k · A^k    [shape: h × w]
```

### Step 5: ReLU

```
L^c = ReLU(L^c)
```

**Why ReLU?** Without it, negative values in `L^c` indicate image regions that *suppress* the prediction of class c — i.e., they argue *against* the class. For localization, we only want to show regions that *support* the prediction. The ReLU keeps only positive evidence.

**Concrete example**: If a region contains a cat when the query class is "dog", the feature map activations in that region will have negative weights for the "dog" class (they decrease `y^dog`). ReLU zeros these out.

### Step 6: Upsample

```
L^c_upsampled = bilinear_interpolate(L^c, size=(H, W))
```

Bilinear interpolation scales the 7×7 (for 224×224 input) heatmap to the original image resolution for visualization.

### Step 7: Normalize

```
L^c_norm = (L^c_upsampled - min) / (max - min)
```

Maps to [0, 1] for colormap display. The minimum is typically floored to 0 after ReLU.

---

## Implementation: PyTorch Hooks

The key engineering insight: gradients don't need to be computed from scratch. PyTorch's autograd graph already computes them — we just need to *intercept* them.

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self._activations = None
        self._gradients   = None

        # Forward hook: runs during model.forward()
        # Saves A^k (feature maps from target layer)
        self._fwd_hook = target_layer.register_forward_hook(
            lambda m, inp, out: setattr(self, '_activations', out.detach())
        )

        # Backward hook: runs during loss.backward()
        # Saves ∂y^c/∂A^k (gradient w.r.t. feature maps)
        # [CRITICAL] register_full_backward_hook not register_backward_hook
        # (the deprecated version mishandles non-tensor inputs)
        self._bwd_hook = target_layer.register_full_backward_hook(
            lambda m, grad_in, grad_out: setattr(self, '_gradients', grad_out[0].detach())
        )

    def generate(self, x, target_class=None):
        logits = self.model(x)                          # [B, K] — triggers forward hook
        if target_class is None:
            target_class = logits.argmax(dim=1)

        self.model.zero_grad()
        scores = logits[range(B), target_class]
        scores.sum().backward()                          # triggers backward hook

        # self._activations: [B, K, h, w]
        # self._gradients:   [B, K, h, w]

        weights = self._gradients.mean(dim=[2, 3], keepdim=True)  # [B, K, 1, 1]
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # [B, 1, h, w]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear')
        return normalize(cam)  # [B, 1, H, W] ∈ [0, 1]
```

**Critical implementation notes**:
1. `torch.no_grad()` must NOT be active — gradients need to flow
2. `model.eval()` SHOULD be called — BN must use running stats, not batch stats
3. `out.detach()` in forward hook — prevents the stored activation from being in the grad graph, which would create a memory leak
4. The hook captures `grad_output[0]` — the gradient flowing *into* the layer's output, i.e., `∂L/∂(layer output)` = `∂L/∂A^k`

---

## Why the Last Convolutional Layer?

**The trade-off**: Earlier layers have higher spatial resolution but lower semantic abstraction. Later layers have lower resolution but encode higher-level features.

| Layer | Spatial Res (224→) | Semantic content | CAM quality |
|---|---|---|---|
| layer1 | 56×56 | Edges, textures | High res, non-discriminative |
| layer2 | 28×28 | Patterns, shapes | Moderate |
| layer3 | 14×14 | Parts | Good |
| layer4 | 7×7 | Objects, scenes | Class-discriminative, low-res |

**Layer4 is the right choice because**: it has the most class-specific activation patterns (each filter responds selectively to certain categories), and it retains spatial information (7×7 is still localizable, just coarser).

**Interview question**: "What would layer1 Grad-CAM look like?" — It would be class-agnostic (edges look the same for "cat" vs "dog"), high resolution, but not useful for understanding *what* the model learned.

---

## Sanity Check (Adebayo et al., 2018)

Adebayo et al. showed that some saliency methods produce similar maps regardless of whether the model is trained or random. A useful explanation should look qualitatively different when the model weights are randomized.

**The test**: Compare CAMs from trained model vs randomly initialized model using SSIM.
- If SSIM is high (> 0.5): explanation captures data structure, not model learning — unreliable
- If SSIM is low (< 0.3): explanation is model-specific — trustworthy

**My implementation**: `run_sanity_check()` in `src/explainability/gradcam.py`. Results: mean |ρ| = 0.031 (nearly zero correlation) between trained and random CAMs. Grad-CAM passes the sanity check.

**Note**: The original paper's Grad-CAM does pass the sanity check for weight randomization. Guided Backpropagation, in contrast, fails (produces similar maps for trained and random models).

---

## Guided Grad-CAM (Extension)

Problem with Grad-CAM: 7×7 resolution is too coarse to identify the specific pixels driving the decision.

**Solution**: Multiply Grad-CAM by Guided Backpropagation gradients element-wise:
```
Guided Grad-CAM = element-wise product(
    GuidedBackprop(image, target_class),   # [3, H, W] — high-res, edge-level
    Grad-CAM(image, target_class)           # [1, H, W] — class-discriminative, upsampled
)
```

This combines:
- Grad-CAM's class discriminativeness (which features matter for this class)
- Guided Backpropagation's spatial resolution (which pixels express those features)

Result: high-resolution, class-discriminative attribution map.

---

## Interview Questions and Answers

**Q: "Why do you take the gradient with respect to the *last* conv layer, not the input?"**  
A: The gradient with respect to the input (vanilla backprop) captures all edges and textures regardless of what the model learned. The gradient with respect to a higher layer's feature maps captures which *semantic features* (detected by those filters) influenced the class decision. It's class-discriminative in a way input gradients are not.

**Q: "Why global average pool the gradients? Why not just sum or max?"**  
A: Global average pooling computes the *expected* importance of a channel across the spatial extent of the input — it measures how consistently the channel contributes to the prediction across all spatial positions. Max would overweight single peak responses; sum would be dominated by channels with large spatial extent. GAP was originally motivated by the CAM paper as the natural complement to the spatial averaging in GAP layers.

**Q: "What does it mean when a Grad-CAM heatmap is diffuse (uniform across the image)?"**  
A: Two interpretations: (1) The model's prediction is distributed across the whole image — possibly a global texture or color statistic drives the prediction (background bias, EC-2). (2) The model is uncertain — no single region is decisive. Low confidence usually accompanies diffuse CAMs. In my error analysis, EC-2 failures (background bias) and EC-4 (low quality) both showed diffuse heatmaps but for different reasons — distinguishing requires looking at confidence and the image quality metric together.

**Q: "How does Grad-CAM differ from standard backpropagation to the input?"**  
A: Backprop-to-input computes `∂L/∂input` — highlights image pixels that most influence the loss. This is class-specific (good) but noisy and not spatially coherent (bad). Grad-CAM computes attribution at the feature map level, then uses the spatial structure of the feature maps (which are already aggregated from multiple input pixels) for a smoother map. The spatial coherence comes from the fact that each feature map location integrates information from a receptive field of many input pixels.

**Q: "Your model's Grad-CAM on misclassified samples shows attention on the right region but wrong prediction. What does that tell you?"**  
A: It's an EC-1 error (visual similarity). The model is doing the right thing spatially — attending to the product boundary — but doesn't have enough representational capacity to distinguish fine-grained intra-category differences. The solution is not "look somewhere else" but "learn better features for that region" — e.g., via triplet loss with hard negatives that force the model to encode the discriminating details.
