# Error Analysis — ResNet-34 on Amazon Berkeley Objects

**Status**: Framework ready. Will be populated with real data after ABO training completes.

---

## What this document will contain

After ABO training finishes and inference runs on the test set, this document will include:

1. **Actual failure count** from the held-out test set
2. **Confusion matrix** — which categories are confused with which
3. **Grad-CAM analysis** — what the model attended to on misclassified images
4. **Categorised failure modes** — grouped by observable cause
5. **Proposed fixes** — linked to specific root causes

---

## Analysis Framework (designed, not yet executed)

When inference runs, each failure will be recorded as:

```
{
  "image_path": "...",
  "true_class": "CHAIR",
  "predicted_class": "STOOL",
  "confidence": 0.87,
  "top5": ["STOOL", "OTTOMAN", "CHAIR", "BENCH", "SIDE_TABLE"],
  "grad_cam_fas": 0.31    ← Foreground Attention Score
}
```

**Foreground Attention Score (FAS)**: fraction of Grad-CAM energy landing on
the product foreground vs background. Computed by thresholding the CAM at 0.5
and comparing to the image border region. FAS < 0.45 = background bias.

---

## Hypothesised failure categories (to be verified)

Based on known ABO dataset properties and general product image classification
literature — **these are predictions, not measured results**.

| Category | Hypothesis | How to verify |
|---|---|---|
| Visually similar pairs | High confusion between furniture subcategories (chair vs stool, vase vs pitcher) | Confusion matrix off-diagonals |
| Background bias | Model attends to studio backdrop, not product | FAS < 0.45 on misclassified samples |
| Occlusion / lifestyle images | ABO mixes hero shots with lifestyle photos | Check image_type metadata field |
| Class imbalance tail | Worst-performing classes are lowest-frequency | Correlation: class frequency vs per-class accuracy |

---

## Methodology (will follow when data is available)

1. Run `python scripts/evaluate.py --compute-grad-cam` on held-out test set
2. Collect all misclassified samples
3. Compute FAS for each failure using `src/explainability/gradcam.py`
4. Manually review 50 highest-confidence failures
5. Group into categories
6. Quantify each category as % of total failures
7. Write proposed fix for each category

---

## Update schedule

This file will be updated with real data after:
- [ ] ABO training completes
- [ ] Test set inference runs
- [ ] Manual review of 50 highest-confidence failures
- [ ] Grad-CAM visualizations generated

*Estimated update: within 1 week of ABO training completion.*
