#!/bin/bash
# STEP 3 — Run this on Day 2 morning after both Colabs finish
# From inside your project folder: bash update_repo.sh

echo "=== Updating repo with real results ==="

# ── 1. Copy Grad-CAM images ───────────────────────────────────────
# Move your downloaded files to this folder first, then run this script
cp ~/Downloads/gradcam_correct.png results/gradcam/correct/
cp ~/Downloads/gradcam_failures.png results/gradcam/failures/
echo "✓ Grad-CAM images copied"

# ── 2. Copy ablation table ────────────────────────────────────────
cp ~/Downloads/ablation_table.csv results/ablation_table.csv
echo "✓ Ablation table updated"

# ── 3. Read real numbers from JSON ───────────────────────────────
echo ""
echo "=== Your real ABO numbers ==="
cat ~/Downloads/abo_results.json
echo ""
echo "=== Your real CIFAR-10 ablation numbers ==="
cat ~/Downloads/all_results.json | python3 -c "
import json,sys
data=json.load(sys.stdin)
for r in data:
    print(f\"  {r['label']:<35} {r['test_acc']:>7.2f}%  F1={r['f1']:.4f}\")
"

# ── 4. Push to GitHub ─────────────────────────────────────────────
git add .
git commit -m "Add real results: ABO training, Grad-CAM visualizations, complete ablation table"
git push

echo ""
echo "=== DONE ==="
echo "GitHub updated with real numbers and Grad-CAM images"
echo ""
echo "Now update your resume bullet with numbers from abo_results.json"
