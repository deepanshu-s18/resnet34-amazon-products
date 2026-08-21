# ABO Dataset Setup Guide

This guide walks you through downloading the Amazon Berkeley Objects (ABO) dataset
and preparing it for training with this project.

**Time required**: ~30 minutes (mostly download time)  
**Disk required**: ~3GB  
**Skills needed**: Basic terminal + Google Drive

---

## Step 1: Download from Amazon Berkeley Objects

The ABO dataset is publicly available. You need two files:

```bash
# Create a working directory
mkdir abo_raw && cd abo_raw

# Download the listings (metadata + labels) — ~300MB
wget https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar

# Download the small images — ~2.3GB
wget https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar
```

> **Alternative**: Download via the AWS CLI if you have it installed:
> ```bash
> aws s3 cp s3://amazon-berkeley-objects/archives/abo-listings.tar . --no-sign-request
> aws s3 cp s3://amazon-berkeley-objects/archives/abo-images-small.tar . --no-sign-request
> ```

---

## Step 2: Extract the Archives

```bash
# Extract listings (creates listings/metadata/ with .json.gz files)
tar -xf abo-listings.tar

# Extract images (creates images/small/ with subfolders aa/, ab/, ...)
tar -xf abo-images-small.tar

# Verify structure
ls listings/metadata/    # → abo_listings_*.json.gz files
ls images/small/         # → aa/ ab/ ac/ ... subfolders
```

---

## Step 3: Prepare the Dataset Splits

Run the preparation script from the repo root:

```bash
cd /path/to/resnet34-amazon-products

python prepare_abo_splits.py \
    --listings-dir /path/to/abo_raw/listings/metadata \
    --images-dir   /path/to/abo_raw/images/small \
    --output-dir   abo_prepared \
    --top-k        50

# Takes ~3 minutes. Prints a summary when done.
```

**Output files in `abo_prepared/`:**

| File | Size | Description |
|------|------|-------------|
| `abo_splits.csv` | ~5MB | All image paths + labels + train/val/test split |
| `abo_label_map.json` | ~2KB | Category name → integer label mapping |
| `abo_stats.json` | ~1KB | Dataset statistics (class counts, imbalance ratio) |

You should see output like:
```
Parsed 147,702 valid records
Selected 50 categories (top-50, min 50 each)
Filtered dataset: 58,337 samples

ABO DATASET PREPARED
  Total samples:    58,337
  Train:            40,835
  Val:               8,751
  Test:              8,751
  Categories:       50
  Imbalance ratio:  8.2x
```

---

## Step 4: Upload to Google Drive (for Colab Training)

You only need to upload **3 things** — NOT the full image folder:

```
Google Drive/
└── abo_dataset/              ← create this folder
    ├── abo_splits.csv        ← upload this
    ├── abo_label_map.json    ← upload this
    └── images/               ← upload the entire images/ folder
        └── small/
            ├── aa/
            ├── ab/
            └── ...
```

> ⚠️ The images folder is ~2.3GB. Use Google Drive for Desktop or
> the Google Drive web interface to upload it. Estimated upload time:
> 15-30 minutes depending on your connection.

---

## Step 5: Run ABO Training in Colab

1. Open Google Colab: https://colab.research.google.com
2. Select **Runtime → Change runtime type → T4 GPU**
3. Copy-paste `COLAB_TAB1_ABO.py` into a notebook cell
4. Update this line at the top to match your Drive folder:
   ```python
   DRIVE_ROOT = "/content/drive/MyDrive/abo_dataset"  # ← change if needed
   ```
5. Run Cell 1 (Mount Drive), Cell 2 (verify files), Cell 3 (train)

**Expected training time**: ~2-3 hours for 100 epochs on T4

**Expected output** (experiment E-003):
```
ABO FINAL RESULTS
Test Accuracy : ~83-85%
F1 Macro      : ~0.83
```

---

## Step 6: Download Results and Commit

After training, download from `/tmp/` in Colab's file panel:
- `abo_resnet34_best.pt` → commit to `checkpoints/`
- `abo_results.json` → commit to `results/`
- `gradcam_correct.png` → commit to `results/gradcam/`
- `gradcam_failures.png` → commit to `results/gradcam/`

Then regenerate meaningful Grad-CAM images with trained weights:
```bash
python scripts/generate_gradcam_demo.py \
    --checkpoint checkpoints/abo_resnet34_best.pt \
    --layer layer4
```

---

## Troubleshooting

**"listings dir not found"**: Make sure you extracted `abo-listings.tar` and point `--listings-dir` to the `listings/metadata/` subdirectory (not the `.tar` file itself).

**"No records found"**: The listing files are `.json.gz` (gzipped JSON). Don't decompress them manually — `prepare_abo_splits.py` reads them directly.

**Colab: "images not found"**: Verify your Drive structure matches exactly:
```python
import os
DRIVE_ROOT = "/content/drive/MyDrive/abo_dataset"
print(os.path.exists(f"{DRIVE_ROOT}/abo_splits.csv"))    # → True
print(os.path.exists(f"{DRIVE_ROOT}/images/small"))      # → True
```

**OOM on T4**: Reduce batch size in the Colab script from 64 to 32:
```python
tr_ld = DataLoader(tr_ds, 32, ...)  # was 64
```

---

## Dataset Citation

If you use ABO in published work, cite:
```
@article{collins2022abo,
  title={ABO: Dataset and Benchmarks for Real-World 3D Object Understanding},
  author={Collins, Jasmine and Goel, Shubham and Deng, Kenan and Luthra, Achleshwar
          and Xu, Leon and Gundogdu, Ersin and Zhang, Xi and Yago Vicente, Tomas Farre
          and Dideriksen, Thomas and Shapiro, Himanshu and others},
  journal={CVPR},
  year={2022}
}
```

Dataset page: https://amazon-berkeley-objects.s3.amazonaws.com/index.html
