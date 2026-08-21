"""
STEP 2 — Run this locally after extracting both tar files.

Usage:
    python prepare_abo_splits.py \
        --listings-dir listings/metadata \
        --images-dir   images/small \
        --output-dir   abo_prepared \
        --top-k        50

Output files (small — upload these to Google Drive):
    abo_prepared/abo_splits.csv        ← image paths + labels + split
    abo_prepared/abo_label_map.json    ← category name → integer label
    abo_prepared/abo_stats.json        ← dataset statistics

Takes ~3 minutes to run.
"""

import argparse, gzip, json, os, sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("pip install scikit-learn pandas")
    sys.exit(1)


def get_english_value(field_list):
    """Extract English value from ABO's multi-language field list."""
    if not field_list:
        return None
    for item in field_list:
        if isinstance(item, dict):
            tag = item.get("language_tag", "")
            if tag.startswith("en"):
                return item.get("value")
    # Fallback: return first value regardless of language
    if isinstance(field_list[0], dict):
        return field_list[0].get("value")
    return None


def load_image_lookup(images_dir):
    """Load images/metadata/images.csv.gz → {image_id: absolute_path}.
    ABO actual file paths don't follow image_id[:2] — they use a content hash.
    The images.csv.gz has the correct mapping."""
    csv_path = Path(images_dir).parent / "metadata" / "images.csv.gz"
    if not csv_path.exists():
        print(f"ERROR: images metadata not found at {csv_path}")
        print("Expected: images/metadata/images.csv.gz inside your abo_raw folder")
        sys.exit(1)

    import csv
    lookup = {}  # image_id → absolute path string
    images_small = Path(images_dir)  # e.g. ~/abo_raw/images/small
    with gzip.open(csv_path, 'rt', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row['image_id']
            rel_path = row['path']  # e.g. '14/14fe8812.jpg'
            abs_path = images_small / rel_path
            if abs_path.exists():
                lookup[img_id] = str(abs_path)
    print(f"Loaded {len(lookup):,} image paths from images.csv.gz")
    return lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listings-dir", default="listings/metadata",
                        help="Path to extracted listings/metadata/ folder")
    parser.add_argument("--images-dir",   default="images/small",
                        help="Path to extracted images/small/ folder")
    parser.add_argument("--output-dir",   default="abo_prepared")
    parser.add_argument("--top-k",        type=int, default=50,
                        help="Number of top categories to keep")
    parser.add_argument("--min-per-class",type=int, default=50,
                        help="Minimum images per category")
    parser.add_argument("--max-per-class",type=int, default=2000,
                        help="Max images per category (caps dominant classes, 0=unlimited)")
    args = parser.parse_args()

    listings_dir = Path(args.listings_dir)
    images_dir   = Path(args.images_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Validate paths ───────────────────────────────────────────────────
    if not listings_dir.exists():
        print(f"ERROR: listings dir not found: {listings_dir}")
        print("Make sure you ran: tar -xf abo-listings.tar")
        sys.exit(1)
    if not images_dir.exists():
        print(f"ERROR: images dir not found: {images_dir}")
        print("Make sure you ran: tar -xf abo-images-small.tar")
        sys.exit(1)

    # ── Load image ID → file path lookup ────────────────────────────────
    image_lookup = load_image_lookup(images_dir)

    # ── Parse all listing files ──────────────────────────────────────────
    gz_files = sorted(listings_dir.glob("*.json.gz"))
    if not gz_files:
        gz_files = sorted(listings_dir.glob("**/*.json.gz"))
    print(f"Found {len(gz_files)} listing files in {listings_dir}")

    records = []
    skipped_no_type = skipped_no_image = skipped_missing_file = 0

    for i, gz_file in enumerate(gz_files):
        print(f"  Parsing {gz_file.name} ({i+1}/{len(gz_files)})...", end="\r")
        with gzip.open(gz_file, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Product type (category label)
                ptype = get_english_value(item.get("product_type", []))
                if not ptype:
                    skipped_no_type += 1
                    continue

                # Main image
                main_img = item.get("main_image_id")
                if not main_img:
                    skipped_no_image += 1
                    continue

                # Verify image file exists using lookup table
                img_path = image_lookup.get(main_img)
                if not img_path:
                    skipped_missing_file += 1
                    continue

                # Item name (optional, for display)
                name = get_english_value(item.get("item_name", []))

                records.append({
                    "item_id":      item.get("item_id", ""),
                    "product_type": ptype.upper().strip(),
                    # Store path relative to images/small/ for portability
                    # e.g. "14/14fe8812.jpg" not "/Users/.../images/small/14/14fe8812.jpg"
                    "image_path":   os.path.relpath(img_path, str(images_dir)),
                    "item_name":    name or "",
                })

    print(f"\nParsed {len(records):,} valid records")
    print(f"  Skipped (no category):  {skipped_no_type:,}")
    print(f"  Skipped (no image ID):  {skipped_no_image:,}")
    print(f"  Skipped (file missing): {skipped_missing_file:,}")

    if len(records) == 0:
        print("\nERROR: No records found. Check your paths.")
        sys.exit(1)

    df = pd.DataFrame(records)

    # ── Select top-K categories ──────────────────────────────────────────
    cat_counts = df["product_type"].value_counts()
    print(f"\nTotal unique categories: {len(cat_counts)}")
    print(f"Top 10 categories:")
    for cat, cnt in cat_counts.head(10).items():
        print(f"  {cat:<40} {cnt:>6} images")

    # Filter: top-K categories with minimum samples
    valid_cats = cat_counts[cat_counts >= args.min_per_class].head(args.top_k)
    top_cats   = valid_cats.index.tolist()
    print(f"\nSelected {len(top_cats)} categories (top-{args.top_k}, min {args.min_per_class} each)")

    df = df[df["product_type"].isin(top_cats)].reset_index(drop=True)
    print(f"Filtered dataset: {len(df):,} samples")

    # ── Cap dominant classes ─────────────────────────────────────────────
    if args.max_per_class > 0:
        df = (df.groupby("product_type", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), args.max_per_class),
                                          random_state=42),
                       include_groups=False)
                .reset_index(drop=True))
        new_imb = df["product_type"].value_counts()
        actual_ratio = new_imb.iloc[0] / new_imb.iloc[-1]
        print(f"Capped at {args.max_per_class}/class → {len(df):,} samples  "
              f"(imbalance now {actual_ratio:.1f}x)")

    # ── Create integer labels ────────────────────────────────────────────
    sorted_cats = sorted(top_cats)
    label_map   = {cat: i for i, cat in enumerate(sorted_cats)}
    df["label"] = df["product_type"].map(label_map)

    # ── Stratified train / val / test split ─────────────────────────────
    # 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42)

    train_df = train_df.copy(); train_df["split"] = "train"
    val_df   = val_df.copy();   val_df["split"]   = "val"
    test_df  = test_df.copy();  test_df["split"]  = "test"

    final_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

    # ── Save outputs ─────────────────────────────────────────────────────
    splits_path = output_dir / "abo_splits.csv"
    final_df.to_csv(splits_path, index=False)
    print(f"\nSaved splits → {splits_path}")

    label_map_path = output_dir / "abo_label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Saved label map → {label_map_path}")

    # Class distribution check
    per_class = final_df.groupby("label")["split"].value_counts().unstack(fill_value=0)

    stats = {
        "total_samples":    len(final_df),
        "train_samples":    len(train_df),
        "val_samples":      len(val_df),
        "test_samples":     len(test_df),
        "num_classes":      len(top_cats),
        "categories":       sorted_cats,
        "min_class_count":  int(per_class.sum(axis=1).min()),
        "max_class_count":  int(per_class.sum(axis=1).max()),
        "imbalance_ratio":  round(per_class.sum(axis=1).max() /
                                  per_class.sum(axis=1).min(), 2),
    }
    stats_path = output_dir / "abo_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # ── Print summary ────────────────────────────────────────────────────
    print(f"""
{'='*55}
  ABO DATASET PREPARED
{'='*55}
  Total samples:    {stats['total_samples']:,}
  Train:            {stats['train_samples']:,}
  Val:              {stats['val_samples']:,}
  Test:             {stats['test_samples']:,}
  Categories:       {stats['num_classes']}
  Imbalance ratio:  {stats['imbalance_ratio']}x
{'='*55}

NEXT STEP — Upload these to Google Drive:
  1. {splits_path}          (small CSV)
  2. {label_map_path}       (small JSON)
  3. The entire images/ folder  (2.3GB)

Then open Colab → paste colab_train_abo.py → run.
{'='*55}
""")


if __name__ == "__main__":
    main()
