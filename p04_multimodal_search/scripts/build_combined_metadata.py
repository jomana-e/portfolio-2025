# scripts/build_combined_metadata.py

"""
Combine all metadata sources (COCO, Fashion, Unsplash) into a unified CSV.
Supports flexible schema detection.
"""

import os
import pandas as pd

DATA_DIR = "data/sources"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

def load_metadata(path, source_name):
    """Load and normalize a metadata file safely."""
    if not os.path.exists(path):
        print(f"⚠️ Missing {path} — skipping.")
        return pd.DataFrame(columns=["image_path", "caption", "source"])

    df = pd.read_csv(path)

    cols = [c.lower() for c in df.columns]
    if "image_path" not in cols:
        print(f"⚠️ Invalid format for {source_name} metadata — skipping.")
        return pd.DataFrame(columns=["image_path", "caption", "source"])

    # Rename columns for consistency
    df.columns = cols
    df = df[["image_path", "caption", "source"]].dropna(subset=["image_path", "caption"])

    print(f"✅ Loaded {len(df)} entries from {source_name}")
    return df

def main():
    print("🧵 Building unified multimodal metadata...")

    coco = load_metadata(os.path.join(DATA_DIR, "coco", "coco_metadata.csv"), "coco")
    fashion = load_metadata(os.path.join(DATA_DIR, "fashion", "fashion_metadata.csv"), "fashion")

    # Prefer normalized Unsplash if available
    unsplash_norm = os.path.join(DATA_DIR, "unsplash", "unsplash_metadata_normalized.csv")
    unsplash_orig = os.path.join(DATA_DIR, "unsplash", "unsplash_metadata_local.csv")
    unsplash_path = unsplash_norm if os.path.exists(unsplash_norm) else unsplash_orig
    unsplash = load_metadata(unsplash_path, "unsplash")

    combined = pd.concat([coco, fashion, unsplash], ignore_index=True)
    combined.to_csv(os.path.join(OUT_DIR, "multimodal_metadata.csv"), index=False)

    print(f"📦 Combined total entries: {len(combined):,}")
    print(f"✅ Saved unified metadata → {OUT_DIR}/multimodal_metadata.csv")
    print(combined.head())

if __name__ == "__main__":
    main()
