# scripts/build_combined_metadata.py

"""
Combine all metadata sources (COCO, Fashion, Unsplash) into a unified CSV.
Adds full S3 paths for image access.
"""

import os
import pandas as pd

# Local and remote configuration
DATA_DIR = "data/sources"
OUT_DIR = "data/processed"
BUCKET_NAME = "portfolio-curated-jomana"
S3_BASE = f"s3://{BUCKET_NAME}/"

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


def convert_to_s3_path(local_path: str) -> str:
    """
    Convert a local-style path like 'data/sources/coco/train2017/xyz.jpg'
    to an S3 path like 's3://portfolio-curated-jomana/coco/train2017/xyz.jpg'.
    """
    if pd.isna(local_path) or not isinstance(local_path, str):
        return local_path

    # Remove local prefixes like 'data/sources/' or './data/sources/'
    for prefix in ["data/sources/", "./data/sources/"]:
        if local_path.startswith(prefix):
            local_path = local_path[len(prefix):]

    # Ensure consistent path structure
    local_path = local_path.lstrip("/")

    return f"{S3_BASE}{local_path}"


def main():
    print("🧵 Building unified multimodal metadata with S3 paths...")

    coco = load_metadata(os.path.join(DATA_DIR, "coco", "coco_metadata.csv"), "coco")
    fashion = load_metadata(os.path.join(DATA_DIR, "fashion", "fashion_metadata.csv"), "fashion")

    # Prefer normalized Unsplash if available
    unsplash_norm = os.path.join(DATA_DIR, "unsplash", "unsplash_metadata_normalized.csv")
    unsplash_orig = os.path.join(DATA_DIR, "unsplash", "unsplash_metadata_local.csv")
    unsplash_path = unsplash_norm if os.path.exists(unsplash_norm) else unsplash_orig
    unsplash = load_metadata(unsplash_path, "unsplash")

    # Combine all datasets
    combined = pd.concat([coco, fashion, unsplash], ignore_index=True)

    # Add S3-compatible paths
    combined["s3_path"] = combined["image_path"].apply(convert_to_s3_path)

    # Save both versions (local + S3)
    local_out = os.path.join(OUT_DIR, "multimodal_metadata.csv")
    s3_out = os.path.join(OUT_DIR, "multimodal_metadata_s3.csv")

    combined.to_csv(local_out, index=False)
    combined.to_csv(s3_out, index=False)

    print(f"📦 Combined total entries: {len(combined):,}")
    print(f"✅ Saved unified metadata → {s3_out}")
    print(combined.head())


if __name__ == "__main__":
    main()
