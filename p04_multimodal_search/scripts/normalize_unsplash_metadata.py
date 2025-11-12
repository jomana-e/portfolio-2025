# scripts/normalize_unsplash_metadata.py

"""
Normalize Unsplash metadata to a consistent schema.
Handles both remote (URLs) and local (paths) formats gracefully.
"""

import os
import pandas as pd

DATA_DIR = "data/sources/unsplash"
CSV_IN = os.path.join(DATA_DIR, "unsplash_metadata.csv")
CSV_OUT = os.path.join(DATA_DIR, "unsplash_metadata_normalized.csv")

def main():
    print("🧹 Normalizing Unsplash metadata...")

    df = pd.read_csv(CSV_IN)
    cols = df.columns.tolist()

    # --- Case 1: Remote metadata (with image_url)
    if "image_url" in cols:
        df = df.rename(columns={"image_url": "image_path"})  # unify naming
        df["image_path"] = df["image_path"].astype(str).str.strip()
        df["caption"] = df["caption"].astype(str).str.strip()
        df["source"] = "unsplash"

    # --- Case 2: Local metadata (already downloaded)
    elif "image_path" in cols:
        df["image_path"] = df["image_path"].astype(str).str.replace("\\", "/", regex=False)
        df["caption"] = df["caption"].astype(str).str.strip()
        df["source"] = "unsplash"

    else:
        raise ValueError(f"❌ Unexpected columns: {cols}")

    # Keep only relevant columns
    df = df[["image_path", "caption", "source"]]

    df.to_csv(CSV_OUT, index=False)
    print(f"✅ Normalized {len(df)} entries → {CSV_OUT}")

if __name__ == "__main__":
    main()
