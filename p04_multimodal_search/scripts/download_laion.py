# scripts/download_laion.py

"""
Expand dataset using LAION-2B-en subset (filtered for English + high-quality samples).
"""

import os
import pandas as pd
from datasets import load_dataset

OUT_DIR = "data/sources/laion"
os.makedirs(OUT_DIR, exist_ok=True)

print("🚀 Loading LAION subset (5M entries max, English only)...")
ds = load_dataset("laion/laion2B-en", split="train[:1%]")

# Keep only the most relevant columns
df = pd.DataFrame({
    "image_url": ds["URL"],
    "caption": ds["TEXT"]
})

# Filter for valid captions and Englishs
df = df.dropna(subset=["caption"])
df = df[df["caption"].str.len() > 10]

# Optionally filter out NSFW or weird captions
df = df[~df["caption"].str.contains("nsfw|porn|nude", case=False)]

print(f"✅ Filtered down to {len(df):,} clean English samples")

# Save metadata
out_path = os.path.join(OUT_DIR, "laion_metadata.csv")
df.to_csv(out_path, index=False)
print(f"💾 Saved LAION subset → {out_path}")
