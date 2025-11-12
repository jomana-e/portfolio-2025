# scripts/build_fashion_metadata.py

"""
Build fashion_metadata.csv (richer captions) from the Kaggle Fashion Product Images dataset.
Combines fields like color, gender, season, and category into descriptive captions.
"""

import os
import pandas as pd

DATA_DIR = "data/sources/fashion"
IMG_DIR = os.path.join(DATA_DIR, "images")
CSV_IN = os.path.join(DATA_DIR, "styles.csv")
CSV_OUT = os.path.join(DATA_DIR, "fashion_metadata.csv")

def main():
    print("🧵 Building fashion metadata with rich captions...")

    df = pd.read_csv(CSV_IN, on_bad_lines="skip")
    df = df.dropna(subset=["id", "productDisplayName"])

    for col in ["gender", "baseColour", "masterCategory", "subCategory", "season", "usage"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
        else:
            df[col] = ""

    records = []
    for row in df.itertuples(index=False):
        image_file = f"{int(row.id)}.jpg"
        image_path = os.path.join(IMG_DIR, image_file)
        if not os.path.exists(image_path):
            continue

        # Build a natural-language caption
        parts = []

        # Gender and category
        if row.gender:
            parts.append(row.gender.capitalize())
        if row.baseColour:
            parts.append(row.baseColour.capitalize())
        if row.subCategory:
            parts.append(row.subCategory.capitalize())
        elif row.masterCategory:
            parts.append(row.masterCategory.capitalize())

        # Product name (main description)
        parts.append(str(row.productDisplayName).strip())

        # Contextual details
        extras = []
        if row.season:
            extras.append(row.season.capitalize())
        if row.usage:
            extras.append(row.usage.capitalize())
        if extras:
            parts.append(f"({' ,'.join(extras)})")

        caption = " ".join(parts).replace("  ", " ").strip()

        records.append((image_path, caption, "fashion"))

    meta_df = pd.DataFrame(records, columns=["image_path", "caption", "source"])
    meta_df.to_csv(CSV_OUT, index=False)

    print(f"✅ Saved {len(meta_df)} richly captioned entries → {CSV_OUT}")
    print("✨ Example captions:")
    print(meta_df.sample(5, random_state=42)["caption"].tolist())

if __name__ == "__main__":
    main()
