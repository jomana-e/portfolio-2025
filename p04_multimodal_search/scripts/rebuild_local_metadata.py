# scripts/rebuild_local_metadata.py

"""
Rebuild the local Unsplash metadata CSV based on existing image files.
"""

import os
import pandas as pd

DATA_DIR = "data/sources/unsplash"
IMG_DIR = os.path.join(DATA_DIR, "images")
META_CSV = os.path.join(DATA_DIR, "unsplash_metadata.csv")
LOCAL_CSV = os.path.join(DATA_DIR, "unsplash_metadata_local.csv")

def main():
    print("🔍 Rebuilding local Unsplash metadata from disk...")

    if not os.path.exists(META_CSV):
        raise FileNotFoundError(f"Missing {META_CSV}")
    if not os.path.exists(IMG_DIR):
        raise FileNotFoundError(f"Missing {IMG_DIR} folder")

    df_meta = pd.read_csv(META_CSV)
    print(f"🗂️  Loaded {len(df_meta)} metadata entries from {META_CSV}")

    image_files = [
        os.path.join(IMG_DIR, f)
        for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    print(f"🖼️  Found {len(image_files)} image files in {IMG_DIR}")

    if len(image_files) == 0:
        print("⚠️ No images found — nothing to rebuild.")
        return

    records = []
    for i, img_path in enumerate(sorted(image_files)):
        # Use metadata row if available
        meta_row = df_meta.iloc[i] if i < len(df_meta) else {}
        caption = meta_row.get("caption", "")
        author = meta_row.get("author", "")
        category = meta_row.get("category", "")
        source = meta_row.get("source", "unsplash")
        records.append((img_path, caption, author, category, source))

    df_local_new = pd.DataFrame(records, columns=["image_path", "caption", "author", "category", "source"])

    os.makedirs(DATA_DIR, exist_ok=True)
    df_local_new.to_csv(LOCAL_CSV, index=False)
    print(f"✅ Rebuilt {LOCAL_CSV} — total {len(df_local_new)} entries synced.")
    print("🎉 Local metadata is now fully in sync with your images folder.")

if __name__ == "__main__":
    main()
