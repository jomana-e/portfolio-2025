# scripts/build_coco_metadata.py

"""
Build COCO metadata CSV (resume-safe, no downloads).
Scans train/val annotations and links captions to image files.
"""

import os
import json
import pandas as pd

DATA_DIR = "data/sources/coco"
ANNOT_DIR = os.path.join(DATA_DIR, "annotations")
TRAIN_DIR = os.path.join(DATA_DIR, "train2017")
VAL_DIR = os.path.join(DATA_DIR, "val2017")
META_CSV = os.path.join(DATA_DIR, "coco_metadata.csv")

def build_metadata(limit=None):
    caption_files = [
        os.path.join(ANNOT_DIR, "captions_train2017.json"),
        os.path.join(ANNOT_DIR, "captions_val2017.json"),
    ]

    records = []
    for file in caption_files:
        if not os.path.exists(file):
            print(f"⚠️ Missing: {file}")
            continue

        print(f"📖 Reading {os.path.basename(file)} ...")
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_map = {img["id"]: img["file_name"] for img in data["images"]}
        for ann in data["annotations"]:
            img_file = img_map.get(ann["image_id"])
            if not img_file:
                continue

            img_path = (
                os.path.join(TRAIN_DIR, img_file)
                if "train" in file
                else os.path.join(VAL_DIR, img_file)
            )

            if not os.path.exists(img_path):
                continue

            records.append((img_path, ann["caption"].strip(), "coco"))

            if limit and len(records) >= limit:
                print(f"⚙️ Limit reached ({limit}), stopping early.")
                break
        if limit and len(records) >= limit:
            break

    df = pd.DataFrame(records, columns=["image_path", "caption", "source"])
    df.to_csv(META_CSV, index=False)
    print(f"✅ Saved {len(df)} entries → {META_CSV}")

def main():
    print("🧩 Building COCO caption metadata (no download)...")
    build_metadata()

if __name__ == "__main__":
    main()
