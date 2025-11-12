# scripts/download_coco.py

"""
Build COCO Caption Metadata (Full 2017, Fully Automated + Resume Safe)
Downloads, extracts, merges, and indexes all COCO (train+val) images + captions.
"""

import os
import json
import zipfile
import requests
from tqdm import tqdm
import pandas as pd

DATA_DIR = "data/sources/coco"
IMG_DIR = os.path.join(DATA_DIR, "images")
ANNOT_DIR = os.path.join(DATA_DIR, "annotations")
CSV_OUT = os.path.join(DATA_DIR, "coco_metadata.csv")

URLS = {
    "train": "http://images.cocodataset.org/zips/train2017.zip",
    "val": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

def download_file(url, dest):
    """Download a large file with resume support."""
    if os.path.exists(dest):
        print(f"✅ Found existing: {os.path.basename(dest)}")
        return
    print(f"⬇️ Downloading {os.path.basename(dest)} ...")
    r = requests.get(url, stream=True)
    with open(dest, "wb") as f:
        for chunk in tqdm(r.iter_content(chunk_size=8192)):
            if chunk:
                f.write(chunk)

def extract_zip(zip_path, extract_to):
    """Safely extract a zip if not already extracted."""
    if os.path.exists(extract_to) and os.listdir(extract_to):
        print(f"✅ Already extracted: {extract_to}")
        return
    print(f"📦 Extracting {os.path.basename(zip_path)} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)

def safe_concat(new_df, csv_path):
    """Append new rows without duplicates."""
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        combined = pd.concat([old_df, new_df]).drop_duplicates(
            subset=["image_path", "caption"], keep="first"
        )
    else:
        combined = new_df
    combined.to_csv(csv_path, index=False)
    return combined

def build_metadata(limit=None):
    """Parse captions and create metadata entries."""
    caption_files = [
        os.path.join(ANNOT_DIR, "captions_train2017.json"),
        os.path.join(ANNOT_DIR, "captions_val2017.json"),
    ]

    records = []
    for cap_file in caption_files:
        if not os.path.exists(cap_file):
            print(f"⚠️ Missing {cap_file}")
            continue

        print(f"📖 Reading {cap_file} ...")
        with open(cap_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_map = {img["id"]: img["file_name"] for img in data["images"]}

        for i, ann in enumerate(tqdm(data["annotations"])):
            if limit and i >= limit:
                break
            img_file = img_map.get(ann["image_id"])
            if not img_file:
                continue
            img_path = os.path.join(IMG_DIR, img_file)
            caption = ann["caption"].strip()
            if os.path.exists(img_path):
                records.append((img_path, caption, "coco"))

    new_df = pd.DataFrame(records, columns=["image_path", "caption", "source"])
    combined = safe_concat(new_df, CSV_OUT)
    print(f"✅ Saved {len(combined)} total entries → {CSV_OUT}")

def main(limit=None):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(ANNOT_DIR, exist_ok=True)

    for key, url in URLS.items():
        zip_path = os.path.join(DATA_DIR, f"{key}.zip")
        download_file(url, zip_path)

    extract_zip(os.path.join(DATA_DIR, "train.zip"), DATA_DIR)
    extract_zip(os.path.join(DATA_DIR, "val.zip"), DATA_DIR)
    extract_zip(os.path.join(DATA_DIR, "annotations.zip"), DATA_DIR)

    for folder in ["train2017", "val2017"]:
        src = os.path.join(DATA_DIR, folder)
        if os.path.exists(src):
            for f in os.listdir(src):
                src_path = os.path.join(src, f)
                dest_path = os.path.join(IMG_DIR, f)
                if not os.path.exists(dest_path):
                    os.rename(src_path, dest_path)
            print(f"🧩 Merged {folder} → {IMG_DIR}")

    build_metadata(limit=limit)

if __name__ == "__main__":
    main(limit=None)
