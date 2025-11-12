# scripts/download_fashion.py

"""
Download Fashion Product Images (Kaggle dataset, resume-aware, polite, continuous)
"""

import os
import requests
import time
import pandas as pd
from tqdm import tqdm

DATA_DIR = "data/sources/fashion"
IMG_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

CSV_IN = os.path.join(DATA_DIR, "fashion-product-images-small.csv")
CSV_OUT = os.path.join(DATA_DIR, "fashion_metadata.csv")

BATCH_SIZE = 5000      # images per batch
SLEEP_BETWEEN = 0.2    # polite pacing
MAX_RETRIES = 3        # retry failed downloads


def safe_concat(new_df, csv_path):
    """Append new rows without duplicates."""
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        combined = pd.concat([old_df, new_df]).drop_duplicates(subset=["image_path"], keep="first")
    else:
        combined = new_df
    combined.to_csv(csv_path, index=False)
    return combined


def download_image(url, path):
    """Download an image safely with retry."""
    for _ in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(path, "wb") as f:
                    f.write(r.content)
                return True
        except Exception:
            time.sleep(1)
    return False


def main():
    df = pd.read_csv(CSV_IN, engine="python", on_bad_lines="skip")
    total_rows = len(df)

    existing_files = set(os.listdir(IMG_DIR))
    existing_count = len(existing_files)

    print(f"🧵 Starting fashion image download — {existing_count} files already present")
    print(f"📦 Dataset total: {total_rows} entries")

    start_index = existing_count

    while start_index < total_rows:
        end_index = min(start_index + BATCH_SIZE, total_rows)
        print(f"\n🚀 Downloading batch {start_index // BATCH_SIZE + 1} — rows {start_index} → {end_index}")

        new_records = []
        for i in tqdm(range(start_index, end_index)):
            row = df.iloc[i]
            filename = f"fashion_{i:05d}.jpg"
            path = os.path.join(IMG_DIR, filename)

            if filename in existing_files:
                continue  # already downloaded

            url = row.get("imageURL")
            caption = row.get("productDisplayName", "")

            ok = download_image(url, path)
            if ok:
                new_records.append((path, caption, "fashion"))
                existing_files.add(filename)

            time.sleep(SLEEP_BETWEEN)  # polite pacing

        # Save metadata for this batch
        if new_records:
            new_df = pd.DataFrame(new_records, columns=["image_path", "caption", "source"])
            combined = safe_concat(new_df, CSV_OUT)
            print(f"✅ Batch complete — {len(new_records)} new images. Total = {len(combined)}")
        else:
            print("⚠️ No new images this batch.")

        # Move to next batch
        start_index = end_index

        # If not finished, pause politely between batches
        if start_index < total_rows:
            print("⏸️ Sleeping 60 seconds before next batch...")
            time.sleep(60)

    print("\n🎉 All images downloaded successfully!")
    print(f"🧾 Total files: {len(os.listdir(IMG_DIR))}")


if __name__ == "__main__":
    main()
