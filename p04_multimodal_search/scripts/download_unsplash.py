# scripts/download_unsplash.py

"""
Fetch Unsplash images + captions safely (auto-resume, API-limit aware)
Now supports --download_only mode to fetch missing images from metadata.
"""

import os
import sys
import time
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

SEARCH_TERMS = [
    "nature", "people", "technology", "architecture", "fashion", "animals",
    "travel", "food", "sports", "art", "business", "education", "music",
    "cars", "abstract", "mountains", "interior", "landscape", "city",
    "portrait", "street", "design", "minimal", "space", "macro", "winter"
]

PER_PAGE = 30
PAGES_PER_TERM = 10
REQUEST_LIMIT_PER_HOUR = 50
TARGET_TOTAL = 5000

DATA_DIR = "data/sources/unsplash"
IMG_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

META_CSV = os.path.join(DATA_DIR, "unsplash_metadata.csv")
LOCAL_CSV = os.path.join(DATA_DIR, "unsplash_metadata_local.csv")

def fetch_photos(query, page):
    """Fetch one page of Unsplash search results."""
    url = "https://api.unsplash.com/search/photos"
    params = {"query": query, "page": page, "per_page": PER_PAGE, "client_id": UNSPLASH_ACCESS_KEY}
    r = requests.get(url, params=params)
    if r.status_code != 200:
        print(f"⚠️ Failed request ({r.status_code}) for '{query}' page {page}: {r.text}")
        return []
    return r.json().get("results", [])


def download_image(url, path):
    """Download one image (skip if already exists)."""
    if os.path.exists(path):
        return True
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            return True
    except Exception:
        pass
    return False


def safe_concat(new_df, csv_path):
    """Append new rows without duplicates."""
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        key_col = "image_url" if "image_url" in old_df.columns else None
        if key_col:
            combined = pd.concat([old_df, new_df]).drop_duplicates(subset=[key_col], keep="first")
        else:
            combined = pd.concat([old_df, new_df])
    else:
        combined = new_df
    combined.to_csv(csv_path, index=False)
    return combined

def download_missing_images():
    """Check metadata and download any missing images."""
    if not os.path.exists(META_CSV):
        print("❌ No metadata file found. Run normally first to collect URLs.")
        return

    df = pd.read_csv(META_CSV)
    print(f"🖼 Found {len(df)} metadata entries. Checking for missing images...")
    downloaded = 0
    skipped = 0

    for i, row in tqdm(df.iterrows(), total=len(df)):
        url = row["image_url"]
        filename = f"unsplash_{i:05d}.jpg"
        path = os.path.join(IMG_DIR, filename)
        if os.path.exists(path):
            skipped += 1
            continue
        ok = download_image(url, path)
        if ok:
            downloaded += 1
            time.sleep(0.2)  # polite pacing

    print(f"✅ Download complete — {downloaded} new, {skipped} skipped (already present).")

def main(download_only=False):
    if download_only:
        return download_missing_images()

    if not UNSPLASH_ACCESS_KEY:
        raise EnvironmentError("Missing UNSPLASH_ACCESS_KEY in .env")

    existing_urls = set()
    if os.path.exists(META_CSV):
        existing_urls.update(pd.read_csv(META_CSV)["image_url"].tolist())

    total_images = len(existing_urls)
    print(f"📸 Starting Unsplash fetch — {total_images} URLs already saved")

    requests_made = 0
    start_time = time.time()
    all_records = []

    for term in SEARCH_TERMS:
        for page in range(1, PAGES_PER_TERM + 1):
            if total_images >= TARGET_TOTAL:
                print(f"🎯 Target of {TARGET_TOTAL} images reached. Stopping.")
                return

            elapsed = time.time() - start_time
            if requests_made >= REQUEST_LIMIT_PER_HOUR:
                if elapsed < 3600:
                    wait_time = 3600 - elapsed
                    print(f"⏸️ Reached hourly limit ({REQUEST_LIMIT_PER_HOUR} requests). Sleeping {int(wait_time // 60)} min...")
                    time.sleep(wait_time)
                requests_made = 0
                start_time = time.time()

            photos = fetch_photos(term, page)
            requests_made += 1
            if not photos:
                continue

            new_count = 0
            for p in photos:
                url = p["urls"]["regular"]
                if url in existing_urls:
                    continue
                caption = p.get("alt_description") or ""
                author = p["user"]["username"]
                all_records.append((url, caption, author, term, "unsplash"))
                existing_urls.add(url)
                new_count += 1

            total_images += new_count
            print(f"🔹 {term} / page {page}: +{new_count} new (total {total_images})")
            time.sleep(1)

            if len(all_records) >= 100:
                new_df = pd.DataFrame(all_records, columns=["image_url", "caption", "author", "category", "source"])
                safe_concat(new_df, META_CSV)
                all_records.clear()

    if all_records:
        new_df = pd.DataFrame(all_records, columns=["image_url", "caption", "author", "category", "source"])
        combined = safe_concat(new_df, META_CSV)
        print(f"✅ Session complete — {len(combined)} total metadata entries saved.")
    else:
        print("⚠️ No new records this run.")


if __name__ == "__main__":
    download_only_flag = "--download_only" in sys.argv
    main(download_only=download_only_flag)
