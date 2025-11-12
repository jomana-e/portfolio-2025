# scripts/generate_embeddings.py

"""
Generate CLIP embeddings for multimodal dataset (images + captions).
Outputs batched Parquet files for efficient FAISS indexing.
"""

import os
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pyarrow.parquet as pq
import pyarrow as pa

DATA_PATH = "data/processed/multimodal_metadata.csv"
OUT_DIR = "data/embeddings"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

print("🚀 Loading CLIP model (openai/clip-vit-base-patch32)...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

df = pd.read_csv(DATA_PATH)
print(f"📦 Loaded {len(df):,} entries from metadata")

# === DETERMINE START POINT (resume) ===
existing_batches = [f for f in os.listdir(OUT_DIR) if f.endswith(".parquet")]
start_batch = len(existing_batches)
print(f"🔁 Resuming from batch {start_batch}")

for batch_idx in range(start_batch, (len(df) // BATCH_SIZE) + 1):
    start, end = batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE
    batch = df.iloc[start:end]
    if batch.empty:
        break

    images, texts, image_paths = [], [], []

    for _, row in batch.iterrows():
        try:
            image = Image.open(row["image_path"]).convert("RGB")
            images.append(image)
            texts.append(str(row["caption"]))
            image_paths.append(row["image_path"])
        except Exception:
            continue

    if not images:
        continue

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        img_embeds = outputs.image_embeds.cpu().numpy()
        txt_embeds = outputs.text_embeds.cpu().numpy()

    # Save batch as Parquet
    data = pa.table({
        "image_path": image_paths,
        "caption": texts,
        "image_embeds": [emb.tolist() for emb in img_embeds],
        "text_embeds": [emb.tolist() for emb in txt_embeds]
    })
    out_path = os.path.join(OUT_DIR, f"embeddings_{batch_idx:05d}.parquet")
    pq.write_table(data, out_path)
    print(f"✅ Saved batch {batch_idx} → {out_path}")

print("🎉 Embedding generation complete!")
