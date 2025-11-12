# scripts/build_faiss_index.py

"""
Build FAISS indexes from CLIP embeddings (image + text).
"""

import os
import numpy as np
import pandas as pd
import faiss
import pyarrow.parquet as pq
from transformers import CLIPProcessor, CLIPModel
import torch

EMB_DIR = "data/embeddings"
INDEX_DIR = "data/indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("📦 Loading all embedding batches...")
tables = [pq.read_table(os.path.join(EMB_DIR, f)) for f in os.listdir(EMB_DIR) if f.endswith(".parquet")]
df = pd.concat([t.to_pandas() for t in tables])
print(f"✅ Loaded {len(df):,} embeddings from {len(tables)} batches")

image_embeds = np.vstack(df["image_embeds"].apply(lambda x: np.array(x, dtype=np.float32)))
text_embeds = np.vstack(df["text_embeds"].apply(lambda x: np.array(x, dtype=np.float32)))

d = image_embeds.shape[1]
index_img = faiss.IndexFlatIP(d)
index_txt = faiss.IndexFlatIP(d)

# Normalize for cosine similarity
faiss.normalize_L2(image_embeds)
faiss.normalize_L2(text_embeds)

index_img.add(image_embeds)
index_txt.add(text_embeds)

faiss.write_index(index_img, os.path.join(INDEX_DIR, "faiss_image.index"))
faiss.write_index(index_txt, os.path.join(INDEX_DIR, "faiss_text.index"))
print("💾 FAISS indexes saved successfully.")

def search(query, top_k=5):
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        q_emb = model.get_text_features(**inputs).cpu().numpy()
    faiss.normalize_L2(q_emb)
    D, indices = index_img.search(q_emb, top_k)
    print(f"\n🔍 Query: {query}")
    for idx in indices[0]:
        print("→", df.iloc[idx]["image_path"], ":", df.iloc[idx]["caption"])

if __name__ == "__main__":
    search("a red sports car", top_k=5)
