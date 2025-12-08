import os
import tempfile
from io import BytesIO

import boto3
import faiss
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer

# --- Config ---
BUCKET_NAME = "portfolio-curated-jomana"
PROCESSED_PATH = "processed/multimodal_metadata_s3.csv"
IMAGE_INDEX_KEY = "indexes/faiss_image.index"
TEXT_INDEX_KEY = "indexes/faiss_text.index"
MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
CACHE_DIR = os.path.join(tempfile.gettempdir(), "faiss_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

st.set_page_config(page_title=" Multimodal Search", layout="wide")

# --- Helpers ---
@st.cache_resource(show_spinner=False)
def get_s3_client():
    try:
        aws = st.secrets.get("aws", None)
        if aws:
            session = boto3.session.Session(
                aws_access_key_id=aws["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=aws["AWS_SECRET_ACCESS_KEY"],
                region_name=aws.get("AWS_DEFAULT_REGION", "us-east-1"),
            )
            st.sidebar.success("✅ AWS credentials loaded from Streamlit secrets.")
            return session.client("s3")
        st.sidebar.warning("⚠️ No AWS credentials; using public URLs.")
        return None
    except Exception as e:
        st.sidebar.error(f"Failed to init AWS client: {e}")
        return None

s3_client = get_s3_client()

def download_http(url: str, dest: str):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)

def fetch_to_cache(key: str, local_name: str) -> str:
    dest = os.path.join(CACHE_DIR, local_name)
    if os.path.exists(dest):
        return dest
    if s3_client:
        s3_client.download_file(BUCKET_NAME, key, dest)
    else:
        url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{key}"
        download_http(url, dest)
    return dest

@st.cache_resource(show_spinner=True)
def load_faiss_indexes():
    img_path = fetch_to_cache(IMAGE_INDEX_KEY, "faiss_image.index")
    txt_path = fetch_to_cache(TEXT_INDEX_KEY, "faiss_text.index")
    try:
        img_idx = faiss.read_index(img_path)
        txt_idx = faiss.read_index(txt_path)
        return img_idx, txt_idx
    except Exception as e:
        st.error(f"Failed to read FAISS indexes: {e}")
        raise

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data(show_spinner=False)
def load_metadata_from_s3(bucket_name, key):
    if s3_client:
        obj = s3_client.get_object(Bucket=bucket_name, Key=key)
        return pd.read_csv(obj["Body"])
    url = f"https://{bucket_name}.s3.amazonaws.com/{key}"
    return pd.read_csv(url)

@st.cache_data(show_spinner=False)
def load_image_from_s3(bucket_name, key):
    try:
        if s3_client:
            obj = s3_client.get_object(Bucket=bucket_name, Key=key)
            return Image.open(BytesIO(obj["Body"].read()))
        url = f"https://{bucket_name}.s3.amazonaws.com/{key}"
        r = requests.get(url, stream=True, timeout=15)
        r.raise_for_status()
        return Image.open(r.raw)
    except Exception as e:
        raise RuntimeError(f"Could not load image {key}: {e}")

def search_index(index, query_vector, top_k=5):
    vec = np.array([query_vector], dtype="float32")
    distances, indices = index.search(vec, top_k)
    return indices[0], distances[0]

def get_s3_key_from_uri(s3_uri):
    if isinstance(s3_uri, str) and s3_uri.startswith("s3://"):
        return s3_uri.split("/", 3)[-1]
    return s3_uri

# --- UI ---
def main():
    st.title(" Multimodal Semantic Search")
    st.caption("Search across image, text, and metadata powered by CLIP and FAISS")

    model = load_model()
    st.sidebar.header(" Data Configuration")

    st.sidebar.write("Loading metadata from S3...")
    metadata = load_metadata_from_s3(BUCKET_NAME, PROCESSED_PATH)

    st.sidebar.write("Loading FAISS indexes (from S3)...")
    try:
        image_index, text_index = load_faiss_indexes()
    except Exception:
        st.stop()

    mode = st.sidebar.radio("Search mode", ["️ Image", " Text"], horizontal=True)

    if mode == " Text":
        query = st.text_input("Enter your text query:", "a stylish red dress")
        if st.button("Search"):
            with st.spinner("Encoding and searching..."):
                qv = model.encode([query], normalize_embeddings=True)[0]
                idxs, dists = search_index(text_index, qv)
                show_results(metadata, idxs, dists)
    else:
        uploaded = st.file_uploader("Upload an image to search similar ones", type=["jpg", "png", "jpeg"])
        if uploaded and st.button("Search"):
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded Image", use_container_width=True)
            with st.spinner("Encoding and searching..."):
                qv = model.encode([np.array(img)], normalize_embeddings=True)[0]
                idxs, dists = search_index(image_index, qv)
                show_results(metadata, idxs, dists)

def show_results(metadata, indices, distances):
    st.subheader(" Search Results")
    for i, idx in enumerate(indices):
        row = metadata.iloc[int(idx)]
        s3_uri = row.get("s3_path") or row.get("image_path")
        if pd.isna(s3_uri):
            continue
        s3_key = get_s3_key_from_uri(s3_uri)
        try:
            image = load_image_from_s3(BUCKET_NAME, s3_key)
            st.image(image, caption=f"{row.get('caption', '')}\n{row.get('source', '')}", use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Failed to load image: {s3_key} ({e})")
        st.write(f"**Source:** {row.get('source', 'Unknown')} | **Distance:** {distances[i]:.4f}")
        st.divider()

if __name__ == "__main__":
    main()
