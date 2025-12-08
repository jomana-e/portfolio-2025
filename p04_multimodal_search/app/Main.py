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

BUCKET_NAME = "portfolio-curated-jomana"
PROCESSED_PATH = "processed/multimodal_metadata_s3.csv"
IMAGE_INDEX_KEY = "indexes/faiss_image.index"
TEXT_INDEX_KEY = "indexes/faiss_text.index"
MODEL_NAME = "sentence-transformers/clip-ViT-B-32"

CACHE_DIR = os.path.join(tempfile.gettempdir(), "faiss_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

st.set_page_config(page_title=" Multimodal Search", layout="wide")

@st.cache_resource(show_spinner=False)
def get_s3_client():
    """Create a boto3 S3 client using Streamlit secrets if available."""
    try:
        aws_secrets = st.secrets.get("aws", None)
        if aws_secrets:
            session = boto3.session.Session(
                aws_access_key_id=aws_secrets["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=aws_secrets["AWS_SECRET_ACCESS_KEY"],
                region_name=aws_secrets.get("AWS_DEFAULT_REGION", "us-east-1"),
            )
            st.sidebar.success("✅ AWS credentials loaded from Streamlit secrets.")
            return session.client("s3")
        else:
            st.sidebar.warning("⚠️ No AWS credentials found in Streamlit secrets — using public URLs.")
            return None
    except Exception as e:
        st.sidebar.error(f"Failed to load AWS credentials: {e}")
        return None

s3_client = get_s3_client()

def download_to(path: str, url: str):
    """Download a file to path if it doesn't exist."""
    if os.path.exists(path):
        return path
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return path

@st.cache_resource(show_spinner=True)
def load_faiss_indexes():
    """Fetch indexes from S3 (public or using creds) into cache, then load FAISS."""
    img_local = os.path.join(CACHE_DIR, "faiss_image.index")
    txt_local = os.path.join(CACHE_DIR, "faiss_text.index")

    if s3_client:
        s3_client.download_file(BUCKET_NAME, IMAGE_INDEX_KEY, img_local)
        s3_client.download_file(BUCKET_NAME, TEXT_INDEX_KEY, txt_local)
    else:
        download_to(img_local, f"https://{BUCKET_NAME}.s3.amazonaws.com/{IMAGE_INDEX_KEY}")
        download_to(txt_local, f"https://{BUCKET_NAME}.s3.amazonaws.com/{TEXT_INDEX_KEY}")

    return faiss.read_index(img_local), faiss.read_index(txt_local)

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data(show_spinner=False)
def load_metadata_from_s3(bucket_name, key):
    if s3_client:
        obj = s3_client.get_object(Bucket=bucket_name, Key=key)
        return pd.read_csv(obj["Body"])
    else:
        url = f"https://{bucket_name}.s3.amazonaws.com/{key}"
        return pd.read_csv(url)

@st.cache_data(show_spinner=False)
def load_image_from_s3(bucket_name, key):
    """Load an image from S3 using credentials or public URL."""
    try:
        if s3_client:
            obj = s3_client.get_object(Bucket=bucket_name, Key=key)
            return Image.open(BytesIO(obj["Body"].read()))
        else:
            url = f"https://{bucket_name}.s3.amazonaws.com/{key}"
            response = requests.get(url, stream=True)
            response.raise_for_status()
            return Image.open(response.raw)
    except Exception as e:
        raise RuntimeError(f"Could not load image {key}: {e}")

def search_index(index, query_vector, top_k=5):
    distances, indices = index.search(np.array([query_vector]).astype("float32"), top_k)
    return indices[0], distances[0]

def get_s3_key_from_uri(s3_uri):
    if s3_uri.startswith("s3://"):
        return s3_uri.split("/", 3)[-1]
    return s3_uri

def main():
    st.title(" Multimodal Semantic Search")
    st.caption("Search across image, text, and metadata powered by CLIP and FAISS")

    model = load_model()
    st.sidebar.header(" Data Configuration")

    st.sidebar.write("Loading metadata from S3...")
    metadata = load_metadata_from_s3(BUCKET_NAME, PROCESSED_PATH)

    st.sidebar.write("Loading FAISS indexes...")
    image_index, text_index = load_faiss_indexes()

    mode = st.sidebar.radio("Search mode", ["️ Image", " Text"], horizontal=True)

    if mode == " Text":
        query = st.text_input("Enter your text query:", "a stylish red dress")
        if st.button("Search"):
            with st.spinner("Encoding and searching..."):
                query_vector = model.encode([query], normalize_embeddings=True)[0]
                indices, distances = search_index(text_index, query_vector)
                show_results(metadata, indices, distances)
    else:
        uploaded = st.file_uploader("Upload an image to search similar ones", type=["jpg", "png", "jpeg"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded Image", use_container_width=True)
            if st.button("Search"):
                with st.spinner("Encoding and searching..."):
                    query_vector = model.encode([np.array(img)], normalize_embeddings=True)[0]
                    indices, distances = search_index(image_index, query_vector)
                    show_results(metadata, indices, distances)

def show_results(metadata, indices, distances):
    st.subheader(" Search Results")
    for i, idx in enumerate(indices):
        row = metadata.iloc[idx]
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
