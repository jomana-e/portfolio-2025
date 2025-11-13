# app/Main.py

import os
import streamlit as st
import pandas as pd
import boto3
from io import BytesIO
from PIL import Image
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BUCKET_NAME = "portfolio-curated-jomana"
PROCESSED_PATH = "processed/multimodal_metadata_s3.csv"
INDEX_DIR = "indexes"
MODEL_NAME = "sentence-transformers/clip-ViT-B-32"

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource(show_spinner=False)
def load_faiss_index(index_path):
    index = faiss.read_index(index_path)
    return index

@st.cache_data(show_spinner=False)
def load_metadata_from_s3(bucket_name, key):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    return pd.read_csv(obj["Body"])

@st.cache_data(show_spinner=False)
def load_image_from_s3(bucket_name, key):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    return Image.open(BytesIO(obj["Body"].read()))

def search_index(index, query_vector, top_k=5):
    distances, indices = index.search(np.array([query_vector]), top_k)
    return indices[0], distances[0]

def get_s3_key_from_uri(s3_uri):
    if s3_uri.startswith("s3://"):
        return s3_uri.split("/", 3)[-1]
    return s3_uri

def main():
    st.set_page_config(page_title="🧠 Multimodal Search", layout="wide")
    st.title("🔍 Multimodal Semantic Search")
    st.caption("Search across image, text, and metadata powered by CLIP and FAISS")

    model = load_model()
    st.sidebar.header("📂 Data Configuration")

    st.sidebar.write("Loading metadata from S3...")
    metadata = load_metadata_from_s3(BUCKET_NAME, PROCESSED_PATH)

    st.sidebar.write("Loading FAISS indexes...")
    image_index_path = os.path.join(INDEX_DIR, "image_index.faiss")
    text_index_path = os.path.join(INDEX_DIR, "text_index.faiss")
    image_index = load_faiss_index(image_index_path)
    text_index = load_faiss_index(text_index_path)

    mode = st.sidebar.radio("Search mode", ["🖼️ Image", "💬 Text"], horizontal=True)

    if mode == "💬 Text":
        query = st.text_input("Enter your text query:", "a stylish red dress")
        if st.button("Search"):
            with st.spinner("Encoding and searching..."):
                query_vector = model.encode([query], normalize_embeddings=True)
                indices, distances = search_index(text_index, query_vector[0])
                show_results(metadata, indices, distances)
    else:
        uploaded = st.file_uploader("Upload an image to search similar ones", type=["jpg", "png", "jpeg"])
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Image", use_container_width=True)
            if st.button("Search"):
                with st.spinner("Encoding and searching..."):
                    query_vector = model.encode([np.array(img)], normalize_embeddings=True)
                    indices, distances = search_index(image_index, query_vector[0])
                    show_results(metadata, indices, distances)

def show_results(metadata, indices, distances):
    st.subheader("📸 Search Results")
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
