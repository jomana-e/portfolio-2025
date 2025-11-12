# app/Main.py

import os
import streamlit as st
import pandas as pd
import torch
import faiss
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

st.set_page_config(page_title="Multimodal AI Search Engine", layout="wide")

# --- Paths ---
META_PATH = "data/processed/multimodal_metadata.csv"
FAISS_DIR = "data/indexes"
FAISS_IMG_INDEX = os.path.join(FAISS_DIR, "faiss_image.index")
FAISS_TXT_INDEX = os.path.join(FAISS_DIR, "faiss_text.index")

os.makedirs(FAISS_DIR, exist_ok=True)

# --- Load CLIP model ---
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


# --- Download and load FAISS indexes from S3 ---
@st.cache_resource
def load_faiss_indexes():
    bucket_name = "portfolio-curated-jomana"
    s3_keys = {
        FAISS_IMG_INDEX: "indexes/faiss_image.index",
        FAISS_TXT_INDEX: "indexes/faiss_text.index",
    }

    st.sidebar.write("🪣 Checking S3 credentials...")

    try:
        # Prefer Streamlit secrets
        if "aws" in st.secrets:
            aws_cfg = st.secrets["aws"]
            s3 = boto3.client(
                "s3",
                aws_access_key_id=aws_cfg["aws_access_key_id"],
                aws_secret_access_key=aws_cfg["aws_secret_access_key"],
                region_name=aws_cfg.get("region_name", "us-east-1"),
            )
            st.sidebar.success("✅ AWS credentials loaded from secrets.")
        else:
            s3 = boto3.client("s3")
            st.sidebar.warning("⚠️ Using default AWS credentials (none found in secrets).")
    except Exception as e:
        st.error(f"❌ Failed to initialize S3 client: {e}")
        raise

    # Download missing indexes
    for local_path, s3_key in s3_keys.items():
        if not os.path.exists(local_path):
            st.sidebar.info(f"📥 Downloading {os.path.basename(local_path)} from S3...")
            try:
                s3.download_file(bucket_name, s3_key, local_path)
                st.sidebar.success(f"✅ {os.path.basename(local_path)} downloaded.")
            except NoCredentialsError:
                st.error("❌ No AWS credentials found. Please set them in Streamlit secrets.")
                raise
            except ClientError as e:
                st.error(f"❌ AWS ClientError: {e}")
                raise
            except Exception as e:
                st.error(f"❌ Unexpected error downloading {s3_key}: {e}")
                raise

    # Load FAISS indexes
    try:
        img_index = faiss.read_index(FAISS_IMG_INDEX)
        txt_index = faiss.read_index(FAISS_TXT_INDEX)
        st.sidebar.success("✅ FAISS indexes loaded successfully.")
        return img_index, txt_index
    except Exception as e:
        st.error(f"❌ Failed to load FAISS indexes: {e}")
        raise


# --- Load metadata ---
@st.cache_data
def load_metadata():
    df = pd.read_csv(META_PATH)
    df["image_path"] = df["image_path"].astype(str).str.replace("\\", "/", regex=False)
    return df


# --- Main ---
st.sidebar.success("🚀 Initializing app...")
model, processor = load_model()
img_index, txt_index = load_faiss_indexes()
metadata = load_metadata()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

st.title("🧠 Multimodal AI Search Engine")
st.markdown("Search across images and captions using **CLIP + FAISS** embeddings.")

mode = st.sidebar.radio("Choose mode:", ["Text → Image", "Image → Text"])
top_k = st.sidebar.slider("Number of results", 1, 20, 5)
st.sidebar.info("✅ Ready for multimodal search.")


# --- TEXT → IMAGE SEARCH ---
if mode == "Text → Image":
    query = st.text_input("🔍 Enter a text query:", "a red sports car")

    if st.button("Search") and query:
        with torch.no_grad():
            inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
            text_emb = model.get_text_features(**inputs).cpu().numpy()

        faiss.normalize_L2(text_emb)
        distances, indices = img_index.search(text_emb, top_k * 3)

        st.subheader(f"Top {top_k} unique image results for: *{query}*")

        seen, results = set(), []
        for idx in indices[0]:
            row = metadata.iloc[idx]
            path = row["image_path"]
            if path not in seen and os.path.exists(path):
                results.append(row)
                seen.add(path)
            if len(results) >= top_k:
                break

        if not results:
            st.warning("No matching images found.")
        else:
            cols = st.columns(min(top_k, 5))
            for i, row in enumerate(results):
                with cols[i % 5]:
                    try:
                        st.image(row["image_path"], caption=row["caption"])
                    except Exception:
                        st.markdown(f"⚠️ Missing image: `{row['image_path']}`")


# --- IMAGE → TEXT SEARCH ---
elif mode == "Image → Text":
    uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=400)

        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
            img_emb = model.get_image_features(**inputs).cpu().numpy()

        faiss.normalize_L2(img_emb)
        distances, indices = txt_index.search(img_emb, top_k)

        st.subheader("Top matching captions:")
        for i, idx in enumerate(indices[0]):
            row = metadata.iloc[idx]
            st.markdown(f"**{i+1}.** {row['caption']}")
