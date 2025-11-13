# app/Main.py

import tempfile
import streamlit as st
import pandas as pd
import torch
import faiss
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

st.set_page_config(page_title="Multimodal AI Search Engine", layout="wide")

BUCKET = "portfolio-curated-jomana"
S3_PATHS = {
    "image_index": "indexes/faiss_image.index",
    "text_index": "indexes/faiss_text.index",
    "metadata": "processed/multimodal_metadata.csv",
}


@st.cache_resource(show_spinner="Loading CLIP model...")
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, processor, device


def init_s3_client():
    try:
        if "aws" in st.secrets:
            aws_cfg = st.secrets["aws"]
            s3 = boto3.client(
                "s3",
                aws_access_key_id=aws_cfg["aws_access_key_id"],
                aws_secret_access_key=aws_cfg["aws_secret_access_key"],
                region_name=aws_cfg.get("region_name", "us-east-1"),
            )
        else:
            s3 = boto3.client("s3")
        return s3
    except Exception as e:
        st.error(f"❌ Failed to initialize S3 client: {e}")
        raise


@st.cache_resource(show_spinner="Fetching FAISS index from S3...")
def load_index_from_s3(index_type: str):
    s3 = init_s3_client()
    s3_key = S3_PATHS[f"{index_type}_index"]
    st.sidebar.info(f"📥 Downloading {index_type} index (~1.3 GB) from S3...")

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            s3.download_fileobj(BUCKET, s3_key, tmp)
            tmp.flush()
            index = faiss.read_index(tmp.name)
        st.sidebar.success(f"✅ {index_type.capitalize()} index loaded successfully.")
        return index
    except NoCredentialsError:
        st.error("❌ No AWS credentials found. Set them in Streamlit secrets.")
        raise
    except ClientError as e:
        st.error(f"❌ AWS ClientError: {e}")
        raise
    except Exception as e:
        st.error(f"❌ Failed to download or read {index_type} index: {e}")
        raise


@st.cache_data(show_spinner="Loading metadata from S3...")
def load_metadata_from_s3():
    s3 = init_s3_client()
    key = S3_PATHS["metadata"]
    st.sidebar.info("📄 Downloading metadata CSV from S3...")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            s3.download_fileobj(BUCKET, key, tmp)
            tmp.flush()
            df = pd.read_csv(tmp.name)

        df["image_path"] = df["image_path"].astype(str).str.replace("\\", "/", regex=False)
        # Construct S3 image URLs based on the stored paths
        df["image_url"] = df["image_path"].apply(
            lambda p: f"https://{BUCKET}.s3.amazonaws.com/{p.lstrip('/')}"
        )

        st.sidebar.success("✅ Metadata loaded successfully.")
        return df
    except Exception as e:
        st.error(f"❌ Failed to load metadata: {e}")
        raise


st.sidebar.success("🚀 Initializing app...")
model, processor, device = load_model()
metadata = load_metadata_from_s3()

st.title("🧠 Multimodal AI Search Engine")
st.markdown("Search across images and captions using **CLIP + FAISS** embeddings.")

mode = st.sidebar.radio("Choose mode:", ["Text → Image", "Image → Text"])
top_k = st.sidebar.slider("Number of results", 1, 20, 5)
st.sidebar.info("⚙️ Index and metadata load from S3 only when needed.")


if mode == "Text → Image":
    query = st.text_input("🔍 Enter a text query:", "a red sports car")

    if st.button("Search") and query:
        img_index = load_index_from_s3("image")

        with torch.no_grad():
            inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
            text_emb = model.get_text_features(**inputs).cpu().numpy()

        faiss.normalize_L2(text_emb)
        distances, indices = img_index.search(text_emb, top_k * 3)

        st.subheader(f"Top {top_k} image results for: *{query}*")
        seen, results = set(), []
        for idx in indices[0]:
            row = metadata.iloc[idx]
            url = row.get("image_url")
            if url and url not in seen:
                results.append(row)
                seen.add(url)
            if len(results) >= top_k:
                break

        if not results:
            st.warning("No matching images found.")
        else:
            cols = st.columns(min(top_k, 5))
            for i, row in enumerate(results):
                with cols[i % 5]:
                    try:
                        st.image(row["image_url"], caption=row["caption"])
                    except Exception:
                        st.markdown(f"⚠️ Missing or invalid image URL: `{row['image_url']}`")


elif mode == "Image → Text":
    uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=400)

        txt_index = load_index_from_s3("text")

        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
            img_emb = model.get_image_features(**inputs).cpu().numpy()

        faiss.normalize_L2(img_emb)
        distances, indices = txt_index.search(img_emb, top_k)

        st.subheader("Top matching captions:")
        for i, idx in enumerate(indices[0]):
            row = metadata.iloc[idx]
            st.markdown(f"**{i+1}.** {row['caption']}")
