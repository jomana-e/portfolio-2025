# scripts/nlp_core.py

from typing import List, Dict, Tuple
import os
import re
import yaml
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
from rapidfuzz import fuzz
import streamlit as st

# Optional dependencies
try:
    from transformers import pipeline
except Exception:
    pipeline = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx
except Exception:
    docx = None

# NEW optional dependencies
try:
    import spacy
    from nltk.corpus import stopwords
except Exception:
    spacy, stopwords = None, None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
taxonomy_path = os.path.join(BASE_DIR, "data", "skills.yaml")

# CONFIGURATION
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HF_TEXTGEN_MODEL = "HuggingFaceH4/zephyr-7b-beta"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", None)
CACHE_DIR = Path(".cache_nlp")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "irrelevant_terms_cache.json"


# HELPERS
def clean_text(text: str) -> str:
    if not text:
        return ""
    t = str(text)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^A-Za-z0-9\s\-\./]", " ", t)
    return t.strip().lower()


# MODEL LOADERS (cached)
@st.cache_resource
def load_embed_model(name: str = EMBED_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(name)

@st.cache_resource
def load_kw_model(embed_name: str = EMBED_MODEL_NAME) -> KeyBERT:
    return KeyBERT(model=embed_name)


# ---------- NEW: DYNAMIC IRRELEVANT TERM FILTER ---------- #

def load_irrelevant_cache() -> set:
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()

def save_irrelevant_cache(terms: set):
    try:
        CACHE_PATH.parent.mkdir(exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(sorted(list(terms)), f, indent=2)
    except Exception:
        pass

def clean_keywords(jd_keywords: List[str], job_input: str) -> List[str]:
    """Dynamic filter that removes irrelevant JD terms & updates cache, with static hard exclusions and verb exclusion."""
    irrelevant_dynamic = load_irrelevant_cache()

    # 1️⃣ NER-based removal
    if spacy is not None:
        try:
            nlp = spacy.load("en_core_web_sm", disable=["parser"])
            doc = nlp(job_input)
            for ent in doc.ents:
                if ent.label_ in {"ORG", "GPE", "LOC", "PERSON"}:
                    irrelevant_dynamic.add(ent.text.lower())
        except Exception:
            pass

    # 2️⃣ Generic job/HR-related words
    generic_terms = {
        "role", "job", "position", "employee", "team", "organization",
        "company", "department", "manager", "associate", "engineer", "staff",
        "worker", "individual", "professional", "hybrid", "remote", "onsite",
        "location", "office", "headquarters", "candidate", "applicant", "employer",
        "developing", "creating"
    }
    irrelevant_dynamic.update(generic_terms)

    # 3️⃣ Stopwords
    if stopwords is not None:
        try:
            irrelevant_dynamic.update(stopwords.words("english"))
        except Exception:
            pass

    # 4️⃣ Add tokens from job title / intro lines
    for line in job_input.splitlines()[:3]:
        for w in line.split():
            if len(w) > 2:
                irrelevant_dynamic.add(w.lower())

    # 5️⃣ 🔒 Hard exclusions (always removed)
    hard_exclusions = {"workplace", "environment", "culture", "diversity", "values"}
    irrelevant_dynamic.update(hard_exclusions)

    # 6️⃣ 🚫 Company & city fallback detection (case-insensitive)
    company_like = re.compile(r"\b(inc|llc|corp|corporation|ltd|co\.|company)\b", re.IGNORECASE)
    city_like_terms = {
        "new york", "los angeles", "chicago", "houston", "phoenix",
        "philadelphia", "san antonio", "san diego", "dallas", "san jose",
        "seattle", "austin", "boston", "denver", "atlanta", "miami",
        "london", "paris", "berlin", "toronto", "vancouver", "tokyo", "singapore"
    }

    for kw in jd_keywords:
        kw_l = kw.lower().strip()
        if company_like.search(kw_l):
            irrelevant_dynamic.add(kw_l)
        if kw_l in city_like_terms:
            irrelevant_dynamic.add(kw_l)

    # 7️⃣ 🚫 Verb exclusions (drop only verbs, keep nouns/adjectives)
    verb_exclusions = {
        "developing", "creating", "leading", "managing", "working",
        "supporting", "building", "designing", "using", "helping", "driving"
    }

    for kw in jd_keywords:
        kw_l = kw.lower().strip()
        if kw_l in verb_exclusions:
            irrelevant_dynamic.add(kw_l)

    # 8️⃣ Filter keywords (fully lowercase matching)
    filtered = [
        kw for kw in jd_keywords
        if kw.lower() not in irrelevant_dynamic
        and kw.isalpha()
        and len(kw) > 2
    ]

    # 9️⃣ Normalize capitalization rules
    normalized = []
    for kw in filtered:
        if kw.lower() == "ml":
            normalized.append("ML")
        else:
            normalized.append(kw)

    # Save updated cache
    save_irrelevant_cache(irrelevant_dynamic)
    return list(dict.fromkeys(normalized))  # unique + preserve orders

# ---------- CORE ANALYSIS PHASES ---------- #

def compute_similarity(resume_text: str, jd_text: str) -> float:
    model = load_embed_model()
    r = clean_text(resume_text)
    j = clean_text(jd_text)
    emb_r = model.encode(r, convert_to_tensor=True)
    emb_j = model.encode(j, convert_to_tensor=True)
    sim = util.cos_sim(emb_r, emb_j).item()
    return round(sim * 100, 2)


def extract_keywords(text: str, top_n: int = 20) -> List[Tuple[str, float]]:
    kw = load_kw_model()
    txt = clean_text(text)
    kws = kw.extract_keywords(txt, top_n=top_n, stop_words="english")
    return kws


def find_missing_semantic(resume_text: str, jd_text: str, top_n: int = 25) -> List[str]:
    jd_kws = extract_keywords(jd_text, top_n=top_n)
    resume_l = clean_text(resume_text)
    missing = [kw for kw, _ in jd_kws if clean_text(kw) not in resume_l]
    return missing


def load_skills_yaml(path: str = taxonomy_path) -> Dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _flatten_skills(raw) -> List[str]:
    if isinstance(raw, dict):
        return [s for v in raw.values() for s in _flatten_skills(v)]
    elif isinstance(raw, list):
        return [s for i in raw for s in _flatten_skills(i)]
    elif isinstance(raw, str):
        return [raw.strip()]
    return []


def taxonomy_coverage(resume_text: str, jd_text: str, yaml_path: str = taxonomy_path, threshold: int = 75) -> Dict:
    raw = load_skills_yaml(yaml_path)
    if not raw:
        return {}
    resume_l, jd_l = clean_text(resume_text), clean_text(jd_text)
    coverage = {}

    for cat, items in raw.items():
        matched_resume, matched_jd = set(), set()
        for s in _flatten_skills(items):
            s_clean = clean_text(s)
            if not s_clean:
                continue
            if fuzz.partial_ratio(s_clean, resume_l) >= threshold:
                matched_resume.add(s_clean)
            if fuzz.partial_ratio(s_clean, jd_l) >= threshold:
                matched_jd.add(s_clean)
        overlap = matched_resume & matched_jd
        missing = sorted(list(matched_jd - matched_resume))
        coverage[cat] = {
            "job_count": len(matched_jd),
            "resume_count": len(matched_resume),
            "overlap_count": len(overlap),
            "coverage_pct": round(len(overlap) / len(matched_jd) * 100, 2) if matched_jd else 0.0,
            "missing": missing,
        }
    return coverage


def _build_generation_prompt(resume_text: str, jd_text: str, missing_skills: List[str]) -> str:
    mk = ", ".join(missing_skills) if missing_skills else "none"
    return (
        "You are an AI resume coach. "
        "Given a resume and a job description, write 3–6 short action-oriented resume bullets "
        "that incorporate the missing skills where possible.\n\n"
        f"Job Description:\n{jd_text}\n\n"
        f"Resume:\n{resume_text}\n\n"
        f"Missing skills to emphasize: {mk}\n\nBullets:\n-"
    )


def suggest_resume_bullets(resume_text: str, jd_text: str, missing_skills: List[str]) -> List[str]:
    prompt = _build_generation_prompt(resume_text, jd_text, missing_skills)

    if HF_API_TOKEN:
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=HF_API_TOKEN)
            result = client.text_generation(model=HF_TEXTGEN_MODEL, inputs=prompt, parameters={"max_new_tokens": 200})
            text = result[0]["generated_text"] if isinstance(result, list) else str(result)
            return [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()][:6]
        except Exception:
            pass

    return [f"Demonstrated expertise in {s} through data-driven projects." for s in missing_skills[:6]]


def extract_text_from_pdf(path: str) -> str:
    if pdfplumber is None:
        return ""
    text = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            txt = p.extract_text()
            if txt:
                text.append(txt)
    return "\n".join(text)


def extract_text_from_docx(path: str) -> str:
    if docx is None:
        return ""
    d = docx.Document(path)
    return "\n".join([p.text for p in d.paragraphs if p.text])


def read_resume(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    ext = p.suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(str(p))
    if ext in [".docx", ".doc"]:
        return extract_text_from_docx(str(p))
    return p.read_text(encoding="utf-8", errors="ignore")


# ONE-CALL ANALYZER
def analyze_resume_vs_jd(resume_text: str, jd_text: str, skills_yaml_path: str = taxonomy_path) -> Dict:
    out = {}
    out["similarity"] = compute_similarity(resume_text, jd_text)
    jd_raw = [k for k, _ in extract_keywords(jd_text, top_n=25)]
    sem_missing_raw = find_missing_semantic(resume_text, jd_text, top_n=25)

    # Apply dynamic cleaning
    out["jd_keywords"] = clean_keywords(jd_raw, jd_text)
    out["semantic_missing"] = clean_keywords(sem_missing_raw, jd_text)

    out["taxonomy_coverage"] = taxonomy_coverage(resume_text, jd_text, yaml_path=skills_yaml_path, threshold=70)

    taxonomy_missing = []
    for cat, info in out["taxonomy_coverage"].items():
        taxonomy_missing += info.get("missing", [])
    taxonomy_missing = sorted(set(taxonomy_missing))
    missing_for_gen = taxonomy_missing or out["semantic_missing"][:8]
    out["suggested_bullets"] = suggest_resume_bullets(resume_text, jd_text, missing_for_gen)
    return out
