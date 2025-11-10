# app/Main.py

import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
from pathlib import Path
from io import BytesIO
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import analyzer and helpers from nlp_core
from scripts.nlp_core import analyze_resume_vs_jd, read_resume, load_skills_yaml, _flatten_skills, clean_text

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
taxonomy_path = os.path.join(BASE_DIR, "data", "skills.yaml")

st.set_page_config(page_title="NLP Resume Analyzer", layout="wide")
st.title("🧠 NLP Resume & Job Description Analyzer — HF-powered")

# ---------------- Sidebar ---------------- #
with st.sidebar.expander("🧭 Model Info"):
    st.markdown("""
    • **Embedding Model:** `all-MiniLM-L6-v2`
    • **Keyword Extractor:** `KeyBERT`
    • **Similarity Metric:** Cosine Similarity
    • **Skill Taxonomy:** `data/skills.yaml`
    • **Dynamic Keyword Filter:** Self-learning (NER + cache)
    • **Optional Generator:** `bigscience/bloomz-1b1`
    """)

skills_yaml_default = taxonomy_path


# ---------------- Utility: Dynamic recommendations ---------------- #
def build_category_skill_map(yaml_path: str = skills_yaml_default) -> Dict[str, set]:
    raw = load_skills_yaml(yaml_path)
    if not raw:
        return {}
    mapping = {}
    for cat, items in raw.items():
        flat = _flatten_skills(items)
        mapping[cat] = set([clean_text(s) for s in flat if s and isinstance(s, str)])
    return mapping


def infer_candidates_for_category(cat: str, out: dict, cat_skill_set: set) -> list:
    cands = []
    sem_missing = out.get("semantic_missing", []) or []
    jd_kws = out.get("jd_keywords", []) or []

    def matches_skill_set(token: str):
        t = clean_text(token)
        for s in cat_skill_set:
            if not t or not s or len(t) < 3 or len(s) < 3:
                continue
            if t in s or s in t:
                return True
        return False

    for tok in sem_missing:
        if matches_skill_set(tok) and tok not in cands:
            cands.append(tok)
    if not cands:
        for tok in jd_kws:
            if matches_skill_set(tok) and tok not in cands:
                cands.append(tok)
    if not cands:
        fallback = [t for t in sem_missing + jd_kws if len(clean_text(t)) > 3]
        for t in fallback[:3]:
            if t not in cands:
                cands.append(t)
    return cands


def generate_recommendations(cov: dict, out: dict, yaml_path: str = skills_yaml_default) -> dict:
    cat_map = build_category_skill_map(yaml_path)
    recs = {}
    threshold = 60.0

    for cat, info in cov.items():
        pct = info.get("coverage_pct", 0.0)
        missing = info.get("missing", []) or []
        recs[cat] = {"coverage_pct": pct, "missing": missing, "inferred": [], "suggestions": []}

        if pct >= threshold:
            continue

        if missing:
            candidates = missing
        else:
            cat_skill_set = cat_map.get(cat, set())
            candidates = infer_candidates_for_category(cat, out, cat_skill_set)

        if not missing:
            recs[cat]["inferred"] = candidates

        suggestions = []
        for cand in candidates:
            c = clean_text(cand)
            if any(tok in c for tok in ["deploy", "deployment", "production", "monitor"]):
                suggestion = "Emphasize model deployment and monitoring practices (MLOps)."
                example = "💡 *Add a line like:* 'Deployed ML models with Docker and CI/CD; monitored drift with MLflow.'"
            elif any(tok in c for tok in ["pipeline", "etl", "airflow", "data pipeline"]):
                suggestion = "Highlight ETL pipelines and data orchestration supporting ML."
                example = "💡 *Add a line like:* 'Built automated ETL pipelines and retraining workflows using Airflow.'"
            elif any(tok in c for tok in ["aws", "gcp", "azure", "sagemaker", "vertex"]):
                suggestion = "Show cloud ML experience (AWS, GCP, Azure)."
                example = "💡 *Add a line like:* 'Trained and deployed ML models using AWS SageMaker and GCP Vertex AI.'"
            elif any(tok in c for tok in ["git", "version", "ci", "cd"]):
                suggestion = "Mention version control and CI/CD for reproducible ML workflows."
                example = "💡 *Add a line like:* 'Used Git and CI/CD to manage model lifecycle and deployment automation.'"
            elif any(tok in c for tok in ["metric", "validate", "test", "evaluation"]):
                suggestion = "Include model evaluation and validation methods."
                example = "💡 *Add a line like:* 'Performed cross-validation and tracked performance metrics to improve accuracy.'"
            elif any(tok in c for tok in ["communicat", "collabor", "present"]):
                suggestion = "Emphasize collaboration and stakeholder communication."
                example = "💡 *Add a line like:* 'Partnered with engineering and product teams to translate insights into deployed ML systems.'"
            else:
                suggestion = f"Add or refine a bullet highlighting experience with **{cand}**."
                example = f"💡 *Add a line like:* 'Applied {cand} in an end-to-end ML project to enhance predictive performance.'"
            suggestions.append({"term": cand, "suggestion": suggestion, "example": example})

        if not suggestions:
            jd_kws = out.get("jd_keywords", []) or []
            sample = jd_kws[:2]
            if sample:
                suggestion = f"The job emphasizes {', '.join(sample)} which aren't clearly reflected in your resume."
                example = f"💡 *Add:* 'Worked with {', '.join(sample)} in production ML workflows.'"
                suggestions.append({"term": ", ".join(sample), "suggestion": suggestion, "example": example})
        recs[cat]["suggestions"] = suggestions

    return recs


# ---------------- Main Inputs ---------------- #
col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Upload resume (pdf/docx/txt) or paste below", type=["pdf", "docx", "txt"])
    resume_input = st.text_area("📄 Paste resume (or leave blank if uploading)", height=300)
with col2:
    job_input = st.text_area("💼 Paste job description here", height=425)

# ---------------- File Handling ---------------- #
if uploaded:
    tmpdir = ROOT / "tmp_uploads"
    tmpdir.mkdir(exist_ok=True)
    upath = tmpdir / uploaded.name
    with open(upath, "wb") as f:
        f.write(uploaded.getbuffer())
    resume_input = read_resume(str(upath))
    st.success(f"Loaded {uploaded.name}")

# ---------------- Analysis ---------------- #
if st.button("🔍 Analyze"):
    if not job_input.strip():
        st.warning("Please paste a job description to analyze.")
    elif not resume_input.strip():
        st.warning("Please paste or upload a resume.")
    else:
        with st.spinner("Running analysis..."):
            st.session_state["analysis_result"] = analyze_resume_vs_jd(resume_input, job_input, skills_yaml_default)
        st.success("Analysis complete ✅")

out = st.session_state.get("analysis_result", None)

# ---------------- Results Rendering ---------------- #
if out:
    similarity = out["similarity"]
    cov = out["taxonomy_coverage"]

    # Summary
    st.markdown("---")
    st.subheader("📊 Overall Fit Summary")
    st.progress(int(similarity))
    st.caption(f"**Overall Resume–JD Match Score:** {similarity:.2f}%")

    if similarity >= 80:
        st.success("✅ Excellent match! Your resume is highly aligned with this JD.")
    elif similarity >= 60:
        st.info("🟡 Good match — a few tweaks could improve alignment.")
    else:
        st.warning("🔴 Significant skill or emphasis gaps detected.")

    st.markdown("---")
    st.markdown("### 🔎 Top Extracted JD Keywords")
    st.write(", ".join(out.get("jd_keywords", [])) or "No keywords extracted.")

    st.markdown("### 🚫 Semantic Missing (from JD)")
    st.write(", ".join(out.get("semantic_missing", [])) or "None detected 🎉")

    st.markdown("### 📊 Taxonomy Coverage by Category")
    if cov:
        df = pd.DataFrame(
            [{"Category": k.replace("_", " ").title(), "Coverage (%)": v["coverage_pct"]} for k, v in cov.items()]
        )
        if not df.empty:
            fig = px.line_polar(df, r="Coverage (%)", theta="Category",
                                line_close=True, markers=True,
                                color_discrete_sequence=["#00B5E2"])
            fig.update_traces(fill="toself")
            fig.update_layout(polar=dict(bgcolor="#111", radialaxis=dict(range=[0, 100], gridcolor="gray")),
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            avg_cov = df["Coverage (%)"].mean()
            st.info(f"🧩 **Average taxonomy coverage:** {avg_cov:.1f}%")

    # ---------------- Improved Targeted Recommendations ---------------- #
    st.markdown("---")
    st.markdown("## 🎯 Targeted Skill Recommendations")

    recs = generate_recommendations(cov, out, skills_yaml_default)
    improvement_cats = [c for c, v in recs.items() if v["coverage_pct"] < 60.0]

    if improvement_cats:
        for cat in improvement_cats:
            meta = recs[cat]
            cat_clean = cat.replace("_", " ").title()
            st.markdown(f"### 🟨 {cat_clean} — Coverage: {meta['coverage_pct']:.1f}%")

            if meta["missing"]:
                st.markdown(f"**🔍 Detected Gaps:** {', '.join(meta['missing'])}")
            if meta["inferred"]:
                st.markdown(f"**📉 Inferred Concepts from JD:** {', '.join(meta['inferred'])}")

            st.markdown("#### ✍️ Suggested Improvements")
            for s in meta["suggestions"]:
                st.markdown(f"- **{s['term']}** → {s['suggestion']}")
                st.markdown(f"  {s['example']}")

            st.divider()
    else:
        st.success("🟩 No major skill gaps detected — resume aligns strongly with the JD!")

    # ---------------- PDF Report ---------------- #
    st.markdown("---")
    st.subheader("📄 Downloadable Report")
    if st.button("Generate PDF Report"):
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        story = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        story.append(Paragraph(f"<b>Report generated:</b> {timestamp}", styles["Normal"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"<b>Resume–JD Similarity:</b> {similarity:.2f}%", styles["Normal"]))
        avg_cov = df["Coverage (%)"].mean() if not df.empty else 0
        story.append(Paragraph(f"<b>Average Taxonomy Coverage:</b> {avg_cov:.1f}%", styles["Normal"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph("<b>Targeted Skill Recommendations:</b>", styles["Heading3"]))
        for cat in improvement_cats:
            meta = recs[cat]
            story.append(Paragraph(f"<b>{cat.replace('_', ' ').title()}</b> — {meta['coverage_pct']:.1f}% coverage", styles["Normal"]))
            if meta["missing"]:
                story.append(Paragraph("Detected Gaps: " + ", ".join(meta["missing"]), styles["Normal"]))
            if meta["inferred"]:
                story.append(Paragraph("Inferred from JD: " + ", ".join(meta["inferred"]), styles["Normal"]))
            for s in meta["suggestions"]:
                story.append(Paragraph(f"- {s['term']}: {s['suggestion']}", styles["Normal"]))
                story.append(Paragraph(f"  {s['example']}", styles["Normal"]))
            story.append(Spacer(1, 8))

        doc.build(story)
        pdf_data = buffer.getvalue()
        filename = f"resume_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        st.download_button("⬇️ Download PDF Report", data=pdf_data, file_name=filename, mime="application/pdf")
