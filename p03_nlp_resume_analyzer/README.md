\# 🧠 NLP Resume \& Job Description Analyzer



\*\*Project 3 — AI Portfolio\*\*

An intelligent, end-to-end NLP web app that analyzes resumes against job descriptions using semantic similarity, keyword extraction, and skill taxonomy coverage.



---



\## 🚀 Overview



This project leverages \*\*transformer-based embeddings\*\* and \*\*keyword extraction models\*\* to evaluate how well a candidate’s resume aligns with a job description.



It performs:

\- \*\*Semantic similarity scoring\*\* between resume and JD text

\- \*\*KeyBERT-based keyword extraction\*\* for important job terms

\- \*\*Dynamic filtering\*\* to remove irrelevant words (verbs, companies, cities, etc.)

\- \*\*Taxonomy-based skill coverage\*\* analysis from a YAML skill hierarchy

\- \*\*Actionable resume improvement suggestions\*\* generated automatically



The web interface, built in \*\*Streamlit\*\*, provides a one-click workflow to analyze resumes and export a professional \*\*PDF report\*\* summarizing alignment, missing skills, and improvement areas.



---



\## 🧩 Tech Stack



| Layer | Technology |

|-------|-------------|

| \*\*NLP \& ML\*\* | Sentence Transformers (`all-MiniLM-L6-v2`), KeyBERT, spaCy, RapidFuzz |

| \*\*App UI\*\* | Streamlit |

| \*\*Visualization\*\* | Plotly (Radar Chart), ReportLab (PDF export) |

| \*\*Data Storage\*\* | YAML-based Skill Taxonomy |

| \*\*Automation\*\* | Pre-commit hooks, Ruff, Verified Streamlit Deployment |

| \*\*Environment\*\* | Conda / Python virtualenv |



---



\## 📂 Project Structure



```bash

p03\_nlp\_resume\_analyzer/

│

├── app/

│ └── Main.py # Streamlit web interface

│

├── scripts/

│ └── nlp\_core.py # Core NLP logic (embeddings, filters, taxonomy)

│

├── data/

│ ├── skills.yaml # Hierarchical taxonomy of AI/ML skills

│ └── irrelevant\_terms\_cache.json # Dynamic term cache

│

├── tmp\_uploads/ # Temporary resume uploads (runtime)

│

├── requirements.txt # Environment dependencies

├── Makefile # Lint/test/run shortcuts

└── README.md # Project documentation

```





---



\## 🧠 Model \& NLP Summary



\- \*\*Embedding Model:\*\* `all-MiniLM-L6-v2` (sentence-transformers)

\- \*\*Keyword Extractor:\*\* `KeyBERT`

\- \*\*Similarity Metric:\*\* Cosine similarity

\- \*\*Skill Taxonomy:\*\* Custom YAML taxonomy (AI, ML, Data, Cloud, etc.)

\- \*\*Filtering Pipeline:\*\*

&nbsp; - Removes verbs (e.g., \*developing, leading\*) but keeps nouns (\*development, leadership\*)

&nbsp; - Excludes company \& city names (case-insensitive)

&nbsp; - Always capitalizes “ML” correctly

&nbsp; - Dynamic self-updating cache via `irrelevant\_terms\_cache.json`



---



\## 📈 Core Features



\- 📄 \*\*PDF Resume Parsing:\*\* Reads `.pdf`, `.docx`, or text input automatically

\- 🧠 \*\*Semantic Analysis:\*\* Measures deep alignment between resume and JD text

\- 🔎 \*\*Keyword Insight:\*\* Highlights missing and overlapping key concepts

\- 📊 \*\*Skill Taxonomy Coverage:\*\* Radar chart visualization of domain-level alignment

\- ✍️ \*\*Targeted Recommendations:\*\* Contextual bullet suggestions to improve the resume

\- 💾 \*\*Report Export:\*\* One-click downloadable professional report (PDF format)



---



\## 🧰 How to Run Locally



\### 1️⃣ Setup Environment



```bash

conda create -n portfolio-py python=3.10

conda activate portfolio-py

pip install -r requirements.txt

```



\### 2️⃣ Launch the Streamlit App



```bash

streamlit run app/Main.py

```

Then open your browser at:

👉 \[http://localhost:8501](http://localhost:8501)



---



\## 🧩 Example Workflow



1. Upload or paste your resume.

2\. Paste a job description.

3\. Click 🔍 Analyze to view:

&nbsp;	- Top job keywords

&nbsp;	- Semantic gaps

&nbsp;	- Skill coverage radar chart

4\. Click 📄 Generate PDF Report for a full summary.



---



\## 🎯 Key Highlights



* Dynamic NLP filters automatically adapt to job context
* Hybrid approach: rule-based + transformer-based matching
* YAML-driven taxonomy ensures explainability and extensibility
* Fully deployable Streamlit app (tested on Streamlit Cloud)
* Strict code quality enforced by Ruff and pre-commit



---



\## 💡 Lessons \& Takeaways



* Combining semantic similarity with taxonomy-based coverage improves resume–JD matching accuracy
* Real-world NLP apps benefit from dynamic filtering and caching for cleaner results
* Structuring ML/NLP apps modularly (UI vs. logic layers) simplifies deployment and maintenance
* Integrating visualization and report export turns raw NLP analysis into user-facing insights



---



\## 🌐 Live Demo



🧠 NLP Resume Analyzer App:

\[https://ja-portfolio-nlp-resume-analyzer.streamlit.app](https://ja-portfolio-nlp-resume-analyzer.streamlit.app)
