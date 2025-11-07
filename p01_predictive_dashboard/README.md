# 🧠 Predictive Customer Churn Dashboard

**Project 1 — AI Portfolio**
A full-stack machine learning dashboard that predicts telecom customer churn using interactive data visualization and model inference.

---

## 🚀 Overview

This project builds an **end-to-end predictive system** using the [Telco Customer Churn dataset](https://huggingface.co/datasets/aai510-group1/telco-customer-churn).
The model predicts whether a customer is likely to churn and provides an interactive dashboard built with **Streamlit**.

It demonstrates:
- Applied **supervised learning** and feature engineering
- Reproducible **ML pipelines**
- Deployment-ready code structure
- **MLOps-style** workflow (linting, automation, pre-commit checks)

---

## 🧩 Tech Stack

| Layer | Technology |
|-------|-------------|
| **Data** | Hugging Face Datasets, Pandas, NumPy |
| **Modeling** | Scikit-learn (`Pipeline`, `RandomForestClassifier`) |
| **App UI** | Streamlit |
| **Automation** | Pre-commit hooks, Makefile, Verified workflows |
| **Environment** | Conda / Python virtualenv |

---

## 📂 Project Structure

```
p01_predictive_dashboard/
│
├── app/ # Streamlit frontend
│ ├── main.py # Web app entry point
│ └── components/ # Reusable UI parts
│
├── models/
│ └── churn_model.joblib # Trained model artifact
│
├── scripts/
│ ├── train.py # Model training pipeline
│ ├── test_model.py # (Optional) Model sanity test
│ ├── data_cleaning.py # Preprocessing helpers
│ └── custom_transformers.py
│
├── data/
│ └── raw/ # (Optional) Data cache
│
├── index.html # Portfolio home redirect
├── requirements.txt # Environment dependencies
├── Makefile # Commands for lint/test/run
└── README.md # Project documentation
```

---

## 🧠 Model Summary

- **Algorithm:** Random Forest Classifier
- **Target Variable:** `Churn` (1 = churned, 0 = retained)
- **Feature Count:** 36
- **Train/Test Split:** 80/20
- **Accuracy:** ~99% on holdout set
- **Feature Types:** Numeric, categorical, and encoded boolean fields
- **Preprocessing:**
  - Missing value imputation
  - One-hot encoding for categorical fields
  - Scaling numeric features

---

## 🧰 How to Run Locally

### 1️⃣ Setup Environment

```bash
conda create -n portfolio-py python=3.10
conda activate portfolio-py
pip install -r requirements.txt
```

### 2️⃣ Launch Dashboard
```bash
streamlit run app/main.py
```

Then open your browser at http://localhost:8501

---

# 🎯 Features

- 📈 Predicts customer churn in real time
- 🧹 Handles missing/unknown data gracefully
- 🌍 Loads pre-trained model for instant inference
- 🎨 Clean, responsive dashboard layout
- ⚙️ Modular, reusable ML and UI codebase

---

# 💡 Lessons & Highlights

- Building production-grade pipelines using scikit-learn Pipeline objects
- Handling data inconsistencies gracefully with preprocessing transformers
- Creating automated reproducibility (verified automation, pre-commit hooks)
- Designing a user-facing ML dashboard with Streamlit

---
