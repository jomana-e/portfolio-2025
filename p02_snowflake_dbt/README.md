# ❄️ Snowflake + dbt Fraud Detection Pipeline

**Project 2 — AI Portfolio**  
A modern **data engineering and analytics project** that builds an end-to-end fraud detection pipeline using **Snowflake**, **dbt**, and **cloud-based data modeling**.

---

## 🚀 Overview

This project transforms raw financial transactions from **S3 → Snowflake → dbt models** to detect and summarize fraudulent activity.  

It demonstrates:
- Cloud-scale **data ingestion and transformation**
- Production-grade **data modeling in dbt**
- **Feature engineering** for fraud analysis
- Automated **data lineage and documentation** via dbt Docs
- Version-controlled, reproducible data workflows

---

## 🧩 Tech Stack

| Layer | Technology |
|-------|-------------|
| **Data Source** | AWS S3 (CSV data) |
| **Data Warehouse** | Snowflake |
| **Transformation Layer** | dbt (Data Build Tool) |
| **Orchestration / Automation** | dbt CLI, Pre-commit hooks |
| **Environment** | Conda (`portfolio-py`), GitHub integration |

---

## 📂 Project Structure

```bash
p02_snowflake_dbt/
│
├── models/
│ ├── staging/
│ │ ├── stg_fraud_data.sql # Base staging model for raw_fraud_data
│ │ └── staging.yml # Source + test definitions
│ │
│ ├── core/
│ │ ├── fraud_union.sql # Unified dataset
│ │ ├── fraud_features.sql # Feature engineering for fraud analysis
│ │ └── core.yml
│ │
│ └── marts/
│ └── fraud_summary.sql # Final analytical model (state/category stats)
│
├── scripts/
│ ├── prepare_data.py # Prepares and uploads data to S3
│ └── upload_to_snowflake.py # Loads data from S3 to Snowflake
│
├── data/
│ ├── fraudTrain.csv
│ └── fraudTest.csv
│
├── dbt_project.yml # Project configuration
├── packages.yml # dbt dependencies
├── Makefile (optional) # Reproducibility helper
└── README.md # Documentation (this file)
```

---

## 🧠 Data Summary

- **Dataset:** Synthetic financial transactions (train/test splits)
- **Source Table:** `RAW_FRAUD_DATA`
- **Destination Models:**
  - `stg_fraud_data` → cleans and standardizes raw data  
  - `fraud_features` → aggregates fraud indicators  
  - `fraud_summary` → summarizes by state and category

🧮 **Sample Output Metrics**
| Column | Description |
|---------|-------------|
| `STATE` | US state abbreviation |
| `CATEGORY` | Merchant category |
| `TOTAL_TRANSACTIONS` | Total transactions in group |
| `TOTAL_FRAUD` | Count of fraud cases |
| `AVG_AMOUNT` | Mean transaction amount |
| `HIGH_VALUE_COUNT` | Count of high-value transactions |

---

## 🧰 How to Run Locally

### 1️⃣ Setup Environment
```bash
conda create -n portfolio-py python=3.10
conda activate portfolio-py
pip install dbt-snowflake
```

### 2️⃣ Set Up Snowflake Credentials

In ~/.dbt/profiles.yml (or Windows %USERPROFILE%\.dbt\profiles.yml):

```bash
portfolio_2025:
  target: dev
  outputs:
    dev:
      type: snowflake
      account: <your_account>
      user: <your_user>
      password: <your_password>
      role: ACCOUNTADMIN
      database: FINANCIAL_TRANSACTIONS_DB
      warehouse: COMPUTE_WH
      schema: PUBLIC
      threads: 1
```

### 3️⃣ Run dbt Models

```bash
cd p02_snowflake_dbt
dbt run
```

### 4️⃣ Generate Documentation

```bash
dbt docs generate
dbt docs serve
```

Then open your browser at [http://localhost:8080](http://localhost:8080).

---

## 🎯 Features

- 🔄 End-to-end Snowflake ingestion from S3
- 🧱 Modular dbt models for layered transformations
- 🧮 Automated fraud feature creation
- 🧭 dbt Docs lineage + metadata visualization
- ⚙️ Pre-commit linting and formatting with Ruff
- ☁️ Reproducible and version-controlled setup

---

### 💡 Lessons & Highlights

- Integrating Snowflake + dbt for production-grade analytics
- Building modular SQL models with sources and dependencies
- Implementing data lineage and documentation via dbt docs
- Automating environment reproducibility and quality checks
- Designing data pipelines as code for scalable fraud detection

---
