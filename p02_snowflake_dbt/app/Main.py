import streamlit as st
import pandas as pd
import plotly.express as px
import snowflake.connector

# Page Config
st.set_page_config(
    page_title="Snowflake + dbt Data Pipeline Showcase",
    page_icon="❄️",
    layout="wide"
)

# Header
st.title("❄️ Snowflake + dbt Data Pipeline Showcase")
st.markdown("""
This project demonstrates how **Snowflake** and **dbt** work together to create a modern analytics pipeline.
The app connects directly to Snowflake using **key-pair authentication**, queries a **dbt-transformed model**,
and presents metrics and visualizations that show the power of modular data transformations.
""")

# Tabs for structure
overview_tab, data_tab, visuals_tab, lineage_tab = st.tabs([
    "📘 Overview",
    "📊 Data",
    "📈 Visuals",
    "🧱 dbt Lineage & Testing"
])

# Overview Tab
with overview_tab:
    st.markdown("""
    ### 💡 Project Overview

    **Pipeline Flow:**
    1. **Raw Layer:** Ingests fraud transactions from S3 → `RAW_DATA.RAW_FRAUD_DATA`
    2. **Staging Models:** Cleaned in dbt → `STG_FRAUD_DATA`
    3. **Core & Mart Models:** Business logic → `FRAUD_FEATURES`, `FRAUD_SUMMARY`
    4. **Analytics Layer:** Streamlit visualizes data live from Snowflake.

    **Tools Used**
    - 🧊 Snowflake — Cloud data warehouse
    - 🧱 dbt — Transformation and testing
    - 🖥 Streamlit — Interactive dashboard
    - 🗝 Key-pair auth — Secure connection

    ---
    """)

# Cached Data Loader
@st.cache_data(ttl=600)
def load_data():
    conn = snowflake.connector.connect(**st.secrets["snowflake"])
    query = """
        SELECT *
        FROM financial_transactions_db.public.fraud_summary
        LIMIT 100
    """
    return pd.read_sql(query, conn)

# Data Tab
with data_tab:
    st.subheader("📊 Sample Data from dbt Model: `FRAUD_SUMMARY`")

    try:
        df = load_data()
        st.success("Connected to Snowflake ✅")

        col1, col2 = st.columns(2)
        col1.metric("Total Records", len(df))
        col2.metric("Columns", len(df.columns))

        st.dataframe(df.head(10), use_container_width=True)

        with st.expander("📄 View Underlying SQL Query"):
            st.code("""
            SELECT *
            FROM financial_transactions_db.public.fraud_summary
            LIMIT 100
            """, language="sql")

    except Exception as e:
        st.error("⚠️ Error executing query on Snowflake.")
        st.exception(e)
        st.stop()

# Visuals Tab
with visuals_tab:
    st.subheader("📈 Data Insights & Visualizations")

    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) > 0:
        metric_col = st.selectbox("Select numeric column to visualize", numeric_cols)
        fig = px.histogram(df, x=metric_col, color="IS_FRAUD" if "IS_FRAUD" in df.columns else None,
                           title=f"Distribution of {metric_col}", nbins=30)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns found for visualization.")

    if "TRANSACTION_AMOUNT" in df.columns:
        st.subheader("💰 Fraud vs. Non-Fraud Transactions")
        fig2 = px.box(df, x="IS_FRAUD", y="TRANSACTION_AMOUNT",
                      color="IS_FRAUD", title="Transaction Amount by Fraud Status")
        st.plotly_chart(fig2, use_container_width=True)

# Lineage & Testing Tab
with lineage_tab:
    st.subheader("🧱 dbt Lineage & Model Quality")

    st.markdown("""
    ### 🔗 Model Lineage
    The dbt models form a clear lineage:
    ```
    RAW_DATA.RAW_FRAUD_DATA
           ↓
    STAGING.STG_FRAUD_DATA
           ↓
    CORE.FRAUD_FEATURES
           ↓
    MARTS.FRAUD_SUMMARY
    ```

    ### ✅ Testing & Documentation
    - Each dbt model includes schema tests (`unique`, `not_null`, `relationships`)
    - Documentation generated with `dbt docs generate`
    - Models are version-controlled and automatically built via `dbt run`

    ### 📚 Key Learnings
    - Key-pair auth avoids storing credentials
    - dbt enforces data reliability before analytics
    - Modular transformations make pipelines easy to maintain
    """)

    st.caption("You can include a dbt lineage diagram here if generated (e.g. via `dbt docs generate`).")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, dbt, and Snowflake | © 2025 Jomana Abdelrahman")
