import streamlit as st
import pandas as pd
import snowflake.connector

st.set_page_config(page_title="Snowflake + dbt Pipeline", page_icon="❄️", layout="wide")

st.title("❄️ Snowflake + dbt Data Pipeline Showcase")
st.markdown("""
This project demonstrates how **Snowflake** and **dbt** work together to create a modern analytics pipeline.

It includes:
- Raw → staging → marts transformation flow
- dbt model lineage and tests
- Example queries from the transformed data
""")

try:
    conn = snowflake.connector.connect(**st.secrets["snowflake"])
    st.success("Connected to Snowflake ✅")

    query = """
        SELECT *
        FROM marts.transactions_summary
        LIMIT 10
    """
    df = pd.read_sql(query, conn)
    st.subheader("Sample Data from mart model")
    st.dataframe(df)

    st.subheader("Quick Metrics")
    st.metric("Total Records", len(df))
    st.metric("Columns", len(df.columns))

except Exception as e:
    st.warning("⚠️ Could not connect to Snowflake (likely running locally without secrets).")
    st.info("Add Snowflake credentials in `.streamlit/secrets.toml` to enable live queries.")
    st.exception(e)

# --- Optional Visualization Example ---
st.subheader("Data Preview / Visualization Example")
if 'df' in locals():
    st.bar_chart(df.select_dtypes(include='number').iloc[:, :1])

# --- Project Overview ---
st.markdown("""
### 🧱 Pipeline Overview
**Layers:**
- **Staging** — cleaned and standardized source data
- **Core Models** — business logic and joins
- **Marts** — analytics-ready tables for BI tools
""")
