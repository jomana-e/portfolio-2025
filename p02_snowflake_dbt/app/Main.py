import streamlit as st
import pandas as pd
import snowflake.connector
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

st.set_page_config(page_title="Snowflake + dbt Pipeline", page_icon="❄️", layout="wide")

st.title("❄️ Snowflake + dbt Data Pipeline Showcase")
st.markdown("""
This project demonstrates how **Snowflake** and **dbt** work together to create a modern analytics pipeline.

It includes:
- Raw → staging → marts transformation flow
- dbt model lineage and tests
- Example queries from the transformed data
""")

# 🔐 Key-pair authentication setup
def get_private_key():
    """Load private key from Streamlit secrets."""
    key_str = st.secrets["snowflake"].get("private_key")
    if not key_str:
        raise ValueError("No private key found in Streamlit secrets.")
    passphrase = st.secrets["snowflake"].get("private_key_passphrase")

    return serialization.load_pem_private_key(
        key_str.encode(),
        password=passphrase.encode() if passphrase else None,
        backend=default_backend()
    )

# ❄️ Connect to Snowflake
@st.cache_resource(show_spinner=False)
def connect_to_snowflake():
    try:
        if "snowflake" not in st.secrets:
            st.warning("No Snowflake credentials found — running in demo mode.")
            return None

        private_key = get_private_key()
        conn = snowflake.connector.connect(
            user=st.secrets["snowflake"]["user"],
            account=st.secrets["snowflake"]["account"],
            warehouse=st.secrets["snowflake"]["warehouse"],
            database=st.secrets["snowflake"]["database"],
            schema=st.secrets["snowflake"]["schema"],
            role=st.secrets["snowflake"].get("role"),
            private_key=private_key,
        )
        st.success("✅ Connected to Snowflake via key-pair authentication")
        return conn
    except Exception as e:
        st.warning("⚠️ Could not connect to Snowflake.")
        st.info("Make sure your `.streamlit/secrets.toml` or Cloud Secrets include a valid key pair and user without MFA.")
        st.exception(e)
        return None


conn = connect_to_snowflake()

# 📊 Load data or show demo mode
st.subheader("Sample Data from mart model")

if conn:
    try:
        query = """
        SELECT *
        FROM financial_transactions_db.public.fraud_summary
        LIMIT 10
        """
        df = pd.read_sql(query, conn)
        st.dataframe(df)

        st.subheader("Quick Metrics")
        col1, col2 = st.columns(2)
        col1.metric("Total Records", len(df))
        col2.metric("Columns", len(df.columns))

    except Exception as e:
        st.error("❌ Error executing query on Snowflake.")
        st.exception(e)
        df = None
else:
    # Demo fallback
    st.info("Showing demo data (no live Snowflake connection).")
    df = pd.DataFrame({
        "customer_id": [1001, 1002, 1003],
        "transaction_amount": [200.5, 450.2, 125.0],
        "is_fraud": [0, 1, 0],
    })
    st.dataframe(df)

# 📈 Visualization
st.subheader("Data Preview / Visualization Example")
if df is not None and not df.empty:
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        st.bar_chart(df[numeric_cols].iloc[:, :1])
    else:
        st.info("No numeric columns available for charting.")

# 🧱 Project Overview
st.markdown("""
### 🧱 Pipeline Overview
**Layers:**
- **Staging** — cleaned and standardized source data
- **Core Models** — business logic and joins
- **Marts** — analytics-ready tables for BI tools
""")
