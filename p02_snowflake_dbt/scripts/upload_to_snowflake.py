import snowflake.connector
from pathlib import Path

# === CONFIGURATION ===
ACCOUNT = "gwnjdko-fmb38685"
USER = "jomana"
WAREHOUSE = "DBT_WH"
DATABASE = "FINANCIAL_TRANSACTIONS_DB"
SCHEMA = "RAW"
DATA_FOLDER = Path(__file__).resolve().parents[1] / "data"  # points to ../data/

# === CONNECT TO SNOWFLAKE ===
print("🔐 Connecting to Snowflake...")
conn = snowflake.connector.connect(
    user=USER,
    account=ACCOUNT,
    authenticator="externalbrowser",  # opens browser for MFA
    warehouse=WAREHOUSE,
    database=DATABASE,
    schema=SCHEMA
)
cur = conn.cursor()

# === ENSURE STAGE EXISTS ===
print("📦 Ensuring stage exists...")
cur.execute("CREATE OR REPLACE STAGE raw_stage COMMENT='Stage for Kaggle financial transactions data';")

# === UPLOAD EACH CSV IN DATA FOLDER ===
print(f"📤 Uploading CSV files from {DATA_FOLDER} ...")
for file in DATA_FOLDER.glob("*.csv"):
    print(f"  → Uploading {file.name} ...")
    cur.execute(f"PUT file://{file} @raw_stage AUTO_COMPRESS=TRUE;")

print("✅ Upload complete.\n")

# === VERIFY FILES IN STAGE ===
print("🔍 Files in @raw_stage:")
cur.execute("LIST @raw_stage;")
for row in cur.fetchall():
    print(f"  {row[0]} ({row[1]} bytes)")

# === CLEANUP ===
cur.close()
conn.close()
print("\n🏁 Done. Data successfully staged in Snowflake!")
