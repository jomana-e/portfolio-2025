import pandas as pd
import joblib
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Custom transformer to clean 'Unknown' and similar placeholders
from sklearn.base import BaseEstimator, TransformerMixin

class CleanUnknowns(BaseEstimator, TransformerMixin):
    def __init__(self, unknown_values=None):
        self.unknown_values = unknown_values or ["Unknown", "None", "?", "", " "]
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.replace(self.unknown_values, pd.NA)

print("📥 Loading Telco Churn dataset from Hugging Face...")
ds = load_dataset("aai510-group1/telco-customer-churn")
df = pd.DataFrame(ds["train"])

target = "Churn"
if target not in df.columns:
    raise KeyError(f"Expected target column '{target}' not found. Found: {df.columns.tolist()}")

# Drop non-informative or ID-like columns
drop_cols = ["Customer ID", "Lat Long", "Zip Code"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Convert numeric-like strings
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

X = df.drop(columns=[target])
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

num_pipe = Pipeline([
    ("clean", CleanUnknowns()),
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

cat_pipe = Pipeline([
    ("clean", CleanUnknowns()),
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(random_state=42))
])

print(f"✅ Training model on {len(X_train)} samples...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n📊 Evaluation:")
print(classification_report(y_test, y_pred))

joblib.dump(model, "models/churn_model.joblib")
print("\n💾 Model saved to models/churn_model.joblib")
