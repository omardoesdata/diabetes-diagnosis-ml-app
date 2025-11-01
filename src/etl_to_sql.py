import pandas as pd
from sqlalchemy import create_engine

# === CONFIG ===
DATA_PATH = "../data/diabetes.csv"   # relative path to your CSV
DB_PATH = "../data/medical.db"
TABLE_NAME = "patients"

# === LOAD CSV ===
df = pd.read_csv(DATA_PATH)

print("Preview of dataset:")
print(df.head())

# === CLEANING ===
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# === LOAD TO SQLITE ===
engine = create_engine(f"sqlite:///{DB_PATH}")
df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)

print(f"\nData loaded successfully into {DB_PATH}, table '{TABLE_NAME}'")
