import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Connect to SQLite database
engine = create_engine(r"sqlite:///../data/medical.db")

# Load dataset
df = pd.read_sql("SELECT * FROM patients", engine)

# Features & Target
X = df.drop("outcome", axis=1)
y = df["outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
xgb.fit(X_train_scaled, y_train)

# Predict
y_pred = xgb.predict(X_test_scaled)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance
plt.figure(figsize=(10, 6))
plt.barh(X.columns, xgb.feature_importances_ ,color = 'red'),
plt.title("Feature Importance (XGBoost)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()
