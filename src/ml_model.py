import pandas as pd
from sqlalchemy import create_engine 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

engine = create_engine(r"sqlite:///../data/medical.db")
df = pd.read_sql_query("SELECT * FROM patients", engine)
print("data loaded", df.shape)

#separating features and targets
X = df.drop("outcome", axis=1)
y = df["outcome"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train size:", X_train.shape,"| Test size:", X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#logisiticRegression

log_regression = LogisticRegression(max_iter=1000)
log_regression.fit(X_train_scaled, y_train)
y_pred_logregression = log_regression.predict(X_test_scaled)

print("\n Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_logregression))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logregression))
print("Classification Report:\n", classification_report(y_test, y_pred_logregression))


#XGBoost
xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("\n===== XGBoost Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))