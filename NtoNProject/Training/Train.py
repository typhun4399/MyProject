import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

df = pd.read_csv(r"C:\Users\phunk\Desktop\MyProject\machine learning\WA_Fn-UseC_-Telco-Customer-Churn.csv")

cat_cols = df.select_dtypes(include='object').columns.tolist()

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=['Churn','customerID'], errors='ignore')
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

report = classification_report(y_test, y_pred, output_dict=True)
roc_auc = roc_auc_score(y_test, y_prob)

print(classification_report(y_test, y_pred))
print(roc_auc_score(y_test, y_prob))

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/churn_model.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")

with open("model/model_evaluation.json", "w") as f:
    import json
    json.dump({"classification_report": report, "roc_auc": roc_auc}, f, indent=4)

print("Model training complete. Artifacts saved.")