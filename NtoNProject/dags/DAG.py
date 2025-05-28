from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import joblib

def score_new_data():
    import pandas as pd
    model = joblib.load("model/churn_model.pkl")
    encoders = joblib.load("model/label_encoders.pkl")

    # Load new data
    df_new = pd.read_csv("data/churn_telco_customers.csv")  # mock new data
    for col, le in encoders.items():
        if col in df_new.columns:
            df_new[col] = le.transform(df_new[col])

    X_new = df_new.drop(columns=["Churn", "customer_id"], errors='ignore')
    df_new["Churn_score"] = model.predict_proba(X_new)[:, 1]
    df_new.to_csv("data/churn_scored.csv", index=False)

    print("Scoring completed.")

# Define DAG
with DAG(
    dag_id="daily_churn_scoring",
    start_date=datetime(2023, 1, 1),
    schedule_interval="@daily",
    catchup=False,
) as dag:

    task_score = PythonOperator(
        task_id="score_new_data",
        python_callable=score_new_data
    )

    task_score
