import joblib
import pandas as pd
import numpy as np

def round_values(x):
    return np.round(x, 0)

# โหลดโมเดล
model = joblib.load('/Users/pacharaporn/Desktop/งานเบ้บๆ /แฟนแอบเล่นคอมอะ/MyProject/machine learning/Student Depress/student_depression_model.pkl')

# โหลดข้อมูล
df = pd.read_excel("machine learning/Student Depress/Use_Test.xlsx")

# แก้ค่า missing
df["Financial Stress"] = df["Financial Stress"].replace("?", np.nan)

# พยากรณ์
predictions = model.predict(df)

print("Prediction : ", predictions)