import pandas as pd

# โหลดไฟล์ต้นฉบับ
df = pd.read_csv(r"C:\Users\phunk\Desktop\MyProject\machine learning\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# สุ่มเลือก 20 แถวเป็น mock data
df_mock = df.sample(n=2000, random_state=42).copy()

# ลบคอลัมน์ Churn (เป้าหมายที่เราต้องการให้โมเดลทำนาย)
if 'Churn' in df_mock.columns:
    df_mock.drop(columns=['Churn'], inplace=True)

# บันทึกเป็นไฟล์ใหม่สำหรับให้โมเดลทำนาย
output_path = "data/churn_telco_customers.csv"
df_mock.to_csv(output_path, index=False)

output_path
