from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import random as r
import pandas as pd

# โหลดข้อมูล
df = pd.read_csv("/Users/pacharaporn/Downloads/winequality-red-2.csv")

# แยกตัวแปรอิสระ (X) และตัวแปรตาม (y)
X = df.drop("quality", axis=1)
y = df["quality"]

def get_max_r2():
    r_array = []  # เก็บค่า R²
    ts_array = []  # เก็บค่า test_size

    for z in range(1, 1000):
        ts = round(r.uniform(0.2, 0.8), 3)  # สุ่มค่า test_size
        
        try:
            # แบ่งข้อมูล
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)

            sc = StandardScaler()
            sc.fit(X_train)
            X_train = sc.transform(X_train)
            X_test = sc.transform(X_test)

            # สร้างและฝึกโมเดล
            model = LinearRegression()
            model.fit(X_train, y_train)

            # ทำนายผลลัพธ์และคำนวณค่า R²
            y_pred = model.predict(X_test)
            r2 = round(r2_score(y_test, y_pred), 3)

            # บันทึกค่า
            r_array.append(r2)
            ts_array.append(ts)

        except ValueError as e:
            print(f"Skipping test_size={ts} due to error: {e}")

    max_r2 = max(r_array)
    best_ts = ts_array[r_array.index(max_r2)]

    print(f"Max R²: {max_r2}")
    print(f"Best Test size: {best_ts}")

def model_test():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = round(r2_score(y_test, y_pred),3)
    print(f"R²: {r2}")

get_max_r2()