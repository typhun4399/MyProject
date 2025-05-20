import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("/Users/pacharaporn/Downloads/student_depression_dataset.csv")
print(df['Depression'].unique())

le = LabelEncoder()

categorical_cols = ['Gender', 'City', 'Profession', 'Degree', 'Dietary Habits', 
                    'Sleep Duration', 'Have you ever had suicidal thoughts ?', 
                    'Family History of Mental Illness']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='ignore')
df.fillna(df.median(), inplace=True)

X = df.drop(['Depression','id'],axis=1)
y = df['Depression']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

lgr = LogisticRegression()
lgr.fit(X_train, y_train)
y_pred = lgr.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")