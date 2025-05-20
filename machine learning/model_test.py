import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("/Users/pacharaporn/Downloads/winequality-red-2.csv")

bins = (2, 6.5, 8)
group_names = ['bad', 'good']
df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)

label_quality = LabelEncoder()
df['quality'] = label_quality.fit_transform(df['quality'])

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()

sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

print(classification_report(y_test, rfc.predict(X_test)))
print(f"Accuracy: {accuracy_score(y_test, rfc.predict(X_test))}")
print(confusion_matrix(y_test, rfc.predict(X_test)))