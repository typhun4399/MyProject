import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.calibration import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("/Users/pacharaporn/Downloads/ObesityDataSet_raw_and_data_sinthetic.csv")

le = LabelEncoder()
df['NObeyesdad'] = le.fit_transform(df['NObeyesdad'])
df['Gender'] = le.fit_transform(df['Gender'])
df['family_history_with_overweight'] = le.fit_transform(df['family_history_with_overweight'])
df['FAVC'] = le.fit_transform(df['FAVC'])
df['SMOKE'] = le.fit_transform(df['SMOKE'])
df['SCC'] = le.fit_transform(df['SCC'])
df['CAEC'] = le.fit_transform(df['CAEC'])
df['CALC'] = le.fit_transform(df['CALC'])
df['MTRANS'] = le.fit_transform(df['MTRANS'])

X = df.drop(['NObeyesdad'], axis=1)
y = df['NObeyesdad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Test set accuracy: {:.2f}".format(rfc.score(X_test, y_test)))
print("Train set accuracy: {:.2f}".format(rfc.score(X_train, y_train)))