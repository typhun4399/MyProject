import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import joblib

df = pd.read_excel("/Users/pacharaporn/Desktop/งานเบ้บๆ /แฟนแอบเล่นคอมอะ/MyProject/machine learning/Student Depress/student_depression_dataset.xlsx")

df["Financial Stress"] = df["Financial Stress"].replace("?",np.nan)

X = df.drop(["Depression","id"],axis=1)
y = df["Depression"]

cat_col = ["Gender","City","Profession","Sleep Duration","Dietary Habits","Degree","Have you ever had suicidal thoughts ?","Family History of Mental Illness"]
num_col = ["Age","Academic Pressure","Work Pressure","CGPA","Study Satisfaction","Job Satisfaction","Work/Study Hours","Financial Stress"]

cat_tranform = Pipeline(steps=[
    ('simple impute',SimpleImputer(strategy='most_frequent')),
    ('one hot encode',OneHotEncoder(drop="first",sparse_output=False, handle_unknown='ignore'))
])

def round_values(x):
    return np.round(x, 0)

num_tranform = Pipeline(steps=[
    ('impute',KNNImputer(n_neighbors=3)),
    ('round',FunctionTransformer(round_values)),
    ('scaler',StandardScaler())
])

prepreocess = ColumnTransformer(transformers=[
    ('cat',cat_tranform,cat_col),
    ('num',num_tranform,num_col)
])

models = {
    "logistic regression": LogisticRegression(max_iter=500),
    "decision tree": DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42),
    "random forest": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=10, random_state=42, n_jobs=-1),
    "SVM": SVC(),
    "KNeighbors": KNeighborsClassifier(n_neighbors=5)
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

voting_clf = VotingClassifier(
    estimators=list(models.items()),
    voting="hard"
)
vclf = Pipeline(steps=[
    ('prepreocess',prepreocess),
    ('classifier',voting_clf)
])

if __name__ == "__main__":
    vclf.fit(X_train, y_train)
    y_pred_voting = vclf.predict(X_test)
    joblib.dump(vclf, "/Users/pacharaporn/Desktop/งานเบ้บๆ /แฟนแอบเล่นคอมอะ/MyProject/machine learning/Student Depress/student_depression_model.pkl")
    print("Model Done")