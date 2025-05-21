import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib

df = pd.read_csv(r"C:\Users\phunk\Desktop\MyProject\machine learning\Obesity\Data\ObesityDataSet.csv")

X = df.drop("NObeyesdad",axis=1)
le = LabelEncoder()
y = le.fit_transform(df["NObeyesdad"])

num_col = ["Age","Height","Weight","FCVC","NCP","CH2O","FAF","TUE"]
cat_col = ["Gender","family_history_with_overweight","FAVC","CAEC","SMOKE","SCC","CALC","MTRANS"]

num_tranform = Pipeline(steps=[
    ("impute",KNNImputer(n_neighbors=5)),
    ("scale",StandardScaler())
])

cat_tranform = Pipeline(steps=[
    ("ordinal",OrdinalEncoder()),
    ("impute",KNNImputer(n_neighbors=5))
])

preprocess = ColumnTransformer(transformers=[
    ("num",num_tranform,num_col),
    ("cat",cat_tranform,cat_col)
])

models = {
    "logistic regression" : LogisticRegression(max_iter=500),
    "decision tree" : DecisionTreeClassifier(),
    "random forrest" : RandomForestClassifier(n_estimators=300),
    "SVM" : SVC(),
    "KNeighbors":KNeighborsClassifier(n_neighbors=3)
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = VotingClassifier(
    estimators=list(models.items()),
    voting="hard"
    
)
vclf = Pipeline(steps=[
    ("preprocess",preprocess),
    ("classifier",clf)
])

if __name__ == "__main__":
    vclf.fit(X_train,y_train)
    y_pred = vclf.predict(X_test)
    joblib.dump(vclf, r"C:\Users\phunk\Desktop\MyProject\machine learning\Obesity\Model\obesity_model.pkl")
    joblib.dump(le, r"C:\Users\phunk\Desktop\MyProject\machine learning\Obesity\Model\label_encoder.pkl")
    print("Model Saved")