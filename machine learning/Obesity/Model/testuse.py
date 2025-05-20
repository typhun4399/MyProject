import joblib
import pandas as pd

model = joblib.load(r"C:\Users\phunk\Desktop\MyProject\machine learning\Obesity\Model\obesity_model.pkl")
le =  joblib.load(r"C:\Users\phunk\Desktop\MyProject\machine learning\Obesity\Model\label_encoder.pkl")

df = pd.read_csv(r"C:\Users\phunk\Desktop\MyProject\machine learning\Obesity\Data\TestData.csv")

predictions = le.inverse_transform(model.predict(df))

print("Predicted obesity categories:", predictions)