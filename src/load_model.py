import mlflow.sklearn

import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv(override=True)

model_uri = os.getenv("MODEL_URI")

model = mlflow.sklearn.load_model(model_uri)

print("Model başarıyla yüklendi!")
print(type(model))

df = pd.read_csv(
    "industrial-machine-condition-classification/student_dataset.csv"
)

X_sample = df.drop(columns=["unit_id", "condition"]).head(5)

predictions = model.predict(X_sample)

print("Tahminler:", predictions)
print("Gerçek değerler:", df["condition"].head(5).to_numpy())