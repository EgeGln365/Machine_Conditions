from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import os
from dotenv import load_dotenv
import mlflow.sklearn

load_dotenv(override=True)

model_uri = os.getenv("MODEL_URI")
model = mlflow.sklearn.load_model(model_uri)


app = FastAPI(title="Machine Condition API")

@app.get("/")
def home():
    return {"message": "Machine Condition API is running"}

class SensorInput(BaseModel):
    unit_type: str
    ambient_temp: float
    process_temp: float
    coolant_flow: float
    shaft_speed: float
    drive_torque: float
    power_load: float
    component_wear: float
    thermal_stress: float
    mechanical_load: float
    vibration_index: float | None = None

@app.post("/test")
def test_sensor(data:SensorInput):
    return{
        "ambient_temp": data.ambient_temp,
        "process_temp": data.process_temp,
        "message": "Sensor data received successfully"
    }

CONDITION_LABELS = {
    0: "Normal",
    1: "Thermal Failure",
    2: "Mechanical Failure"
}

@app.post("/predict")
def predict(data:SensorInput):
    
    input_df = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_df)

    condition = int(prediction[0])

    return {
        "condition": int(prediction[0]),
        "label": CONDITION_LABELS[condition]
    }