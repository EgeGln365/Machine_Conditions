from fastapi.testclient import TestClient
from src.api import app
import pandas as pd


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Machine Condition API is running"


def test_predict():
    df = pd.read_csv(
        "/Users/egegulunay/PROJECTS/Machine_Cond/industrial-machine-condition-classification/student_dataset.csv"
    )

    sample = df.drop(
        columns=["unit_id", "condition"]
    ).iloc[0]

    payload = sample.to_dict()

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["condition"] in [0, 1, 2]
    expected_labels = {
        0: "Normal",
        1: "Thermal Failure",
        2: "Mechanical Failure"
    }

    assert result["label"] == expected_labels[result["condition"]]
    assert result["label"] in [
        "Normal",
        "Thermal Failure",
        "Mechanical Failure"
    ]

def test_invalid_input():
    payload = {
        "unit_type": "M",
        "ambient_temp": "çok sıcak",
        "process_temp": 60.2
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

def test_missing_vibration():
    df = pd.read_csv(
        "industrial-machine-condition-classification/student_dataset.csv"
    )

    sample = df.drop(
        columns=["unit_id", "condition"]
    ).dropna().iloc[0]

    payload = sample.to_dict()

    payload.pop("vibration_index")

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["condition"] in [0, 1, 2]