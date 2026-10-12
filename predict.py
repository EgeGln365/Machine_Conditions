import joblib
import pandas as pd

# Eğitilmiş modeli yükle
model = joblib.load("machine_model.joblib")

# Örnek makine verisi
machine = pd.DataFrame([{
    "unit_type": "M",
    "ambient_temp": 25.0,
    "process_temp": 65.0,
    "coolant_flow": 12.0,
    "shaft_speed": 1500,
    "drive_torque": 40.0,
    "power_load": 6.0,
    "component_wear": 100,
    "thermal_stress": 50.0,
    "mechanical_load": 80.0,
    "vibration_index": 3.0
}])

prediction = model.predict(machine)
probabilities = model.predict_proba(machine)

labels = {
    0: "Normal",
    1: "Thermal Failure",
    2: "Mechanical Failure"
}

print("Tahmin:", labels[int(prediction[0])])
print("Olasılıklar:", probabilities[0]) 