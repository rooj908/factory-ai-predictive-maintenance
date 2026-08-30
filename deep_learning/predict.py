import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


MODEL_PATH = "deep_learning/model/factory_failure_mlp.keras"
PREPROCESSOR_PATH = "deep_learning/model/preprocessor.pkl"


def predict_failure(machine_data):
    """
    Predict failure risk within the next 24 hours.

    machine_data should contain the same features used during training.
    """

    # Load trained model and preprocessor
    model = load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Convert input to DataFrame
    df = pd.DataFrame([machine_data])

    # Features used during training
    features = [
        "vibration_rms",
        "temperature_motor",
        "current_phase_avg",
        "pressure_level",
        "rpm",
        "operating_mode",
        "hours_since_maintenance",
        "ambient_temp",
        "machine_type"
    ]

    df = df[features]

    # Handle missing values
    numeric_features = [
        "vibration_rms",
        "temperature_motor",
        "current_phase_avg",
        "pressure_level",
        "rpm",
        "hours_since_maintenance",
        "ambient_temp"
    ]

    categorical_features = [
        "operating_mode",
        "machine_type"
    ]

    for col in numeric_features:
        if df[col].isnull().any():
            df[col] = df[col].fillna(0)

    for col in categorical_features:
        if df[col].isnull().any():
            df[col] = df[col].fillna("unknown")

    # Preprocess
    X = preprocessor.transform(df)
    X = np.asarray(X).astype("float32")

    # Prediction
    probability = float(model.predict(X, verbose=0)[0][0])

    failure_percentage = probability * 100

    # Risk classification
    if failure_percentage >= 70:
        risk = "HIGH"
    elif failure_percentage >= 30:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "failure_probability": round(failure_percentage, 2),
        "risk_level": risk
    }


if __name__ == "__main__":

    # Example machine
    sample_machine = {
        "vibration_rms": 0.88,
        "temperature_motor": 41.39,
        "current_phase_avg": 4.44,
        "pressure_level": 22.2,
        "rpm": 881.9,
        "operating_mode": "idle",
        "hours_since_maintenance": 274.70,
        "ambient_temp": 60.10,
        "machine_type": "CNC"
    }

    result = predict_failure(sample_machine)

    print("\n" + "=" * 50)
    print("FACTORY AI - DEEP LEARNING PREDICTION")
    print("=" * 50)

    print(f"\nFailure Probability: {result['failure_probability']}%")
    print(f"Risk Level: {result['risk_level']}")

    print("\nPrediction completed successfully.")