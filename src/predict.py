import joblib
import numpy as np
from src.preprocessing import load_data


def load_files():
  variance_selector = joblib.load(
      "models/variance_selector.pkl"
  )

  model = joblib.load("models/best_model.pkl")
  scaler = joblib.load("models/scaler.pkl")
  selector = joblib.load("models/feature_selector.pkl")
  encoder = joblib.load("models/label_encoder.pkl")

  return (
    model,
    scaler,
    variance_selector,
    selector,
    encoder
 )
    

def predict_sample(sample):

    model, scaler, variance_selector, selector, encoder = load_files()

    sample = np.array(sample).reshape(1, -1)

    sample = variance_selector.transform(sample)

    sample = scaler.transform(sample)

    sample = selector.transform(sample)
    
    prediction = model.predict(sample)

    predicted_class = encoder.inverse_transform(prediction)[0]

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)
        confidence = float(np.max(probabilities))

    return predicted_class, confidence

def predict_from_array(sample):

    prediction, confidence = predict_sample(sample)

    print("=" * 60)

    print("Prediction")

    print("=" * 60)

    print("Cancer Type :", prediction)

    if confidence is not None:

        print(f"Confidence : {confidence * 100:.2f}%")

    print("=" * 60)

    return prediction

import pandas as pd
from src.preprocessing import load_data


def predict_existing_patient(sample_name):

    data = load_data()

    row = data[data["Sample"] == sample_name]

    true_label = row["Class"].values[0]

    sample = row.drop(
        columns=["Sample", "Class"]
    ).values[0]

    prediction, confidence = predict_sample(sample)

    return prediction, confidence, true_label


def predict_uploaded_csv(file):

    data = pd.read_csv(file)

    sample = data.iloc[0].values

    prediction, confidence = predict_sample(sample)

    return prediction, confidence