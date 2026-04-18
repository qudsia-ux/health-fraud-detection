from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)

# -------------------------------
# LOAD MODELS + PREPROCESSORS
# -------------------------------
rf_model = pickle.load(open("rf_model.pkl", "rb"))
lr_model = pickle.load(open("lr_model.pkl", "rb"))
dt_model = pickle.load(open("dt_model.pkl", "rb"))

imputer = pickle.load(open("imputer.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# -------------------------------
# HELPER FUNCTION
# -------------------------------
def get_prediction_label(prob):
    return "Fraudulent" if prob >= 0.4 else "Genuine"

# -------------------------------
# HOME ROUTE
# -------------------------------
@app.route("/")
def home():
    return "Fraud Detection API Running"

# -------------------------------
# PREDICTION ROUTE
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    print("Incoming Data:", data)  # DEBUG

    # Convert input to DataFrame
    df = pd.DataFrame([data])

    # One-hot encoding ONLY for Diagnosis
    df = pd.get_dummies(df, columns=["Diagnosis"])

    # FIX: Ensure all training columns exist
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    # Ensure correct order
    df = df[columns]

    print("Processed DF:", df)  # DEBUG

    # Apply preprocessing
    df = imputer.transform(df)
    df = scaler.transform(df)

    # -------------------------------
    # MODEL PREDICTIONS
    # -------------------------------
    rf_prob = rf_model.predict_proba(df)[0][1]
    lr_prob = lr_model.predict_proba(df)[0][1]
    dt_prob = dt_model.predict_proba(df)[0][1]

    print("RF:", rf_prob, "LR:", lr_prob, "DT:", dt_prob)  # DEBUG

    rf_label = get_prediction_label(rf_prob)
    lr_label = get_prediction_label(lr_prob)
    dt_label = get_prediction_label(dt_prob)

    return jsonify({
        "Random Forest": {
            "prediction": rf_label,
            "probability": round(rf_prob * 100, 2)
        },
        "Logistic Regression": {
            "prediction": lr_label,
            "probability": round(lr_prob * 100, 2)
        },
        "Decision Tree": {
            "prediction": dt_label,
            "probability": round(dt_prob * 100, 2)
        }
    })

# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)