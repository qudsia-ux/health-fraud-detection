from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load saved files
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route("/")
def home():
    return "Fraud Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Convert input to DataFrame
    df = pd.DataFrame([data])

    # Match training format
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    # Apply preprocessing
    df = pd.DataFrame([data])
    df = df.reindex(columns=columns, fill_value=0)

    # Predict

    # Get probability
    probability = model.predict_proba(df)[0][1]  # fraud probability

    if probability >= 0.25:
        prediction = 1
    else:
        prediction = 0
    return jsonify({
        "prediction": "Fraudulent Claim" if prediction == 1 else "Genuine Claim",
        "confidence": round(probability * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)