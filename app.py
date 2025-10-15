from flask import Flask, request, jsonify
import joblib
import os
from functools import wraps
import numpy as np

app = Flask(__name__)

# Load the trained XGBoost model
model = joblib.load("model/xgboost_cv.pkl")

# Example API key (store securely using .env in production)
API_KEY = os.getenv("API_KEY", "aadit123securekey")

# --- API Key Decorator ---
def require_api_key(view_function):
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        key = request.headers.get("x-api-key")
        if key and key == API_KEY:
            return view_function(*args, **kwargs)
        else:
            return jsonify({"error": "Unauthorized"}), 401
    return decorated_function


# --- Home Route ---
@app.route('/')
def home():
    return jsonify({"message": "Health Monitoring XGBoost API Running 🚀"})


# --- Prediction Route ---
@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    try:
        data = request.get_json()

        # Expecting JSON input in this format:
        # { "bpm": 85, "spo2": 98, "temperature_C": 98.6 }

        bpm = data.get('bpm')
        spo2 = data.get('spo2')
        temperature = data.get('temperature_C')

        # Validate input
        if bpm is None or spo2 is None or temperature is None:
            return jsonify({
                "error": "Missing one or more required fields: bpm, spo2, temperature"
            }), 400

        # Prepare data for model prediction
        features = np.array([[bpm, spo2, temperature]])

        # Run model prediction
        prediction = model.predict(features)

        # Map model output to readable labels
        if int(prediction[0]) == 1:
            result = "Anomaly Detected"
        else:
            result = "Normal"

        return jsonify({
            "status": "success",
            "prediction": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
