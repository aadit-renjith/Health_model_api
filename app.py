from flask import Flask, request, jsonify
import joblib
import os
from functools import wraps
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model/xgboost_cv.pkl")

# API Key (use .env in production)
API_KEY = os.getenv("API_KEY", "aadit123securekey")

# --- Authentication decorator ---
def require_api_key(view_function):
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        key = request.headers.get("x-api-key")
        if key and key == API_KEY:
            return view_function(*args, **kwargs)
        else:
            return jsonify({"error": "Unauthorized"}), 401
    return decorated_function


@app.route('/')
def home():
    return jsonify({"message": "Health Monitoring XGBoost API Running 🚀"})


@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    try:
        data = request.get_json()

        # Expected JSON:
        # {
        #   "bpm": 85,
        #   "systolic_bp": 120,
        #   "diastolic_bp": 80,
        #   "spo2": 98,
        #   "temperature": 98.6
        # }

        bpm = data.get('bpm')
        systolic_bp = data.get('systolic_bp')
        diastolic_bp = data.get('diastolic_bp')
        spo2 = data.get('spo2')
        temperature = data.get('temperature')

        # Validate input
        if None in [bpm, systolic_bp, diastolic_bp, spo2, temperature]:
            return jsonify({"error": "Missing one or more required fields"}), 400

        # Convert to numpy array (order matches model training)
        features = np.array([[bpm, systolic_bp, diastolic_bp, spo2, temperature]])

        # Model prediction (0 = normal, 1 = anomaly)
        prediction = int(model.predict(features)[0])

        # --- Check which vital(s) caused the anomaly ---
        anomalies = []
        if bpm > 200:
            anomalies.append("High Heart Rate")
        elif bpm < 40:
            anomalies.append("Low Heart Rate")

        if spo2 < 90:
            anomalies.append("Low Oxygen Level (SpO₂)")

        if temperature > 105:
            anomalies.append("High Body Temperature")
        elif temperature < 90:
            anomalies.append("Low Body Temperature")

        # Optional: can also use BP limits if needed
        if systolic_bp > 180 or diastolic_bp > 120:
            anomalies.append("High Blood Pressure")
        elif systolic_bp < 90 or diastolic_bp < 60:
            anomalies.append("Low Blood Pressure")

        # Prepare response
        if prediction == 0:
            result = {
                "status": "success",
                "prediction": "normal",
                "details": "All vitals are within normal range ✅"
            }
        else:
            result = {
                "status": "success",
                "prediction": "anomaly detected 🚨",
                "abnormal_vitals": anomalies if anomalies else ["Model detected irregular pattern"]
            }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
