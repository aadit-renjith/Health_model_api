from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
from functools import wraps

# --- Configuration ---
app = Flask(__name__)

# Use environment variable for API Key, default to a secure key if not set
# IMPORTANT: In Render, set the API_KEY environment variable for production security.
API_KEY = os.getenv("API_KEY", "aadit123securekey") 
MODEL_PATH = "model/xgboost_anomaly_model.pkl"

# --- Model Loading ---
try:
    # Ensure the model is loaded from the correct, final .pkl file path
    model = joblib.load(MODEL_PATH)
    print(f"Successfully loaded model from {MODEL_PATH}")
except FileNotFoundError:
    print(f"FATAL ERROR: Model not found at {MODEL_PATH}.")
    print("Please ensure you have run xgboost_anomaly_detection.py and moved the model file into a 'model/' directory.")
    model = None # Set model to None to handle errors gracefully later

# --- Helper Functions ---

def convert_celsius_to_fahrenheit(celsius):
    """Converts Celsius to Fahrenheit, as the model was trained on °F."""
    return celsius * 9/5 + 32

# --- Authentication decorator ---
def require_api_key(view_function):
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        key = request.headers.get("x-api-key")
        if key and key == API_KEY:
            return view_function(*args, **kwargs)
        else:
            return jsonify({"error": "Unauthorized. Missing or invalid 'x-api-key' header."}), 401
    return decorated_function

# --- API Routes ---

@app.route('/')
def home():
    return jsonify({"message": "Health Monitoring XGBoost API Running 🚀", "model_status": "Loaded" if model else "Error"})

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    if model is None:
        return jsonify({"error": "Model failed to load at startup."}), 500
        
    try:
        data = request.get_json()

        # Extract only the 3 required features
        bpm = data.get('bpm')
        spo2 = data.get('spo2')
        temp_c = data.get('temperature') # Assume input is Celsius

        # 5. Input Validation
        if not all(isinstance(v, (int, float)) for v in [bpm, spo2, temp_c]) or any(v is None for v in [bpm, spo2, temp_c]):
            return jsonify({
                "error": "Invalid or missing vital signs. Required fields: 'bpm', 'spo2', 'temperature' (in °C). All must be numeric."
            }), 400

        # Feature Engineering: Convert C to F for the model
        temp_f = convert_celsius_to_fahrenheit(temp_c)
        
        # 1. Convert to numpy array in the exact order the model expects: 
        # [Heart Rate (bpm), Body Temperature (°F), Blood Oxygen Level (SpO2 %)]
        features = np.array([[bpm, temp_f, spo2]])

        # Model prediction (0 = normal, 1 = anomaly)
        prediction = int(model.predict(features)[0])

        # --- Optional: Check which vital(s) triggered the anomaly (using the original F thresholds) ---
        anomalies = []
        
        # Heart Rate Check (Thresholds: < 40 or > 200)
        if bpm < 40:
            anomalies.append(f"Low Heart Rate ({bpm} bpm)")
        elif bpm > 200:
            anomalies.append(f"High Heart Rate ({bpm} bpm)")

        # SpO2 Check (Threshold: < 90%)
        if spo2 < 90:
            anomalies.append(f"Low Oxygen Level ({spo2} %)")

        # Temperature Check (Thresholds: < 90°F or > 105°F)
        if temp_f < 90:
            anomalies.append(f"Low Body Temperature ({temp_f:.1f}°F / {temp_c:.1f}°C)")
        elif temp_f > 105:
            anomalies.append(f"High Body Temperature ({temp_f:.1f}°F / {temp_c:.1f}°C)")

        # 4. Prepare response based on model output
        if prediction == 0:
            result = {
                "status": "success",
                "prediction": "normal",
                "details": "All vitals are within safe limits according to the model. ✅"
            }
        else:
            result = {
                "status": "success",
                "prediction": "anomaly detected 🚨",
                # If the model predicts an anomaly, but the rule-based check found nothing 
                # (due to the model catching complex interactions), use the model's catch-all.
                "abnormal_vitals": anomalies if anomalies else ["Model detected an irregular pattern."]
            }

        return jsonify(result)

    except Exception as e:
        # A broader exception catch for parsing errors, etc.
        return jsonify({"error": f"An unexpected error occurred during prediction: {str(e)}"}), 500


if __name__ == '__main__':
    # Use the Gunicorn command for local testing (production-like environment)
    # Render's server will automatically handle setting the HOST and PORT from environment variables.
    # For local testing, ensure Gunicorn is installed: pip install gunicorn
    print("Starting Flask application using Gunicorn for local testing (http://127.0.0.1:5000)")
    os.system("gunicorn app:app --bind 0.0.0.0:5000")
