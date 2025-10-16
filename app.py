from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
from functools import wraps
import firebase_admin
from firebase_admin import credentials, firestore
import json

# --- Configuration ---
app = Flask(__name__)

# API Key
API_KEY = os.getenv("API_KEY", "aadit123securekey") 
MODEL_PATH = "model/xgboost_anomaly_model.pkl"

# Hardcoded User ID (matches your Firestore path)
HARDCODED_USER_ID = "KNC4mIXcZ0Vp3t8SvhUk841FOvF2"

# --- Firebase Initialization ---
try:
    if os.path.exists("firebase_credentials.json"):
        cred = credentials.Certificate("firebase_credentials.json")
    elif os.getenv("FIREBASE_CREDENTIALS"):
        cred_dict = json.loads(os.getenv("FIREBASE_CREDENTIALS"))
        cred = credentials.Certificate(cred_dict)
    else:
        raise FileNotFoundError("Firebase credentials not found")
    
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase Firestore initialized successfully (READ-ONLY MODE)")
    print(f"📌 Hardcoded User ID: {HARDCODED_USER_ID}")
except Exception as e:
    print(f"⚠️ Firebase initialization failed: {str(e)}")
    db = None

# --- Model Loading ---
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Successfully loaded model from {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Model not found at {MODEL_PATH}")
    model = None

# --- Helper Functions ---
def convert_celsius_to_fahrenheit(celsius):
    """Converts Celsius to Fahrenheit, as model uses °F."""
    return celsius * 9/5 + 32

def get_severity_level(anomalies, bpm, spo2, temp_f):
    if not anomalies:
        return 'normal'
    critical_conditions = [
        bpm < 40 or bpm > 180,
        spo2 < 85,
        temp_f > 104 or temp_f < 92
    ]
    return 'critical' if any(critical_conditions) else 'warning'

def read_vitals_from_firebase(reading_id):
    if db is None:
        print("⚠️ Firestore not initialized")
        return None
    try:
        doc_ref = db.collection('users').document(HARDCODED_USER_ID).collection('health_readings').document(reading_id)
        doc = doc_ref.get()
        if doc.exists:
            print(f"✅ Read data from Firebase: {reading_id}")
            data = doc.to_dict()
            data['reading_id'] = doc.id
            return data
        else:
            print(f"⚠️ Reading not found: {reading_id}")
            return None
    except Exception as e:
        print(f"❌ Error reading from Firebase: {str(e)}")
        return None

def read_latest_vitals_from_firebase():
    if db is None:
        print("⚠️ Firestore not initialized")
        return None
    try:
        readings_ref = db.collection('users').document(HARDCODED_USER_ID).collection('health_readings')
        docs = readings_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).stream()
        for doc in docs:
            data = doc.to_dict()
            data['reading_id'] = doc.id
            print(f"✅ Read latest data from Firebase: {doc.id}")
            return data
        print("⚠️ No readings found in Firebase")
        return None
    except Exception as e:
        print(f"❌ Error reading from Firebase: {str(e)}")
        return None

# --- Authentication ---
def require_api_key(view_function):
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        key = request.headers.get("x-api-key")
        if key and key == API_KEY:
            return view_function(*args, **kwargs)
        else:
            return jsonify({"error": "Unauthorized. Missing or invalid 'x-api-key' header."}), 401
    return decorated_function

# --- Routes ---
@app.route('/')
def home():
    return jsonify({
        "message": "Health Monitoring XGBoost API Running 🚀",
        "model_status": "Loaded" if model else "Error",
        "firebase_status": "Connected (READ-ONLY)" if db else "Disconnected",
        "user_id": HARDCODED_USER_ID,
        "features": ["bpm", "temperature_F", "spo2"],
        "mode": "Firebase Read → Predict → Return to Flutter"
    })

@app.route('/predict/latest', methods=['GET'])
@require_api_key
def predict_latest():
    if model is None:
        return jsonify({"error": "Model failed to load at startup."}), 500
    if db is None:
        return jsonify({"error": "Firebase not initialized."}), 500

    try:
        firebase_data = read_latest_vitals_from_firebase()
        if firebase_data is None:
            return jsonify({"error": "No health readings found in Firebase."}), 404

        # Extract from Firestore fields
        bpm = firebase_data.get('bpm')
        spo2 = firebase_data.get('spo2')
        temp_c = firebase_data.get('temperature_C')
        temp_f = firebase_data.get('temperature_F')

        if temp_c is None and temp_f is not None:
            temp_c = (temp_f - 32) * 5 / 9
        elif temp_c is None:
            return jsonify({"error": "Temperature data not found in Firebase."}), 400

        if temp_f is None:
            temp_f = convert_celsius_to_fahrenheit(temp_c)

        if not all(isinstance(v, (int, float)) for v in [bpm, spo2, temp_f]) or any(v is None for v in [bpm, spo2, temp_f]):
            return jsonify({"error": "Invalid or missing vital signs in Firebase."}), 400

        features = np.array([[bpm, temp_f, spo2]])
        prediction = int(model.predict(features)[0])

        anomalies = []
        if bpm < 40:
            anomalies.append(f"Low Heart Rate ({bpm} bpm)")
        elif bpm > 200:
            anomalies.append(f"High Heart Rate ({bpm} bpm)")
        if spo2 < 90:
            anomalies.append(f"Low Oxygen Level ({spo2}%)")
        if temp_f < 90:
            anomalies.append(f"Low Body Temperature ({temp_f:.1f}°F)")
        elif temp_f > 105:
            anomalies.append(f"High Body Temperature ({temp_f:.1f}°F)")

        severity = get_severity_level(anomalies, bpm, spo2, temp_f)

        result = {
            "status": "success",
            "reading_id": firebase_data.get('reading_id'),
            "timestamp": firebase_data.get('timestamp'),
            "prediction": "normal" if prediction == 0 else "anomaly detected",
            "severity": severity,
            "vitals": {
                "bpm": int(bpm),
                "spo2": int(spo2),
                "temperature_celsius": round(temp_c, 1),
                "temperature_fahrenheit": round(temp_f, 1)
            }
        }

        if prediction == 0:
            result["details"] = "All vitals are within safe limits."
        else:
            result["abnormal_vitals"] = anomalies if anomalies else ["Model detected irregular patterns."]
            result["recommendation"] = (
                "Consult a healthcare professional immediately." if severity == "critical" else "Monitor vitals closely."
            )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Unexpected error during prediction: {str(e)}"}), 500


@app.route('/predict/<reading_id>', methods=['GET'])
@require_api_key
def predict_by_id(reading_id):
    if model is None:
        return jsonify({"error": "Model failed to load at startup."}), 500
    if db is None:
        return jsonify({"error": "Firebase not initialized."}), 500

    try:
        firebase_data = read_vitals_from_firebase(reading_id)
        if firebase_data is None:
            return jsonify({"error": f"Health reading '{reading_id}' not found."}), 404

        bpm = firebase_data.get('bpm')
        spo2 = firebase_data.get('spo2')
        temp_c = firebase_data.get('temperature_C')
        temp_f = firebase_data.get('temperature_F')

        if temp_c is None and temp_f is not None:
            temp_c = (temp_f - 32) * 5 / 9
        elif temp_c is None:
            return jsonify({"error": "Temperature data not found in Firebase."}), 400

        if temp_f is None:
            temp_f = convert_celsius_to_fahrenheit(temp_c)

        features = np.array([[bpm, temp_f, spo2]])
        prediction = int(model.predict(features)[0])

        anomalies = []
        if bpm < 40:
            anomalies.append(f"Low Heart Rate ({bpm} bpm)")
        elif bpm > 200:
            anomalies.append(f"High Heart Rate ({bpm} bpm)")
        if spo2 < 90:
            anomalies.append(f"Low Oxygen Level ({spo2}%)")
        if temp_f < 90:
            anomalies.append(f"Low Body Temperature ({temp_f:.1f}°F)")
        elif temp_f > 105:
            anomalies.append(f"High Body Temperature ({temp_f:.1f}°F)")

        severity = get_severity_level(anomalies, bpm, spo2, temp_f)

        result = {
            "status": "success",
            "reading_id": reading_id,
            "timestamp": firebase_data.get('timestamp'),
            "prediction": "normal" if prediction == 0 else "anomaly detected",
            "severity": severity,
            "vitals": {
                "bpm": int(bpm),
                "spo2": int(spo2),
                "temperature_celsius": round(temp_c, 1),
                "temperature_fahrenheit": round(temp_f, 1)
            }
        }

        if prediction == 0:
            result["details"] = "All vitals are within safe limits."
        else:
            result["abnormal_vitals"] = anomalies if anomalies else ["Model detected irregular patterns."]
            result["recommendation"] = (
                "Consult a healthcare professional immediately." if severity == "critical" else "Monitor vitals closely."
            )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Unexpected error during prediction: {str(e)}"}), 500


if __name__ == '__main__':
    print("🚀 Starting Flask application with Firebase Firestore integration")
    print(f"📊 Model Status: {'Loaded' if model else 'Failed'}")
    print(f"🔥 Firebase Status: {'Connected (READ-ONLY)' if db else 'Disconnected'}")
    print(f"👤 Hardcoded User: {HARDCODED_USER_ID}")
    print("📖 Mode: Read from Firebase → Predict → Return to Flutter")
    os.system("gunicorn app:app --bind 0.0.0.0:$PORT")
