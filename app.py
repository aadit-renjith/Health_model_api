from flask import Flask, request, jsonify
import joblib
import os
from functools import wraps
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model/xgboost_cv.pkl")

# Example API key (store securely in env variable)
API_KEY = os.getenv("API_KEY", "aadit123securekey")

# Authentication decorator
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

        # Extract numerical features
        heart_rate = data['heart_rate']
        systolic_bp = data['systolic_bp']
        diastolic_bp = data['diastolic_bp']
        spo2 = data['spo2']
        temperature = data['temperature']

        # Arrange features in correct order
        features = np.array([[heart_rate, systolic_bp, diastolic_bp, spo2, temperature]])

        prediction = model.predict(features)

        return jsonify({
            "status": "success",
            "prediction": int(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400



if __name__ == '__main__':
    app.run(debug=True)
