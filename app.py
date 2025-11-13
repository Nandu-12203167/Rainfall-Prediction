from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# === Absolute Paths (for Windows) ===
# BASE_DIR = r".\models"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "models")

MODEL_PATH = os.path.join(BASE_DIR, "rainfall_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.pkl")

LOG_DIR = os.path.join(SCRIPT_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'app.log'), 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === Initialize variables ===
model, scaler, metadata = None, None, None

# === Load model, scaler, metadata ===
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully: {MODEL_PATH}")
    else:
        print(f"❌ Model not found at {MODEL_PATH}")

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print(f"✅ Scaler loaded successfully: {SCALER_PATH}")
    else:
        print(f"❌ Scaler not found at {SCALER_PATH}")

    if os.path.exists(METADATA_PATH):
        metadata = joblib.load(METADATA_PATH)
        print(f"✅ Metadata loaded successfully: {METADATA_PATH}")
        print(f"📊 Model Name: {metadata.get('model_name', 'Unknown')}")
        print(f"🎯 Model Accuracy: {metadata.get('accuracy', 0):.4f}")
    else:
        print(f"❌ Metadata not found at {METADATA_PATH}")

except Exception as e:
    print(f"⚠️ Error loading model/scaler/metadata: {e}")
    model, scaler, metadata = None, None, None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Predict rainfall based on input features."""
    try:
        # Check if model/scaler loaded
        if model is None or scaler is None:
            return jsonify({'error': 'Model or scaler not loaded properly. Please check .pkl paths.'}), 500

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data received.'}), 400

        # Extract and format features
        features = [
            float(data.get('pressure', 0)),
            float(data.get('maxtemp', 0)),
            float(data.get('temperature', 0)),
            float(data.get('mintemp', 0)),
            float(data.get('dewpoint', 0)),
            float(data.get('humidity', 0)),
            float(data.get('cloud', 0)),
            float(data.get('sunshine', 0)),
            float(data.get('winddirection', 0)),
            float(data.get('windspeed', 0))
        ]

        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        # Predict
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]

        # Build result
        result = {
            'prediction': 'Rain' if prediction == 1 else 'No Rain',
            'confidence': float(max(prediction_proba) * 100),
            'probability': {
                'no_rain': float(prediction_proba[0] * 100),
                'rain': float(prediction_proba[1] * 100)
            }
        }

        return jsonify(result)

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/model-info', methods=['GET'])
def model_info():
    """Display model information."""
    if metadata:
        return jsonify({
            'model_name': metadata.get('model_name', 'Unknown'),
            'accuracy': f"{metadata.get('accuracy', 0):.2%}",
            'features': metadata.get('features', [])
        })
    else:
        return jsonify({'error': 'Metadata not loaded.'}), 500


if __name__ == '__main__':
    print("🚀 Flask app starting on http://127.0.0.1:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
