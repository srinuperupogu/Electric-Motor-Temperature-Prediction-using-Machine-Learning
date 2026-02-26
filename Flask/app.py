from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Get current directory (Flask folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level → Laddu → then into Model folder
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Model"))

print("Model directory:", MODEL_DIR)  # Debug line

model_path = os.path.join(MODEL_DIR, "model.save")
scaler_path = os.path.join(MODEL_DIR, "transform.save")

print("Loading model from:", model_path)

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        print("Incoming data:", data)

        features = [[
            float(data["ambient"]),
            float(data["coolant"]),
            float(data["u_d"]),
            float(data["u_q"]),
            float(data["motor_speed"]),
            float(data["i_d"]),
            float(data["i_q"])
        ]]

        print("Feature length:", len(features[0]))

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)

        return jsonify({
            "pm_temperature_prediction": float(prediction[0])
        })

    except Exception as e:
        print("ERROR OCCURRED:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)