from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Correct model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Model"))

model = joblib.load(os.path.join(MODEL_DIR, "model.save"))
scaler = joblib.load(os.path.join(MODEL_DIR, "transform.save"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = [
            data["ambient"],
            data["coolant"],
            data["u_d"],
            data["u_q"],
            data["motor_speed"],
            data["i_d"],
            data["i_q"]
        ]

        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)

        return jsonify({
            "pm_temperature_prediction": float(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)