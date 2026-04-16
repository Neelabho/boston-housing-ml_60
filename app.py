# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Author: Neelabho Chakraborty - 241005004308 
# PURPOSE: Flask web server for the Boston Housing Price Predictor
# RUN THIS: (after running train_model.py at least once)
# ROUTES:
#   GET  /         → serves the index.html webpage
#   POST /predict  → receives 13 feature values, returns predicted price as JSON
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

# __name__ tells Flask where to find templates/ and static/ folders
app = Flask(__name__)

try:
    model  = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("model.pkl and scaler.pkl loaded successfully")
except FileNotFoundError:
    print("ERROR: model.pkl or scaler.pkl not found.")
    print("Please run: python train_model.py  first.")
    model  = None
    scaler = None

# Route 1: Serve the main page 
@app.route("/")
def home():
    # render_template looks in the templates/ folder for index.html
    return render_template("index.html")

# Route 2: Handle prediction requests 
@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 500

    try:
        # Get the JSON body sent by JavaScript's fetch() call
        data = request.get_json()

        features = [
            float(data["crim"]),     # Per capita crime rate
            float(data["zn"]),       # Residential land zone %
            float(data["indus"]),    # Non-retail business acres %
            float(data["chas"]),     # Charles River dummy (0 or 1)
            float(data["nox"]),      # Nitric oxides concentration
            float(data["rm"]),       # Average rooms per dwelling
            float(data["age"]),      # Units built before 1940 %
            float(data["dis"]),      # Distance to employment centres
            float(data["rad"]),      # Highway accessibility index
            float(data["tax"]),      # Property tax rate
            float(data["ptratio"]), # Pupil-teacher ratio
            float(data["b"]),       # Demographic index
            float(data["lstat"]),   # % lower-status population
        ]

        # Reshape to 2D array [[v1, v2, ..., v13]] 
        features_array = np.array(features).reshape(1, -1)

        # Scale using the same scaler used during training
        features_scaled = scaler.transform(features_array)

        # Run prediction 
        prediction = model.predict(features_scaled)[0]

        # Round to 2 decimal places and return as JSON
        return jsonify({"prediction": round(float(prediction), 2)})

    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e}"}), 400

    except ValueError as e:
        return jsonify({"error": f"Invalid value: {e}"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start the server
if __name__ == "__main__":
    # debug=True → Flask auto-restarts when you save changes to app.py
    # debug=True → Shows detailed error messages in the browser
    # IMPORTANT: Set debug=False before deploying to PythonAnywhere
    app.run(debug=True, port=5000)