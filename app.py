from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import requests                  # ← ADDED: needed for the Groq proxy route
import joblib
import numpy as np
from predict import predict_from_base64

app = Flask(__name__)
CORS(app)

# Diabetes
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Heart Disease
heart_model = joblib.load("heart_model.pkl")
heart_scaler = joblib.load("heart_scaler.pkl")

# Hypertension
bp_model = joblib.load("bp_model.pkl")
bp_scaler = joblib.load("bp_scaler.pkl")

# Stroke
stroke_model = joblib.load("stroke_model.pkl")
stroke_scaler = joblib.load("stroke_scaler.pkl")

# Obesity
obesity_model = joblib.load("obesity_model.pkl")
obesity_scaler = joblib.load("obesity_scaler.pkl")

# ── Page Routes ──────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")           # home page

@app.route("/risk-analysis")
def risk_analysis():
    return render_template("risk_analysis.html")

@app.route("/symptom-analyzer")
def symptom_analyzer():
    return render_template("symptom_analyzer.html")

@app.route("/first-aid-assistant")
def first_aid_assistant():
    return render_template("first_aid_assistant.html")

@app.route("/mindguard-ai")
def mindguard_ai():
    return render_template("mindguard_ai.html")

@app.route("/skin-detect")
def skin_detect():
    return render_template("skin_detect.html")

# ── Groq Proxy ───────────────────────────────  ← ADDED
# All HTML files call /api/groq instead of Groq directly.
# The API key is read from the environment variable GROQ_API_KEY,
# which comes from .env locally and from Render's environment tab in production.
@app.route("/api/groq", methods=["POST"])
def groq_proxy():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return jsonify({"error": "GROQ_API_KEY not configured"}), 500

    payload = request.json
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json=payload,
        timeout=30
    )
    return jsonify(response.json()), response.status_code

# ── ML Prediction ────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # ── Diabetes (16 features) ──────────────────────────────
    diabetes_features = np.array([[
        data["HighBP"], data["HighChol"], data["BMI"],
        data["Smoker"], data["PhysActivity"], data["Fruits"],
        data["Veggies"], data["HvyAlcoholConsump"], data["GenHlth"],
        data["MentHlth"], data["PhysHlth"], data["DiffWalk"],
        data["Sex"], data["Age"], data["Education"], data["Income"]
    ]])
    diabetes_prob = model.predict_proba(
        scaler.transform(diabetes_features)
    )[0][2]  # index 2 = diabetes class

    # ── Heart Disease (13 features) ─────────────────────────
    heart_features = np.array([[
        data["HighBP"], data["HighChol"], data["BMI"],
        data["Smoker"], data["PhysActivity"], data["Fruits"],
        data["Veggies"], data["HvyAlcoholConsump"], data["GenHlth"],
        data["PhysHlth"], data["DiffWalk"], data["Sex"], data["Age"]
    ]])
    heart_prob = heart_model.predict_proba(
        heart_scaler.transform(heart_features)
    )[0][1]  # index 1 = heart disease class

    # ── Hypertension (13 features) ──────────────────────────
    bp_features = np.array([[
        data["HighChol"], data["BMI"], data["Smoker"],
        data["PhysActivity"], data["Fruits"], data["Veggies"],
        data["HvyAlcoholConsump"], data["GenHlth"], data["MentHlth"],
        data["PhysHlth"], data["DiffWalk"], data["Sex"], data["Age"]
    ]])
    bp_prob = bp_model.predict_proba(
        bp_scaler.transform(bp_features)
    )[0][1]  # index 1 = high BP class

    # ── Stroke (12 features) ────────────────────────────────
    stroke_features = np.array([[
        data["HighBP"], data["HighChol"], data["BMI"],
        data["Smoker"], data["PhysActivity"], data["Fruits"],
        data["Veggies"], data["HvyAlcoholConsump"], data["GenHlth"],
        data["PhysHlth"], data["Sex"], data["Age"]
    ]])
    stroke_prob = stroke_model.predict_proba(
        stroke_scaler.transform(stroke_features)
    )[0][1]  # index 1 = stroke class

    # ── Obesity (12 features) ───────────────────────────────
    obesity_features = np.array([[
        data["HighBP"], data["HighChol"], data["PhysActivity"],
        data["Fruits"], data["Veggies"], data["HvyAlcoholConsump"],
        data["GenHlth"], data["MentHlth"], data["PhysHlth"],
        data["DiffWalk"], data["Sex"], data["Age"]
    ]])
    obesity_prob = obesity_model.predict_proba(
        obesity_scaler.transform(obesity_features)
    )[0][1]  # index 1 = obese class

    # ── Return all 5 scores ─────────────────────────────────
    return jsonify({
        "risk_probability": float(diabetes_prob),   # kept for backward compatibility
        "diabetes":   round(float(diabetes_prob) * 10, 1),
        "heart":      round(float(heart_prob) * 10, 1),
        "bp":         round(float(bp_prob) * 10, 1),
        "stroke":     round(float(stroke_prob) * 10, 1),
        "obesity":    round(float(obesity_prob) * 10, 1),
    })

@app.route("/skin-predict", methods=["POST"])
def skin_predict():
    data = request.json
    b64_image = data.get("image", "")

    if not b64_image:
        return jsonify({"error": "No image provided"}), 400

    try:
        label_code, readable_name, confidence = predict_from_base64(b64_image)
        return jsonify({
            "label_code": label_code,       # e.g. "mel"
            "disease": readable_name,        # e.g. "Melanoma"
            "confidence": confidence         # e.g. 91.4
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)