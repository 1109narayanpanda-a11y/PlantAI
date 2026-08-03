from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict_image

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return "✅ PlantAI Backend Running"


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        plant, disease, confidence = predict_image(file)
        return jsonify({
            "plant_name": plant,
            "disease": disease,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
