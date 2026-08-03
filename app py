from flask import Flask, request, jsonify
from model import predict_image

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ PlantAI Backend Running"

@app.route("/api/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    plant, disease, confidence = predict_image(file)

    return jsonify({
        "plant_name": plant,
        "disease": disease,
        "confidence": confidence
    })
