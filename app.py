from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from openai import OpenAI
from model import predict_image

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route("/")
def home():
    return "✅ DoctorAI Backend Running"


@app.route("/api/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    plant, disease, confidence = predict_image(file)

    return jsonify({
        "plant_name": plant,
        "disease": disease,
        "confidence": confidence
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are FloraBot, an AI plant health assistant helping farmers."},
            {"role": "user", "content": user_message}
        ],
        temperature=0.5
    )

    reply = response.choices[0].message.content

    return jsonify({"reply": reply})
