import requests
import os

HF_API_URL = "https://api-inference.huggingface.co/models/narayna4545/plantai-disease-model"
HF_TOKEN = os.environ.get("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def predict_image(file):
    image_bytes = file.read()

    response = requests.post(
        HF_API_URL,
        headers=headers,
        data=image_bytes
    )

    result = response.json()

    if isinstance(result, dict) and "error" in result:
        raise Exception(result["error"])

    top = result[0]

    label = top["label"]
    confidence = round(top["score"] * 100, 2)

    if "___" in label:
        plant, disease = label.split("___")
    else:
        plant, disease = label, "Unknown"

    return plant, disease.replace("_", " "), confidence
