import numpy as np
import json
import os
import gdown
from PIL import Image
from tensorflow.keras.models import load_model

MODEL_FILE = "plant_model.keras"
CLASS_FILE = "class_names.json"

MODEL_ID = "1FhG9QuYGmuiC7Ki8xZod_ZicS2F7ayJU"
CLASS_ID = "1b-ZyA6OI8SFmh8WmJvophnwPTpqLTrCH"

if not os.path.exists(MODEL_FILE):
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_FILE, quiet=False)

if not os.path.exists(CLASS_FILE):
    gdown.download(f"https://drive.google.com/uc?id={CLASS_ID}", CLASS_FILE, quiet=False)

model = load_model(MODEL_FILE)

with open(CLASS_FILE) as f:
    class_names = json.load(f)["class_names"]

def predict_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0)

    preds = model.predict(arr)[0]
    index = int(np.argmax(preds))

    label = class_names[index]
    confidence = float(preds[index]) * 100

    if "___" in label:
        plant, disease = label.split("___")
    else:
        plant, disease = label, "Unknown"

    return plant, disease.replace("_", " "), round(confidence, 2)
