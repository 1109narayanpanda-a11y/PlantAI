import numpy as np
import json
from PIL import Image
from tensorflow.keras.models import load_model

# Model load later (file Render server pe hoga)
model = load_model("plant_model.keras")

with open("class_names.json") as f:
    class_names = json.load(f)["class_names"]

def predict_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize((224,224))
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr,0)

    preds = model.predict(arr)[0]
    index = np.argmax(preds)

    label = class_names[index]
    confidence = float(preds[index])*100

    plant, disease = label.split("___")

    return plant, disease, round(confidence,2)
