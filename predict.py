import os
import json
import base64
import io
import numpy as np
from PIL import Image
import onnxruntime as ort

IMG_SIZE = 224

# Load class names
with open("classes.json") as f:
    class_names = json.load(f)

label_mapping = {
    "nv":    "Melanocytic Nevus (Mole)",
    "mel":   "Melanoma",
    "bcc":   "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "vasc":  "Vascular Lesion"
}

# Load ONNX model — uses ~50MB RAM vs ~500MB for PyTorch
# Both skin_model.onnx and skin_model_onnx.data must be in the same folder
session = ort.InferenceSession(
    "skin_model.onnx",
    providers=["CPUExecutionProvider"]
)

def preprocess(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5                   # normalize same as training
    arr = arr.transpose(2, 0, 1)              # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)         # add batch dim
    return arr

def predict_from_base64(b64_string):
    """
    Accepts a base64 image string (as sent from the browser via fetch).
    Strips the data URL prefix if present, decodes, and runs inference.
    """
    # Strip the "data:image/jpeg;base64," prefix if present
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]

    image_bytes = base64.b64decode(b64_string)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_arr = preprocess(image)
    outputs = session.run(None, {"input": input_arr})
    logits = outputs[0][0]

    # Softmax
    e = np.exp(logits - np.max(logits))
    probs = e / e.sum()

    predicted = int(np.argmax(probs))
    confidence = float(probs[predicted]) * 100

    label_code = class_names[predicted]
    readable_name = label_mapping.get(label_code, label_code)
    return label_code, readable_name, round(confidence, 2)


# Keep file-path function for local testing from terminal
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_arr = preprocess(image)
    outputs = session.run(None, {"input": input_arr})
    logits = outputs[0][0]

    e = np.exp(logits - np.max(logits))
    probs = e / e.sum()

    predicted = int(np.argmax(probs))
    confidence = float(probs[predicted]) * 100

    label_code = class_names[predicted]
    readable_name = label_mapping.get(label_code, label_code)
    return readable_name, round(confidence, 2)


if __name__ == "__main__":
    path = input("Enter image path: ")
    disease, conf = predict_image(path)
    print(f"\nPredicted Disease: {disease}")
    print(f"Confidence: {conf}%")