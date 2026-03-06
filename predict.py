import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""        # force CPU — no GPU on Render
os.environ["OMP_NUM_THREADS"] = "1"            # limit OpenMP threads
os.environ["MKL_NUM_THREADS"] = "1"            # limit MKL threads

import torch
torch.set_num_threads(1)                        # limit PyTorch threads for free tier
torch.set_num_interop_threads(1)

import torchvision
from torchvision import transforms
from PIL import Image
from torch import nn
import json
import base64
import io

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

# Load model
model = torchvision.models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load("skin_model.pth", map_location="cpu"))
model.eval()

# Pre-warm the model with a dummy pass so first real request is faster
with torch.no_grad():
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    model(dummy)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def predict_from_base64(b64_string):
    # Strip the "data:image/jpeg;base64," prefix if present
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]

    image_bytes = base64.b64decode(b64_string)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probs, 0)

    label_code = class_names[predicted.item()]
    readable_name = label_mapping.get(label_code, label_code)
    return label_code, readable_name, round(confidence.item() * 100, 2)


# Keep original file-path function for testing from terminal
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probs, 0)
    label_code = class_names[predicted.item()]
    readable_name = label_mapping.get(label_code, label_code)
    return readable_name, round(confidence.item() * 100, 2)

if __name__ == "__main__":
    path = input("Enter image path: ")
    disease, conf = predict_image(path)
    print(f"\nPredicted Disease: {disease}")
    print(f"Confidence: {conf}%")