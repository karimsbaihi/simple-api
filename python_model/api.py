# api.py
import os
import pickle
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from flask import Flask, request, jsonify
from PIL import Image

# -------------------- Setup --------------------
app = Flask(__name__)

# Device
device = torch.device("cpu")  # Render generally doesn't provide GPU

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")
MAPPINGS_PATH = os.path.join(BASE_DIR, "class_mappings.pkl")

# -------------------- Load Model --------------------
model = models.resnet34(pretrained=False)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.fc.in_features, 200)
)

# Load checkpoint (weights + extra metadata)
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# -------------------- Load Class Mappings --------------------
with open(MAPPINGS_PATH, "rb") as f:
    class_to_idx, wnid_to_words = pickle.load(f)
idx_to_word = {v: wnid_to_words[k] for k, v in class_to_idx.items()}

# -------------------- Image Transform --------------------
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                         std=[0.2296, 0.2263, 0.2255])
])

# -------------------- Prediction Function --------------------
def predict_image(img: Image.Image):
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)[0] * 100
        _, pred = torch.max(output, 1)

    # Top 5 predictions
    top5_probs, top5_classes = torch.topk(probabilities, 5)
    top5 = [{"class": idx_to_word[i.item()], "confidence": float(p.item())}
            for i, p in zip(top5_classes, top5_probs)]
    return {"prediction": idx_to_word[pred.item()], "top5": top5}

# -------------------- Routes --------------------
@app.route("/")
def home():
    return jsonify({"message": "Tiny ImageNet prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
        result = predict_image(img)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- Run --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
