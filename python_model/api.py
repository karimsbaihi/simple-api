import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify
import pickle
import os
import logging
import gc

# ------------------- Logging -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cpu')

# ------------------- Paths -------------------
def get_file_path(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

# ------------------- Load Model -------------------
def load_model():
    try:
        checkpoint = torch.load(get_file_path('simple_model.pt'), map_location=device)
        model = models.resnet34(weights=None)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()
        model.fc = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(model.fc.in_features, 200))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device).eval()
        del checkpoint
        gc.collect()

        with open(get_file_path('class_mappings.pkl'), 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, tuple):  # (class_to_idx, wnid_to_words)
                class_to_idx, wnid_to_words = data
            elif isinstance(data, dict):
                class_to_idx = data.get('class_to_idx', data)
                wnid_to_words = data.get('wnid_to_words', {k: f"class_{v}" for k, v in class_to_idx.items()})
            else:
                raise ValueError("Unknown mapping format")
        idx_to_word = {v: wnid_to_words[k] for k, v in class_to_idx.items()}

        logger.info(f"Model loaded: {len(idx_to_word)} classes")
        return model, idx_to_word
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise

model, idx_to_word = load_model()

# ------------------- Transform -------------------
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2296, 0.2263, 0.2255])
])

# ------------------- Prediction -------------------
def predict_top5(image_file):
    img = Image.open(image_file).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)[0] * 100
        top5_probs, top5_idx = torch.topk(probs, 5)
    return [{"class": idx_to_word[i.item()], "confidence": float(p.item())} for i, p in zip(top5_idx, top5_probs)]

# ------------------- Flask App -------------------
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": True, "num_classes": len(idx_to_word)})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    try:
        top5 = predict_top5(request.files['image'])
        return jsonify({"predictions": top5})
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Python model service...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
