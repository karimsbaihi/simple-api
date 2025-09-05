import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify
import pickle
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cpu')

# Check files
def check_files():
    if not os.path.exists('simple_model.pt'):
        logger.error("simple_model.pt not found!")
        return False
    if not os.path.exists('class_mappings.pkl'):
        logger.error("class_mappings.pkl not found!")
        return False
    return True

logger.info("Checking files...")
if not check_files():
    raise RuntimeError("Files missing")

# Load model simply
logger.info("Loading model...")
checkpoint = torch.load('simple_model.pt', map_location=device, weights_only=False)

#checkpoints debug
logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
if 'class_to_idx' in checkpoint:
    logger.info(f"class_to_idx type: {type(checkpoint['class_to_idx'])}")
    logger.info(f"class_to_idx value: {checkpoint['class_to_idx']}")

model = models.resnet34(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.fc.in_features, 200)
)

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Load class mappings - FIXED THIS PART
class_to_idx = None
wnid_to_words = None

# Try to get from checkpoint first
if 'class_to_idx' in checkpoint and checkpoint['class_to_idx'] is not None:
    class_to_idx = checkpoint['class_to_idx']
    wnid_to_words = checkpoint['wnid_to_words']
    logger.info("Loaded class mappings from model checkpoint")
else:
    # Fallback to separate file
    try:
        with open('class_mappings.pkl', 'rb') as f:
            class_to_idx, wnid_to_words = pickle.load(f)
        logger.info("Loaded class mappings from separate file")
    except Exception as e:
        logger.error(f"Failed to load class mappings: {e}")
        raise

# DOUBLE CHECK that we have the mappings
if class_to_idx is None or wnid_to_words is None:
    raise RuntimeError("Class mappings are None - check your model file or class_mappings.pkl")

idx_to_word = {v: wnid_to_words[k] for k, v in class_to_idx.items()}
logger.info("Model loaded successfully!")

# Transform and prediction
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2296, 0.2263, 0.2255])
])

def predict_top5(image):
    img = Image.open(image).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)[0] * 100
        top5_probs, top5_idx = torch.topk(probs, 5)
    
    return [{"class": idx_to_word[i.item()], "confidence": float(p.item())} 
            for i, p in zip(top5_idx, top5_probs)]

# Flask app
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "model_loaded": True,
        "class_mappings_loaded": class_to_idx is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    try:
        top5 = predict_top5(request.files['image'])
        return jsonify({"predictions": top5})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting server...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)