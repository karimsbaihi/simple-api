import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify
import pickle
import os

# ------------------- CONFIG -------------------
device = torch.device('cpu')

# Paths
results_path = ''

# ------------------- MODEL SETUP -------------------
model = models.resnet34(pretrained=False)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.fc.in_features, 200)
)

# Load trained checkpoint
checkpoint = torch.load(results_path+'best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# ------------------- CLASS MAPPINGS -------------------
with open(results_path+'class_mappings.pkl', 'rb') as f:
    class_to_idx, wnid_to_words = pickle.load(f)
idx_to_word = {v: wnid_to_words[k] for k, v in class_to_idx.items()}

# ------------------- IMAGE TRANSFORMS -------------------
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], 
                         std=[0.2296, 0.2263, 0.2255])
])

# ------------------- PREDICTION FUNCTION -------------------
def predict_top5(image):
    img = Image.open(image).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)[0] * 100
        top5_probs, top5_idx = torch.topk(probs, 5)
    
    top5 = []
    for i, p in zip(top5_idx, top5_probs):
        top5.append({
            "class": idx_to_word[i.item()],
            "confidence": float(p.item())
        })
    return top5

# ------------------- FLASK APP -------------------
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    try:
        top5 = predict_top5(file)
        return jsonify({"predictions": top5})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------- RUN SERVER -------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
