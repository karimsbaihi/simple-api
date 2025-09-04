from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import pickle
import os

# --- Load model ---
device = torch.device('cpu')
model = models.resnet34(pretrained=False)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.fc.in_features, 200)
)

results_path = '/home/karim/Documents/GitHub/TinyImageNet-Image-Classification/results/'
checkpoint = torch.load('best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# --- Load class mappings ---
with open(results_path+'class_mappings.pkl', 'rb') as f:
    class_to_idx, wnid_to_words = pickle.load(f)
idx_to_word = {v: wnid_to_words[k] for k, v in class_to_idx.items()}

# --- Flask app ---
app = Flask(__name__)

def predict(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                             std=[0.2296, 0.2263, 0.2255])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)[0] * 100
        _, pred = torch.max(output, 1)
    return idx_to_word[pred.item()], probabilities[pred.item()].item()

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    filepath = os.path.join('/tmp', file.filename)
    file.save(filepath)
    pred_class, confidence = predict(filepath)
    return jsonify({'class': pred_class, 'confidence': confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
