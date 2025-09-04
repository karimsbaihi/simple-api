import os
from flask import Flask, request, jsonify
import torch
from torchvision import transforms, models
from PIL import Image
import pickle

app = Flask(__name__)
PORT = int(os.environ.get("PORT", 5000))  # Render sets this automatically

# Load model
device = torch.device('cpu')
model = models.resnet34(pretrained=False)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.fc.in_features, 200)
)
results_path = './'  # Adjust if your files are in another folder
checkpoint = torch.load(os.path.join(results_path,'best_model.pt'), map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Load class mappings
with open(os.path.join(results_path,'class_mappings.pkl'), 'rb') as f:
    class_to_idx, wnid_to_words = pickle.load(f)
idx_to_word = {v: wnid_to_words[k] for k,v in class_to_idx.items()}

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    img = Image.open(file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4802,0.4481,0.3975],
                             std=[0.2296,0.2263,0.2255])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        _, pred = torch.max(output,1)
    return jsonify({'prediction': idx_to_word[pred.item()]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
