import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify
import pickle
import os
import logging
import requests
import gdown  # For Google Drive downloads

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------- CONFIG -------------------
device = torch.device('cpu')

#mapping link https://drive.google.com/file/d/1jrU9q1TMhVU3GgK_wVQ1dimq6NiC3961/view?usp=sharing
#model link https://drive.google.com/file/d/1TOZ2U-KuYc_9kFdqztWISbaJAu8BiTof/view?usp=drive_link

# Google Drive file IDs (extract from your URLs)
MODEL_FILE_ID = "1TOZ2U-KuYc_9kFdqztWISbaJAu8BiTof"
MAPPINGS_FILE_ID = "1jrU9q1TMhVU3GgK_wVQ1dimq6NiC3961"  # Replace with actual mappings file ID if different

# Direct download URLs (Google Drive format)
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}&export=download"
MAPPINGS_URL = f"https://drive.google.com/uc?id={MAPPINGS_FILE_ID}&export=download"  # If you have separate file

def download_from_google_drive(file_id, filename):
    """Download file from Google Drive using gdown"""
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)
        
        if os.path.exists(filename) and os.path.getsize(filename) > 1000:
            logger.info(f"Successfully downloaded {filename}. Size: {os.path.getsize(filename)} bytes")
            return True
        else:
            logger.error(f"Download failed or file is too small: {filename}")
            return False
    except Exception as e:
        logger.error(f"Error downloading {filename}: {e}")
        return False

def download_file_direct(url, filename):
    """Alternative direct download method"""
    try:
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # Handle Google Drive virus scan warning
        if "virus scan" in response.text.lower():
            # Extract confirm token
            confirm = None
            for line in response.text.split('\n'):
                if "confirm=" in line:
                    confirm = line.split('confirm=')[1].split('"')[0]
                    break
            
            if confirm:
                url = f"{url}&confirm={confirm}"
                response = session.get(url, stream=True)
        
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"Downloaded {filename}. Size: {os.path.getsize(filename)} bytes")
        return True
        
    except Exception as e:
        logger.error(f"Direct download failed: {e}")
        return False

def ensure_files_exist():
    """Ensure model files exist, download if missing"""
    files_to_check = [
        ('best_model.pt', MODEL_FILE_ID, 10000000),  # ~10MB minimum
        ('class_mappings.pkl', MODEL_FILE_ID, 1000)   # Replace with actual mappings ID
    ]
    
    for filename, file_id, min_size in files_to_check:
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            if file_size >= min_size:
                logger.info(f"{filename} already exists ({file_size} bytes)")
                continue
            else:
                logger.warning(f"{filename} exists but is too small ({file_size} bytes), re-downloading")
                os.remove(filename)
        
        # Try gdown first
        logger.info(f"Downloading {filename}...")
        if not download_from_google_drive(file_id, filename):
            # Fallback to direct download
            url = f"https://drive.google.com/uc?id={file_id}&export=download"
            if not download_file_direct(url, filename):
                raise RuntimeError(f"Failed to download {filename}")

# ------------------- DOWNLOAD FILES -------------------
logger.info("Checking for required model files...")
ensure_files_exist()

# Verify files are valid
model_path = 'best_model.pt'
mappings_path = 'class_mappings.pkl'

if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
    raise RuntimeError("Model file is missing or invalid")

if not os.path.exists(mappings_path) or os.path.getsize(mappings_path) < 1000:
    raise RuntimeError("Class mappings file is missing or invalid")

# ------------------- MODEL SETUP -------------------
model = models.resnet34(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.fc.in_features, 200)
)

def load_model_checkpoint(path, device):
    """Load model checkpoint with compatibility for different PyTorch versions"""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception as e:
        logger.error(f"Error loading with weights_only=False: {e}")
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except Exception as e2:
            logger.error(f"Error loading with weights_only=True: {e2}")
            raise

# Load trained checkpoint
try:
    logger.info("Loading model checkpoint...")
    checkpoint = load_model_checkpoint(model_path, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# ------------------- CLASS MAPPINGS -------------------
try:
    logger.info("Loading class mappings...")
    with open(mappings_path, 'rb') as f:
        class_to_idx, wnid_to_words = pickle.load(f)
    idx_to_word = {v: wnid_to_words[k] for k, v in class_to_idx.items()}
    logger.info("Class mappings loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load class mappings: {e}")
    raise

# ------------------- IMAGE TRANSFORMS -------------------
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], 
                         std=[0.2296, 0.2263, 0.2255])
])

# ------------------- PREDICTION FUNCTION -------------------
def predict_top5(image):
    try:
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
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise

# ------------------- FLASK APP -------------------
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "message": "API is running",
        "model_loaded": True,
        "files": {
            "model_size": os.path.getsize('best_model.pt'),
            "mappings_size": os.path.getsize('class_mappings.pkl')
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    try:
        logger.info(f"Received prediction request for file: {file.filename}")
        top5 = predict_top5(file)
        logger.info("Prediction completed successfully")
        return jsonify({"predictions": top5})
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Endpoint to get model information"""
    return jsonify({
        "model": "ResNet34",
        "dataset": "TinyImageNet-200",
        "device": str(device),
        "status": "loaded",
        "files": {
            "model": os.path.exists('best_model.pt'),
            "mappings": os.path.exists('class_mappings.pkl')
        }
    })

# ------------------- RUN SERVER -------------------
if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False)