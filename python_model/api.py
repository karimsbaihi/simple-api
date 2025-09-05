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

# ------------------- DEBUG -------------------
def debug_files():
    """Debug function to check file status"""
    results_path = ''
    model_path = os.path.join(results_path, 'best_model.pt')
    mappings_path = os.path.join(results_path, 'class_mappings.pkl')
    
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Files in directory: {os.listdir('.')}")
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        logger.info(f"Model file exists. Size: {file_size} bytes")
        
        # Check if it's a Git LFS pointer
        try:
            with open(model_path, 'r', encoding='utf-8') as f:
                content = f.read(200)
                if 'version https://git-lfs.github.com/spec/' in content:
                    logger.error("ERROR: Model file is a Git LFS pointer!")
                    logger.error("Content: %s", content[:100])
                    return False
        except UnicodeDecodeError:
            # File is binary (good sign)
            pass
        except Exception as e:
            logger.error(f"Error reading model file: {e}")
    else:
        logger.error("Model file does not exist!")
        return False
        
    if os.path.exists(mappings_path):
        logger.info(f"Class mappings file exists. Size: {os.path.getsize(mappings_path)} bytes")
    else:
        logger.error("Class mappings file does not exist!")
        return False
        
    return True

# ------------------- CONFIG -------------------
device = torch.device('cpu')
results_path = ''

def load_model_checkpoint(path, device):
    """Load model checkpoint with compatibility for different PyTorch versions"""
    try:
        # Try with weights_only=False first
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions
        return torch.load(path, map_location=device)
    except Exception as e:
        logger.error(f"Error loading with weights_only=False: {e}")
        # Try with weights_only=True as last resort
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except Exception as e2:
            logger.error(f"Error loading with weights_only=True: {e2}")
            raise

# ------------------- FILE VALIDATION -------------------
model_path = os.path.join(results_path, 'best_model.pt')
mappings_path = os.path.join(results_path, 'class_mappings.pkl')

# Check files before proceeding
if not debug_files():
    raise FileNotFoundError("Required model files are missing or corrupted")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

if not os.path.exists(mappings_path):
    raise FileNotFoundError(f"Class mappings file not found: {mappings_path}")

# ------------------- MODEL SETUP -------------------
model = models.resnet34(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.fc.in_features, 200)
)

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
    return jsonify({"status": "healthy", "message": "API is running"})

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
        "status": "loaded"
    })

# ------------------- RUN SERVER -------------------
if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False)