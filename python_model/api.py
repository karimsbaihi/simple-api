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

# ------------------- CONFIG -------------------
device = torch.device('cpu')

# ------------------- MEMORY OPTIMIZATION -------------------
# Reduce memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32' if torch.cuda.is_available() else ''

def check_files():
    """Check if required files exist and are valid"""
    files = {
        'optimized_model.pt': 30000000,  # ~30MB minimum (for 40MB file)
        'class_mappings.pkl': 1000
    }
    
    for filename, min_size in files.items():
        if not os.path.exists(filename):
            logger.error(f"{filename} not found!")
            return False
        
        file_size = os.path.getsize(filename)
        logger.info(f"{filename} size: {file_size} bytes")
        
        if file_size < min_size:
            logger.error(f"{filename} is too small ({file_size} bytes). Expected at least {min_size} bytes.")
            return False
    
    return True

# ------------------- VERIFY FILES -------------------
logger.info("Checking for required model files...")
if not check_files():
    raise RuntimeError("Required model files are missing or invalid")

# ------------------- MEMORY EFFICIENT MODEL LOADING -------------------
def load_model_memory_efficient():
    """Load model with minimal memory usage"""
    try:
        # Load checkpoint to CPU first (memory efficient)
        checkpoint = torch.load('optimized_model.pt', map_location='cpu', weights_only=False)
        
        # Create model architecture
        model = models.resnet34(weights=None)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(model.fc.in_features, 200)
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Convert to half precision for memory savings (if model supports it)
        try:
            model = model.half()
            logger.info("Model converted to half precision")
        except:
            logger.info("Model does not support half precision, using float32")
        
        model.to(device)
        model.eval()
        
        # Clean up memory
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model loaded with memory optimization")
        return model, checkpoint.get('class_to_idx'), checkpoint.get('wnid_to_words')
        
    except Exception as e:
        logger.error(f"Memory efficient loading failed: {e}")
        # Fallback to regular loading
        checkpoint = torch.load('optimized_model.pt', map_location=device, weights_only=False)
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
        return model, checkpoint.get('class_to_idx'), checkpoint.get('wnid_to_words')

# ------------------- LOAD MODEL -------------------
logger.info("Loading model with memory optimization...")
model, class_to_idx, wnid_to_words = load_model_memory_efficient()

# Handle class mappings
if class_to_idx and wnid_to_words:
    idx_to_word = {v: wnid_to_words[k] for k, v in class_to_idx.items()}
    logger.info("Class mappings loaded from model checkpoint")
else:
    # Load from separate file
    try:
        with open('class_mappings.pkl', 'rb') as f:
            class_to_idx, wnid_to_words = pickle.load(f)
        idx_to_word = {v: wnid_to_words[k] for k, v in class_to_idx.items()}
        logger.info("Class mappings loaded from separate file")
    except Exception as e:
        logger.error(f"Failed to load class mappings: {e}")
        raise

logger.info("Model and mappings loaded successfully!")

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
        
        # Convert input to half precision to match model (if model is half)
        if next(model.parameters()).dtype == torch.float16:
            img_tensor = img_tensor.half()
        
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
        "memory_optimized": True,
        "files": {
            "model_size": os.path.getsize('optimized_model.pt'),
            "mappings_size": os.path.getsize('class_mappings.pkl') if os.path.exists('class_mappings.pkl') else 0
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
        "optimized": True,
        "precision": str(next(model.parameters()).dtype)
    })

# ------------------- RUN SERVER -------------------
if __name__ == '__main__':
    logger.info("Starting memory-optimized Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)