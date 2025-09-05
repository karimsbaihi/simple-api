import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify
import pickle
import os
import logging
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------- RENDER-SPECIFIC FIXES -------------------
device = torch.device('cpu')

# Force memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# ------------------- FILE PATH FIXES -------------------
def get_file_path(filename):
    """Get absolute path to file, works on both local and Render"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, filename)

def check_files():
    """Check if required files exist with absolute paths"""
    files = {
        'simple_model.pt': 30000000,
        'class_mappings.pkl': 1000
    }
    
    for filename, min_size in files.items():
        file_path = get_file_path(filename)
        if not os.path.exists(file_path):
            logger.error(f"{filename} not found at: {file_path}")
            logger.error(f"Current directory: {os.getcwd()}")
            logger.error(f"Directory contents: {os.listdir('.')}")
            return False
        
        file_size = os.path.getsize(file_path)
        logger.info(f"{filename} size: {file_size} bytes")
        
        if file_size < min_size:
            logger.error(f"{filename} is too small ({file_size} bytes)")
            return False
    
    return True

# ------------------- MEMORY-EFFICIENT LOADING -------------------
def load_model_with_memory_guard():
    """Load model with memory protection for Render"""
    try:
        model_path = get_file_path('simple_model.pt')
        logger.info(f"Loading model from: {model_path}")
        
        # Load with memory optimization
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Debug checkpoint contents
        logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Build model
        model = models.resnet34(weights=None)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(model.fc.in_features, 200)
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Free memory immediately
        del checkpoint
        gc.collect()
        
        # Load class mappings
        mappings_path = get_file_path('class_mappings.pkl')
        logger.info(f"Loading mappings from: {mappings_path}")
        
        with open(mappings_path, 'rb') as f:
            class_mappings_data = pickle.load(f)
        
        # Handle different formats
        if isinstance(class_mappings_data, tuple) and len(class_mappings_data) == 2:
            class_to_idx, wnid_to_words = class_mappings_data
        elif isinstance(class_mappings_data, dict):
            if 'class_to_idx' in class_mappings_data and 'wnid_to_words' in class_mappings_data:
                class_to_idx = class_mappings_data['class_to_idx']
                wnid_to_words = class_mappings_data['wnid_to_words']
            else:
                class_to_idx = class_mappings_data
                wnid_to_words = {k: f"class_{v}" for k, v in class_mappings_data.items()}
        else:
            raise ValueError("Unknown class mappings format")
        
        idx_to_word = {v: wnid_to_words[k] for k, v in class_to_idx.items()}
        
        logger.info(f"Loaded {len(idx_to_word)} classes")
        return model, idx_to_word
        
    except Exception as e:
        logger.error(f"Loading failed: {e}")
        # One more try with absolute paths
        try:
            checkpoint = torch.load(get_file_path('simple_model.pt'), map_location='cpu')
            # ... rest of loading logic
        except Exception as e2:
            logger.error(f"Complete failure: {e2}")
            raise

# ------------------- INITIALIZATION -------------------
logger.info("=== STARTING APP ===")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Files here: {os.listdir('.')}")

if not check_files():
    raise RuntimeError("Missing required files")

logger.info("Loading model with memory guard...")
model, idx_to_word = load_model_with_memory_guard()
logger.info("Model loaded successfully!")

# ------------------- PREDICTION -------------------
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

# ------------------- FLASK APP -------------------
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "model_loaded": True,
        "num_classes": len(idx_to_word),
        "environment": "render" if "render" in os.getcwd().lower() else "local"
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