import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify
import pickle
import os

# ------------------- CONFIG -------------------
device = torch.device('cpu')
results_path = ''

def load_model_checkpoint(path, device):
    """Load model checkpoint with compatibility for different PyTorch versions"""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

# ------------------- MODEL SETUP -------------------
model = models.resnet34(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.fc.in_features, 200)
)

# Verify files exist
model_path = os.path.join(results_path, 'best_model.pt')
mappings_path = os.path.join(results_path, 'class_mappings.pkl')

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load trained checkpoint
checkpoint = load_model_checkpoint(model_path, device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# ------------------- CLASS MAPPINGS -------------------
if not os.path.exists(mappings_path):
    raise FileNotFoundError(f"Class mappings file not found: {mappings_path}")

with open(mappings_path, 'rb') as f:
    class_to_idx, wnid_to_words = pickle.load(f)
idx_to_word = {v: wnid_to_words[k] for k, v in class_to_idx.items()}

# ... rest of the code remains the same ...