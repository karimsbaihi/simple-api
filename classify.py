import sys
import base64
import os
import torch
from PIL import Image
from torchvision import transforms

# Add the specific path to the tiny_cnn module
sys.path.insert(0, '/home/karim/Documents/GitHub/TinyImageNet-Image-Classification/results/')

from best_model import predict  # your CNN model logic
# Load the PyTorch model
model = torch.load('/home/karim/Documents/GitHub/TinyImageNet-Image-Classification/results/best_model.pt')
model.eval()

def predict(image_data):
    # Preprocess the image_data to match the model's input requirements
    # Example: Convert to tensor, normalize, resize, etc.
    # Assuming image_data is a raw image in bytes

    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to the model's expected input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust mean and std as needed
    ])

    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)

    return predicted_class.item()

img_data = sys.argv[1]

# Decode and preprocess the image here
# Example: Assuming img_data is base64 encoded
decoded_img = base64.b64decode(img_data)

# Pass the decoded image to the model for prediction
prediction = predict(decoded_img)
print(prediction)