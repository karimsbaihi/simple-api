from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from torchvision import transforms
import io

app = FastAPI()

# Allow Node.js frontend to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Old code
# model = torch.load("best_model.pt")

# New code for CPU-only
model = torch.load("best_model.pt", map_location=torch.device('cpu'))

model.eval()

# Replace with your Tiny ImageNet labels
classes = ["class_0", "class_1", "...", "class_199"]

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        pred = model(img_tensor)
        class_idx = pred.argmax().item()
    
    return {"class": classes[class_idx]}
