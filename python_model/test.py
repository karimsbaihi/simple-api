import torch

checkpoint = torch.load("best_model.pt", map_location="cpu")
print(type(checkpoint))
