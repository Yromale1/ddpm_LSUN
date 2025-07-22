import torch
from model import ConditionalUNet
from utils import tensor_to_base64_img
from utils import sample_ddpm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le modèle
model = ConditionalUNet(input_c=3, base_c=64, cond_dim=128, n_classes=10).to(device)
checkpoint = torch.load("./models/checkpoint_epoch_100.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def generate_image(label):
    with torch.no_grad():
        image_tensor = sample_ddpm(model, shape=(1, 3, 64, 64), label=label, device=device)
        return tensor_to_base64_img(image_tensor)
