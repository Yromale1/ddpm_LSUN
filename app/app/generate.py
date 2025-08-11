import torch
from model import ConditionalUNet
from utils import tensor_to_base64_img, sample_ddpm, init_weights
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_file = os.environ.get("MODEL_FILE", "checkpoint_epoch_100.pth")
model_path = os.path.join("models", model_file)

# Charger le modèle
model = ConditionalUNet(input_c=3, base_c=64, cond_dim=128, n_classes=6).to(device)
model.apply(init_weights)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def generate_image(label):
    with torch.no_grad():
        image_tensor = sample_ddpm(model, shape=(1, 3, 128, 128), label=label, device=device)
        return tensor_to_base64_img(image_tensor)
