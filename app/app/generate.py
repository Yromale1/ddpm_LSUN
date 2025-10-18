import torch
import os
import math

from utils import init_weights, extract, tensor_to_base64_img
from model import ConditionalUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_file = os.environ.get("MODEL_FILE", "checkpoint_epoch_100.pth")
model_path = os.path.join("models", model_file)

# Charger le modèle
model = ConditionalUNet(input_c=1, base_c=64, cond_dim=64, n_classes=10).to(device)
model.apply(init_weights)
checkpoint = torch.load("./models/checkpoint_epoch_200.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def generate_image(label):
    T = 1000
    t_vals = torch.arange(T + 1, device=device) / T
    alpha_bars = torch.cos((t_vals + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    betas = torch.clamp(betas, min=1e-5, max=0.999)
    alphas = 1. - betas
    assert (alphas > 0).all() and (betas > 0).all()

    # pas de batch : (C,H,W)
    x_t = torch.randn(1, 28, 28, device=device)
    y_label = torch.tensor(label, device=device, dtype=torch.long)

    with torch.no_grad():
        for t_inv in reversed(range(T)):
            t = torch.tensor(t_inv, device=device, dtype=torch.long)
            eps_cond = model(x_t.unsqueeze(0), t.unsqueeze(0), y_label.unsqueeze(0))
            eps_uncond = model(x_t.unsqueeze(0), t.unsqueeze(0), None)
            guidance_scale = 1.0
            eps_theta = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            beta_t = extract(betas, t.unsqueeze(0), x_t.unsqueeze(0).shape)
            alpha_t = extract(alphas, t.unsqueeze(0), x_t.unsqueeze(0).shape)
            alpha_bar_t = extract(alpha_bars, t.unsqueeze(0), x_t.unsqueeze(0).shape).clamp(min=1e-5)

            noise = torch.randn_like(x_t) if t_inv > 0 else torch.zeros_like(x_t)
            x_t = (1 / torch.sqrt(alpha_t.squeeze(0))) * (
                x_t - ((1 - alpha_t.squeeze(0)) / torch.sqrt(1 - alpha_bar_t.squeeze(0))) * eps_theta.squeeze(0)
            ) + torch.sqrt(beta_t.squeeze(0)) * noise

    # map [-1,1] -> [0,1]
    img = torch.clamp(x_t, -1, 1) * 0.5 + 0.5
    return tensor_to_base64_img(img)
