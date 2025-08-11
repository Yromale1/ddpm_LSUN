import torch
import io
import base64
import torchvision.transforms

def extract(a, t, x_shape):
    """ Extract t-indexed coefficients from precomputed arrays. """
    a = a.to(t.device)
    out = a.gather(-1, t)
    return out.view(-1, 1, 1, 1).expand(x_shape)

@torch.no_grad()
def sample_ddpm(model, shape, label, device):
    model.eval()
    T = 1000
    betas = torch.linspace(1e-4**0.5, 0.02**0.5, T, device=device) ** 2
    alphas = 1. - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    x_t = torch.randn(shape).to(device)
    y_label = torch.tensor([label], device=device)

    for t_inv in reversed(range(T)):
        t = torch.full((1,), t_inv, device=device, dtype=torch.long)
        eps_theta = model(x_t, t, y_label)

        beta_t = betas[t_inv]
        alpha_t = alphas[t_inv]
        alpha_bar_t = alpha_bars[t_inv]

        noise = torch.randn_like(x_t) if t_inv > 0 else torch.zeros_like(x_t)

        x_t = (1 / torch.sqrt(alpha_t)) * (
            x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_theta
        ) + torch.sqrt(beta_t) * noise

    img = x_t.clamp(-1, 1) * 0.5 + 0.5
    return img

                   
def tensor_to_base64_img(tensor):
    # Suppose tensor shape: (1, 1, 64, 64) ou (1, 3, 64, 64)
    tensor = tensor.squeeze(0).cpu().clamp(0, 1)  # shape: (C, H, W)
    to_pil = torchvision.transforms.ToPILImage()
    image = to_pil(tensor)
    
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    return img_str
