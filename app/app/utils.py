import torch
import io
import base64
from PIL import Image
import torchvision.transforms

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((x + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.999)


T = 1000
betas = cosine_beta_schedule(T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

def extract(a, t, x_shape):
    """ Extract t-indexed coefficients from precomputed arrays. """
    a = a.to(t.device)
    out = a.gather(-1, t)
    return out.view(-1, 1, 1, 1).expand(x_shape)

def q_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    sqrt_alpha_cumprod = extract(alphas_cumprod.sqrt(), t, x_0.shape)
    sqrt_one_minus_alpha_cumprod = extract((1 - alphas_cumprod).sqrt(), t, x_0.shape)
    return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise

@torch.no_grad()
def p_sample(model, x, t, y):
    beta_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract((1 - alphas_cumprod).sqrt(), t, x.shape)
    sqrt_recip_alphas_t = extract((1.0 / alphas).sqrt(), t, x.shape)

    pred_noise = model(x, t, y)
    model_mean = sqrt_recip_alphas_t * (x - beta_t / sqrt_one_minus_alphas_cumprod_t * pred_noise)

    if t[0] == 0:
        return model_mean
    noise = torch.randn_like(x)
    posterior_variance_t = extract(betas, t, x.shape)
    return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample_ddpm(model, shape, label, device):
    model.eval()
    T = 1000
    betas = torch.linspace(1e-4, 0.02, T).to(device)
    alphas = 1.0 - betas
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
