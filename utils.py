import torch
import torch.nn.functional as F

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((x + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.999)

def extract(a, t, x_shape):
    """ Extract t-indexed coefficients from precomputed arrays. """
    a = a.to(t.device)
    out = a.gather(-1, t)
    return out.view(-1, 1, 1, 1).expand(x_shape)

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
    x = torch.randn(shape, device=device)
    y = torch.full((shape[0],), label, device=device, dtype=torch.long)

    for t_ in reversed(range(T)):
        t = torch.full((shape[0],), t_, device=device, dtype=torch.long)
        x = p_sample(model, x, t, y)
    return x

def q_sample(x_0, t, noise, alpha_bars):
    """x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise"""
    alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1).to(x_0.device)
    return torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise

@torch.no_grad()
def ddim_sample(model, y_label, img_size, T=250, ddim_steps=50, eta=0.0, device="cuda",alpha_bars=None):
    x_t = torch.randn(1, 3, img_size, img_size).to(device)
    times = torch.linspace(T - 1, 0, ddim_steps, dtype=torch.long).to(device)

    for i in range(len(times) - 1):
        t = times[i].unsqueeze(0)
        t_next = times[i + 1].unsqueeze(0)

        eps = model(x_t, t, y_label)

        alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
        alpha_bar_next = alpha_bars[t_next].view(-1, 1, 1, 1)

        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
        sigma = eta * torch.sqrt((1 - alpha_bar_t / alpha_bar_next) * (1 - alpha_bar_next) / (1 - alpha_bar_t))
        noise = torch.randn_like(x_t) if eta > 0 else 0

        x_t = (
            torch.sqrt(alpha_bar_next) * x0_pred +
            torch.sqrt(1 - alpha_bar_next - sigma ** 2) * eps +
            sigma * noise
        )

    return x_t

def vlb_loss(x_0, x_t, t, pred_noise, alpha_bars, betas):
    alphas = 1. - betas
    x_shape = x_0.shape

    sqrt_recip_alpha = extract(torch.sqrt(1. / alphas), t, x_shape)
    sqrt_recipm1_alpha_bar = extract(torch.sqrt(1. / alpha_bars - 1), t, x_shape)
    
    safe_t_prev = (t - 1).clamp(min=0)
    alpha_bar_prev = extract(alpha_bars, safe_t_prev, x_shape)

    posterior_mean = (
        torch.sqrt(alpha_bar_prev) * x_0 +
        torch.sqrt(1. - alpha_bar_prev) * pred_noise
    )

    model_mean = (1. / sqrt_recip_alpha) * (
        x_t - sqrt_recipm1_alpha_bar * pred_noise
    )

    # KL divergence approx: MSE between means
    kl = F.mse_loss(model_mean, posterior_mean, reduction='none')
    return kl.mean()

def diffusion_loss(model, x_0, t, y, alpha_bars, betas, vlb_weight=0.001):
    noise = torch.randn_like(x_0)
    x_noisy = q_sample(x_0, t, noise, alpha_bars)
    noise_pred = model(x_noisy, t, y)

    mse = F.mse_loss(noise_pred, noise)

    with torch.no_grad():
        vlb = vlb_loss(x_0, x_noisy, t, noise_pred, alpha_bars, betas)

    return mse + vlb_weight * vlb