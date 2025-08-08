import torch
import torch.nn.functional as F

def extract(a, t, x_shape):
    """ Extract t-indexed coefficients from precomputed arrays. """
    a = a.to(t.device)
    out = a.gather(-1, t)
    return out.view(-1, 1, 1, 1).expand(x_shape)


def q_sample(x_0, t, noise, alpha_bars):
    """x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise"""
    alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1).to(x_0.device)
    alpha_bar_t = torch.clamp(alpha_bar_t, min=1e-5)
    return torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise

def init_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)