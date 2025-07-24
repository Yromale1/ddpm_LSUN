import os
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from tqdm import tqdm

from utils import q_sample

def train_ddpm(model, dataloader, validation_dataloader, optimizer, scheduler, device, 
               epochs=100, save_every=10, T=1000):

    # Precompute the noise schedule
    betas = torch.linspace(1e-4, 0.02, T).to(device)
    alphas = 1. - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    loss_history = []

    os.makedirs("./samples", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"[Epoch {epoch+1}/{epochs}]")
        for x, y in progress_bar:
            x = x.to(device)
            y = y.to(device)
            B = x.size(0)

            # Sample t and noise ε
            t = torch.randint(0, T, (B,), device=device).long()
            noise = torch.randn_like(x)

            # Forward diffusion: get x_t
            x_t = q_sample(x, t, noise, alpha_bars)

            # Predict ε
            pred_noise = model(x_t, t, y)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(dataloader)
        loss_history.append(avg_train_loss)

        if (epoch + 1) % save_every == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_val, y_val in tqdm(validation_dataloader, desc="Validation"):
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    B = x_val.size(0)
                    t_val = torch.randint(0, T, (B,), device=device).long()
                    noise_val = torch.randn_like(x_val)
                    x_t_val = q_sample(x_val, t_val, noise_val, alpha_bars)

                    pred_noise_val = model(x_t_val, t_val, y_val)
                    loss_val = F.mse_loss(pred_noise_val, noise_val)
                    val_loss += loss_val.item() * B

            avg_val_loss = val_loss / len(validation_dataloader.dataset)
            print(f"Validation Loss: {avg_val_loss:.4f}")

            # Checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_train_loss
            }, f"./models/checkpoint_epoch_{epoch + 1}.pth")
            print(f"Saved checkpoint at epoch {epoch + 1}")

            # Sampling
            with torch.no_grad():
                img_size = x.shape[-1]
                for label in range(len(validation_dataloader.dataset.classes)):
                    x_t = torch.randn(1, 3, img_size, img_size).to(device)
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
                    vutils.save_image(img, f"./samples/epoch_{epoch+1}_label_{label}.png")

    return loss_history
