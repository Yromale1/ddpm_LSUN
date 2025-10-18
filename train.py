import os
import math
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from utils import q_sample, extract  # q_sample pour forward diffusion, extract pour alpha/beta

def train_ddpm(model, dataloader, validation_dataloader, optimizer, scheduler, device,
               epochs=100, save_every=10, T=1000, start_epoch=0):

    # Cosine schedule
    t_vals = torch.arange(T + 1, device=device) / T
    alpha_bars = torch.cos((t_vals + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    betas = torch.clamp(betas, min=1e-5, max=0.999)
    alphas = 1. - betas
    assert (alphas > 0).all() and (betas > 0).all()

    loss_history = []
    scaler = GradScaler()

    os.makedirs("./samples", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    for epoch in range(start_epoch, epochs + start_epoch):
        model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"[Epoch {epoch+1}/{epochs + start_epoch}]")

        for i, (x, y) in enumerate(progress_bar):
            x = x.to(device)
            y = y.to(device)
            B = x.size(0)

            # Sample random timesteps
            t = torch.randint(1, T, (B,), device=device).long()
            noise = torch.randn_like(x)
            x_t = q_sample(x, t, noise, alpha_bars)

            # Class dropout for classifier-free guidance
            drop_prob = 0.1
            y_train = None if torch.rand(1).item() < drop_prob else y

            optimizer.zero_grad()

            with autocast(device_type=device):
                pred_noise = model(x_t, t, y_train)

                # Clamp alpha_bar pour éviter NaN
                alpha_bar_t = extract(alpha_bars, t, x_t.shape).clamp(min=1e-5)
                x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

                loss = F.mse_loss(pred_noise, noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(dataloader)
        loss_history.append(avg_train_loss)

        # Save training loss plot
        plt.figure(figsize=(8, 5))
        plt.plot(loss_history, label="Training loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("./samples/loss_curve.png")
        plt.close()

        # Validation + reconstruction images
        if (epoch + 1) % save_every == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_val, y_val in tqdm(validation_dataloader, desc="Validation"):
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    B = x_val.size(0)

                    t_val = torch.randint(1, T, (B,), device=device).long()
                    noise_val = torch.randn_like(x_val)
                    x_t_val = q_sample(x_val, t_val, noise_val, alpha_bars)

                    pred_noise_val = model(x_t_val, t_val, y_val)
                    alpha_bar_val = extract(alpha_bars, t_val, x_t_val.shape).clamp(min=1e-5)
                    x0_pred_val = (x_t_val - torch.sqrt(1 - alpha_bar_val) * pred_noise_val) / torch.sqrt(alpha_bar_val)

                    loss_mse_val = F.mse_loss(pred_noise_val, noise_val)
                    val_loss += loss_mse_val.item() * B

                avg_val_loss = val_loss / len(validation_dataloader.dataset)
                print(f"Validation Loss: {avg_val_loss:.4f}")

                # Save reconstruction images
                n_vis = min(8, x_val.size(0))
                vis_input = x_val[:n_vis].clamp(-1, 1) * 0.5 + 0.5
                vis_noisy = x_t_val[:n_vis].clamp(-1, 1) * 0.5 + 0.5
                vis_recon = x0_pred_val[:n_vis].clamp(-1, 1) * 0.5 + 0.5
                vis_concat = torch.cat([vis_input, vis_noisy, vis_recon], dim=0)
                vutils.save_image(vis_concat, f"./samples/val_recon_epoch_{epoch+1}.png", nrow=n_vis)

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_train_loss
            }, f"./models/checkpoint_epoch_{epoch + 1}.pth")
            print(f"Saved checkpoint at epoch {epoch + 1}")

            # Sampling images par label avec guidance
            n_samples_per_label = 4
            num_classes = len(validation_dataloader.dataset.classes)
            img_size = 28

            with torch.no_grad():
                for label in range(num_classes):
                    x_t = torch.randn(n_samples_per_label, 1, img_size, img_size, device=device)
                    y_label = torch.full((n_samples_per_label,), label, device=device, dtype=torch.long)

                    for t_inv in reversed(range(T)):
                        t = torch.full((n_samples_per_label,), t_inv, device=device, dtype=torch.long)
                        eps_cond = model(x_t, t, y_label)
                        eps_uncond = model(x_t, t, None)
                        guidance_scale = 0.0
                        eps_theta = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

                        beta_t = extract(betas, t, x_t.shape)
                        alpha_t = extract(alphas, t, x_t.shape)
                        alpha_bar_t = extract(alpha_bars, t, x_t.shape).clamp(min=1e-5)

                        noise = torch.randn_like(x_t) if t_inv > 0 else torch.zeros_like(x_t)
                        x_t = (1 / torch.sqrt(alpha_t)) * (
                            x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_theta
                        ) + torch.sqrt(beta_t) * noise

                        if t_inv % 100 == 0 or t_inv == T-1:
                            print(f"[label={label}, t={t_inv}] x_t mean={x_t.mean():.4f}, std={x_t.std():.4f}")

                    # Map to [0,1] et save
                    img = torch.clamp(x_t, -1, 1) * 0.5 + 0.5
                    vutils.save_image(img, f"./samples/epoch_{epoch+1}_label_{label}.png",
                                      nrow=n_samples_per_label)

    return loss_history
