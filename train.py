import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from tqdm import tqdm
import lpips
import matplotlib.pyplot as plt
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from utils import q_sample, extract


def train_ddpm(model, dataloader, validation_dataloader, optimizer, scheduler, device,
               epochs=100, save_every=10, T=1000, start_epoch=0):

    betas = torch.linspace(1e-4**0.5, 0.02**0.5, T, device=device) ** 2
    alphas = 1. - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    loss_history = []
    scaler = GradScaler()

    # LPIPS loss
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    os.makedirs("./samples", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    for epoch in range(start_epoch, epochs + start_epoch):
        model.train()
        total_loss = 0
        lpips_lambda = min(0.1, epoch * 0.01)
        mse_per_t = {}
        count_per_t = {}

        progress_bar = tqdm(dataloader, desc=f"[Epoch {epoch+1}/{epochs + start_epoch}]")

        optimizer.zero_grad() 

        for i, (x, y) in enumerate(progress_bar):
            x = x.to(device)
            y = y.to(device)
            B = x.size(0)
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("NaN in input images")

            t = torch.randint(1, T, (B,), device=device).long()
            noise = torch.randn_like(x)
            x_t = q_sample(x, t, noise, alpha_bars)

            drop_prob = 0.1
            y_train = None if torch.rand(1).item() < drop_prob else y
            
            optimizer.zero_grad()

            with autocast(device_type=device):
                pred_noise = model(x_t, t, y_train)

                alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
                alpha_bar_t = torch.clamp(alpha_bar_t, min=1e-3)
                x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

                loss_mse = F.mse_loss(pred_noise, noise)
                with torch.no_grad():
                    loss_lpips = lpips_model(x0_pred, x).mean()
                loss = loss_mse + lpips_lambda * loss_lpips

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            for name, param in model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"[NaN detected] {name} contains NaN or Inf")
                    exit()

            total_loss += loss.item()
            progress_bar.set_postfix(mse=loss_mse.item(), lpips=loss_lpips.item(), total=loss.item())

            # MSE(t) logging
            with torch.no_grad():
                for i_batch in range(B):
                    t_i = t[i_batch].item()
                    mse_val = F.mse_loss(pred_noise[i_batch], noise[i_batch], reduction='mean').item()
                    if t_i not in mse_per_t:
                        mse_per_t[t_i] = 0.0
                        count_per_t[t_i] = 0
                    mse_per_t[t_i] += mse_val
                    count_per_t[t_i] += 1

        # Log epoch loss
        avg_train_loss = total_loss / len(dataloader)
        loss_history.append(avg_train_loss)

        # Plot training loss
        plt.figure(figsize=(8, 5))
        plt.plot(loss_history, label="Training loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("./samples/loss_curve.png")
        plt.close()

        # Plot MSE vs t
        mse_by_t = np.zeros(T)
        for t_i in mse_per_t:
            mse_by_t[t_i] = mse_per_t[t_i] / max(count_per_t[t_i], 1)

        plt.figure(figsize=(10, 4))
        plt.plot(mse_by_t, label="MSE vs t", linewidth=1)
        plt.xlabel("Timestep t")
        plt.ylabel("MSE")
        plt.title(f"MSE vs Timestep (Epoch {epoch+1})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"./samples/mse_vs_t_epoch_{epoch+1}.png")
        plt.close()

        # Validation
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

                    alpha_bar_val = alpha_bars[t_val].view(-1, 1, 1, 1)
                    x0_pred_val = (x_t_val - torch.sqrt(1 - alpha_bar_val) * pred_noise_val) / torch.sqrt(alpha_bar_val)

                    loss_mse_val = F.mse_loss(pred_noise_val, noise_val)
                    loss_lpips_val = lpips_model(x0_pred_val, x_val).mean()
                    loss_val = loss_mse_val + lpips_lambda * loss_lpips_val
                    val_loss += loss_val.item() * B

                avg_val_loss = val_loss / len(validation_dataloader.dataset)
                print(f"Validation Loss: {avg_val_loss:.4f}")

                # Save validation reconstructions
                n_vis = min(8, x_val.size(0))
                vis_input = x_val[:n_vis].clamp(-1, 1) * 0.5 + 0.5
                vis_noisy = x_t_val[:n_vis].clamp(-1, 1) * 0.5 + 0.5
                vis_recon = x0_pred_val[:n_vis].clamp(-1, 1) * 0.5 + 0.5
                vis_concat = torch.cat([vis_input, vis_noisy, vis_recon], dim=0)
                vutils.save_image(vis_concat, f"./samples/val_recon_epoch_{epoch+1}.png", nrow=n_vis)

            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_train_loss
            }, f"./models/checkpoint_epoch_{epoch + 1}.pth")
            print(f"Saved checkpoint at epoch {epoch + 1}")

            # Sampling
            model.eval()
            n_samples_per_label = 4
            num_classes = len(validation_dataloader.dataset.classes)
            img_size = 128

            with torch.no_grad():
                for label in range(num_classes):
                    x_t = torch.randn(n_samples_per_label, 3, img_size, img_size).to(device)
                    y_label = torch.full((n_samples_per_label,), label, device=device)

                    for t_inv in reversed(range(T)):
                        t = torch.full((n_samples_per_label,), t_inv, device=device, dtype=torch.long)
                        eps_cond = model(x_t, t, y_label)
                        eps_uncond = model(x_t, t, None)
                        guidance_scale = 1.5
                        eps_theta = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

                        beta_t = extract(betas, t, x_t.shape)
                        alpha_t = extract(alphas, t, x_t.shape)
                        alpha_bar_t = extract(alpha_bars, t, x_t.shape).clamp(min=1e-4)

                        noise = torch.randn_like(x_t) if t_inv > 0 else torch.zeros_like(x_t)
                        x_t = (1 / torch.sqrt(alpha_t)) * (
                            x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_theta
                        ) + torch.sqrt(beta_t) * noise
                        
                        if t_inv % 100 == 0 or t_inv == T - 1:
                            x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_theta) / torch.sqrt(alpha_bar_t)
                            print(f"[t={t_inv}] x0_pred mean={x0_pred.mean():.3f}, std={x0_pred.std():.3f}")


                    img = x_t.clamp(-1, 1) * 0.5 + 0.5
                    vutils.save_image(img, f"./samples/epoch_{epoch+1}_label_{label}.png", nrow=n_samples_per_label)

    return loss_history
