import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from tqdm import tqdm
import lpips
import imageio
import matplotlib.pyplot as plt

from utils import q_sample, cosine_beta_schedule, extract
from loss import compute_lpips_loss


def train_ddpm(model, dataloader, validation_dataloader, optimizer, scheduler, device,
               epochs=100, save_every=10, T=1000, start_epoch=0):

    betas = cosine_beta_schedule(T).to(device)
    alphas = 1. - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    loss_history = []

    # LPIPS model
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    
    vlb_weight = 0.001

    os.makedirs("./samples", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    for epoch in range(start_epoch, epochs + start_epoch):
        model.train()
        total_loss = 0
        
        lpips_lambda = min(0.01, epoch * 0.0001)

        progress_bar = tqdm(dataloader, desc=f"[Epoch {epoch+1}/{epochs + start_epoch}]")
        for x, y in progress_bar:
            x = x.to(device)
            y = y.to(device)
            B = x.size(0)

            # Sample t and noise
            t = torch.randint(0, T, (B,), device=device).long()
            noise = torch.randn_like(x)

            # Forward diffusion
            x_t = q_sample(x, t, noise, alpha_bars)
            
            drop_prob = 0.1
            if torch.rand(1).item() < drop_prob:
                y_train = None
            else:
                y_train = y

            # Predict noise
            pred_noise = model(x_t, t, y_train)

            # Reconstruct x0
            alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
            x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

            loss_mse = F.mse_loss(pred_noise, noise)
            loss_lpips = compute_lpips_loss(x0_pred, x, lpips_model)
            # loss_vlb = vlb_loss(x, x_t, t, pred_noise, alpha_bars, betas)
            loss = loss_mse + lpips_lambda * loss_lpips # + vlb_weight * loss_vlb

            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            progress_bar.set_postfix(mse=loss_mse.item(), lpips=loss_lpips.item() ,total=loss.item())

        avg_train_loss = total_loss / len(dataloader)
        loss_history.append(avg_train_loss)
        
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

        # Validation + checkpoint + sampling
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
                    loss_lpips_val = compute_lpips_loss(x0_pred_val, x_val, lpips_model)
                    # loss_vlb_val = vlb_loss(x_val, x_t_val, t_val, pred_noise_val, alpha_bars, betas)
                    loss_val = loss_mse_val + lpips_lambda * loss_lpips_val # + vlb_weight * loss_vlb_val
                    val_loss += loss_val.item() * B

                avg_val_loss = val_loss / len(validation_dataloader.dataset)
                print(f"Validation Loss: {avg_val_loss:.4f}")

                # Visualisation validation
                n_val_vis = min(8, x_val.size(0))
                vis_val_pred = x0_pred_val[:n_val_vis].clamp(-1, 1) * 0.5 + 0.5
                vis_val_target = x_val[:n_val_vis].clamp(-1, 1) * 0.5 + 0.5
                vis_val_concat = torch.cat([vis_val_target, vis_val_pred], dim=0)
                vutils.save_image(vis_val_concat, f"./samples/val_recon_epoch_{epoch+1}.png", nrow=n_val_vis)

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
                img_size = 128
                for label in range(len(validation_dataloader.dataset.classes)):
                    x_t = torch.randn(1, 3, img_size, img_size).to(device)
                    y_label = torch.tensor([label], device=device)

                    for t_inv in reversed(range(T)):
                        t = torch.full((1,), t_inv, device=device, dtype=torch.long)
                        eps_cond = model(x_t, t, y_label)
                        eps_uncond = model(x_t, t, None)
                        guidance_scale = 3.0 
                        eps_theta = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

                        beta_t = extract(betas, t, x_t.shape)
                        alpha_t = extract(alphas, t, x_t.shape)
                        alpha_bar_t = extract(alpha_bars, t, x_t.shape)

                        noise = torch.randn_like(x_t) if t_inv > 0 else torch.zeros_like(x_t)

                        x_t = (1 / torch.sqrt(alpha_t)) * (
                            x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_theta
                        ) + torch.sqrt(beta_t) * noise

                    img = x_t.clamp(-1, 1) * 0.5 + 0.5
                    vutils.save_image(img, f"./samples/epoch_{epoch+1}_label_{label}.png")

    return loss_history
