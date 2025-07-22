import os
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt

from model import ConditionalUNet
from train import train_ddpm
from utils import cosine_beta_schedule

# ───────────── ARGUMENT PARSER ─────────────
parser = argparse.ArgumentParser(description="Train Conditional DDPM on LSUN")

parser.add_argument("--checkpoint", type=str, default=None, help="Path of the last checkpoint to resume training")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and validation")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--save_every", type=int, default=10, help="Model checkpoint save interval (in epochs)")
parser.add_argument("--image_size", type=int, default=128, help="Image size for resizing")
parser.add_argument("--data_root", type=str, default="./data", help="Root directory for LSUN dataset")

args = parser.parse_args()

# ───────────── DEVICE ─────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device utilisé :", device)

# ───────────── CATÉGORIES LSUN ─────────────
categories = [
    "bridge",
    "church_outdoor",
    "classroom",
    "conference_room",
    "dining_room",
    "kitchen",
    "living_room",
    "restaurant",
    "tower",
]

categories_train = [k + '_train' for k in categories]
categories_val = [k + '_val' for k in categories]

# ───────────── TRANSFORMATIONS ─────────────
transform = transforms.Compose([
    transforms.Resize(args.image_size),
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ───────────── DATASETS ─────────────
train_datasets = [datasets.LSUN(root=args.data_root, classes=[categories_train], transform=transform) for cat in categories_train]
val_datasets = [datasets.LSUN(root=args.data_root, classes=[categories_val], transform=transform)]

train_dataset = ConcatDataset(train_datasets)
val_dataset = ConcatDataset(val_datasets)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
validation_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# ───────────── DIFFUSION PARAMS ─────────────
T = 1000
betas = cosine_beta_schedule(T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

if args.checkpoint:
    # ───────────── MODÈLE + OPTIM ─────────────
    model = ConditionalUNet(input_c=3, base_c=64, cond_dim=128, n_classes=len(train_dataset.datasets[0].classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # ───────────── ENTRAÎNEMENT ─────────────
    loss_history = train_ddpm(
        model, train_loader, validation_loader, optimizer, scheduler,
        device, epochs=args.epochs, save_every=args.save_every
    )

    # ───────────── PLOT ─────────────
    os.makedirs('./plots', exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DDPM Training Loss per Epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('DDPM_training_loss.png')
    
else:
    # ───────────── MODÈLE + OPTIM ─────────────
    model = ConditionalUNet(input_c=3, base_c=64, cond_dim=128, n_classes=len(train_dataset.datasets[0].classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ───────────── ENTRAÎNEMENT ─────────────
    loss_history = train_ddpm(
        model, train_loader, validation_loader, optimizer, scheduler,
        device, epochs=args.epochs, save_every=args.save_every
    )

    # ───────────── PLOT ─────────────
    os.makedirs('./plots', exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DDPM Training Loss per Epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('DDPM_training_loss.png')
