import os
import datetime
import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import MultiLMDBDataset
from model import ConditionalUNet
from train import train_ddpm
from utils import init_weights

def main():
    
    # ───────────── ARGUMENT PARSER ─────────────
    parser = argparse.ArgumentParser(description="Train Conditional DDPM on LSUN")

    parser.add_argument("--checkpoint", type=str, default=None, help="Path of the last checkpoint to resume training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--save_every", type=int, default=10, help="Model checkpoint save interval (in epochs)")
    parser.add_argument("--image_size", type=int, default=128, help="Image size for resizing")
    parser.add_argument("--data_root", type=str, default="./data/scenes", help="Root directory for LSUN dataset")

    args = parser.parse_args()

    # ───────────── DEVICE ─────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device utilisé :", device)

    # ───────────── TRANSFORMATIONS ─────────────
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    start = datetime.datetime.now()
    # ───────────── DATASETS ─────────────
    train_dataset = MultiLMDBDataset(root_dir=args.data_root, split="train", max_per_class=10000, transform=transform)
    val_dataset = MultiLMDBDataset(root_dir=args.data_root, split="val", transform=transform)
    
    num_classes = len(train_dataset.classes)
    print(f"Nombre de classes détectées : {num_classes}")
    
    end = datetime.datetime.now() - start
    end = end.seconds + end.days / (24 * 60 * 60)
    print(f"Train and Validation Datasets created in {end // 3600} hours, {(end % 3600) // 60} mins and {end % 60} seconds")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    validation_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)
    
    all_labels = [class_idx for _, _, class_idx in train_dataset.samples]
    max_label = max(all_labels)
    if max_label >= num_classes:
        print(f"[ERREUR] Label invalide détecté : {max_label} >= {num_classes}")

    if args.checkpoint:
        # ───────────── MODÈLE + OPTIM ─────────────
        model = ConditionalUNet(input_c=3, base_c=64, cond_dim=128, n_classes=len(train_dataset.classes)).to(device)
        model.apply(init_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        # ───────────── ENTRAÎNEMENT ─────────────
        loss_history = train_ddpm(
            model, train_loader, validation_loader, optimizer, scheduler,
            device, epochs=args.epochs, save_every=args.save_every, start_epoch=start_epoch
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
        plt.savefig('./plots/DDPM_training_loss.png')
        
    else:
        # ───────────── MODÈLE + OPTIM ─────────────
        model = ConditionalUNet(input_c=3, base_c=64, cond_dim=128, n_classes=len(train_dataset.classes)).to(device)
        model.apply(init_weights)
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
        plt.savefig('./plots/DDPM_training_loss.png')

if __name__ == "__main__":
    main()