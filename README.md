# Conditional DDPM on LSUN

This repository contains an implementation of a **Conditional Denoising Diffusion Probabilistic Model (DDPM)** for generating realistic images from the **LSUN** dataset.  

The project is organized into two main components:
1. **Model Training** (`main.py`) — Train a Conditional U-Net with class conditioning on LSUN.
2. **Dockerized Application** (`app/`) — Deploy a pre-trained model and generate images through an API.

---

## 📌 Key Features
- **Conditional DDPM** built on a U-Net backbone with sinusoidal time embeddings and class conditioning.
- **Multi-class LSUN dataset loader** (LMDB format).
- **Checkpoint support** for resuming interrupted training.
- Runs seamlessly on **GPU** or **CPU** (PyTorch).
- **Docker-ready** inference application.
- **Automatic model retrieval** from Hugging Face.

---

## 📂 Project Structure

```
.
├── main.py                # Main training entry point
├── dataset.py              # LSUN (LMDB) dataset loader
├── model.py                # Conditional U-Net architecture
├── train.py                # DDPM training loop
├── utils.py                # Utility functions
├── app/                         # Dockerized inference application
│   ├── Dockerfile               # Docker build configuration
│   ├── requirements.txt         # Python dependencies for the app
│   ├── startup.sh               # Startup script + model download
│   └── app/
│       ├── app.py               # Main web API server (Flask/FastAPI)
│       ├── generate.py          # Image generation logic
│       ├── model.py             # Inference-time model definition
│       ├── utils.py             # Helper functions for inference
│       ├── models/              # Pre-trained model checkpoints
│       └── static/              # Static frontend assets
│           └── index.html       # Simple web UI for interaction
├── requirements.txt             # Training dependencies
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Yromale1/ddpm_LSUN.git
cd ddpm_LSUN
```

### 2️⃣ Install dependencies
> **Tip:** Use a virtual environment or Conda for better dependency management.
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📂 Dataset

This project uses the **LSUN dataset** ([download link](https://academictorrents.com/details/c53c374bd6de76da7fe76ed5c9e3c7c6c691c489)).

- Classes used:  
  `church_outdoors`, `classroom`, `conference_room`, `dining_room`, `restaurant`, `tower`
- Training data: **10,000 images per class**, resized to **64×64** pixels.

---

## 🚀 Training

### Basic command
```bash
python main.py --data_root /path/to/lsun --epochs 200 --batch_size 32
```

### Available options
| Argument       | Description |
|----------------|-------------|
| `--checkpoint` | Path to a checkpoint to resume training |
| `--batch_size` | Batch size (default: 32) |
| `--lr`         | Learning rate (default: 1e-4) |
| `--epochs`     | Number of training epochs (default: 200) |
| `--save_every` | Save a checkpoint every N epochs |
| `--image_size` | Image resize dimension (default: 128) |
| `--data_root`  | Path to the LSUN dataset |

---

## 🐳 Docker Deployment

### 1️⃣ Build the image
```bash
docker build -t ddpm_lsun_app ./app
```

### 2️⃣ Run the container
```bash
docker run -p 5000:5000 ddpm_lsun_app
```

### 3️⃣ Environment variables
| Variable     | Description |
|--------------|-------------|
| `MODEL_FILE` | Model file name to download (default: `checkpoint_epoch_50.pth`) |
| `HF_TOKEN`   | Hugging Face token if the model is private |

**Available models:**  
- `checkpoint_epoch_20.pth` — Early training (20 epochs)  
- `checkpoint_epoch_50.pth` — Improved training (50 epochs)  

Models are hosted on:  
[https://huggingface.co/Yromale/ddpm](https://huggingface.co/Yromale/ddpm)

---

## 📊 Results

Due to hardware limitations, only early-stage training results are available:  
- **Epoch 20** — Initial recognizable shapes and patterns  
- **Epoch 50** — Improved quality but not fully converged  

Loss curves are saved automatically in `plots/DDPM_training_loss.png`.

---

## ✏️ Author
Developed by **Amory Hervet**.

