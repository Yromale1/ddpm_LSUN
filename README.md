# Conditional DDPM on MNIST

This repository contains an implementation of a **Conditional Denoising Diffusion Probabilistic Model (DDPM)** for generating realistic images from the **MNIST** dataset.  

The project is organized into two main components:
1. **Model Training** (`main.py`) — Train a Conditional U-Net with class conditioning on MNIST.
2. **Dockerized Application** (`app/`) — Deploy a pre-trained model and generate images through an API.

---

## Key Features
- **Conditional DDPM** built on a U-Net backbone with sinusoidal time embeddings and class conditioning.
- **Checkpoint support** for resuming interrupted training.
- Runs seamlessly on **GPU** or **CPU** (PyTorch).
- **Docker-ready** inference application.
- **Automatic model retrieval** from Hugging Face.

---

## Project Structure

```
.
├── main.py                # Main training entry point
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

## Installation

### Clone the repository
```bash
git clone https://github.com/Yromale1/ddpm_MNIST.git
cd ddpm_MNIST
```

### Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dataset

This project uses the **MNIST dataset** ([download link](http://yann.lecun.com/exdb/mnist/)).

---

## Training

### Basic command
```bash
python main.py
```

### Available options
| Argument       | Description |
|----------------|-------------|
| `--checkpoint` | Path to a checkpoint to resume training (default: None) |
| `--batch_size` | Batch size (default: 128) |
| `--lr`         | Learning rate (default: 1e-4) |
| `--epochs`     | Number of training epochs (default: 200) |
| `--save_every` | Save a checkpoint every N epochs (defualt: 10) |
| `--image_size` | Image resize dimension (default: 128) |

---

## Docker Deployment

### Build the image
```bash
docker build -t ddpm_MNIST_app ./app
```

### Run the container
```bash
docker run -p 5000:5000 ddpm_MNIST_app
```

### Environment variables
| Variable     | Description |
|--------------|-------------|
| `MODEL_FILE` | Model file name to download (default: `checkpoint_epoch_200.pth`) |
| `HF_TOKEN`   | Hugging Face token if the model is private |

**Available models:**  
- `checkpoint_epoch_200.pth`

Models are hosted on:  
[https://huggingface.co/Yromale/ddpm](https://huggingface.co/Yromale/ddpm)

---

## Author

**Amory Hervet**  
Student at **EPITA**, specialization in *Cognitive Science & Advanced Computing (SCIA-G)*.  
[amory.rv@gmail.com](mailto:amory.rv@gmail.com) · [LinkedIn](https://www.linkedin.com/in/amory-hervet/)
