"""
train_ddpm.py

This script trains a Denoising Diffusion Probabilistic Model (DDPM) for image generation using the Unet and 
GaussianDiffusion classes from the denoising_diffusion_pytorch library. The model progressively removes noise 
from an image to generate realistic samples from a Gaussian noise distribution. This code includes data loading, 
training loop, loss calculation, WandB logging, Ray Tune for hyperparameter tuning, and model checkpointing.

Requirements:
    - PyTorch
    - torchvision
    - denoising_diffusion_pytorch
    - wandb
    - ray[tune]

Author: Youwei Chen
Date: 2024-10-07
"""

import os
import time
import torch
import wandb
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# Define the dataset class
class SingleClassImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Define the training function with Ray Tune and WandB logging
def train_ddpm(config):
    # Initialize WandB with project name 'DDPM Trail' and add a label
    wandb_run = wandb.init(
        project="DDPM Trail", 
        name="DDPM Experiment",
        config=config
    ) if not wandb.run else wandb.run
    wandb_run._label(repo='DDPM')
    
    image_dir = '/home/ychen/Documents/project/Data-Project/datasets/0928_suspecious_equall_split/trainB'
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    dataset = SingleClassImageDataset(image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Model and Diffusion Setup
    model = Unet(dim=64, dim_mults=(1, 2, 4, 8), flash_attn=True).to(config["device"])
    diffusion = GaussianDiffusion(model, image_size=128, timesteps=config["timesteps"]).to(config["device"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        total_loss = 0
        for i, images in enumerate(dataloader):
            images = images.to(config["device"])
            optimizer.zero_grad()
            loss = diffusion(images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Log metrics to WandB
            if i % 2 == 0:
                wandb.log({"Epoch": epoch + 1, "Step": i, "Loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"End of Epoch {epoch+1}/{config['epochs']} - Avg Loss: {avg_loss:.4f}")
        wandb.log({"Avg Loss": avg_loss})

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"/home/ychen/Documents/project/Data-Project/checkpoints/DDPM/ddpm_epoch_{epoch+1}.pth")
            print(f"Model checkpoint saved at epoch {epoch+1}")

    result_folder = "/home/ychen/Documents/project/Data-Project/results/DDPM/DDPM2"
    os.makedirs(result_folder, exist_ok=True)
    sampled_images = diffusion.sample(batch_size=4)
    for idx, img in enumerate(sampled_images):
        save_image(img, os.path.join(result_folder, f"sampled_image_{idx + 1}.png"))
    print(f"Sampled images saved to '{result_folder}' folder.")

# Configure Ray Tune
config = {
    "epochs": 10,
    "batch_size": tune.choice([4, 8, 16]),
    "lr": tune.loguniform(1e-5, 1e-3),
    "timesteps": 1000,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Run Ray Tune with ASHA Scheduler
tune.run(
    train_ddpm,
    config=config,
    resources_per_trial={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0},
    scheduler=ASHAScheduler(metric="Avg Loss", mode="min"),
    num_samples=10  # Number of hyperparameter samples to try
)
