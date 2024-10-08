"""
train_ddpm.py

This script trains a Denoising Diffusion Probabilistic Model (DDPM) for image generation using the Unet and 
GaussianDiffusion classes from the denoising_diffusion_pytorch library. The model progressively removes noise 
from an image to generate realistic samples from a Gaussian noise distribution. This code includes data loading, 
training loop, loss calculation, and model checkpointing.

Requirements:
    - PyTorch
    - torchvision
    - denoising_diffusion_pytorch

Author: Youwei Chen
Date: 2024-10-07
"""
import os
import torch
from torch import optim
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from torchvision.utils import save_image

# 1. Model and Diffusion Process Initialization
# Define the Unet model and the GaussianDiffusion process.
# The model dimensions and multiplier are adjustable for different architectures.

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),  # Defines the channel multipliers for different Unet layers
    flash_attn=True  # Enables flash attention for faster processing
)

diffusion = GaussianDiffusion(
    model,
    image_size=128,  # Set the image resolution for diffusion
    timesteps=1000   # Define the number of diffusion steps
)

# 2. Data Preparation
# Use ImageFolder to load images from a directory and apply transformations.
# The dataset assumes images are organized in subdirectories for each class.

transform = transforms.Compose([
    transforms.Resize(128),  # Resize images to the required size
    transforms.ToTensor(),   # Convert images to tensors normalized to [0, 1]
])

#for cycleGAN it need two domain images, diffusion model only need one domain
dataset = datasets.ImageFolder('/home/ychen/Documents/project/Data-Project/datasets/0928_suspecious_equall_split/trainB', transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 3. Optimizer Configuration
# Initialize an Adam optimizer with a learning rate of 1e-4 for training the model.

optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 4. Training Loop
# Set up the main training loop over a specified number of epochs. Each epoch processes the entire dataset.

n_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)          # Move model to the selected device
diffusion.to(device)      # Move diffusion process to the selected device

for epoch in range(n_epochs):
    epoch_start_time = time.time()
    total_loss = 0        # Accumulate total loss over an epoch for logging

    # Iterate over data batches
    for i, (images, _) in enumerate(dataloader):
        images = images.to(device)  # Move images to the selected device
        optimizer.zero_grad()       # Zero gradients to avoid accumulation

        # Forward Pass: Calculate the diffusion loss
        loss = diffusion(images)
        
        # Backward Pass: Compute gradients and update model parameters
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # Track the cumulative loss for the epoch

        # Print loss every 100 batches
        if i % 2 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Step {i}, Loss: {loss.item()}")

    # Calculate and print the average loss for the epoch
    avg_loss = total_loss / len(dataloader)
    print(f"End of Epoch {epoch+1}/{n_epochs} - Avg Loss: {avg_loss:.4f} - Time Taken: {time.time() - epoch_start_time:.2f}s")

    # 5. Model Checkpointing
    # Save the model state every 10 epochs for later use or resuming training
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"ddpm_epoch_{epoch+1}.pth")
        print(f"Model checkpoint saved at epoch {epoch+1}")

# 6. Image Sampling (Post-training)
# After training, generate images by sampling from the diffusion process
sampled_images = diffusion.sample(batch_size=4)
print(f"Sampled images shape: {sampled_images.shape}")


result_folder = "/home/ychen/Documents/project/Data-Project/results/DDPM"

# Ensure the results folder exists
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# Save each sampled image
for idx, img in enumerate(sampled_images):
    save_image(img, os.path.join(result_folder, f"sampled_image_{idx + 1}.png"))

print(f"Sampled images saved to '{result_folder}' folder.")