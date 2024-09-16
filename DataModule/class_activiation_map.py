# -*- coding: utf-8 -*-
"""
@Author: ychen, Yanir, Ragini
Date: 2024-09-10
Purpose: Script for loading model weights and performing inference based on user-specified options.
"""



import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from PIL import Image
import os

# Define the directory to save images, replace with your own image
save_dir = '/home/david/workingDIR/Data-Project/outputs'
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# Load the model and checkpoint
checkpoint = "/home/david/workingDIR/Data-Project/checkpoints/imgAssist/ImgAssist (1).pt"
model = imgassist.ImgAssistCNN()
model.load_state_dict(torch.load(checkpoint))
model.eval()

# Transform the input image (assuming `patch` is your input tensor/image)
patch = transforms.ToPILImage()(patch)
data_transform = transforms.Compose([
    transforms.Resize((188, 188)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.Grayscale(num_output_channels=1)
])
input_tensor = data_transform(patch)
input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

# Forward pass to get the model output
with torch.no_grad():
    output = model(input_tensor)

# Set up GradCAM
cam = GradCAM(model=model, target_layer='features')
class_idx = output.squeeze(0).argmax().item()  # Get the predicted class index

# Get the CAM for the predicted class
cam_activation_map = cam(class_idx, output)

# Visualize and save the CAM and heatmap overlay
plt.imshow(cam_activation_map[0].squeeze(0).detach().cpu().numpy(), cmap='jet')
plt.title(f'Class: {class_idx}')
plt.axis('off')

# Save CAM image
cam_image_path = os.path.join(save_dir, f'CAM_class_{class_idx}.png')
plt.savefig(cam_image_path, bbox_inches='tight', pad_inches=0)
plt.close()

# Generate heatmap overlay
heatmap = overlay_mask(transforms.ToPILImage()(patch), cam_activation_map[0])

# Visualize and save the heatmap overlay
plt.imshow(heatmap)
plt.axis('off')

# Save heatmap image
heatmap_image_path = os.path.join(save_dir, f'Heatmap_class_{class_idx}.png')
plt.savefig(heatmap_image_path, bbox_inches='tight', pad_inches=0)
plt.close()

print(f"Saved CAM image to {cam_image_path}")
print(f"Saved heatmap overlay image to {heatmap_image_path}")
