# -*- coding: utf-8 -*-
"""
@Author: ychen, adpoted from Yanir, Ragini
Date: 2024-09-10
Purpose: Script for loading model weights and performing inference based on user-specified options.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchcam.methods import GradCAM, GradCAMpp
from torchcam.utils import overlay_mask
import cv2
from commonTools import imgassist



# Define paths (removing extra spaces in directory paths)
scripted_model_path = '/home/david/workingDIR/Data-Project/checkpoints/imgAssist/ImgAssist_scripted.pt'
model_path = 'C:/Users/Yanir/PerimeterProjects/imgAssist/CP81.pth'
image_dir = 'C:/Users/Yanir/Downloads/datasets_3/datasets/train/images'
output_dir = 'C:/Users/Yanir/Documents/Outputscam'




# Load model assuming it is unscripted
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    # Use torch.load for an unscripted model
    # model = torch.load(model_path, map_location=device)
    # model = model.to(device)
    # model.eval()
    
    # Load the imgassist model
    model = imgassist.ImgAssistCNN()

    # Load the model weights from the checkpoint
    model.load_state_dict(torch.load(model_path))
    
    if torch.cuda.is_available():
        model.to(device)
        print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Load and preprocess the patch (image) to be used for inference
def load_image_as_patch(image_path):
    try:
        # Load image with cv2e
        img = cv2.imread(image_path)
        #apply the prprocess transform

        # Preprocess images
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((188, 188)),  # Size as specified in your original script
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        input_img = preprocess(img)
        input_img = input_img.unsqueeze(0)  # Add batch dimension
        input_img = input_img.to(device)  # Move to device
    
        return input_img
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Grad-CAM function
def generate_grad_cam(model, patch):
    #try:
    #patch = patch.unsqueeze(0).to(device)  # Add batch dimension and move to device
    model.eval()
    cam = GradCAM(model=model, target_layer='features')
    
    # Forward pass
    output = model(patch)
    output = output.to(device)
    #print(f'output: {output}')
    #probs = torch.softmax(output, dim=1)
    #print(f'probs: {probs}')
    # Set up GradCAM and get the activation map
    target_class = output.squeeze(0).argmax().item()
    #print(f'target_class: {target_class}')
    
    #print(f'cam: {cam}')    
    # Get the predicted class by argmaxing the output
    cam_activation_map = cam(target_class,output)
    #show the CAM
    plt.imshow(cam_activation_map[0].squeeze().detach().cpu().numpy(), cmap='jet')
    plt.show()
    return cam_activation_map, target_class
    # except Exception as e:
    #     print(f"Error generating Grad-CAM: {e}")
    #     return None, None

# Function to visualize and save Grad-CAM
def plot_grad_cam(image_path, cam_activation_map, output_path, target_class):
    try:
        img = Image.open(image_path).convert('L')
        plt.figure(figsize=(8, 8))
        plt.imshow(img, alpha=0.5, cmap='gray')  # Original image
        plt.imshow(cam_activation_map[0].squeeze(0).detach().cpu().numpy(), cmap='jet', alpha=0.5)  # CAM overlay
        plt.axis('off')
        plt.title(f'Class: {target_class}')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saved CAM visualization for class {target_class} to {output_path}")
    except Exception as e:
        print(f"Error saving Grad-CAM image {output_path}: {e}")

# Process images in the directory
def process_images(image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    for img_name in os.listdir(image_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):  # Support multiple image formats
            img_path = os.path.join(image_dir, img_name)

            # Load image as patch
            patch = load_image_as_patch(img_path)
            if patch is None:
                continue  # Skip if there was an error processing the image
       

            # Generate Grad-CAM
            cam_activation_map, target_class = generate_grad_cam(model, patch)
            if cam_activation_map is None or target_class is None:
                continue  # Skip if there was an error generating Grad-CAM

            # Save Grad-CAM visualization
            output_path = os.path.join(output_dir, f"cam_{img_name}")
            plot_grad_cam(img_path, cam_activation_map, output_path, target_class)

# Call the processing function
process_images(image_dir, output_dir)