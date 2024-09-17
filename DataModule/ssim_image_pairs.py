#!/usr/bin/env python3
"""
Script to calculate the top 50 lowest SSIM (Structural Similarity Index Measure) image pairs in a given folder.
This script scans a folder for image pairs with names like '*_fake_B.png' and '*_real_A.png', calculates the SSIM
between them, and reports the 50 image pairs with the lowest SSIM scores.

change ssim_scores[100:120] index to decide range

Usage:
    python ssim_image_pairs.py

Requirements:
    - scikit-image
    - Pillow
    - tqdm
    - numpy

Author: Your Name
Date: YYYY-MM-DD
"""

import os
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
from tqdm import tqdm  # For the progress bar

def calculate_ssim(imageA_path, imageB_path):
    """
    Calculate the SSIM between two images.

    Parameters:
        imageA_path (str): Path to the first image (fake).
        imageB_path (str): Path to the second image (real).
    
    Returns:
        float: SSIM value between the two images.
    """
    # Open the images and convert them to grayscale
    imageA = Image.open(imageA_path).convert('L')
    imageB = Image.open(imageB_path).convert('L')
    
    # Convert the images to numpy arrays
    imageA_np = np.array(imageA)
    imageB_np = np.array(imageB)
    
    # Calculate SSIM
    ssim_value, _ = ssim(imageA_np, imageB_np, full=True)
    return ssim_value

def find_image_pairs(directory):
    """
    Find pairs of images in a directory. Looks for images with 'fake_B' and 'real_A' in their filenames.

    Parameters:
        directory (str): Path to the folder containing images.

    Returns:
        list: A list of tuples containing paths to the fake and real image pairs.
    """
    fake_images = []
    real_images = []
    
    # Walk through the directory to collect image paths
    for root, dirs, files in os.walk(directory):
        for file in files:
            if 'fake_B' in file:
                fake_images.append(os.path.join(root, file))
            elif 'real_A' in file:
                real_images.append(os.path.join(root, file))
    
    # Pair images based on the matching parts of their filenames
    image_pairs = []
    for fake_image in fake_images:
        base_name = fake_image.replace('fake_B', '')
        for real_image in real_images:
            if real_image.replace('real_A', '') == base_name:
                image_pairs.append((fake_image, real_image))
                break
    
    return image_pairs

def get_top_50_lowest_ssim(image_pairs):
    """
    Get the top 50 image pairs with the lowest SSIM.

    Parameters:
        image_pairs (list): A list of tuples containing paths to image pairs.

    Returns:
        list: The top 50 image pairs with the lowest SSIM values.
    """
    ssim_scores = []
    
    # Adding a progress bar using tqdm
    for fake_image, real_image in tqdm(image_pairs, desc="Calculating SSIM", unit="pair"):
        score = calculate_ssim(fake_image, real_image)
        ssim_scores.append((fake_image, real_image, score))
    
    # Sort by SSIM value (ascending order, since lower SSIM means less similarity)
    ssim_scores.sort(key=lambda x: x[2])
    
    # Return the top 50 image pairs with the lowest SSIM
    #TODO change here to decide range
    return ssim_scores[0:120]

if __name__ == '__main__':
    # Specify the folder containing the images
    folder_path = '/home/ychen/Documents/project/Data-Project/results/DCIS_IDC_cyclegan/test_62/images/testA'
    
    # Find image pairs
    image_pairs = find_image_pairs(folder_path)
    
    # Get the top 50 pairs with the lowest SSIM
    top_50_lowest_ssim = get_top_50_lowest_ssim(image_pairs)
    
    # Print the results
    print("\nTop 50 Image Pairs with the Lowest SSIM:")
    for fake_image, real_image, score in top_50_lowest_ssim:
        print(f"Fake Image: {fake_image}")
        print(f"Real Image: {real_image}")
        print(f"SSIM: {score}")
        print("-" * 40)
