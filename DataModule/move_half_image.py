#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
move_half_images.py

Description:
    This script moves half of the image files from a source folder to a destination folder.
    It supports common image file formats such as .jpg, .jpeg, .png, .gif, .bmp, and .tiff.
    The script randomly selects which files to move.

Usage:
    1. Set the source and destination folder paths.
    2. Run the script to move half of the images from the source to the destination folder.

Requirements:
    - Python 3.x
    - os, shutil, random libraries (which are standard in Python)

Author: Youwei Chen
Date: 20240928
"""

import os
import shutil
import random

# Function to move half of the images from source_folder to destination_folder
def move_half_images(source_folder, destination_folder):
    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get the list of all files in the source folder
    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    # Filter out image files by checking extensions (you can add more extensions if needed)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in image_extensions]
    
    # Shuffle the list to randomize which images will be moved
    random.shuffle(image_files)
    
    # Calculate half of the image files
    num_to_move = len(image_files) // 2
    
    # Move the files
    for i in range(num_to_move):
        file_to_move = image_files[i]
        src_path = os.path.join(source_folder, file_to_move)
        dest_path = os.path.join(destination_folder, file_to_move)
        
        # Move the file
        shutil.move(src_path, dest_path)
        print(f"Moved {file_to_move} to {destination_folder}")

# Define the source and destination folders
source_folder = '/home/ychen/Documents/project/Data-Project/results/DCIS_cyclegan/test_latest/images/testA'  # Replace with your source folder path
destination_folder = '/home/ychen/Documents/project/Data-Project/results/DCIS_cyclegan/DCIS_20240808'  # Replace with your destination folder path

# Move half of the image files
move_half_images(source_folder, destination_folder)
