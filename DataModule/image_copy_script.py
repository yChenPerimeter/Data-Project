"""
File Name: image_copy_script.py
Author: Youwei Chen
Description: 
    This script recursively searches for all image files in a specified folder and its subfolders, 
    and then copies them to a destination folder. The script preserves the file metadata when copying.
"""

import os
import shutil
from pathlib import Path

# Define the source folder and destination folder
source_folder = '/home/ychen/Documents/project/mother_data/datasets'
destination_folder = '/home/ychen/Documents/project/Data-Project/datasets/0922_Duct_fiber_suspeciouse/testB'

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Define the image extensions to search for
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

def copy_images(source_dir, destination_dir):
    """
    Recursively search for image files in source_dir and copy them to destination_dir.
    
    Parameters:
    - source_dir: The root folder to start searching for image files.
    - destination_dir: The folder where the images will be copied.
    """
    # Traverse the source directory and its subdirectories
    for root, _, files in os.walk(source_dir):
        for file in files:
            # Check if the file has an image extension
            if any(file.lower().endswith(ext) for ext in image_extensions):
                # Construct the full path of the source file
                source_file = os.path.join(root, file)
                
                # Construct the full path of the destination file
                destination_file = os.path.join(destination_dir, file)
                
                # Copy the image to the destination folder
                shutil.copy2(source_file, destination_file)
                print(f'Copied: {source_file} to {destination_file}')

# Call the function
copy_images(source_folder, destination_folder)
