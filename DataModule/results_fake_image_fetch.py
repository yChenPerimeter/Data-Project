# file_header.py
"""
File: results_fake_image_fetch.py
Author: youwei
Date: [Insert Date]
Description: 
    This script copies only the image files from a specified source folder to a destination folder,
    where the filenames contain either 'fake A' or 'Rec A'. Supported image formats include .png, 
    .jpg, .jpeg, .gif, .bmp, and .tiff.
"""

import os
import shutil

# Function to copy image files that contain 'fake A' or 'Rec A' in the name
def copy_images_with_fake_or_rec(source_folder, destination_folder):
    """
    Copies image files from the source folder to the destination folder if 
    the filename contains 'fake A' or 'Rec A'.
    
    Parameters:
    - source_folder (str): Path to the folder where images are located.
    - destination_folder (str): Path to the folder where images will be copied.
    
    Supported formats: .png, .jpg, .jpeg, .gif, .bmp, .tiff
    """

    # Check if the destination folder exists, create if not
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created destination folder: {destination_folder}")

    # Loop through all files in the source folder
    for filename in os.listdir(source_folder):
        # Full path to the source file
        source_file = os.path.join(source_folder, filename)

        # Check if it's a file and if it's an image (you can add more formats if needed)
        if os.path.isfile(source_file) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            # Check if the filename contains 'fake A' or 'Rec A'
       
            if 'fake_a' in filename.lower() or 'rec_a' in filename.lower() or 'fake_b' in filename.lower() or 'rec_b' in filename.lower():
                # Full path to the destination file
                destination_file = os.path.join(destination_folder, filename)
                # Copy the file to the destination folder
                shutil.copy2(source_file, destination_file)
                print(f"Copied: {filename}")
            else:
                print(f"Skipped: {filename}")


# Example usage
if __name__ == "__main__":
    source_folder = '/home/ychen/Documents/project/Data-Project/results/DCIS_IDC_cyclegan/test_latest/images/testA'  # Replace with your source folder path
    destination_folder = '/home/ychen/Documents/project/Data-Project/results/DCIS_IDC_cyclegan/DCIS_IDC_20240826_all_fake_types'  # Replace with your destination folder path # Replace with the path to your destination folder
    
    copy_images_with_fake_or_rec(source_folder, destination_folder)
