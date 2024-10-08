# Script to filter, print, and move PNG images that do not contain the '_bb' suffix
# Author: [Youwei Chen]
# Date: [Date] 
# Description:
# This script scans a specified folder for PNG files, filters out any that contain '_bb'
# in their name, prints the remaining filenames, and moves them to a destination folder.

import os
import shutil

# Define the source folder containing the images
source_folder = '/home/ychen/Documents/project/mother_data/S04/filtered_patches'

# Define the destination folder where filtered files will be moved
destination_folder = '/home/ychen/Documents/project/Data-Project/datasets/S04_test/testB'

# Ensure the destination folder exists, if not, create it
os.makedirs(destination_folder, exist_ok=True)

# Loop through each file in the source folder
for filename in os.listdir(source_folder):
    # Check if the file is a PNG and does not contain '_bb' in its name
    if filename.endswith('.png') and '_bb' not in filename:
        # Print the name of the file if it meets the criteria
        print(f"Moving file: {filename}")
        
        # Construct the full file paths for moving
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        
        # Move or copy the file to the destination folder
        shutil.move(source_path, destination_path)
        #shutil.copy(source_path, destination_path)

        print(f"File {filename} has been moved to {destination_folder}")
