"""
File Name: compare_files.py
Author: Youwei Chen
Description:
    This script compares all files in two specified folders based only on their names (without including paths).
    It lists the file names that are common between both folders and those that are unique to each folder.
"""

import os

# Define the two folders to compare
folder1 = '/home/ychen/Documents/project/mother_data/Data_youwei_non-sus/NormalDuct/Training'
folder2 = '/home/ychen/Documents/project/mother_data/Data_youwei_non-sus/NormalDuct/Validation'

def get_file_names(folder):
    """
    Returns a set of file names in the specified folder (non-recursively).
    
    Parameters:
    - folder: Path to the folder whose files will be listed.
    
    Returns:
    - A set containing the names of the files in the folder.
    """
    return {f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))}

def compare_folders(folder1, folder2):
    """
    Compares files in folder1 and folder2 based only on file names and prints the matching and non-matching file names.
    
    Parameters:
    - folder1: Path to the first folder.
    - folder2: Path to the second folder.
    """
    # Get the set of file names from both folders
    files1 = get_file_names(folder1)
    files2 = get_file_names(folder2)
    
    # Find common and unique files
    common_files = files1.intersection(files2)
    unique_to_folder1 = files1 - files2
    unique_to_folder2 = files2 - files1
    
    # Print results
    print(f"Files common in both folders: {len(common_files)}")
    for file in common_files:
        print(f"  {file}")
    
    # print(f"\nFiles only in {folder1}: {len(unique_to_folder1)}")
    # for file in unique_to_folder1:
    #     print(f"  {file}")
    
    # print(f"\nFiles only in {folder2}: {len(unique_to_folder2)}")
    # for file in unique_to_folder2:
    #     print(f"  {file}")
    print(len(common_files))
# Call the function to compare the folders
compare_folders(folder1, folder2)
