#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# sample_images.py
# Created by: Youwei Chen
# Date: 2024-07-17
#
# Description:
# This script samples images from a specified input folder and copies them to an output folder.
# The user can specify the number of images to sample and whether to filter out files ending with '_lr'.
#
# Usage:
# python sample_images.py <input_folder> <output_folder> --sample_size <number_of_images> --filter_lr
# python sample_images.py /home/ychen/Documents/project/TrainingDataOG/Training/DCIS- /home/ychen/Documents/project/Data-Project/datasets/capstone/trainA --sample_size 1824 --filter_lr

# without Flip 1869
# python sample_images.py /home/ychen/Documents/Training_Testing/Train_A_DCIS+ve  /home/ychen/Documents/project/Data-Project/datasets/imgAssit20240729_DCIS/trainB --sample_size 3738 --filter_lr



import os
import shutil
import argparse

def sample_images(input_folder, output_folder, sample_size=2000, filter_lr=False):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # List all files in the input folder
    files = os.listdir(input_folder)
    
    # Filter for image files (e.g., .jpg, .png, etc.)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    images = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
    
    # Optionally filter out files ending with '_lr'
    if filter_lr:
        images = [f for f in images if not f.split('_')[-1].startswith('lr')]
    
    # Sample the first 'sample_size' images
    sampled_images = images[:sample_size]
    
    # Copy sampled images to the output folder
    for image in sampled_images:
        shutil.copy(os.path.join(input_folder, image), os.path.join(output_folder, image))
    
    print(f'Sampled {len(sampled_images)} images to {output_folder}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample images from a folder')
    parser.add_argument('input_folder', type=str, help='Path to the input folder containing images')
    parser.add_argument('output_folder', type=str, help='Path to the output folder to save sampled images')
    parser.add_argument('--sample_size', type=int, default=2000, help='Number of images to sample (default: 2000)')
    parser.add_argument('--filter_lr', action='store_true', help='Filter out files with "_lr" at the end')

    args = parser.parse_args()
    
    sample_images(args.input_folder, args.output_folder, args.sample_size, args.filter_lr)
