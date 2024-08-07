#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# get_image_info.py
# Created by: Youwei Chen
# Date: 2024-07-17
#
# Description:
# This script provides information about an image file, including its dimensions (width and height)
# and file size in bytes. The script uses the Pillow library to open and analyze the image.
#
# Usage:
# python get_image_info.py /path/to/your/image.jpg
# Dimensions: 228x515 for capstone data

from PIL import Image
import os
import sys

def get_image_info(image_path):
    try:
        # Open an image file
        with Image.open(image_path) as img:
            # Get image dimensions
            width, height = img.size
            print(f"Dimensions: {width}x{height}")
            
            # Get file size in bytes
            file_size = os.path.getsize(image_path)
            print(f"File size: {file_size} bytes")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python get_image_info.py /path/to/your/image.jpg")
    else:
        image_path = sys.argv[1]
        get_image_info(image_path)
