"""
Author: Yanir Levy
Date: 03/22/2024
Description: This Tool is used to analyze the interim image data, save the image metadata to a csv file, and report on
the average image statistics.

Functionality:
- read in the png images from the preprocessed folders

"""

import os
import cv2
import numpy as np
import pandas as pd
import logging
from DataModule import data_module_utils
def local_dev_mean(image, m, n):
    #calculate the local mean of a pixel
    kernel_size = 3
    k = kernel_size // 2
    neighborhood = image[m-k:m+k+1, n-k:n+k+1]
    local_dev = np.max(neighborhood) - np.min(neighborhood)
    local_mean = np.mean(neighborhood)
    return local_dev, local_mean

def speckle_index(image):
    #calculate the speckle index of an image
    m,n = image.shape
    sigma_sum = 0
    mu_sum = 0

    #calculate the sum of local deviations and local mean for all pixels
    for m in range(1 , m-1):
        for n in range(1, n-1):
            local_dev, local_mean = local_dev_mean(image, m, n)

            sigma_sum += local_dev
            mu_sum += local_mean

    #calculate the speckle index
    speckle_index = sigma_sum / mu_sum

    return speckle_index

def variance_of_laplacian(img):
    #calculate the sharpness of an image
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    variance = np.var(laplacian)
    return variance

def signal_to_noise_ratio(img):
    image_height = img.shape[0]
    image_width = img.shape[1]
    roi_height = 150
    glass_start = 75
    #calculate the signal to noise ratio of an image
    signal_region = img[glass_start:glass_start + roi_height, 0:image_width]
    avg_pixel_value_signal = np.mean(signal_region)

    # Take the standard dev pixel value of the noise region
    noise_region = img[image_height - roi_height:image_height, 0:image_width]
    std_pixel_value_noise = np.std(noise_region)

    # Calculate the signal to noise ratio
    signal_to_noise = avg_pixel_value_signal / std_pixel_value_noise

    # Display the image with both the signal and noise regions
    '''cv2.rectangle(img, (0, glass_start), (image_width, glass_start + roi_height), (0, 255, 0), 2)
    cv2.rectangle(img, (0, image_height - roi_height), (image_width, image_height), (0, 0, 255), 2)
    cv2.imshow(str(signal_to_noise), img)  # Set signal region in image as a rectangle
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    return signal_to_noise

def calculate_image_statistics(image):
    # Read in the image
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # Calculate the average pixel value
    avg_pixel_value = np.mean(img)

    # Calculate the standard deviation of the pixel value
    std_pixel_value = np.std(img)

    # Calculate the variance of the laplacian
    image_sharpness = variance_of_laplacian(img)

    # Calculate the speckle index
    speckle = speckle_index(img)

    # Calculate the signal-to-noise ratio
    snr = signal_to_noise_ratio(img)

    return avg_pixel_value, std_pixel_value, image_sharpness, speckle, snr
def data_prep(paramfile):
    # Create a list to store the image path and the image statistics
    image_path = []
    image_statistics = []
    # Load the configuration file
    config = data_module_utils.load_imgclear_config(paramfile)
    # Set the path to the preprocessed images paths then preprocessed images
    images_root = config['paths']
    image_folder = images_root['image_output']
    # Read in the image names
    images = data_module_utils.read_image_names(image_folder)
    logging.info(f"Found {len(images)} images, starting image analysis...")

    # Create a dataframe to store the image statistics
    df = pd.DataFrame()

    #populate the df with image path data
    i = 0
    for image in images:
        image_path.append(image)
        image_stats = calculate_image_statistics(image)
        image_statistics.append(image_stats)
        #print progress of images processed so far for every 25 images
        if i % 25 == 0:
            print(f"Analyzed {i} of {len(images)} images", end = '\r')
        i += 1

    df['Image Path'] = image_path

    # split the image path to get the image type and region
    df['Image Type'] = df['Image Path'].apply(lambda x: x.split('/')[-1].split('_')[0])
    df['Region'] = df['Image Path'].apply(lambda x: x.split('/')[-3])
    df['Average Number'] = df['Image Path'].apply(lambda x: x.split('/')[-2])
    df['Image Number'] = df['Image Path'].apply(lambda x: x.split('/')[-1].split('_')[-1].split('.')[0])
    df['Average Pixel Value'] = [x[0] for x in image_statistics]
    df['Standard Deviation'] = [x[1] for x in image_statistics]
    df['Image Sharpness'] = [x[2] for x in image_statistics]
    df['Speckle Index'] = [x[3] for x in image_statistics]
    df['Signal to Noise Ratio'] = [x[4] for x in image_statistics]



    #create a csv file to store the image statistics
    csv_file_path = image_folder + 'processed_image_statistics.csv'
    df.to_csv(csv_file_path, index=False)
    logging.info(f"Image statistics saved to {csv_file_path}, analysis complete.")