"""
Author: ychen 03/16/2024 adapted from Yanir Levy data_analysis_tool.py
TODO:Description: This Tool is used to analyze the cGAN model refference image data, save the image metadata to a csv file, and report on
the average image statistics.

Functionality:
- read in the png images from the preprocessed folders
- Calculate different image statisitics for each image
    1. Average pixel value
    2. Standard deviation of pixel value
    3. Variance of laplacian or "sharpness" of the image
    4. Signal to Noise Ratio
    5. Entropy (Texture characterization)

- Save each image's metadata to a csv file
- Report the average statisitics for each image specimen type, region and average number
- Produce a report on the average statistics for each image type as a .txt file


"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.stats import entropy
# import the configuration file
def load_config():
    with open('conf/config.json') as f:
        config = json.load(f)
    return config

def read_image_names(images_root):
    # Read in .png images from lowest level of the preprocessed folder save image name and path to a list
    images = []
    for root, dirs, files in os.walk(images_root):
        for file in files:
            if file.endswith('.png'):
                images.append(os.path.join(root, file))
    return images

def calculate_image_statistics(image):
    # Read in the image
    img = cv2.imread(image)

    # Calculate the average pixel value
    avg_pixel_value = np.mean(img)

    # Calculate the standard deviation of the pixel value
    std_pixel_value = np.std(img)

    # Calculate the variance of the laplacian
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    variance = laplacian.var()

    #Take the average pixel value of the signal region
    #signal_region = img[75:225, 2:670]
    signal_region = img[75:225, 2:500]
    avg_pixel_value_signal = np.mean(signal_region)

    #Take the standard dev pixel value of the noise region
    #noise_region = img[845:995, 2:670]
    noise_region = img[650:750, 2:500]
    std_pixel_value_noise = np.std(noise_region)

    #Calculate the signal to noise ratio
    signal_to_noise = avg_pixel_value_signal/std_pixel_value_noise
    print(signal_to_noise)
    image_height = img.shape[0]
    image_width = img.shape[1]
    # Display the image with both the signal and noise regions
    '''cv2.rectangle(img, (2, 75), (image_height - 2, 225), (0, 255, 0), 2)
    cv2.rectangle(img, (2, image_height - 155), (image_width - 2, image_height - 5), (0, 0, 255), 2)
    cv2.imshow(str(signal_to_noise), img) # Set signal region in image as a rectangle
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    #Calculate average pixel value of 1000 maximum pixel value in signal region
    signal_region = signal_region.ravel()
    signal_region = signal_region[signal_region.argsort()[-10000:]]

    avg_pixel_value_signal_high = np.mean(signal_region)
    #print(avg_pixel_value_signal_high)
    #print(avg_pixel_value_signal)

    #calculate CNR
    contrast_to_noise = (avg_pixel_value_signal_high - avg_pixel_value_signal)/std_pixel_value_noise
    #print(contrast_to_noise)


    # Calculate the entropy
    _bins = 256
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    image_entropy = entropy(hist, base=2)

    return avg_pixel_value, std_pixel_value, variance, signal_to_noise, image_entropy, contrast_to_noise



def main():
    # Load the configuration file
    config = load_config()

    # Set the path to the preprocessed images paths then preprocessed images
    images_root = config['paths']
    images_processed = images_root['pre_processed_images'] + '/magnitude_processed_avg_data'
    # Read in the image names
    images = read_image_names(images_processed)
    # Create a list to store the image statistics
    image_stats = []
    image_path = []
    # Loop through each image and calculate the statistics
    for image in images:
        stats = calculate_image_statistics(image)
        image_stats.append(stats)
        image_path.append(image)


    print(image_stats)
    # Create a pandas dataframe to store the image statistics by image path
    df = pd.DataFrame(image_stats, columns=['Average Pixel Value', 'Standard Deviation of Pixel Value', 'Variance of Laplacian', 'Signal to Noise Ratio', 'Entropy', 'Contrast to Noise'])
    df['Image Path'] = image_path
    #split the image path to get the image type and region
    df['Image Type'] = df['Image Path'].apply(lambda x: x.split('/')[-3])
    print(df['Image Type'])
    df['Region'] = df['Image Path'].apply(lambda x: x.split('/')[-2])
    df['Image Number'] = df['Image Path'].apply(lambda x: x.split('/')[-1].split('_')[0])

    #create a csv file to store the image statistics
    csv_file_path = images_root['output_csv_path'] + '/image_statistics_1.csv'
    df.to_csv(csv_file_path, index=False)

    # Report the average statistics for each image type as a .txt file
    #avg_stats = df.mean()


    breakpoint()
    # Set the path to the output csv file
    output_path = config['output_csv_path']

    # Set the path to the output txt file
    output_txt_path = config['output_txt_path']



main() # Run the main function