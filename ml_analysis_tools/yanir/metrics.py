'spe'"""
File Name: metrics.py
Description: This file contains the functions used to calculate the metrics for the images.
Author: Yanir Levy
Date: 10/06/2021
Usage: This file is used by .npy images and .u16 image files.`
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from skimage.draw import polygon2mask
import pandas as pd
#import torch


if __name__ == "__main__":
    import constants

def measure_image_sharpness(image): # Blur detection metric - variance of Laplacian
    """Calculate the sharpness of an image using Laplacian method."""
    gray = image
    return cv2.Laplacian(gray, cv2.CV_32F).var() # 64 bit float potentially need to change to 32 bit float

def local_deviation(image, m, n):
    """
    Calculate the local deviation in the neighborhood of the pixel (m, n).
    """
    kernel_size = 3
    k = kernel_size // 2
    neighborhood = image[m - k:m + k + 1, n - k:n + k + 1]
    return np.max(neighborhood) - np.min(neighborhood)


def local_mean(image, m, n):
    """
    Calculate the local mean in the neighborhood of the pixel (m, n).
    """
    kernel_size = 3
    k = kernel_size // 2
    neighborhood = image[m - k:m + k + 1, n - k:n + k + 1]
    return np.mean(neighborhood)


def speckle_index(image):
    """
    Calculate the speckle index for the given image.
    """
    M, N = image.shape
    sigma_sum = 0
    mu_sum = 0

    # Calculate the sum of local deviation and local mean for all pixels
    for m in range(1, M - 1):  # Avoid the border for simplicity
        for n in range(1, N - 1):  # Avoid the border for simplicity
            local_dev = local_deviation(image, m, n)
            local_mean_val = local_mean(image, m, n)

            sigma_sum += local_dev
            mu_sum += local_mean_val

    # Calculate the speckle index
    SI = sigma_sum / mu_sum
    return SI



def calculate_mse(image1, image2):
    """Calculate the mean squared error between two images."""
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2) # converted to float to prevent any problems with modulus operations "Wrapping Around"
    err /= float(image1.shape[0] * image1.shape[1])
    return err

def calculate_snr(image, signal, background_noise):
    """Calculate the average SNR of the image using given ROI size."""
    roi_snr_signal = image[signal[1]:signal[1] + signal[3], signal[0]:signal[0] + signal[2]]
    #print(roi_snr_signal)
    roi_snr_noise = image[background_noise[1]:background_noise[1] + background_noise[3],
                         background_noise[0]:background_noise[0] + background_noise[2]]
    #print(roi_snr_noise)
    signal = np.mean(roi_snr_signal)
    background = np.std(roi_snr_noise)
    snr = abs(signal / background)
    return snr
def create_cnr_rois(image):
    "Usage: Select the signal, background tissue, and background noise regions of interest on image and press enter"
    # open image and allow user to select 3 points to define the signal region, the background tissue region and the background noise region respectively, each point should create a 32 by 32 ROI
    #signal = cv2.selectROI(" CNR Signal", image)
    #background_tissue = cv2.selectROI(" CNR Background Tissue", image)
    #background_noise = cv2.selectROI("CNR Background Noise", image)
    signal_points = create_cnr_mask(image, "CNR Signal")
    back_tiss_points = create_cnr_mask(image, "CNR Background Tissue")
    back_noise_points = create_cnr_mask(image, "CNR Background Noise")

    return signal_points, back_tiss_points, back_noise_points
def create_cnr_mask(image,region_name):
    # Initialize variables
    drawing = False
    points1 = []
    points2 = []
    roi = []
    point1 = ()
    point2 = ()

    # Mouse callback function
    def draw_line(event, x, y, flags, param):
        global drawing, point1, point2
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            point1 = (x, y)
            points1.append(point1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            point2 = (x, y)
            points2.append(point2)
            cv2.line(mask, point1, point2, (255), 2)

    # Create a black image and a window
    mask = np.zeros_like(image)
    cv2.namedWindow(region_name)
    cv2.setMouseCallback(region_name, draw_line)


    while True:

        # Overlay the mask onto the image
        img_display = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
        cv2.imshow(region_name, img_display)
        #combine poinrs into a single array representing the ROI

        # Close the window when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #add points so that points1[i] and points2[i] are alternating points
    points = []
    for i in range(len(points1)):
        points.append(points1[i])
        points.append(points2[i])
    #print(points)
    coordinates = [[y, x] for [x, y] in points]
    #print(coordinates)
    #show the final roi on image
    polygon  = coordinates
    mask = polygon2mask(image.shape, polygon)
    roi = image * mask
    #display on image with a blue overlay
    #img_display = cv2.addWeighted(image, 0.7, roi, 0.9, 0)
    #img_display = cv2.addWeighted(image, 0.7, roi, 0.7, 0)
    #cv2.imshow(region_name, img_display)
    #cv2.waitKey(0)



    #print just the roi values just inside the roi


    cv2.destroyAllWindows()
    return coordinates

def calculate_cnr(image, signal_points, back_tiss_points, back_noise_points):
    """Calculate the average CNR of the image using given ROI size."""
    roi_CNR_signal_mask = polygon2mask(image.shape, signal_points)
    roi_CNR_back_tissue_mask = polygon2mask(image.shape, back_tiss_points)
    roi_CNR_noise_mask = polygon2mask(image.shape, back_noise_points)

    roi_CNR_signal = image * roi_CNR_signal_mask
    roi_CNR_back_tissue = image * roi_CNR_back_tissue_mask
    roi_CNR_noise = image * roi_CNR_noise_mask


    #get just the values inside the roi
    roi_CNR_signal = roi_CNR_signal[roi_CNR_signal != 0]
    roi_CNR_back_tissue = roi_CNR_back_tissue[roi_CNR_back_tissue != 0]
    roi_CNR_noise = roi_CNR_noise[roi_CNR_noise != 0]

    signal = np.mean(roi_CNR_signal)
    background_tissue = np.mean(roi_CNR_back_tissue)
    background_noise = np.std(roi_CNR_noise)
    cnr = abs((signal - background_tissue) / background_noise)
    return cnr
def create_snr_rois(image):
    "Usage: Select the signal and background noise regions of interest on image and press enter"
    snr_signal = cv2.selectROI("SNR Signal", image)
    snr_background_noise = cv2.selectROI("SNR Background Noise", image)
    #set the window size to reduce aspect ration by 3x

    cv2.waitKey(0)
    #print("SNR Signal ROI:", snr_signal)
    #print("SNR Background Noise ROI:", snr_background_noise)
    return snr_signal, snr_background_noise
def calculate_structural_similarity(image1, image2):
    """Calculate the Structural Similarity Index (SSIM) between two images."""
    score, diff = structural_similarity(image1, image2, full=True, data_range=255. - 0.)
    return score, diff
def image_metrics(image1, image2):
    """Provide all metrics for two images."""
    cnr_signal, background_tissue, background_noise = create_cnr_rois(image1)
    cnr_noisy = calculate_cnr(image1, cnr_signal, background_tissue, background_noise)
    cnr_clean = calculate_cnr(image2, cnr_signal, background_tissue, background_noise)

    signal, background_noise = create_snr_rois(image1)
    snr_noisy = calculate_snr(image1, signal, background_noise)
    snr_clean = calculate_snr(image2, signal, background_noise)


    return snr_noisy, snr_clean, cnr_noisy, cnr_clean
def compute_cdf(histogram):
    cdf = np.cumsum(histogram)
    return cdf / cdf[-1] #normalize the cdf
def match_histograms(source, template):
    noisy_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
    clean_hist, _ = np.histogram(template.flatten(), 256, [0, 256])

    #calculate the cdf
    noisy_cdf = compute_cdf(noisy_hist)
    clean_cdf = compute_cdf(clean_hist)

    mapping = np.zeros(256)
    clean_value = 0
    for noisy_value in range(256):
        while clean_value < 256 and clean_cdf[clean_value] < noisy_cdf[noisy_value]:
            clean_value += 1
        mapping[noisy_value] = clean_value

    #map the noisy image to the clean image
    return mapping[source.astype(np.uint8)]

def main():

    # Load the images in root directory and assign them a variable based on their file name
    root = constants.png_image_metrics
    image_type = '.png'
    image_1x = cv2.imread(root + '1x_image' + image_type, cv2.IMREAD_GRAYSCALE)
    image_2x = cv2.imread(root + '2x_image' + image_type, cv2.IMREAD_GRAYSCALE)
    image_ground_truth = cv2.imread(root + 'ground_truth' + image_type, cv2.IMREAD_GRAYSCALE)
    image_denoised_GAN = cv2.imread(root + 'denoised_image_GAN' + image_type, cv2.IMREAD_GRAYSCALE)
    image_denoised_Unet = cv2.imread(root + 'denoised_image_Unet' + image_type, cv2.IMREAD_GRAYSCALE)

    # match the histogram of the noisy image to the clean image
    #image_1x = match_histograms(image_1x, image_ground_truth)
    #image_2x = match_histograms(image_2x, image_ground_truth)
    #image_denoised_GAN = match_histograms(image_denoised_GAN, image_ground_truth)
    #image_denoised_Unet = match_histograms(image_denoised_Unet, image_ground_truth)


    #create snr rois
    signal, background_noise = create_snr_rois(image_1x)
    # Calculate the SNR, SSIM, MSE, and Matching error for the 1x, 2x, and denoised images
    SNR_1x = calculate_snr(image_1x, signal, background_noise)
    SNR_averaged_2 = calculate_snr(image_2x, signal, background_noise)
    SNR_ground_truth = calculate_snr(image_ground_truth, signal, background_noise)
    SNR_denoised_GAN = calculate_snr(image_denoised_GAN, signal, background_noise)
    SNR_denoised_Unet = calculate_snr(image_denoised_Unet, signal, background_noise)

    #calculate the SSIM - this should be compared to the ground truth
    SSIM1X, diff1x = calculate_structural_similarity(image_ground_truth, image_1x)
    SSIM2X, diff2x = calculate_structural_similarity(image_ground_truth, image_2x)
    SSIMGT, diffgt = calculate_structural_similarity(image_ground_truth, image_ground_truth)
    SSIMDL_GAN, diffdl_GAN = calculate_structural_similarity(image_ground_truth, image_denoised_GAN)
    SSIMDL_Unet, diffdl_Unet = calculate_structural_similarity(image_ground_truth, image_denoised_Unet)

    # Calculate the CNR for the 1x, 2x, and denoised images - incorrect as of right now
    cnr_signal, background_tissue, background_noise = create_cnr_rois(image_1x)
    cnr_1x = calculate_cnr(image_1x, cnr_signal, background_tissue, background_noise)
    cnr_averaged_2 = calculate_cnr(image_2x,cnr_signal, background_tissue, background_noise)
    cnr_ground_truth = calculate_cnr(image_ground_truth, cnr_signal, background_tissue, background_noise)
    cnr_denoised_GAN = calculate_cnr(image_denoised_GAN, cnr_signal, background_tissue, background_noise)
    cnr_denoised_Unet = calculate_cnr(image_denoised_Unet, cnr_signal, background_tissue, background_noise)

    #Calculate the MSE - This should be compared to ground truth
    mse_1x = calculate_mse(image_ground_truth, image_1x)
    mse_2x = calculate_mse(image_ground_truth, image_2x)
    mse_ground_truth = calculate_mse(image_ground_truth, image_ground_truth)
    mse_denoised_GAN = calculate_mse(image_ground_truth, image_denoised_GAN)
    mse_denoised_Unet = calculate_mse(image_ground_truth, image_denoised_Unet)

    # Calculate the sharpness of the 1x, 2x, and denoised images
    sharpness_1x = measure_image_sharpness(image_1x)
    sharpness_2x = measure_image_sharpness(image_2x)
    sharpness_ground_truth = measure_image_sharpness(image_ground_truth)
    sharpness_denoised_GAN = measure_image_sharpness(image_denoised_GAN)
    sharpness_denoised_Unet = measure_image_sharpness(image_denoised_Unet)

    # create a list based on the image names

    images = [image_1x, image_2x,image_ground_truth, image_denoised_GAN,image_denoised_Unet]  # Assuming these images are loaded into these variables
    titles = ['1x', '2x','Ground Truth', 'Denoised-GAN', 'Denoised-Unet']

    # Plot the SSIM, SNR, Matching error, Pixelmatch score, MSE, and CNR values on bar charts
    fig, axes = plt.subplots(1, 5, figsize=(15, 10))

    metrics = [('SSIM', [SSIM1X, SSIM2X, SSIMGT, SSIMDL_GAN, SSIMDL_Unet]),
               ('MSE', [mse_1x, mse_2x, mse_ground_truth, mse_denoised_GAN, mse_denoised_Unet]),
               ('SNR', [SNR_1x, SNR_averaged_2, SNR_ground_truth, SNR_denoised_GAN, SNR_denoised_Unet]),
               ('CNR', [cnr_1x, cnr_averaged_2, cnr_ground_truth, cnr_denoised_GAN, cnr_denoised_Unet]),
               ('Sharpness', [sharpness_1x, sharpness_2x, sharpness_ground_truth,sharpness_denoised_GAN, sharpness_denoised_Unet])]

    for ax, (title, values) in zip(axes.ravel(), metrics):
        bars = ax.bar([1, 2, 3, 4, 5], values)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(['1x', '2x', 'GT', 'GAN', 'Unet'])
        bars[-1].set_color('green')
        bars[-2].set_color('red')
        bars[-3].set_color('yellow')
        ax.set_title(title)

    plt.tight_layout()
    plt.show()
    # Show images on a subplot side by side with no borders or axis and with a tight layout
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    #apply histogram equalization to the images
    image_1x_eq = cv2.equalizeHist(image_1x)
    image_2x_eq = cv2.equalizeHist(image_2x)
    image_ground_truth_eq = cv2.equalizeHist(image_ground_truth)
    image_denoised_GAN_eq = cv2.equalizeHist(image_denoised_GAN)
    image_denoised_Unet_eq = cv2.equalizeHist(image_denoised_Unet)
    equal_images = [image_1x_eq, image_2x_eq, image_ground_truth_eq, image_denoised_GAN_eq, image_denoised_Unet_eq]
    titles = ['1x', '2x', 'Ground Truth', 'Denoised-GAN', 'Denoised-Unet']
    # Show equalized hist images on a subplot side by side with no borders or axis and with a tight layout
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for ax, img, title in zip(axes, equal_images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    image_1x_adapteq = clahe.apply(image_1x)
    image_2x_adapteq = clahe.apply(image_2x)
    image_ground_truth_adapteq = clahe.apply(image_ground_truth)
    image_denoised_adapteq_GAN = clahe.apply(image_denoised_GAN)
    image_denoised_adapteq_Unet = clahe.apply(image_denoised_Unet)
    equal_images = [image_1x_adapteq, image_2x_adapteq, image_ground_truth_adapteq, image_denoised_adapteq_GAN, image_denoised_adapteq_Unet]
    titles = ['1x', '2x', 'Ground Truth', 'Denoised-GAN', 'Denoised-Unet']
    # Show equalized hist images on a subplot side by side with no borders or axis and with a tight layout
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for ax, img, title in zip(axes, equal_images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    # Calculate the speckle index
    SI_1x = speckle_index(image_1x)
    SI_2x = speckle_index(image_2x)
    SI_GT = speckle_index(image_ground_truth)
    SI_DL_GAN = speckle_index(image_denoised_GAN)
    SI_DL_Unet = speckle_index(image_denoised_Unet)

    #create a dataframe of SSIM, MSE, SNR, CNR, and Sharpness values

    df = pd.DataFrame({'SSIM': [SSIM1X, SSIM2X, SSIMGT, SSIMDL_GAN, SSIMDL_Unet],
                       'MSE': [mse_1x, mse_2x, mse_ground_truth, mse_denoised_GAN, mse_denoised_Unet],
                       'SNR': [SNR_1x, SNR_averaged_2, SNR_ground_truth, SNR_denoised_GAN, SNR_denoised_Unet],
                       'CNR': [cnr_1x, cnr_averaged_2, cnr_ground_truth, cnr_denoised_GAN, cnr_denoised_Unet],
                       'Blurring': [sharpness_1x, sharpness_2x, sharpness_ground_truth, sharpness_denoised_GAN, sharpness_denoised_Unet],
                       'Speckle Index': [SI_1x, SI_2x, SI_GT, SI_DL_GAN, SI_DL_Unet]},
                      index=['1x', '2x', 'Ground Truth', 'Denoised-GAN', 'Denoised-Unet'])

    #save as csv
    df.to_csv('C:/Users/Yanir/PerimeterProjects/ImgClear/Research/Image_test_data/Comparison_metrics.csv')
    print('saved')

if __name__ == "__main__":
    main()


