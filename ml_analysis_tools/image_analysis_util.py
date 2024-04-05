import cv2
import numpy as np
import pandas as pd

from skimage.metrics import structural_similarity as ssim

from pathlib import Path
import matplotlib.pyplot as plt

def generate_boxplot(dataframe, metrics, plot_output_path):
    """
    Generates and saves a box plot for selected metrics.
    :param dataframe: DataFrame containing the metrics data.
    :param metrics: List of metric names to be plotted.
    :param plot_output_path: Path where the plot image will be saved.
    """
    dataframe[metrics].plot(kind='box', vert=False)
    plt.title('Distribution of Image Quality Metrics')
    plt.tight_layout()
    plt.savefig(plot_output_path)
    plt.close()

def draw_regions(image_path, signal_roi, noise_or_background_roi):
    """
    Draw rectangles around the signal and noise/background regions on an image.

    :param image_path: Path to the image file.
    :param signal_roi: Tuple of (x1, y1, width, height) for the signal region.
    :param noise_or_background_roi: Tuple of (x1, y1, width, height) for the noise or background region.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found.")
        return

    # Convert to a color image if it's grayscale
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw rectangles around the regions
    cv2.rectangle(image, (signal_roi[0], signal_roi[1]), (signal_roi[0]+signal_roi[2], signal_roi[1]+signal_roi[3]), (0, 255, 0), 2)
    cv2.rectangle(image, (noise_or_background_roi[0], noise_or_background_roi[1]), (noise_or_background_roi[0]+noise_or_background_roi[2], noise_or_background_roi[1]+noise_or_background_roi[3]), (0, 0, 255), 2)

    # Display the image
    cv2.imshow("Region Validation", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to save or append DataFrame to CSV
def save_to_csv(df, file_path):
    # Check if the file exists
    file = Path(file_path)
    if file.exists():
        # File exists, append data without header
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        # File does not exist, write new file with header
        df.to_csv(file_path, index=False)
        

def load_image(image_path, grayscale=True):
    """
    Load an image from the specified path.
    """
    # Ensure the path is a string
    image_path_str = str(image_path)
    if grayscale:
        return cv2.imread(image_path_str, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(image_path_str)

def calculate_ssim(imageA, imageB):
    """
    Compute the Structural Similarity Index (SSIM) between two images.
    """
    score, _ = ssim(imageA, imageB, full=True)
    return score

def calculate_rmse(imageA, imageB):
    """
    Compute the Root Mean Squared Error (RMSE) between two images.
    """
    return np.sqrt(calculate_mse(imageA, imageB))    
def calculate_mse(imageA, imageB):
    """
    Compute the Mean Squared Error (MSE) between two images. 
    """
    err = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)
    return err

def calculate_psnr(imageA, imageB):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Parameters:
    - imageA (numpy.ndarray): The original image.
    - imageB (numpy.ndarray): The reconstructed or modified image.

    Returns:
    - float: The PSNR value.
    """
    rmse = calculate_rmse(imageA, imageB)
    
    # Avoid division by zero
    if rmse == 0:
        return float('inf')
    
    MAX_I = 255.0  # Maximum pixel value for an 8-bit image
    psnr = 20 * np.log10(MAX_I / rmse)
    return psnr


def measure_image_sharpness(image):
    """
    Measure the sharpness of an image using the variance of the Laplacian.
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()

def speckle_index(image):
    """
    Calculate the speckle index of an image.
    """
    m, n = image.shape
    sigma_sum = 0
    mu_sum = 0

    for i in range(1, m-1):
        for j in range(1, n-1):
            neighborhood = image[i-1:i+2, j-1:j+2]
            local_dev = np.max(neighborhood) - np.min(neighborhood)
            local_mean = np.mean(neighborhood)
            sigma_sum += local_dev
            mu_sum += local_mean

    return sigma_sum / mu_sum if mu_sum else 0


def calculate_cnr_roi_coord(image, signal_roi, background_roi):
    """
    Calculate the Contrast-to-Noise Ratio (CNR) for given regions of interest within an image.

    The CNR is a measure of the contrast between a signal region and a background region,
    normalized by the standard deviation of the pixel values in the background region.
    This metric is useful for assessing the quality of images, especially in fields like
    medical imaging where distinguishing features from the background is crucial.

    Parameters:
    - image (numpy.ndarray): The image to analyze, expected to be a grayscale image 
      represented as a 2D NumPy array.
    - signal_roi (tuple): The region of interest for the signal, specified as a tuple 
      (x, y, width, height), where (x, y) are the coordinates of the top-left corner, 
      and 'width' and 'height' are the dimensions of the region.
    - background_roi (tuple): The region of interest for the background, specified in 
      the same format as 'signal_roi'.

    Returns:
    - float: The calculated CNR value. If the standard deviation of the background 
      region is zero, the function returns 0 to avoid division by zero.

    Example usage:
    ```
    image = cv2.imread("path/to/image.png", cv2.IMREAD_GRAYSCALE)
    signal_roi = (10, 20, 50, 50)  # Example: x=10, y=20, width=50, height=50
    background_roi = (100, 200, 50, 50)  # Another region for background
    cnr = calculate_cnr_roi_coord(image, signal_roi, background_roi)
    print("CNR:", cnr)
    ```
    """
    signal_region = image[signal_roi[1]:signal_roi[1]+signal_roi[3], signal_roi[0]:signal_roi[0]+signal_roi[2]]
    background_region = image[background_roi[1]:background_roi[1]+background_roi[3], background_roi[0]:background_roi[0]+background_roi[2]]
    
    mu_signal = np.mean(signal_region)
    mu_background = np.mean(background_region)
    sigma_background = np.std(background_region)
    
    return abs(mu_signal - mu_background) / sigma_background if sigma_background else 0



def calculate_signal_to_noise_ratio(image, signal_roi, noise_roi):
    """
    Calculate the signal-to-noise ratio (SNR) for an image using specified ROIs for the signal and noise.

    Parameters:
    - image (numpy.ndarray): The image to analyze, expected to be a grayscale image represented as a 2D NumPy array.
    - signal_roi (tuple): The region of interest for the signal, specified as a tuple (x, y, width, height).
    - noise_roi (tuple): The region of interest for the noise, specified in the same format as 'signal_roi'.

    Returns:
    - float: The calculated SNR, or None if the calculation cannot be performed due to issues like division by zero.
    """
    
    # Extract the signal and noise regions from the image
    signal_region = image[signal_roi[1]:signal_roi[1] + signal_roi[3], signal_roi[0]:signal_roi[0] + signal_roi[2]]
    noise_region = image[noise_roi[1]:noise_roi[1] + noise_roi[3], noise_roi[0]:noise_roi[0] + noise_roi[2]]

    # Ensure the regions are not empty to avoid division by zero
    if signal_region.size == 0 or noise_region.size == 0:
        print("One or both specified ROIs are empty.")
        return None

    # Calculate the mean and standard deviation
    mean_signal = np.mean(signal_region)
    std_noise = np.std(noise_region)

    # Compute the SNR
    if std_noise == 0:
        print("Standard deviation of the noise region is zero; SNR cannot be calculated.")
        return None

    snr = mean_signal / std_noise
    return snr



def cacluate_signal_to_noise_ratio_naive(image):
    """
    Calculate the signal-to-noise ratio (SNR) of an image.
    """
    signal = np.mean(image)
    noise = np.std(image)
    return signal / noise