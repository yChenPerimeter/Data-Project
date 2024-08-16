from PIL import Image
import numpy as np
import os
import pandas as pd

def resize_image(image, size):
    return image.resize(size)

def calculate_pixel_difference_statistics(image1_path, image2_path):
    # Open the images and convert them to grayscale
    image1 = Image.open(image1_path).convert('L')
    image2 = Image.open(image2_path).convert('L')

    # Resize images to the same dimensions
    size = (min(image1.width, image2.width), min(image1.height, image2.height))
    image1 = resize_image(image1, size)
    image2 = resize_image(image2, size)

    # Convert images to numpy arrays
    image1_array = np.array(image1)
    image2_array = np.array(image2)

    # Compute pixel-wise differences
    difference_array = image1_array - image2_array

    # Calculate average and standard deviation of differences
    avg_difference = np.mean(difference_array)
    std_dev_difference = np.std(difference_array)

    return avg_difference, std_dev_difference

def process_images(real_images_dir, synthetic_images_dir, output_excel_path):
    results = []

    # Iterate over all real images in the directory
    for real_image_name in os.listdir(real_images_dir):
        if real_image_name.endswith(".png"):  # Ensure the file is a PNG image
            # Construct the corresponding synthetic image file name
            synthetic_image_name = real_image_name.replace(".png", "_fake_a.png")
            real_image_path = os.path.join(real_images_dir, real_image_name)
            synthetic_image_path = os.path.join(synthetic_images_dir, synthetic_image_name)

            # Check if the corresponding synthetic image exists
            if os.path.exists(synthetic_image_path):
                avg_diff, std_diff = calculate_pixel_difference_statistics(real_image_path, synthetic_image_path)
                results.append([real_image_name, synthetic_image_name, std_diff, avg_diff])

    # Create a DataFrame and save it to an Excel file
    df = pd.DataFrame(results, columns=["Real Image", "Synthetic Image", "Standard Deviation", "Average Pixel Difference"])
    df.to_excel(output_excel_path, index=False)

# Example usage
if __name__ == "__main__":
    real_images_dir = '/Users/ragini/Desktop/Perimeter Medical Imaging AI /My stuff /Training_Testing_DCIS_8_8_2024/Train_B_DCIS+'
    synthetic_images_dir = '/Users/ragini/Desktop/Perimeter Medical Imaging AI /My stuff /scripted model + output  /GAN synthetic data from scripted model /B_ScriptedmodelDCIS+ve_output '
    output_excel_path = '/Users/ragini/Desktop/Perimeter Medical Imaging AI /My stuff /scripted model + output  /GAN synthetic data from scripted model /Fake_AvsReal_DCIS+ve.xlsx'

    process_images(real_images_dir, synthetic_images_dir, output_excel_path)
