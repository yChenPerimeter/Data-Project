import os
from pathlib import Path
import pandas as pd
from pathlib import Path
from image_analysis_util import (load_image, calculate_ssim, calculate_rmse,
                                      measure_image_sharpness, speckle_index,
                                      calculate_signal_to_noise_ratio, calculate_cnr_roi_coord, save_to_csv)

def analyze_image(image_path, reference_path,  sig_roi, bg_roi):
    image = load_image(image_path, grayscale=True)
    reference = load_image(reference_path, grayscale=True)
    
    metrics = {
        "SSIM": calculate_ssim(image, reference),
        "RMSE": calculate_rmse(image, reference),
        "Sharpness": measure_image_sharpness(image),
        "Speckle Index": speckle_index(image),
        "SNR": calculate_signal_to_noise_ratio(image, sig_roi, bg_roi),
        "CNR": calculate_cnr_roi_coord(image, sig_roi, bg_roi)
    }
    
    return metrics

def main(input_directory, reference_path, model_name, signal_roi, background_roi):

    
    results = []
    
    # Specifically target 'fake_B' images within the '/images' subdirectory
    images_dir = Path(input_directory) / 'images'
    for image_subdir in images_dir.iterdir():
        if image_subdir.is_dir():  # Ensure it's a directory
            for image_path in image_subdir.glob('*_fake_B.png'):
                print(f"Analyzing {image_path}")
                # refference image reffer to 1x image, as we want to see the difference between the 1x and denoised image
                reference_path = image_path.parent / image_path.name.replace('_fake_B', '_real_A')
                metrics = analyze_image(image_path, reference_path, signal_roi, background_roi)
                results.append(metrics)

    if results:
        # Calculate mean of the metrics
        df = pd.DataFrame(results).mean().to_frame().T
        # Add a column for the model name
        df.insert(0, 'Model Name', model_name)
        # Define the full path for the CSV file
        csv_file_path = CSV_DIR + "/" +'image_analysis_results.csv'
        # Save or append the DataFrame to the CSV
        save_to_csv(df, csv_file_path)
        print(f"Mean analysis complete. Results saved to {csv_file_path}.")
    else:
        print("No 'fake_B' images found for analysis.")


def one_times_main(input_directory, reference_path):
    """
    main function for one times image analysis
    """
    roi_height=150
    glass_start=75
    # signal_roi = (10, 10, 50, 50)  # Example ROI
    # background_roi = (60, 60, 50, 50)
    
    results = []
    
    # Specifically target 'fake_B' images within the '/images' subdirectory
    images_dir = Path(input_directory) / 'images'
    for image_subdir in images_dir.iterdir():
        if image_subdir.is_dir():  # Ensure it's a directory
            for image_path in image_subdir.glob('*_real_A.png'):
                print(f"Analyzing {image_path}")
                # refference image reffer to 8x image/ signal image/ clean image
                reference_path = image_path.parent / image_path.name.replace('_real_A', '_real_B')
                metrics = analyze_image(image_path, reference_path, roi_height, glass_start)
                results.append(metrics)

    if results:
        # Calculate mean of the metrics
        df = pd.DataFrame(results).mean().to_frame().T
        # Add a column for the model name
        df.insert(0, 'Model Name', model_name)
        # Define the full path for the CSV file
        csv_file_path = CSV_DIR + "/" +'image_analysis_results.csv'
        # Save or append the DataFrame to the CSV
        save_to_csv(df, csv_file_path)
        print(f"Mean analysis complete. Results saved to {csv_file_path}.")
    else:
        print("No 'fake_B' images found for analysis.")

if __name__ == "__main__":
    
    results_directory = "/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/results/"
    CSV_DIR = "/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/analysis_results"
   
    best_epoch_set = {"production_O21CVPL00001_13_01_16": "test_38" , "production_O21CVPL00001_13_01_16_v1":"test_28","production_O21CV00001_13_01_16":"test_15" }
    thumb_test_signal_roi = (0, 65,672, 105)  # Example coordinates for the signal region
    thumb_test_noise_or_background_roi = (0, 520, 672, 105)  # Example coordinates for the noise/background region
    for model_name in best_epoch_set.keys():
        epoch = best_epoch_set[model_name]
        input_directory = os.path.join(results_directory, model_name, epoch)
        # refference directory is the same as the input directory in this case
        reference_dir =  input_directory
        main(input_directory, reference_dir,model_name, thumb_test_signal_roi, thumb_test_noise_or_background_roi)
    
    
    # # test special case 1x averaging result 
    # model_name = "production_O21CVPL00001_13_01_16"
    # epoch = best_epoch_set[model_name]
    # input_directory = os.path.join(results_directory, model_name, epoch)
    # # refference directory is the same as the input directory in this case
    # reference_dir =  input_directory
    # main(input_directory, reference_dir, "1x Averaging")
