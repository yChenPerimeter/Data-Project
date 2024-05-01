import os
from pathlib import Path
import pandas as pd
import numpy as np
from image_analysis_util import (load_image, calculate_ssim, calculate_rmse,
                                      measure_image_sharpness, speckle_index,
                                      calculate_signal_to_noise_ratio, calculate_cnr_roi_coord,calculate_psnr, save_to_csv, generate_boxplot)

def analyze_image_pairs(input_folder, enhanced_folder, signal_roi, noise_roi):
    """
    Analyze all corresponding image pairs between the input and enhanced folders,
    compute evaluation metrics, and return the results as a DataFrame.
    """
    results = []

    # Ensure folders exist
    if not Path(input_folder).exists() or not Path(enhanced_folder).exists():
        print("Input or enhanced folder does not exist.")
        return pd.DataFrame()

    # Loop through each image in the input folder
    for input_image_path in Path(input_folder).glob('*.png'):  # Adjust glob pattern as needed
        # Construct the path to the corresponding enhanced image
        enhanced_image_name = input_image_path.name.replace('_real_A', '_fake_B')  # Adjust naming convention as needed
        enhanced_image_path = Path(enhanced_folder) / enhanced_image_name

        if not enhanced_image_path.exists():
            print(f"Enhanced image not found for {input_image_path.name}")
            continue

        # Load images and calculate metrics
        input_image = load_image(input_image_path, grayscale=True)
        enhanced_image = load_image(enhanced_image_path, grayscale=True)
        metrics = {
            "Image Name": input_image_path.name,
            #"RMSE": calculate_rmse(input_image, enhanced_image),
            "Speckle Index ehanced": speckle_index(enhanced_image),
            "Speckle Index input": speckle_index(input_image),
            #"PSNR": calculate_psnr(input_image, enhanced_image),
            #"SSIM": calculate_ssim(input_image, enhanced_image),
            #"Sharpness": measure_image_sharpness(enhanced_image),
            #"SNR": calculate_signal_to_noise_ratio(enhanced_image, signal_roi, noise_roi),
            #"CNR": calculate_cnr_roi_coord(enhanced_image, signal_roi, noise_roi)
        }
        results.append(metrics)

    return pd.DataFrame(results)

def main(input_directory, enhanced_directory, csv_output_path, signal_roi, noise_roi, average_only=False, plot_metrics=None):
    """
    Main function to process the images, save the metrics to a CSV file,
    and optionally generate a box plot for selected metrics.
    :param average_only: If True, only saves the average of each metric to CSV.
    :param plot_metrics: List of metric names to be included in the box plot.
    """
    df_metrics = analyze_image_pairs(input_directory, enhanced_directory, signal_roi, noise_roi)
    
    if not df_metrics.empty:
        # Exclude non-numeric columns before calculating mean
        numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns.tolist()
        
        if average_only:
            # Calculate mean of the metrics (numeric columns only) and save to a single-row CSV
            df_avg = df_metrics[numeric_cols].mean().to_frame().T
            save_to_csv(df_avg, csv_output_path)
        else:
            # Save the DataFrame with individual results to CSV
            save_to_csv(df_metrics, csv_output_path)

        print(f"Analysis complete. Results saved to {csv_output_path}.")

        if plot_metrics:
            plot_output_path = csv_output_path.replace('.csv', '_boxplot.png')
            generate_boxplot(df_metrics, plot_metrics, plot_output_path)
            print(f"Box plot saved to {plot_output_path}.")
    else:
        print("No images were processed.")

if __name__ == "__main__":
    #input_directory = "/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/analysis_results/ClinicalCase/DCIS"
    #input_directory = "/home/david/workingDIR/datasets_clinical/DCIS+"
    input_directory = "/home/david/workingDIR/datasets_clinical/334_DCIS+_source"
    
    #enhanced_directory = "/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/analysis_results/ClinicalCase/DCIS"
    #enhanced_directory = "/home/david/workingDIR/datasets_clinical/cGAN_denoised/cGAN_cvpl_uint8_ep5"
    enhanced_directory = "/home/david/workingDIR/datasets_clinical/cGAN_denoised_DCIS+/cGAN_cvpl_uint8_ep25"
    
    csv_output_path = "/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/analysis_results/ClinicalCase/DCIS+_image_analysis_results.csv"
    
    # Define ROIs for signal and noise
    signal_roi = (0, 65, 672, 105)  # Example signal ROI
    noise_roi = (0, 520, 672, 105)  # Example noise/background ROI

    # Specify whether to save average metrics only and which metrics to plot
    average_only = True  # Change to True to save only the average results
    #plot_metrics = ["Speckle Index"]  # Specify metrics to include in the box plot

    main(input_directory, enhanced_directory, csv_output_path, signal_roi, noise_roi, average_only)
