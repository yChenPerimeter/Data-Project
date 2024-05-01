import os
import sys
from pathlib import Path
import pandas as pd
from pathlib import Path
from image_analysis_util import (load_image, calculate_ssim, calculate_rmse,
                                      measure_image_sharpness, speckle_index,
                                      calculate_signal_to_noise_ratio_db, calculate_psnr,calculate_cnr_roi_coord, save_to_csv)

def analyze_image(image_path, reference_path,  sig_roi, noise_roi):
    image = load_image(image_path, grayscale=True)
    reference = load_image(reference_path, grayscale=True)
    
    metrics = {

        # "RMSE": calculate_rmse(image, reference),
        # "Speckle Index": speckle_index(image),
        "SNR": calculate_signal_to_noise_ratio_db(image, sig_roi, noise_roi),
       # "CNR3_roi": calculate_cnr_roi_coord(image, tis_roi1, tis_roi2,noise_roi),
        # Change maximum pixel value to 255 for PSNR calculation
        # "PSNR": calculate_psnr(image, reference),
        # "SSIM": calculate_ssim(image, reference),
        # "Sharpness": measure_image_sharpness(image),
        #"fid": fid_find 
        # "CNR": calculate_cnr_roi_coord(image, sig_roi, bg_roi)
    }

    return metrics

def analyze_image(image_path, reference_path,  sig_roi, tis_roi1, tis_roi2, noise_roi):
    image = load_image(image_path, grayscale=True)
    reference = load_image(reference_path, grayscale=True)
    
    metrics = {
        "SNR": calculate_signal_to_noise_ratio_db(image, sig_roi, noise_roi),
        "CNR2_roi": calculate_cnr_roi_coord(image, sig_roi,noise_roi, noise_roi),
        "CNR3_roi": calculate_cnr_roi_coord(image, tis_roi1, tis_roi2, noise_roi),

    }

    return metrics

def main(input_directory, reference_path, model_name, signal_roi, background_roi, comparison= ("fake_B","real_A"), epoch= "N/A", fid = "N/A"):

    
    results = [] 
    #comparsion of G(x) with 8x image
    if comparison == ("fake_B","real_B"):
        # Specifically target 'fake_B' images within the '/images' subdirectory
        images_dir = Path(input_directory) / 'images'
        for image_subdir in images_dir.iterdir():
            if image_subdir.is_dir():  # Ensure it's a directory
                for image_path in image_subdir.glob('*_fake_B.png'):

                    # refference image reffer to 8x/GT image, as we want to see the difference between the denoised and GT image
                    reference_path = image_path.parent / image_path.name.replace('_fake_B', '_real_B')
                    metrics = analyze_image(image_path, reference_path, signal_roi, background_roi)
                    results.append(metrics)
    #comparsion of 1x with 8x image
    elif comparison == ("real_A","real_B"):
        # Specifically target 'real_A'/1x images within the '/images' subdirectory
        images_dir = Path(input_directory) / 'images'
        for image_subdir in images_dir.iterdir():
            if image_subdir.is_dir():  # Ensure it's a directory
                for image_path in image_subdir.glob('*_real_A.png'):

                    # refference image reffer to 8x/GT image, as we want to see the difference between the denoised and GT image
                    reference_path = image_path.parent / image_path.name.replace('_real_A', '_real_B')
                    metrics = analyze_image(image_path, reference_path, signal_roi, background_roi)
                    results.append(metrics)
    elif comparison == ("1x","real_B"):
        images_dir = Path(input_directory)
        for image_path in images_dir.iterdir():
            file_name = image_path.name
            image_reference_path = Path(reference_path) / file_name
            metrics = analyze_image(image_path, image_reference_path, signal_roi, background_roi)
            results.append(metrics)
    elif comparison == ("2x","real_B") or comparison == ("3x","real_B") or comparison == ("4x","real_B") or comparison == ("5x","real_B") or comparison == ("6x","real_B") or comparison == ("7x","real_B"):
        images_dir = Path(input_directory) 
        for image_path in images_dir.iterdir():
            file_name = image_path.name
            image_reference_path = Path(reference_path) / file_name
            metrics = analyze_image(image_path, image_reference_path, signal_roi, background_roi)
            results.append(metrics)

        

    if results:
        # Calculate mean of the metrics
        df = pd.DataFrame(results).mean().to_frame().T
        # Add a column for the model name, this is actually datasource than model name
        df.insert(0, 'Data Source', model_name)
        if comparison[0] not in ["real_A","2x"]:
        
            df.insert(1, "Model Name", "cGAN")
            df.insert(2, 'epoch', epoch)
        else:
            
            df.insert(1, "Model Name", "N/A")
            df.insert(2, 'epoch', "N/A")
        
        df.insert(3, 'fid', fid)
        # Define the full path for the CSV file
        # csv_file_path = CSV_DIR + "/" +'all_FID_image_analysis_results.csv'
        csv_file_path = os.path.join(CSV_DIR, "uint8_FID_image_analysis_results.csv")
        # Save or append the DataFrame to the CSV
        save_to_csv(df, csv_file_path)
        print(f"Mean analysis complete for model {model_name}. Results saved to {csv_file_path}.")
    else:
        print("No 'fake_B' images found for analysis.")


if __name__ == "__main__":
    results_directory = "/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/results/"
    CSV_DIR = "/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/analysis_results"
    ROOT_DIR = "/home/david/workingDIR/datasets_productionUint8/Paired_uint8_thumb_nail/more_test/test_01_16_thumb_nail/thumb_nail_B"
    #ROOT_DIR = "/home/david/workingDIR/datasets_productionUint8/Paired_uint8_thumb_nail/test/test_01_16_thumb_nail/thumb_nail_B"
    # Updated structure: Using list to handle multiple epochs for the same model
    best_epoch_list = {
        "production_O21CVPL00001_13_01_16": [
        #     ("test_20", "24.167971220528536"),
        #     ("test_11", "17.251750253966513"),
            ("test_31", "17.03868161665262"),
        #     ("test_3", "12.965241914417346")
        ],
        "production_O21CVPL00001_13_01_16_v1": [
        #     ("test_28", "35.05690471838181"),
        #     ("test_12", "32.20152541605574"),
            ("test_33", "33.58137053520801"),
        #     ("test_1", "24.776094959556577")
        ],
        "production_O21CVPL_rmNans_v2": [
        #     ("test_13", "20.611053209862362"),
        #     ("test_19", "15.535419047432761"),
            ("test_45", "15.659139522790246"),
        #    ("test_1", "12.57191164762748")
        ],
        "production_O21CV00001_13_01_16": [
        #     ("test_15", "34.340431082160535"),
        #     ("test_32", "29.780200929008537"),
             ("test_35", "29.666050032804243"),
        #     ("test_1", "28.8319221642879145")
         ],
        "production_uint8_O21CVPL00001_13_01_16": [
        #     # ("test_5", "7.534026632322463"),
        #     # ("test_9", "8.599858021256397"),
        #     # ("test_14", "8.360215355327655"),
        #     # ("test_19", "10.652370377257641"),
        ("test_25", "11.954550476610443"),
        ],
        # "production_1x": [],
        # "production_1x_uin8": [],
        "1x": [],
        "2x": [],
        "3x": [],
        "4x": [],
        "5x": [],
        "6x": [],
        "7x": [],
        "noisy": [],
        
    }

    thumb_test_signal_roi = (0, 65, 672, 105)
    thumb_test_noise_or_background_roi = (0, 520, 672, 105)
    comparison = ("fake_B", "real_B")

for model_name, epochs in best_epoch_list.items():
    print("analyse data source: ", model_name)
    if model_name == "production_1x":
        # Handling for "production_1x"
        epoch = "NaN"
        fid = "NaN"
        input_directory = os.path.join(results_directory, "production_O21CVPL00001_13_01_16", "test_38")
        reference_dir = input_directory
        comparison = ("real_A", "real_B")
    elif model_name == "production_1x_uin8":
        # Handling for "production_1x_uin8"
        epoch = "NaN"
        fid = "NaN"
        input_directory = os.path.join(results_directory, "production_uint8_O21CVPL00001_13_01_16", "test_8")
        reference_dir = input_directory
        comparison = ("real_A", "real_B")
    elif model_name in["1x","2x", "3x","4x","5x","6x","7x"]:
        epoch = "NaN"
        fid = "NaN"
        input_directory = os.path.join(ROOT_DIR, model_name)
        reference_dir = os.path.join(ROOT_DIR, "8x")
        comparison = (model_name, "real_B")
    elif model_name == "noisy":
        epoch = "NaN"
        fid = "NaN"
        input_directory = os.path.join(ROOT_DIR, "noisy")
        reference_dir = os.path.join(ROOT_DIR, "8x")
        comparison = ("noisy", "real_B")
    else:
        for epoch, fid in epochs:
            input_directory = os.path.join(results_directory, model_name, epoch)
            reference_dir = input_directory
            main(input_directory, reference_dir, model_name, thumb_test_signal_roi, thumb_test_noise_or_background_roi, comparison, epoch, fid)
    main(input_directory, reference_dir,model_name, thumb_test_signal_roi, thumb_test_noise_or_background_roi, comparison=comparison, epoch=epoch, fid=fid)





# if __name__ == "__main__":
    
#     results_directory = "/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/results/"
#     CSV_DIR = "/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/analysis_results"
    
#     # after converge,  global maximum
#     best_epoch_set = {"production_O21CVPL00001_13_01_16": ("test_20","24.167971220528536") , "production_O21CVPL00001_13_01_16_v1":("test_28","35.05690471838181"),"production_O21CVPL_rmNans_v2":("test_13","20.611053209862362"),"production_O21CV00001_13_01_16":("test_15","34.340431082160535"),"production_1x":"" }
#     #1st local mimum
#     best_epoch_set = {"production_O21CVPL00001_13_01_16": ("test_11","17.251750253966513") , "production_O21CVPL00001_13_01_16_v1":("test_12","32.20152541605574"),"production_O21CVPL_rmNans_v2":("test_19","15.535419047432761"),"production_O21CV00001_13_01_16":("test_32","29.780200929008537s"),"production_1x":"" }
    
#     #2nd local mimum
#     best_epoch_set = {"production_O21CVPL00001_13_01_16": ("test_31","17.03868161665262") , "production_O21CVPL00001_13_01_16_v1":("test_33","33.58137053520801"),"production_O21CVPL_rmNans_v2":("test_45","15.659139522790246"),"production_O21CV00001_13_01_16":("test_35","29.666050032804243"),"production_1x":"" }
    
#     #3rd Global minium
#     best_epoch_set = {"production_O21CVPL00001_13_01_16": ("test_3","12.965241914417346") , "production_O21CVPL00001_13_01_16_v1":("test_1","24.776094959556577"),"production_O21CVPL_rmNans_v2":("test_1","12.57191164762748"),"production_O21CV00001_13_01_16":("test_1"," 28.8319221642879145"),"production_1x":"" }
    
#     # #1st locaal minumu (same as global), then second and third and fourth for 4410_cvpl model
#     best_epoch_set = {"production_uint8_O21CVPL00001_13_01_16": ("test_5","7.534026632322463") , "production_uint8_O21CVPL00001_13_01_16":("test_9","8.599858021256397"),"production_uint8_O21CVPL00001_13_01_16":("test_14","8.360215355327655"),"production_uint8_O21CVPL00001_13_01_16":("test_19"," 10.652370377257641"),"production_1x_uin8":""}
#     # Adjusting the structure to allow multiple epochs per model
#     # best_epoch_set = {
#     #     "production_uint8_O21CVPL00001_13_01_16": [
#     #         ("test_5", "7.534026632322463"),
#     #         ("test_9", "8.599858021256397"),
#     #         ("test_14", "8.360215355327655"),
#     #         ("test_19", "10.652370377257641")
#     #     ],
#     #     "production_1x_uin8": []  # Assuming you might want to add something here later
#     # }

    
    
#     #best_epoch_set = {"production_1x":"" }
#     thumb_test_signal_roi = (0, 65,672, 105)  # Example coordinates for the signal region
#     thumb_test_noise_or_background_roi = (0, 520, 672, 105)  # Example coordinates for the noise/background region
    
#     #default comparsion is between fake_B and real_B, as we always want to compare noisy and "clean" image
#     comparison = ("fake_B","real_B")
    
    
    
#     for model_name in best_epoch_set.keys():
#         if model_name == "production_1x":
#             # the 1x data being contained in the same directory as test_38
#             epoch = "NaN"
#             fid = "NaN"
#             input_directory = os.path.join(results_directory,"production_O21CVPL00001_13_01_16", "test_38" )
#             reference_dir =  input_directory
#             comparison = ("real_A","real_B")
#         elif model_name == "production_1x_uin8":
#             # the 1x data being contained in the same directory as test_38
#             epoch = "NaN"
#             fid = "NaN"
#             input_directory = os.path.join(results_directory,"production_uint8_O21CVPL00001_13_01_16", "test_8" )
#             reference_dir =  input_directory
#             comparison = ("real_A","real_B")
#         else:
#             epoch = best_epoch_set[model_name][0]
#             fid = best_epoch_set[model_name][1]
#             input_directory = os.path.join(results_directory, model_name, epoch)
#             # refference directory is the same as the input directory in this case
#             reference_dir =  input_directory
#         main(input_directory, reference_dir,model_name, thumb_test_signal_roi, thumb_test_noise_or_background_roi, comparison=comparison, epoch=epoch, fid=fid)
        
    
#     # # test special case 1x averaging result 
#     # model_name = "production_O21CVPL00001_13_01_16"
#     # epoch = best_epoch_set[model_name]
#     # input_directory = os.path.join(results_directory, model_name, epoch)
#     # # refference directory is the same as the input directory in this case
#     # reference_dir =  input_directory
#     # main(input_directory, reference_dir, "1x Averaging")
