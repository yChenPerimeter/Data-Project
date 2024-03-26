# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 15:26:35 2023

@author: Yanir Levy
"""


# Global constants
tool = 'Image Data Comparison Tool'
version = '1.0.0'

# Tool operating modes
execMode = 1
initMode = 2
trainMode = 3
testMode = 4


#for testing + finetuning
'''Models'''
#model_name = "C:/Users/Yanir/PerimeterProjects/Research/Image_test_data/checkpoints_scripted_3layer3.pt" #pix2pix +cylce gan resnet 9 block [best]
#model_name = "./Research/Image_test_data/Test_models/New_UNet_sharpened.pt" #Mark 10/30/2023
#model_name = "./Research/Image_test_data/Test_models/New_UNet_no_sharpened.pt" #Mark 11/08/2023
#model_name = "./Research/Image_test_data/Test_models/FloatTest_lr10-4_checkpoints_scriptedlatest_float.pt" #Youwei 11/15/2023
#model_name = "./Research/Image_test_data/Test_models/best_model_larger_Unet.pt" #Mark larger Unet 12/05/2023 -- Best for WFBSCAN
#model_name = "C:/Users/Yanir/PerimeterProjects/Research/Image_test_data/Test_models/v4_FloatTest_lr10-4_batch1_checkpoints_scripted39.pt" #Youwei Float 12/08/2023 - FID 12 - 15
#model_name = "C:/Users/Yanir/PerimeterProjects/Research/Image_test_data/Test_models//v4_FloatTest_lr10-4_batch1_checkpoints_scripted8.pt" #Youwei Float 12/08/2023 - FID 9
#model_name = "C:/Users/Yanir/PerimeterProjects/Research/Image_test_data/Test_models/Large_UNet_8x_raw.pt" #Mark float 12/14/2023
#model_name = "C:/Users/Yanir/PerimeterProjects/Research/Image_test_data/Test_models/New_UNet_sharpened.pt" #Mark 12/18/2023
#model_name = "C:/Users/Yanir/PerimeterProjects/ImgClear/production_O21CVPL00001_13_01_16_v1_checkpoints_scripted52.pt" #Youwei 03/18/2023
#model_name = "C:/Users/Yanir/PerimeterProjects/ImgClear/best_model_8x_L1_without_notarget.pt" #Mark 03/18/2023
#model_name = "C:/Users/Yanir/PerimeterProjects/ImgClear/best_model_cGAN_vgg.pt" #Mark + Youwei 03/18/2023
model_name = "C:/Users/Yanir/PerimeterProjects/ImgClear/Unet_float_8x_L1_png_averaging_without_notarget.pt" #Mark 03/22/2023 - float
#model_name = "C:/Users/Yanir/PerimeterProjects/ImgClear/UNet_8x_L1_png_averaging_without_notarget_int.pt" #Mark 03/22/2023 - int

"""U16 data"""
#root_test_folder = './Research/Image_test_data/u16Data'#add the directory of the test case / cases
#root_test_folder = 'C:/Users/Yanir/Downloads/SEL00003_P000016/resources/OTIS/GAN_Image/SEL00003_P000016_Native/SEL00003_P000016/S03/'
#root_test_folder = 'C:/Users/Yanir/Documents/204/Q1/O2100009_P000046_Native_Combined/O2100009_P000046/S11' --1
#root_test_folder = 'C:/Users/Yanir/Documents/204/Q1/O20PR00002_P000041/resources/OTIS/O20PR00002_P000041/S02' #2
#root_test_folder = 'C:/Users/Yanir/Documents/204/Q1/O20PR00002_P000049/resources/OTIS/O20PR00002_P000049/S02' #3
#root_test_folder = 'C:/Users/Yanir/Documents/204/Q1/O20PR00002_P000143/resources/OTIS/O20PR00002_P000143/S02' #4
#root_test_folder = 'C:/Users/Yanir/Documents/204/Q1/O20PR00004_P000262/resources/OTIS/O20PR00004_P000262/S03' #5 --start here
#root_test_folder = 'C:/Users/Yanir/Documents/204/Q1/O20PR00004_P000275/resources/OTIS/O20PR00004_P000275/S06' #6
#root_test_folder = 'C:/Users/Yanir/Documents/204/Q1/O20PR00004_P000275/resources/OTIS/O20PR00004_P000275/S06' #7
#root_test_folder = 'C:/Users/Yanir/Downloads/O21CV00002_P000137/resources/OTIS/O21CV00002_P000137_Native_Combined/O21CV00002_P000137/S02/' # Calcification 173
#root_test_folder = 'C:/Users/Yanir/Documents/Q4_2023/Capstone/O21FK00003_P000040/resources/OTIS/O21FK00003_P000040_Native/S11/' # IDC 220
#root_test_folder = 'C:/Users/Yanir/Documents/Q4_2023/Capstone/O21FK00003_P000040/resources/O21FK00003_P000040/S10/' # IDC 220
#root_test_folder = 'C:/Users/Yanir/Documents/Q4_2023/Denoising_updates/O20PR00002_P000041/resources/OTIS/O20PR00002_P000041/S05/' # IDC 220
#root_test_folder = 'C:/Users/Yanir/Documents/Q4_2023/Denoising_updates/O20PR00002_P000050/resources/OTIS/O20PR00002_P000050/S02/' # IDC 220
#root_test_folder = 'C:/Users/Yanir/Documents/Q4_2023/Capstone/O20PR00002_P000050/resources/OTIS/O20PR00002_P000050_Native/O20PR00002_P000050/S04'
root_test_folder = 'C:/Users/Yanir/Documents/204/Q1/SNR/Issues/O21PL00002_P000122/S06'
#root_test_folder = 'D:/O21PL00001_P000470/S02/'
#root_test_folder = 'D:/O21PL00001_P000465/S02'
#root_test_folder = 'D:/O21PL00001_P000429/S02' #normal
#root_test_folder = 'D:/O21PL00001_P000443/S04' #new device normal

"""Output for clean denoised images"""
destination_test_png = './Research/Image_test_data/png_images/' #Destination of where the ouput images after DN
destination_test_png = 'D:/ImgclearData/Test/output_images' #Destination of where the ouput images after DN3
#destination_test_png = 'C:/Users/Yanir/Downloads/Testing/Testing/Suspicious-clean/' #Destination of where the ouput images after DN
"""
Inside png_image_metrics folder should be the following:
    1. 1x image - File name: /1x_image.png
    2. 2x image - File name: /2x_image.png
    3. Denoised images - File name: /denoised_image.png
    4. Ground truth images - File name: /ground_truth_image.png
"""

"""Image metrics --> need to take denoised image as well as different averages and compare"""
#png_image_metrics = 'C:/Users/Yanir/PerimeterProjects/ImgClear/Research/Image_test_data/png_image_metrics/youwei_images/'#Destination of where png images which do not need to be denoised are located (for metrics)
png_image_metrics = 'C:/Users/Yanir/PerimeterProjects/ImgClear/Research/Image_test_data/png_image_metrics/'#Destination of where png images which do not need to be denoised are located (for metrics)
png_image_metrics = 'C:/Users/Yanir/PerimeterProjects/ImgClear/Research/Image_test_data/png_image_metrics/youwei_images/' #Destination of where raw png images are located (for metrics)
png_image_metrics = 'C:/Users/Yanir/Downloads/wedge/WedgeSNR/' #Destination of where raw png images are located (for metrics)

"""were you take the .png images to denoise"""
raw_png_images = 'C:/Users/Yanir/PerimeterProjects/ImgClear/Research/Image_test_data/raw_png_images/' #Destination of where raw png images are located (for metrics)
raw_png_images = 'C:/Users/Yanir/PerimeterProjects/ImgClear/Research/Image_test_data/png_image_metrics/youwei_images/' #Destination of where raw png images are located (for metrics)
raw_png_images = 'D:/ImgclearData/Test/input_images/' #Destination of where raw png images are located (for metrics)
#raw_png_images = 'C:/Users/Yanir/PerimeterProjects/ImgClear/Research/Image_test_data/raw_png_images/IDC/DCIS/' #Destination of where raw png images are located (for metrics)
#raw_png_images = 'C:/Users/Yanir/PerimeterProjects/ImgClear/Research/Image_test_data/Test_models/Image_sets/'
#raw_png_images = 'C:/Users/Yanir/PerimeterProjects/ImgClear/Research/Image_test_data/raw_png_images/IDC/DCIS/'
#raw_png_images = 'C:/Users/Yanir/Documents/Q4_2023/Denoising_updates/floatGrape1x8x/home/david/workingDIR/datasets/Paired_FloatGWAD_CNGT_brain_v4/test/Grape/A-scanAvg/1x/'
#raw_png_images = 'C:/Users/Yanir/Documents/Q4_2023/Denoising_updates/Float_Ginger/'
#raw_png_images = 'C:/Users/Yanir/Downloads/Train_folders/'
#raw_png_images = 'C:/Users/Yanir/PerimeterProjects/ImgClear/Research/Image_test_data/png_image_metrics/youwei_images/1x2x/'
#raw_png_images = 'C:/Users/Yanir/Documents/Q4_2023/Capstone/O21FK00002_P000053/resources/O21FK00002_P000053/Images/'
#raw_png_images = 'C:/Users/Yanir/Documents/Q4_2023/Capstone/O21FK00003_P000040/resources/O21FK00003_P000040/'
#png_image_metrics = "C:/Users/Yanir/PerimeterProjects/ImgClear/Research/Image_test_data/png_image_metrics/mark_images/"

destination_1x = 'D:/ImgclearData/Test/input_images' #Destination of where the ouput images after DN

#interim data
interim_data = 'D:/ImgclearData/Test/' #Destination of where the ouput images after DN
#interim_data = 'D:/ImgclearData/Test/01_16'
#interim_data = 'D:/ImgclearData/CV1_01_016/CV1_01_016/ChickenThigh/Region3' #Destination of where the ouput images after DN
#interim_data = 'C:/Users/Yanir/Downloads/wedge/wedge' #Destination of where the ouput images after DN