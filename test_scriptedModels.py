"""
File Name: test_scriptedModels.py
Description: 
    quick denoise test using scripted cGAN model

"""

import cv2
import numpy as np
import os
import torch
import torchvision as tv
import matplotlib.pyplot as plt
from PIL import Image

from torch.profiler import profile, record_function, ProfilerActivity
import sys

"""a path creator function that checks if the path exists, if not, create one
"""
def path_creator(path):

    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

class Constants:
    """Constants used in the processing of the data."""
    def __init__(self):

       
        #self.destination_folder = "/home/david/workingDIR/datasets_pro_test/denoised_1x" # replace with your own destination path
        self.destination_folder = "/home/david/workingDIR/datasets_productionUint8/noise_adding_experiment/denoised" # replace with your own destination path
        self.destination_folder = "/home/david/workingDIR/datasets_clinical/tomatoes/cGAN_denoised"
        
        #self.root_test_folder = "/home/david/workingDIR/datasets_pro_test/testset_1x" # replace with your own destination path
        self.root_test_folder = "/home/david/workingDIR/datasets_productionUint8/noise_adding_experiment/noise_added" # replace with your own destination path
        self.root_test_folder = "/home/david/workingDIR/datasets_clinical/tomatoes/tomatoes_1x"
        
        # self.model_path = "/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/checkpoints_scripted/production_uint8_O21CVPL00001_13_01_16/production_uint8_O21CVPL00001_13_01_16_checkpoints_scripted25.pt" # replace with your own model path
        # self.model_name = "cvpl_uint8"

def main():
    """Main function to run the processing"""
    cfg = Constants()
    processing(cfg)

# processing Function
def processing(cfg):
    # Set model path and test images directory
    input_dir = cfg.root_test_folder
    print(input_dir)
    
    destination_test_folder = os.path.join(cfg.destination_folder, cfg.model_name)
    
    path_creator(destination_test_folder)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    
    #print the gpu device being used
    print('GPU being used: ', torch.cuda.get_device_name(0))


    net = torch.jit.load(cfg.model_path)    
    net.to(device)
    
    net.eval()


    #run the model on the images
    with torch.no_grad():
        for file in os.listdir(input_dir):
            if file.endswith('.jpeg') or file.endswith('.png'):
                image_path = os.path.join(input_dir, file)
                # img read here is RGBA images TODO change result from int to float
                #image_path = "/home/david/workingDIR/datasets_pro_test/testset_1x/average_ChickenThigh_1_1.png"
                rgba_image = Image.open(image_path)
            
                rgb_image = rgba_image.convert('RGB')
                image = rgb_image
                # image = plt.imread(image_path)
                transform = tv.transforms.Compose([tv.transforms.ToTensor(),tv.transforms.Normalize((.5),(.5)), tv.transforms.Grayscale(1)])#, limit all values to a std deviation
                tensor_images = transform(image)
                tensor_images = torch.unsqueeze(tensor_images, dim=0)  # add a dimension for the batch
                tensor_images = tensor_images.to(device)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("model_inference"):
                    y = net.forward(tensor_images)
                    # tensor to numpy
                    y = np.squeeze(y.cpu().data.numpy())
            print(y)
            # image_pil = PIL.Image.fromarray(y)
            # image_pil.save(image_path)
            plt.imsave(os.path.join(destination_test_folder, file), y, cmap='gray')
    #print on activity
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == "__main__":
    main()