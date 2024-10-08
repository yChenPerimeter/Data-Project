"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import sys

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
        # if image_numpy.shape[0] == 1:  # grayscale to RGB
        #     image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)



def tensor2FloatGrayIm(input_image, imtype=np.float32):
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
        #print_info(image_numpy)
        
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)))/ 2.0  * np.finfo(np.float32).max # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

def print_info(arr):
    print("dtype: ", arr.dtype)
    print("range: ", f'({arr.min(), arr.max()})')
    print("shape: ", arr.shape)
    

def normalize_image(image):
    """
    Normalize a float32 image to the range [0, 1].

    Parameters:
    - image: A numpy array representing the image.

    Returns:
    - Normalized image as a numpy array.
    """
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image


def save_image(image_numpy, image_path, aspect_ratio=1.0, output_size = ()):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    
    # since fromarray() only accepts shape: (1024, 672) for 1 channel , we have (1, 1024, 672)  
    image_numpy = np.squeeze(image_numpy)
    #print_info(image_numpy)

    image_pil = Image.fromarray(image_numpy)
    if len(image_numpy.shape) == 2:
        h, w = image_numpy.shape
    else:
        h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    
    
    if len(image_numpy.shape) == 3:
        
        image_pil.save(image_path)
    else:
        # Changed from image_pil to image_numpy #TODO can be used other way?
        # Special case for float grayscale images
        # print("image_path:", image_path)
        # print("ori: ", image_numpy[0:5, 0:5])
        image_numpy = normalize_image(image_numpy)

        #print_info(image_numpy)

        #print("sample: ", image_numpy[0:5, 0:5])
        image =  Image.fromarray(image_numpy)

        if output_size != ():
            # Resize the image to 564x411
            image = image.resize(output_size)
            # print("resize in util")
            # print(image.size)
       
        #print(image.getextrema())
        
        plt.imsave(image_path, image, cmap='gray')
        
        
        # sys.exit(1)
        
    
    #uint8?
    # print("img.dtype:", (np.asarray(image_pil)).dtype)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
