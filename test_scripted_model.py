"""
File Name: test_scriptedModels.py
Description: 
    Quick denoise test using scripted CycleGAN model with configurable inference type
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

# Path creator function that checks if the path exists, if not, creates one
def path_creator(path):
    if not os.path.exists(path):
        os.mkdir(path)

def tensor2im(input_image, imtype=np.uint8):
    """Converts a Tensor array into a numpy image array."""
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: transpose and scaling
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)

class Constants:
    """Constants used in the processing of the data."""
    def __init__(self):
        self.destination_folder = "/home/david/workingDIR/datasets_pro_test/denoised_1x" # replace with your own destination path
        self.root_test_folder_A = "/home/david/workingDIR/datasets_pro_test/testset_A" # replace with your own domain A path
        self.root_test_folder_B = "/home/david/workingDIR/datasets_pro_test/testset_B" # replace with your own domain B path
        self.model_path_G_A = "/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/checkpoints_scripted/production_G_A.pt" # replace with your own G_A model path
        self.model_path_G_B = "/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/checkpoints_scripted/production_G_B.pt" # replace with your own G_B model path
        self.model_name = "cGAN_v1"
        self.inference_type = "fake_B"  # Options: 'fake_B', 'rec_A', 'fake_A', 'rec_B'
        self.cycle_gan = True  # Option to decide if CycleGAN model is used

def main():
    """Main function to run the processing"""
    cfg = Constants()
    processing(cfg)

def load_model(cfg, device):
    """Load appropriate models based on the inference type."""
    netG_A, netG_B = None, None
    if cfg.cycle_gan:
        if cfg.inference_type in ['fake_B', 'rec_A']:
            netG_A = torch.jit.load(cfg.model_path_G_A)
            netG_A.to(device)
            netG_A.eval()
        if cfg.inference_type in ['fake_A', 'rec_B']:
            netG_B = torch.jit.load(cfg.model_path_G_B)
            netG_B.to(device)
            netG_B.eval()
    return netG_A, netG_B

def process_image(image_path, device):
    """Load and transform an image for model inference."""
    rgba_image = Image.open(image_path)
    rgb_image = rgba_image.convert('RGB')
    image = rgb_image
    transform = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((.5), (.5)), tv.transforms.Grayscale(1)])
    tensor_images = transform(image)
    tensor_images = torch.unsqueeze(tensor_images, dim=0)  # add a dimension for the batch
    return tensor_images.to(device)

def run_inference(cfg, netG_A, netG_B, input_dir, device, destination_test_folder):
    """Run inference based on the specified inference type."""
    with torch.no_grad():
        for file in os.listdir(input_dir):
            if file.endswith('.jpeg') or file.endswith('.png'):
                image_path = os.path.join(input_dir, file)
                tensor_images = process_image(image_path, device)

                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                    with record_function("model_inference"):
                        if cfg.cycle_gan:
                            if cfg.inference_type == 'fake_B':
                                output = netG_A(tensor_images)  # G_A(A)
                                suffix = '_fakeB'
                            elif cfg.inference_type == 'rec_A':
                                fake_B = netG_A(tensor_images)  # G_A(A)
                                output = netG_B(fake_B)           # G_B(G_A(A))
                                suffix = '_recA'
                            elif cfg.inference_type == 'fake_A':
                                output = netG_B(tensor_images)  # G_B(B)
                                suffix = '_fakeA'
                            elif cfg.inference_type == 'rec_B':
                                fake_A = netG_B(tensor_images)  # G_B(B)
                                output = netG_A(fake_A)           # G_A(G_B(B))
                                suffix = '_recB'
                            else:
                                raise ValueError("Invalid inference type. Options are: 'fake_B', 'rec_A', 'fake_A', 'rec_B'")

                        # Convert tensor to numpy and save the output image
                        output = np.squeeze(output.cpu().data.numpy())
                        output_file = os.path.splitext(file)[0] + suffix + os.path.splitext(file)[1]
                
                plt.imsave(os.path.join(destination_test_folder, output_file), output, cmap='gray')

    # Print profiling information
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

def processing(cfg):
    input_dir_A = cfg.root_test_folder_A
    input_dir_B = cfg.root_test_folder_B
    destination_test_folder = os.path.join(cfg.destination_folder, cfg.model_name)
    path_creator(destination_test_folder)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    print('GPU being used: ', torch.cuda.get_device_name(0))

    # Load appropriate model based on inference type
    netG_A, netG_B = load_model(cfg, device)

    # Check if input directories exist
    try:
        if cfg.inference_type in ['fake_B', 'rec_A'] and not os.path.exists(input_dir_A):
            raise FileNotFoundError(f"Input directory for domain A not found: {input_dir_A}")
        if cfg.inference_type in ['fake_A', 'rec_B'] and not os.path.exists(input_dir_B):
            raise FileNotFoundError(f"Input directory for domain B not found: {input_dir_B}")
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # Run inference for domain A
    if cfg.inference_type in ['fake_B', 'rec_A']:
        run_inference(cfg, netG_A, netG_B, input_dir_A, device, destination_test_folder)

    # Run inference for domain B
    if cfg.inference_type in ['fake_A', 'rec_B']:
        run_inference(cfg, netG_A, netG_B, input_dir_B, device, destination_test_folder)

if __name__ == "__main__":
    main()