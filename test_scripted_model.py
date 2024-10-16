import cv2
import numpy as np
import os
import torch
import torchvision as tv
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import zipfile

from torch.profiler import profile, record_function, ProfilerActivity
import sys

# Path creator function that checks if the path exists, if not, creates one
def path_creator(path):
    if not os.path.exists(path):
        os.makedirs(path)


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
        self.destination_folder = r"/home/ychen/Documents/project/Data-Project/results/DCIS_IDC_cyclegan/DCIS_IDC20240826CycleGAN_DCIS+splitedData"  # replace with your own destination path
        self.root_test_folder_A = r"/home/ychen/Documents/project/Data-Project/datasets/object_dection_data/DCIS+_images_split_feature_wise"  # replace with your own domain A path
        self.root_test_folder_B = ""  # replace with your own domain B path
        self.model_path_G_A = r"/home/ychen/Documents/project/Data-Project/checkpoints_scripted/cycleGAN/DCIS_IDC_cyclegan_G_A_checkpoints_scriptedlatest.pt"  # replace with your own G_A model path
        self.model_path_G_B = r"/home/ychen/Documents/project/Data-Project/checkpoints_scripted/cycleGAN/DCIS_IDC_cyclegan_G_B_checkpoints_scriptedlatest.pt"  # replace with your own G_B model path
        self.model_name = "cycleGAN_DCIS_IDC"
        self.inference_types = ["fake_B", "rec_A", "fake_A", "rec_B"]  # Options: 'fake_B', 'rec_A', 'fake_A', 'rec_B'
        self.cycle_gan = True  # Option to decide if CycleGAN model is used
        self.preserve_structure = True  # Option to preserve original folder structure
        self.compress_results = True  # Option to compress the destination folder after processing


def main():
    """Main function to run the processing"""
    cfg = Constants()
    processing(cfg)


def load_model(cfg, device):
    """Load appropriate models based on the inference types."""
    netG_A, netG_B = None, None
    if cfg.cycle_gan:
        if any(inf_type in ['fake_B', 'rec_A'] for inf_type in cfg.inference_types):
            netG_A = torch.jit.load(cfg.model_path_G_A)
            netG_A.to(device)
            netG_A.eval()
        if any(inf_type in ['fake_A', 'rec_B'] for inf_type in cfg.inference_types):
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


def run_inference(cfg, netG_A, netG_B, input_dir, device, destination_test_folder, inference_types):
    """Run inference based on the specified inference types."""
    with torch.no_grad():
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.jpeg') or file.endswith('.png'):
                    image_path = os.path.join(root, file)
                    tensor_images = process_image(image_path, device)

                    # Determine the relative path for preserving folder structure
                    relative_path = os.path.relpath(root, input_dir)
                    dest_folder = os.path.join(destination_test_folder, relative_path) if cfg.preserve_structure else destination_test_folder
                    path_creator(dest_folder)

                    # Copy the original image to the destination folder
                    shutil.copy(image_path, os.path.join(dest_folder, file))

                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU], record_shapes=True) as prof:
                        with record_function("model_inference"):
                            output_dict = {}
                            for inference_type in inference_types:
                                if cfg.cycle_gan:
                                    if inference_type == 'fake_B':
                                        output = netG_A(tensor_images)  # G_A(A)
                                        suffix = '_fakeB'
                                    elif inference_type == 'rec_A':
                                        fake_B = netG_A(tensor_images)  # G_A(A)
                                        output = netG_B(fake_B)           # G_B(G_A(A))
                                        suffix = '_recA'
                                    elif inference_type == 'fake_A':
                                        output = netG_B(tensor_images)  # G_B(B)
                                        suffix = '_fakeA'
                                    elif inference_type == 'rec_B':
                                        fake_A = netG_B(tensor_images)  # G_B(B)
                                        output = netG_A(fake_A)           # G_A(G_B(B))
                                        suffix = '_recB'
                                    else:
                                        raise ValueError("Invalid inference type. Options are: 'fake_B', 'rec_A', 'fake_A', 'rec_B'")

                                # Convert tensor to numpy and save the output image
                                output = np.squeeze(output.cpu().data.numpy())
                                output_file = os.path.splitext(file)[0] + suffix + os.path.splitext(file)[1]
                                plt.imsave(os.path.join(dest_folder, output_file), output, cmap='gray')
                                output_dict[inference_type] = output_file

    # Print profiling information
    if 'prof' in locals():
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


def compress_results(cfg, destination_test_folder):
    """Compress the destination folder if the flag is set."""
    if cfg.compress_results:
        zip_path = f"{destination_test_folder}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(destination_test_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=destination_test_folder)
                    zipf.write(file_path, arcname)
        print(f"Results compressed to: {zip_path}")


def processing(cfg):
    input_dir_A = cfg.root_test_folder_A
    input_dir_B = cfg.root_test_folder_B
    destination_test_folder = os.path.join(cfg.destination_folder, cfg.model_name)
    path_creator(destination_test_folder)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    if torch.cuda.is_available():
        print('GPU being used: ', torch.cuda.get_device_name(0))

    # Load appropriate model based on inference types
    netG_A, netG_B = load_model(cfg, device)

    # Check if input directories exist
    try:
        if not os.path.exists(input_dir_A) and any(inf_type in ['fake_B', 'rec_A'] for inf_type in cfg.inference_types):
            raise FileNotFoundError(f"Input directory for domain A not found: {input_dir_A}")
        if not os.path.exists(input_dir_B) and any(inf_type in ['fake_A', 'rec_B'] for inf_type in cfg.inference_types):
            raise FileNotFoundError(f"Input directory for domain B not found: {input_dir_B}")
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # Run inference for domain A
    domain_A_inference_types = [inf for inf in cfg.inference_types if inf in ['fake_B', 'rec_A']]
    if domain_A_inference_types:
        run_inference(cfg, netG_A, netG_B, input_dir_A, device, destination_test_folder, domain_A_inference_types)

    # Run inference for domain B
    domain_B_inference_types = [inf for inf in cfg.inference_types if inf in ['fake_A', 'rec_B']]
    if domain_B_inference_types:
        run_inference(cfg, netG_A, netG_B, input_dir_B, device, destination_test_folder, domain_B_inference_types)

    # Compress the results if the option is enabled
    compress_results(cfg, destination_test_folder)


if __name__ == "__main__":
    main()