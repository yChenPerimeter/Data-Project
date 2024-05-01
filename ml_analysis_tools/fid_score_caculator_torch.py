# import os
# from options.test_options import TestOptions
# from data import create_dataset
# from models import create_model
# from util.visualizer import save_images, save_FloatGrayImages
# from util import html
# import sys

# import torch
# import torch.nn as nn
# from torchvision.models import inception_v3
# from matplotlib import pyplot as plt
# import numpy as np
# from scipy import linalg
# import sys

# from PIL import Image

# try:
#     import wandb
# except ImportError:
#     print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')
    


# # Define the InceptionV3 model for feature extraction
# class InceptionFeatureExtractor(nn.Module):
#     def __init__(self, transform_input=False):
#         super().__init__()
#         self.model = inception_v3(pretrained=True, transform_input=transform_input)
#         self.model.fc = nn.Identity()

#     def forward(self, x):
#         return self.model(x)

# # Function to calculate FID
# def calculate_fid(act1, act2):
#     """ Calculate FID between two sets of activations """
#     mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
#     mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
#     ssdiff = np.sum((mu1 - mu2)**2.0)
#     covmean = linalg.sqrtm(sigma1.dot(sigma2))
#     if np.iscomplexobj(covmean):
#         covmean = covmean.real
#     fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
#     return fid

# if __name__ == '__main__':
#     input_folder = "/home/david/workingDIR/datasets_productionUint8/Paired_uint8_thumb_nail/more_test/test_01_16_thumb_nail/thumb_nail_A/2x"
#     target_folder = "/home/david/workingDIR/datasets_productionUint8/Paired_uint8_thumb_nail/more_test/test_01_16_thumb_nail/thumb_nail_A/8x"
    
#     # Define the device for computation
#     device = torch.device('cuda:0') if torch.cuda.is_available() else "cpu"
#     # Initialize the feature extractor
#     feature_extractor = InceptionFeatureExtractor().to(device)
#     feature_extractor.eval()
#     # Lists to store features
#     target_features = []
#     input_features = []
    
#     #FID
#     input_tensors = .to(device).repeat(1, 3, 1, 1)
#     target_tensors = .to(device).repeat(1, 3, 1, 1)
        
#     # Extract features
#     with torch.no_grad():
#         input_feature = feature_extractor(input_tensors).cpu().numpy()
#         target_feature = feature_extractor(target_tensors).cpu().numpy()

            

#         input_features.append(input_feature)
#         target_features.append(target_feature)    

#     # Convert lists to numpy arrays
#     target_features = np.concatenate(target_features, axis=0)
#     input_features = np.concatenate(input_features, axis=0)

#     # Calculate FID
#     fid_value = calculate_fid(input_features, target_features)
#     print(f"FID: {fid_value}")

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# Define the InceptionV3 model for feature extraction
class InceptionFeatureExtractor(nn.Module):
    def __init__(self, transform_input=False):
        super().__init__()
        self.model = inception_v3(pretrained=True, transform_input=transform_input)
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)

# Function to calculate FID
def calculate_fid(act1, act2):
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


class SimpleImageLoader(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = [os.path.join(self.root_dir, fname) for fname in os.listdir(self.root_dir)
                       if os.path.isfile(os.path.join(self.root_dir, fname))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def load_data(input_folder, target_folder, batch_size):
    # transform = transforms.Compose([
    #     transforms.Resize((299, 299)),  # Resize to the input size required by Inception v3
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((.5),(.5))])

    input_dataset = SimpleImageLoader(root_dir=input_folder, transform=transform)
    target_dataset = SimpleImageLoader(root_dir=target_folder, transform=transform)

    input_loader = DataLoader(input_dataset, batch_size=batch_size, shuffle=False)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False)

    return input_loader, target_loader

def main():
    times = ""
    
    
    times = "5x"
    #times = "1x"
    root = "/home/david/workingDIR/datasets_productionUint8/Paired_uint8_thumb_nail/more_test/test_01_16_thumb_nail/thumb_nail_A"
    #root = "/home/david/workingDIR/datasets_productionUint8/Paired_uint8_thumb_nail/more_test/test_01_16_thumb_nail/thumb_nail_B"
    #root = "/home/david/workingDIR/datasets_productionUint8/Paired_uint8_thumb_nail/test/test_01_16_thumb_nail/thumb_nail_B/"
    if times: input_folder = os.path.join(root, times)
    else:
        input_folder = "/home/david/workingDIR/datasets_productionUint8/Paired_uint8_thumb_nail/more_test/test_01_16_thumb_nail/thumb_nail_A/2x"
    
    target_folder = os.path.join(root, "8x")
    batch_size = 32

    # Define the device for computation
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    # Initialize the feature extractor
    feature_extractor = InceptionFeatureExtractor().to(device)
    feature_extractor.eval()

    # Load data
    input_loader, target_loader = load_data(input_folder, target_folder, batch_size)

    # Lists to store features
    target_features = []
    input_features = []

    # Extract features
    # print(input_loader)
    with torch.no_grad():
        for input_batch in input_loader:
            input_tensors = input_batch.to(device)
            input_features.append(feature_extractor(input_tensors).cpu().numpy())

        for target_batch in target_loader:
            target_tensors = target_batch.to(device)
            target_features.append(feature_extractor(target_tensors).cpu().numpy())
            
        

    # Convert lists to numpy arrays
    target_features = np.concatenate(target_features, axis=0)
    input_features = np.concatenate(input_features, axis=0)

    # Calculate FID
    fid_value = calculate_fid(input_features, target_features)
    print(f"FID: {fid_value}")

# Uncomment the following line to run the program
main()
