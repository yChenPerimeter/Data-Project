
"The directory "testA" inputted into the code has rec_A, rec_B, fake_A, and fake_B images all in the same folder. The code is written to compare rec_A with fake_B pairs and rec_B with fake_A pairs for the GAN images."
import os
import re
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from scipy import linalg
import random

class InceptionFeatureExtractor(nn.Module):
    def __init__(self):
        super(InceptionFeatureExtractor, self).__init__()
        # Load the InceptionV3 model with pre-trained weights from a local file
        self.model = models.inception_v3(weights=None, init_weights=True)  # Initialize without weights
        state_dict = torch.load('/Users/ragini/Downloads/inception_v3_google-0cc3c7bd.pth', weights_only=True)  # Load local weights
        self.model.load_state_dict(state_dict)
        self.model.fc = nn.Identity()  # Remove the classification head

    def forward(self, x):
        return self.model(x)

# Calculation of sFID
def calculate_sfid(sem_act1, sem_act2):
    # Print shapes for debugging
    print(f"Shape of sem_act1: {sem_act1.shape}")  # printed to debug torch weight error
    print(f"Shape of sem_act2: {sem_act2.shape}")

    # Ensure at least 2 samples for covariance matrix calculation
    if sem_act1.shape[0] < 2 or sem_act2.shape[0] < 2:
        raise ValueError("At least 2 samples are required for each set of activations to calculate covariance matrix.")

    # Compute the mean and covariance matrices
    mu1, sigma1 = np.mean(sem_act1, axis=0), np.cov(sem_act1, rowvar=False)
    mu2, sigma2 = np.mean(sem_act2, axis=0), np.cov(sem_act2, rowvar=False)

    # Print shapes of covariance matrices for debugging
    print(f"Shape of sigma1: {sigma1.shape}")
    print(f"Shape of sigma2: {sigma2.shape}")

    # Ensure sigma1 and sigma2 are square matrices
    if sigma1.shape[0] != sigma1.shape[1] or sigma2.shape[0] != sigma2.shape[1]:
        raise ValueError("Covariance matrices must be square.")

    # Compute the squared difference of means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # Compute the matrix square root
    try:
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
    except ValueError as e:
        print(f"Error computing matrix square root: {e}")
        covmean = linalg.sqrtm(np.dot(sigma1, sigma2))

    # Ensure that covmean is a real matrix
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate sFID
    sfid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return sfid

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # InceptionV3 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img

def get_paired_images(directory, num_pairs):
    real_b_images = {}
    rec_b_images = {}

    # Iterate through the files in the directory and categorize them
    for filename in os.listdir(directory):
        if 'real_B' in filename:
            key = re.sub(r'_real_B.*', '', filename)  # Extract the common part of the filename
            real_b_images[key] = os.path.join(directory, filename)
        elif 'rec_B' in filename:
            key = re.sub(r'_rec_B.*', '', filename)  # Extract the common part of the filename
            rec_b_images[key] = os.path.join(directory, filename)

    # Find common keys and create pairs
    common_keys = list(set(real_b_images.keys()) & set(rec_b_images.keys()))
    if len(common_keys) < num_pairs:
        raise ValueError("Not enough paired images to select the desired number of pairs.")

    selected_keys = random.sample(common_keys, num_pairs)
    real_images = [real_b_images[key] for key in selected_keys]
    rec_images = [rec_b_images[key] for key in selected_keys]

    return real_images, rec_images

if __name__ == '__main__':
    # Set device to CUDA if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the feature extractor and move to the appropriate device
    feature_extractor = InceptionFeatureExtractor().to(device)
    feature_extractor.eval()

    # Directory containing all images
    image_dir = '/Users/ragini/Desktop/Perimeter Medical Imaging AI /My stuff /256_images/testA'

    # Get 10 paired real_A and fake_B images
    real_images, rec_images = get_paired_images(image_dir, 10)

    # Print selected images
    print("Selected real_B images:")
    for img in real_images:
        print(img)
    print("\nSelected rec_B images:")
    for img in rec_images:
        print(img)

    real_semantic_features = []
    fake_semantic_features = []

    with torch.no_grad():
        for real_img_path, fake_img_path in zip(real_images, rec_images):
            real_image = preprocess_image(real_img_path).to(device)
            fake_image = preprocess_image(fake_img_path).to(device)

            real_features = feature_extractor(real_image).cpu().numpy()
            fake_features = feature_extractor(fake_image).cpu().numpy()

            real_semantic_features.append(real_features)
            fake_semantic_features.append(fake_features)

    # Stack features
    real_semantic_features = np.vstack(real_semantic_features)
    fake_semantic_features = np.vstack(fake_semantic_features)

    sfid_value = calculate_sfid(real_semantic_features, fake_semantic_features)

    print(f"sFID: {sfid_value}")
