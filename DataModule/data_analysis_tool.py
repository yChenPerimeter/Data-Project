from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
import os
import pandas as pd
import cv2
import json
from tqdm import tqdm

# Load configuration from the JSON file
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

# Inception feature extractor for FID and sFID
class InceptionFeatureExtractor(nn.Module):
    def __init__(self, device, model_path):
        super(InceptionFeatureExtractor, self).__init__()
        self.model = models.inception_v3(weights=None, init_weights=True)

        # Load the state_dict to CPU first before transferring to device
        try:
            state_dict = torch.load(model_path, weights_only=True, map_location='cpu')
            self.model.load_state_dict(state_dict)
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            raise

        self.model.fc = nn.Identity()  # Replace the final layer with an identity layer
        self.model.to(device)
        self.model.eval()

    def forward(self, x):
        return self.model(x)

#preprocess image function
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image)
# Custom dataloader and handler for images
class ImageDataLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.real_images = [f for f in os.listdir(folder_path) if f.endswith(".png") and not f.endswith("_fakeB.png")]
        self.fake_images = [f for f in os.listdir(folder_path) if f.endswith("_fakeB.png")]

    def __len__(self):
        return len(self.real_images)

    def load_image_pair(self, index):
        real_image_name = self.real_images[index]
        fake_image_name = real_image_name.replace(".png", "_fakeB.png")
        real_image_path = os.path.join(self.folder_path, real_image_name)
        fake_image_path = os.path.join(self.folder_path, fake_image_name)
        if os.path.exists(fake_image_path):
            real_image = Image.open(real_image_path).convert('RGB')
            fake_image = Image.open(fake_image_path).convert('RGB')
            return real_image, fake_image, real_image_name, fake_image_name
        else:
            raise FileNotFoundError(f"Fake image {fake_image_name} not found for {real_image_name}")


def calculate_sfid(sem_act1, sem_act2):
    if sem_act1.shape[0] < 2 or sem_act2.shape[0] < 2:
        mu1, mu2 = np.mean(sem_act1, axis=0), np.mean(sem_act2, axis=0)
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        return ssdiff

    mu1, sigma1 = np.mean(sem_act1, axis=0), np.cov(sem_act1, rowvar=False)
    mu2, sigma2 = np.mean(sem_act2, axis=0), np.cov(sem_act2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    sfid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return sfid

# FID calculation function
def calculate_fid(activations1, activations2):
    mu1, mu2 = np.mean(activations1, axis=0), np.mean(activations2, axis=0)
    sigma1 = np.cov(activations1, rowvar=False) + 1e-6 * np.eye(activations1.shape[1])
    sigma2 = np.cov(activations2, rowvar=False) + 1e-6 * np.eye(activations2.shape[1])

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Calculate pSNR
def calculate_psnr(image):
    height, width = image.shape[:2]
    signal = image[:height // 2, :, :]
    noise = image[height // 2:, :, :]
    mse = np.mean((signal - np.mean(signal, axis=(0, 1), keepdims=True)) ** 2)
    return 20 * np.log10(255 / np.sqrt(mse)) if mse != 0 else float('inf')

#IS
def calculate_inception_score_per_image(images, device):
    # Instantiate the Inception model with weights
    inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()
    epsilon = 1e-10  # Small constant to avoid log(0)

    # Preprocess images and move to the appropriate device
    images = [preprocess_image(image).unsqueeze(0).to(device) for image in images]
    all_probs = []

    # Get probabilities for each image
    for idx, image in enumerate(images):
        with torch.no_grad():
            # Forward pass through the model
            features = inception_model(image)  # Use inception_model instead of feature_extractor
            softmax_probs = torch.softmax(features, dim=1).cpu().numpy()
            all_probs.append(softmax_probs)

            # Debugging: Print softmax probabilities and their shapes
            #print(f"Image {idx} softmax probabilities: {softmax_probs}, shape: {softmax_probs.shape}")

    # Convert list to tensor for easier manipulation
    probabilities = torch.tensor(all_probs).squeeze()  # Remove extra dimension

    # Check if probabilities are valid
    if probabilities.size(0) == 0:
        raise ValueError("No probabilities calculated, check input images.")

    # Calculate Inception Score
    mean_probs = probabilities.mean(dim=0) + epsilon  # Mean probabilities for IS calculation
    scores = []
    for i in range(probabilities.size(0)):
        p = probabilities[i] + epsilon  # Adding epsilon to avoid log(0)
        score = torch.exp(torch.sum(p * torch.log(p / mean_probs)))  # Inception Score formula
        scores.append(score.item())

    return np.mean(scores), np.std(scores)

# Calculate CSNR
def calculate_csnr(image):
    height, width = image.shape[:2]
    signal = image[:height // 2, :, :]
    noise = image[height // 2:, :, :]
    signal_mean = np.mean(signal)
    noise_mean = np.mean(noise)
    return 20 * np.log10(signal_mean / noise_mean) if noise_mean != 0 else float('inf')

# Calculate image sharpness
def calculate_sharpness(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return laplacian.var()

# SSIM as the inspection score calculation
def calculate_inspection_score(real_image, fake_image):
    # Convert images to grayscale
    real_image_gray = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
    fake_image_gray = cv2.cvtColor(fake_image, cv2.COLOR_BGR2GRAY)

    # Debugging: Check dimensions of the images
    #print(f"Real image shape: {real_image_gray.shape}, Fake image shape: {fake_image_gray.shape}")

    # Ensure both images are the same size
    if real_image_gray.shape != fake_image_gray.shape:
        # Resize fake image to match real image
        fake_image_gray = cv2.resize(fake_image_gray, (real_image_gray.shape[1], real_image_gray.shape[0]))
        #print(f"Resized fake image shape: {fake_image_gray.shape}")

    # Calculate SSIM
    ssim_score, _ = ssim(real_image_gray, fake_image_gray, full=True)
    return ssim_score


def calculate_metrics_for_all(folder_path, json_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = InceptionFeatureExtractor(device, model_path)
    feature_extractor.eval()

    fid_results = []
    dataloader = ImageDataLoader(folder_path)

    for i in tqdm(range(len(dataloader))):
        try:
            real_image, fake_image, real_image_name, fake_image_name = dataloader.load_image_pair(i)
            print(f"Processing pair: {real_image_name} <-> {fake_image_name}")

            # Collect image metadata for real and fake images
            format_real = real_image.format
            format_fake = fake_image.format
            mode_real = real_image.mode
            mode_fake = fake_image.mode
            size_real = real_image.size
            size_fake = fake_image.size
            dtype_real = np.array(real_image).dtype
            dtype_fake = np.array(fake_image).dtype
            shape_real = np.array(real_image).shape
            shape_fake = np.array(fake_image).shape
            channels_real = 1 if len(shape_real) == 2 else shape_real[2]
            channels_fake = 1 if len(shape_fake) == 2 else shape_fake[2]

            # Preprocess images for FID and sFID
            real_image_torch = preprocess_image(real_image).to(device)
            fake_image_torch = preprocess_image(fake_image).to(device)

            # Add batch dimension to tensors
            real_image_torch = real_image_torch.unsqueeze(0)  # Shape becomes [1, channels, height, width]
            fake_image_torch = fake_image_torch.unsqueeze(0)  # Shape becomes [1, channels, height, width]

            with torch.no_grad():
                real_features = feature_extractor(real_image_torch).cpu().numpy()
                fake_features = feature_extractor(fake_image_torch).cpu().numpy()

            # Calculate FID and sFID
            fid_value = calculate_fid(real_features, fake_features)
            sfid_value = calculate_sfid(real_features, fake_features)

            # Convert images to numpy array for pSNR, cSNR, sharpness, and SSIM calculations
            real_image_np = np.array(real_image)
            fake_image_np = np.array(fake_image)

            # pSNR, cSNR, and sharpness for real and fake images
            real_psnr = calculate_psnr(real_image_np)
            fake_psnr = calculate_psnr(fake_image_np)

            real_csnr = calculate_csnr(real_image_np)
            fake_csnr = calculate_csnr(fake_image_np)

            real_sharpness = calculate_sharpness(real_image_np)
            fake_sharpness = calculate_sharpness(fake_image_np)

            # Calculate Inception Score for fake and real images
            is_fake_value = calculate_inception_score_per_image([fake_image], device)
            is_real_value = calculate_inception_score_per_image([real_image], device)

            # SSIM calculation as the inspection score
            inspection_score = calculate_inspection_score(real_image_np, fake_image_np)

            # results
            def convert_to_serializable(value):
                if isinstance(value, np.ndarray):
                    return value.tolist()  # Convert NumPy arrays to lists
                elif isinstance(value, (np.generic, np.integer)):
                    return value.item()  # Convert NumPy scalars to standard Python types
                elif isinstance(value, (np.float64, float)):
                    return float(value)  # Convert NumPy float to standard float
                elif isinstance(value, (np.uint8, np.uint16)):
                    return int(value)  # Convert unsigned integers to standard Python int
                elif isinstance(value, (list, tuple)):
                    return [convert_to_serializable(item) for item in value]  # Recursively convert list/tuple items
                return value

            fid_results = []
            fid_results.append({
                'Real_Image_Name': real_image_name,
                'Fake_Image_Name': fake_image_name,
                'Format_Real': convert_to_serializable(format_real),
                'Format_Fake': convert_to_serializable(format_fake),
                'Mode_Real': convert_to_serializable(mode_real),
                'Mode_Fake': convert_to_serializable(mode_fake),
                'Size_Real': convert_to_serializable(size_real),
                'Size_Fake': convert_to_serializable(size_fake),
                'Data_Type_Real': convert_to_serializable(dtype_real),
                'Data_Type_Fake': convert_to_serializable(dtype_fake),
                'Shape_Real': convert_to_serializable(shape_real),
                'Shape_Fake': convert_to_serializable(shape_fake),
                'Channels_Real': convert_to_serializable(channels_real),
                'Channels_Fake': convert_to_serializable(channels_fake),
                'FID': convert_to_serializable(fid_value),
                'sFID': convert_to_serializable(sfid_value),
                'Real pSNR': convert_to_serializable(real_psnr),
                'Fake pSNR': convert_to_serializable(fake_psnr),
                'Real cSNR': convert_to_serializable(real_csnr),
                'Fake cSNR': convert_to_serializable(fake_csnr),
                'Real Sharpness': convert_to_serializable(real_sharpness),
                'Fake Sharpness': convert_to_serializable(fake_sharpness),
                'SSIM': convert_to_serializable(inspection_score),
                'Fake IS & STD': convert_to_serializable(is_fake_value),
                'Real IS & STD': convert_to_serializable(is_real_value)
            })

        except FileNotFoundError as e:
            print(e)

    # Convert to DataFrame for statistical calculations
    df = pd.DataFrame(fid_results)

    # Extracting Inception Score and Standard Deviation values
    df['Real IS'] = df['Real IS & STD'].apply(lambda x: x[0])  # Extracting IS value (first element)
    df['Fake IS'] = df['Fake IS & STD'].apply(lambda x: x[0])

    # Calculate mean and standard deviation for all numeric columns
    mean_values = df.mean(numeric_only=True).to_dict()
    std_values = df.std(numeric_only=True).to_dict()

    # Append the calculated mean and std to the results
    result_summary = {
        'results': fid_results,
        'statistics': {
            'Mean': mean_values,
            'STD': std_values
        }
    }
    for entry in result_summary:
        if isinstance(entry, dict):
            for key, value in entry.items():
                try:
                    convert_to_serializable(value)
                except Exception as e:
                    print(f"Error converting {key}: {value} -> {e}")
        else:
            print(f"Unexpected entry type: {entry}")

    # Save the results to a JSON file
    with open(json_path, 'w') as json_file:
        result_summary_serializable = [convert_to_serializable(entry) for entry in result_summary]
        json.dump(result_summary_serializable, json_file, indent=4)
    print(f"Metrics calculation complete and saved to {json_path}.")


if __name__ == "__main__":
    config_file = "config.json"

    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)

    folder_path = config["input_folder_path"]
    model_path = config["inception_model_path"]
    json_path = "metrics.json"
    calculate_metrics_for_all(folder_path, json_path, model_path)