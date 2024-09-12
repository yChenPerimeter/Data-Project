import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask

# Convolutional neural network architecture
class ImgAssistCNN(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(ImgAssistCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 188, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(188 * 11 * 11, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 8),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(8, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.classifier(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# Define paths
model_path = '/home/ragini/Downloads/CP.pth'
image_dir = '/mnt/Data4/Summer2024/RNarasimha/All_Model_Outputs/model_output_original_imgassist2/failure analysis /False_Negatives_images/IDC/'
output_dir = '/mnt/Data4/Summer2024/RNarasimha/All_Model_Outputs/model_output_original_imgassist2/failure analysis /False_Negatives_images/CAM_IDC_trial2/'

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImgAssistCNN()  # Replace with actual model architecture if different
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()
print("Model loaded successfully.")

# Preprocess images
preprocess = transforms.Compose([
    transforms.Resize((188, 188)),  # Size according to the input shape required by the model
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Load and preprocess the patch (image) to be used for inference
def load_image_as_patch(image_path):
    try:
        image = Image.open(image_path).convert('L')  # Grayscale conversion
        patch = preprocess(image)  # Apply the preprocessing transforms
        return patch
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Grad-CAM function
def generate_grad_cam(model, patch):
    patch = patch.unsqueeze(0).to(device)  # Add batch dimension and move to device
    patch.requires_grad_()  # Enable gradients for Grad-CAM

    # Set up GradCAM for a specific layer
    cam = GradCAM(model=model, target_layer='features.4')  # Use the correct index for the convolutional layer

    # Forward pass to capture hooks
    output = model(patch)
    target_class = output.argmax().item()  # Get the predicted class

    # Generate CAM for the predicted class
    cam_activation_map = cam(class_idx=target_class, scores=output)

    return cam_activation_map, target_class

# Function to visualize and save Grad-CAM
def plot_grad_cam(image_path, cam_activation_map, output_path, target_class):
    try:
        img = Image.open(image_path).convert('L')
        plt.figure(figsize=(8, 8))
        plt.imshow(img, alpha=0.5, cmap='gray')  # Original image
        plt.imshow(cam_activation_map[0].squeeze(0).detach().cpu().numpy(), cmap='jet', alpha=0.5)  # CAM overlay
        plt.axis('off')
        plt.title(f'Class: {target_class}')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saved CAM visualization for class {target_class} to {output_path}")
    except Exception as e:
        print(f"Error saving Grad-CAM image {output_path}: {e}")

# Process images in the directory
def process_images(image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    for img_name in os.listdir(image_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):  # Support multiple image formats
            img_path = os.path.join(image_dir, img_name)

            # Load image as patch
            patch = load_image_as_patch(img_path)
            if patch is None:
                continue  # Skip if there was an error processing the image

            # Generate Grad-CAM
            cam_activation_map, target_class = generate_grad_cam(model, patch)
            if cam_activation_map is None or target_class is None:
                continue  # Skip if there was an error generating Grad-CAM

            # Save Grad-CAM visualization
            output_path = os.path.join(output_dir, f"cam_{img_name}")
            plot_grad_cam(img_path, cam_activation_map, output_path, target_class)

# Call the processing function
process_images(image_dir, output_dir)
