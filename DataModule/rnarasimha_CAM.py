"rnarasimha_CAM.py is a trial but the CAM has issues. Refer to ychen_CAM.py for the final debugged CAM version"
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


# Define the CustomModel
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Calculate the output size of the features part
        def get_conv_output_size(model, input_size):
            with torch.no_grad():
                dummy_input = torch.zeros(1, 1, input_size, input_size)
                output = model.features(dummy_input)
                return output.view(1, -1).size(1)

        conv_output_size = get_conv_output_size(self, 224)

        self.classifier = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 13)  # Adjust to match number of classes
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Load the checkpoint and initialize the model
checkpoint_path = '/home/ragini/Downloads/CP.pth'
checkpoint = torch.load(checkpoint_path)

model = CustomModel()

# Load state_dict with strict=False
model.load_state_dict(checkpoint, strict=False)
model.eval()

# Rest of your Grad-CAM code remains the same
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def generate_grad_cam(model, image_tensor, target_class):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    image_tensor.requires_grad_()

    # Forward pass
    output = model(image_tensor)

    # Ensure the target_class index is within the bounds of the output
    if target_class >= output.size(1):
        raise ValueError(f"Target class index {target_class} exceeds output size {output.size(1)}")

    # Target the specific class
    target = output[0, target_class]

    # Zero gradients
    model.zero_grad()

    # Backward pass
    target.backward(retain_graph=True)

    # Extract activations and gradients
    activations = image_tensor.grad
    if activations is None:
        raise ValueError("Activations are None. Ensure gradients are properly computed.")

    # Extract gradients of the output w.r.t the input
    gradients = torch.autograd.grad(outputs=target, inputs=image_tensor, grad_outputs=torch.ones_like(target))[0]

    if gradients is None:
        raise ValueError("Gradients are None. Ensure gradients are computed correctly.")

    # Global average pooling
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Apply gradients to activations
    for i in range(activations.size(1)):
        activations[:, i, :, :] *= pooled_gradients[i]

    # Create heatmap
    heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = Image.fromarray(np.uint8(heatmap * 255)).resize((224, 224))

    return heatmap


def plot_grad_cam(image_path, cam, output_path):
    img = Image.open(image_path).convert('L')
    plt.figure(figsize=(8, 8))
    plt.imshow(img, alpha=0.5, cmap='gray')
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def process_fn_images(fp_fn_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for img_name in os.listdir(fp_fn_dir):
        if img_name.endswith('.png'):
            img_path = os.path.join(fp_fn_dir, img_name)
            img = Image.open(img_path).convert('L')
            img_tensor = preprocess(img)

            target_class = 1
            heatmap = generate_grad_cam(model, img_tensor, target_class)

            output_path = os.path.join(output_dir, f"cam_{img_name}")
            plot_grad_cam(img_path, heatmap, output_path)


# Process images
false_negative_dir = '/mnt/Data4/Summer2024/RNarasimha/ALLdata+csv/TrainingDataOG/Training/DCIS+/'
output_fn_dir = '/home/ragini/Desktop/DCIS+'

process_fn_images(false_negative_dir, output_fn_dir)
