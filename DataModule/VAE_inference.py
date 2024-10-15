import torch
import os
import torch.nn.functional as F
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from torchvision import transforms

# Define the Decoder class (as in your VAE model)
class Decoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = torch.nn.Linear(latent_dim, 512 * 8 * 8)
        self.deconv1 = torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv5 = torch.nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 512, 8, 8)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = torch.sigmoid(self.deconv5(x))
        # Resize output to 564x411
        return F.interpolate(x, size=(411, 564), mode='bilinear', align_corners=False)

# Count images for each class based on the first number in labels.txt
def count_images_by_class(label_dir):
    counts = {'DCIS-': 0, 'DCIS+': 0, 'IDC': 0}

    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            label_path = os.path.join(label_dir, label_file)
            with open(label_path, 'r') as file:
                for line in file:
                    label = int(line.split()[0])  # Read the first number in each line
                    if label == 1:
                        counts['DCIS-'] += 1
                    elif label == 2:
                        counts['DCIS+'] += 1
                    elif label == 3:
                        counts['IDC'] += 1

    return counts['DCIS-'], counts['DCIS+'], counts['IDC']

# Function to sharpen an image
def sharpen_image(image, factor=0.0):
    pil_image = Image.fromarray((image * 255).astype('uint8'), mode='L')  # Convert to PIL Image
    enhancer = ImageEnhance.Sharpness(pil_image)
    sharpened_image = enhancer.enhance(factor)
    return sharpened_image

# Function to generate images and save them
def generate_images(model, latent_dim, num_images, output_dir, class_name, device):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_images):
        z = torch.randn(1, latent_dim).to(device)
        with torch.no_grad():
            generated_img = model(z)

        img = generated_img.squeeze().cpu().numpy()

        # Sharpen the image before saving
        sharpened_img = sharpen_image(img)

        # Save the sharpened image
        img_path = os.path.join(output_dir, f'{class_name}_generated_{i}.png')
        sharpened_img.save(img_path)

# Load the trained model and perform inference
def run_inference():
    latent_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    decoder = Decoder(latent_dim).to(device)

    # Load the model state dictionary
    state_dict = torch.load(
        '/mnt/Data4/Summer2024/RNarasimha/All_Model_Outputs/model_output_5_9_24_VAE/aspect_Yanir_3/vae_model.pth',
        map_location=device)

    # Filter the decoder weights
    decoder_state_dict = {k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')}

    # Load the decoder state_dict
    decoder.load_state_dict(decoder_state_dict)

    decoder.eval()

    # Define paths
    label_dir = '/mnt/Data4/Summer2024/RNarasimha/ALLdata+csv/aspect_ratio_new_yanir/Split_3_classes/datasets/train/labels/'
    output_dir_base = '/mnt/Data4/Summer2024/RNarasimha/All_Model_Outputs/model_output_5_9_24_VAE/aspect_Yanir_3/Gen_3/'

    # Count the number of images for each class
    num_images_DCI_minus, num_images_DCI_plus, num_images_IDC = count_images_by_class(label_dir)

    # Define output directories for each class
    output_dir_DCI_minus = os.path.join(output_dir_base, 'DCIS_minus')
    output_dir_DCI_plus = os.path.join(output_dir_base, 'DCIS_plus')
    output_dir_IDC = os.path.join(output_dir_base, 'IDC')

    # Generate and save the images
    generate_images(decoder, latent_dim, num_images_DCI_minus, output_dir_DCI_minus, 'DCIS_minus', device)
    generate_images(decoder, latent_dim, num_images_DCI_plus, output_dir_DCI_plus, 'DCIS_plus', device)
    generate_images(decoder, latent_dim, num_images_IDC, output_dir_IDC, 'IDC', device)

if __name__ == '__main__':
    run_inference()
