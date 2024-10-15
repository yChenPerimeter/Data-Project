'''Custom Loss Function: Replaced with a combination of MSE/NLL/BCE and Kullback-Leibler divergence.
Gradient Clipping: Implemented to prevent exploding gradients.
Dropout Layers: Added in the encoder for regularization.
Learning Rate Scheduler: Implemented ReduceLROnPlateau to adjust the learning rate based on the validation loss.
Early Stopping: Added logic to stop training when validation loss doesn't improve for a specified number of epochs.
WandB Logging: Enhanced to include original and sharpened reconstructed images.
Also implemented is the resnet VAE nd its hyper parameter tuning'''



#libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import numpy as np
from PIL import Image

# Initialize wandb
wandb.init(project="ImgGen")

# Calculate the flattened dimension after the convolutional layers
def calculate_flattened_dim(image_size, layers):
    x = torch.zeros(1, 1, *image_size)  # one channel data grayscale
    for layer in layers:
        x = layer(x)
    return int(np.prod(x.shape[1:]))

#VAE convolution model  definition
# Strong encoder written by Ragini
# class Encoder(nn.Module):
#     def __init__(self, latent_dim):
#         super(Encoder, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.pool3 = nn.MaxPool2d(2, 2)
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.pool4 = nn.MaxPool2d(2, 2)
#         self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
#         self.pool5 = nn.MaxPool2d(2, 2)
#         self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
#
#         image_size = (564, 411)
#         conv_layers = [
#             self.conv1, self.pool1, self.conv2, self.pool2,
#             self.conv3, self.pool3, self.conv4, self.pool4,
#             self.conv5, self.pool5, self.conv6,
#         ]
#         flattened_dim = calculate_flattened_dim(image_size, conv_layers)
#
#         self.fc_mu = nn.Linear(flattened_dim, latent_dim)
#         self.fc_logvar = nn.Linear(flattened_dim, latent_dim)
#         self.dropout = nn.Dropout(p=0.3)  # Dropout layer for regularization
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = self.dropout(x)  # Apply dropout
#         x = F.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = self.dropout(x)  # Apply dropout
#         x = F.relu(self.conv3(x))
#         x = self.pool3(x)
#         x = self.dropout(x)  # Apply dropout
#         x = F.relu(self.conv4(x))
#         x = self.pool4(x)
#         x = self.dropout(x)  # Apply dropout
#         x = F.relu(self.conv5(x))
#         x = self.pool5(x)
#         x = self.dropout(x)  # Apply dropout
#         x = F.relu(self.conv6(x))
#         x = torch.flatten(x, start_dim=1)
#         mu = self.fc_mu(x)
#         logvar = self.fc_logvar(x)
#         return mu, logvar
#
#
# # Strong decoder written by Ragini; works for large high-res images
# class Decoder(nn.Module):
#     def __init__(self, latent_dim):
#         super(Decoder, self).__init__()
#         self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
#         self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
#         self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
#         self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
#         self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
#         self.deconv5 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
#
#         self.output_size = (564, 411)
#
#     def forward(self, z):
#         x = F.relu(self.fc(z))
#         x = x.view(-1, 512, 8, 8)
#         x = F.relu(self.deconv1(x))
#         x = F.relu(self.deconv2(x))
#         x = F.relu(self.deconv3(x))
#         x = F.relu(self.deconv4(x))
#         x = torch.sigmoid(self.deconv5(x))
#         return F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
#
# # Variational Autoencoder Definition
# class VAE(nn.Module):
#     def __init__(self, latent_dim):
#         super(VAE, self).__init__()
#         self.encoder = Encoder(latent_dim)
#         self.decoder = Decoder(latent_dim)
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def forward(self, x):
#         mu, logvar = self.encoder(x)
#         z = self.reparameterize(mu, logvar)
#         return self.decoder(z), mu, logvar

#Dataset definition
#Our dataset is split into images and the labels.txt files, writing this class to combine them to be called into the code VAE
#VAE resnet model definition

# Define the ResNet residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


# Encoder using ResNet-like architecture
class ResNetEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(ResNetEncoder, self).__init__()
        self.initial_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual Blocks (Increase the number of channels gradually)
        self.layer1 = self._make_layer(64, 128, stride=2)
        self.layer2 = self._make_layer(128, 256, stride=2)
        self.layer3 = self._make_layer(256, 512, stride=2)
        self.layer4 = self._make_layer(512, 1024, stride=2)

        # Calculate flattened size after convolution layers
        image_size = (564, 411)
        conv_layers = [self.initial_conv, self.layer1, self.layer2, self.layer3, self.layer4]
        flattened_dim = calculate_flattened_dim(image_size, conv_layers)

        # Latent space
        self.fc_mu = nn.Linear(flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(flattened_dim, latent_dim)

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.initial_conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# Decoder with ConvTranspose layers
class ResNetDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(ResNetDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 1024 * 8 * 8)

        # Deconvolution layers to upsample the image
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

        self.output_size = (564, 411)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 1024, 8, 8)  # Reshape to start the deconvolution process
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = torch.sigmoid(self.deconv5(x))
        return F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)


# Define the ResNet VAE
class ResNetVAE(nn.Module):
    def __init__(self, latent_dim):
        super(ResNetVAE, self).__init__()
        self.encoder = ResNetEncoder(latent_dim)
        self.decoder = ResNetDecoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

#Data loader
class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]
        self.image_labels = {f: self._load_label(f) for f in self.image_files}

    def _load_label(self, filename):
        label_file = os.path.join(self.label_dir, filename.replace('.png', '.txt'))
        if os.path.exists(label_file):
            with open(label_file, 'r') as file:
                return file.read().strip()
        return 'unknown'

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('L') #grayscale
        label = self.image_labels[self.image_files[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

#All the functionalities
# Sharpening function to sharpen the genertaed images coming out the decoder before they can go into wandb
#def sharpen_image(image):
    # sharpening kernel
   # kernel = torch.tensor([[0, -1, 0],
                          # [-1, 0, -1],
                        #   [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    #kernel = kernel.to(image.device)
    #sharpened_image: Tensor = F.conv2d(image, kernel, padding=1)
    #return sharpened_image

# Trying various loss functions

#MSE
#def loss_function(recon_x, x, mu, logvar):
   # MSE = F.mse_loss(recon_x.view(-1, 564 * 411), x.view(-1, 564 * 411), reduction='sum')
   # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
   # return MSE + KLD

#Total variation loss function
# def loss_function(recon_x, x, mu, logvar):
#     def total_variation_loss(x):
#         tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
#         tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
#         return tv_h + tv_w
#
#     # Reconstruction loss (Binary Cross-Entropy)
#     reconstruction_loss = F.binary_cross_entropy(
#         recon_x.view(-1, 564 * 411), x.view(-1, 564 * 411), reduction='sum'
#     )
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     tv_loss = total_variation_loss(recon_x)
#     return reconstruction_loss + KLD + tv_loss


#NLL
#def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (MSE loss or BCE loss)
    #NLL = F.mse_loss(recon_x.view(-1, 564 * 411), x.view(-1, 564 * 411), reduction='sum')
    # Kullback-Leibler Divergence loss
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #return NLL + KLD

#BCE+KLD
# def loss_function(recon_x, x, mu, logvar):
#     BCE = F.mse_loss(recon_x, x, reduction='sum')
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return BCE + KLD

#Resnet specific loss function
def kl_divergence_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def vae_loss_function(reconstructed, input_image, mu, logvar):
    # Reconstruction loss (choose BCE or MSE)
    reconstruction_loss = F.binary_cross_entropy(reconstructed, input_image, reduction='sum')
    kld_loss = kl_divergence_loss(mu, logvar)
    total_loss = reconstruction_loss + kld_loss
    return total_loss, reconstruction_loss, kld_loss


#Validation
def validate(model, val_loader, device):
    model.eval()
    running_val_total_loss = 0.0
    running_val_recon_loss = 0.0
    running_val_kld_loss = 0.0

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(val_loader):
            data = data.to(device)

            recon_batch, mu, logvar = model(data)

            # Unpack the loss function output
            val_total_loss, val_recon_loss, val_kld_loss = vae_loss_function(recon_batch, data, mu, logvar)

            # Accumulate validation losses
            running_val_total_loss += val_total_loss.item()
            running_val_recon_loss += val_recon_loss.item()
            running_val_kld_loss += val_kld_loss.item()

    avg_val_total_loss = running_val_total_loss / len(val_loader.dataset)
    avg_val_recon_loss = running_val_recon_loss / len(val_loader.dataset)
    avg_val_kld_loss = running_val_kld_loss / len(val_loader.dataset)

    print(f"Validation: Total Loss: {avg_val_total_loss:.4f}, Recon Loss: {avg_val_recon_loss:.4f}, KLD Loss: {avg_val_kld_loss:.4f}")

    return avg_val_total_loss, avg_val_recon_loss, avg_val_kld_loss


def train(model, train_loader, optimizer, device, epoch, scheduler, epochs):
    model.train()
    running_total_loss = 0.0
    running_recon_loss = 0.0
    running_kld_loss = 0.0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)

        # Unpack the three outputs from the loss function
        total_loss, recon_loss, kld_loss = vae_loss_function(recon_batch, data, mu, logvar)

        # Backpropagation
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate losses for tracking
        running_total_loss += total_loss.item()
        running_recon_loss += recon_loss.item()
        running_kld_loss += kld_loss.item()

        # Log images to WandB every 20 batches
        if batch_idx % 20 == 0:
            indices = np.random.choice(data.size(0), size=min(3, data.size(0)), replace=False)
            wandb.log({
                "original": [wandb.Image(data[idx].cpu().numpy().transpose(1, 2, 0), caption=f"Original_{idx}") for idx
                             in indices],
                "reconstructed": [wandb.Image(recon_batch[idx].detach().cpu().numpy().transpose(1, 2, 0),
                                              caption=f"Reconstructed_{idx}") for idx in indices],
            })

    # Calculate average losses
    avg_total_loss = running_total_loss / len(train_loader.dataset)
    avg_recon_loss = running_recon_loss / len(train_loader.dataset)
    avg_kld_loss = running_kld_loss / len(train_loader.dataset)

    print(
        f"Training: Epoch {epoch}/{epochs}, Total Loss: {avg_total_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, KLD Loss: {avg_kld_loss:.4f}")

    # Step scheduler with the average total loss
    scheduler.step(avg_total_loss)

    return avg_total_loss, avg_recon_loss, avg_kld_loss


# Main function (continued)
# Main function (continued)
def main():
    # Hyperparameters
    latent_dim = 64
    batch_size = 32
    epochs = 8000
    learning_rate = 1e-4
    early_stopping_patience = 4000

    # Directories for train and validation data
    train_data_dir = '/mnt/Data4/Summer2024/RNarasimha/ALLdata+csv/aspect_ratio_new_yanir/Split_3_classes/datasets/train/images/'
    train_label_dir = '/mnt/Data4/Summer2024/RNarasimha/ALLdata+csv/aspect_ratio_new_yanir/Split_3_classes/datasets/train/labels/'
    val_data_dir = '/mnt/Data4/Summer2024/RNarasimha/ALLdata+csv/aspect_ratio_new_yanir/Split_3_classes/datasets/validation/images/'
    val_label_dir = '/mnt/Data4/Summer2024/RNarasimha/ALLdata+csv/aspect_ratio_new_yanir/Split_3_classes/datasets/validation/labels/'
    output_dir = '/mnt/Data4/Summer2024/RNarasimha/All_Model_Outputs/model_output_5_9_24_VAE/Resnet_VAE_8000_ie-4_64_oct9_BCE_KLD/'

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetVAE(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define transforms for the dataset
    transform = transforms.Compose([
        transforms.Resize((564, 411)),
        transforms.ToTensor(),
    ])

    train_dataset = CustomDataset(train_data_dir, train_label_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = CustomDataset(val_data_dir, val_label_dir, transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    # Initialize min_loss and epochs_without_improvement
    min_loss = float('inf')
    epochs_without_improvement = 0

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, device, epoch, scheduler, epochs)
        val_total_loss, val_recon_loss, val_kld_loss = validate(model, val_loader, device)  # Unpack validation losses

        # Log training and validation losses to WandB
        wandb.log({
            "train_loss": train_loss,
            "val_total_loss": val_total_loss,
            "val_recon_loss": val_recon_loss,
            "val_kld_loss": val_kld_loss,
            "epoch": epoch,
        })

        # Check for early stopping based on total validation loss
        if val_total_loss < min_loss:
            min_loss = val_total_loss
            epochs_without_improvement = 10
            # Save the model when validation loss improves
            torch.save(model.state_dict(), os.path.join(output_dir, f"vae_epoch_{epoch}.pth"))
        else:
            epochs_without_improvement += 1

        # Stop if validation loss hasn't improved for early_stopping_patience epochs
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print("Training complete.")


if __name__ == "__main__":
    main()
