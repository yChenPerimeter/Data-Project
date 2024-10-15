" the VAE model which works very well,but it does not have noise matching"



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
import os
from PIL import Image
import wandb

# Initialize wandb
wandb.init(project="ImgGen")


def calculate_flattened_dim(image_size, conv_layers):
    x = torch.randn(1, 1, image_size[0], image_size[1])
    for layer in conv_layers:
        x = layer(x)
    return x.numel()


# Define the VAE model
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)

        image_size = (564, 411)
        conv_layers = [
            self.conv1, self.pool1, self.conv2, self.pool2,
            self.conv3, self.pool3, self.conv4, self.pool4,
            self.conv5, self.pool5, self.conv6,
        ]
        flattened_dim = calculate_flattened_dim(image_size, conv_layers)

        self.fc_mu = nn.Linear(flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(flattened_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        x = F.relu(self.conv6(x))
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

        self.output_size = (564, 411)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 512, 8, 8)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = torch.sigmoid(self.deconv5(x))
        return F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


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
        image = Image.open(img_name).convert('L')
        label = self.image_labels[self.image_files[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label


def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(model, dataloader, optimizer, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        # Log images to wandb every few batches
        if batch_idx % 40 == 0:
            wandb.log({
                "reconstructed": [wandb.Image(img.cpu()) for img in recon_batch[:3]],
            })

    wandb.log({"train_loss": train_loss / len(dataloader.dataset)})
    return train_loss / len(dataloader.dataset)


def validate(model, dataloader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            val_loss += loss.item()

    wandb.log({"val_loss": val_loss / len(dataloader.dataset)})
    return val_loss / len(dataloader.dataset)


def main():
    #hyperparamater tuning
    latent_dim = 128 #64 #128 #256
    batch_size = 32 #16 #64
    epochs = 8000 #32 #64 #128 #256 #512 #1024 #2000 #3000 #4000 #5000
    learning_rate = 5e-4 #1e-3 #1e-4
    #early_stopping_patience = 50
    #step_counter = 0
    #min_loss = float('inf')
    #epochs_without_improvement = 0

    # Directories for train and validation data
    train_data_dir = '/mnt/Data4/Summer2024/RNarasimha/ALLdata+csv/aspect_ratio_new_yanir/Split_3_classes/datasets/train/images/'
    train_label_dir = '/mnt/Data4/Summer2024/RNarasimha/ALLdata+csv/aspect_ratio_new_yanir/Split_3_classes/datasets/train/labels/'

    val_data_dir = '/mnt/Data4/Summer2024/RNarasimha/ALLdata+csv/aspect_ratio_new_yanir/Split_3_classes/datasets/validation/images/'
    val_label_dir = '/mnt/Data4/Summer2024/RNarasimha/ALLdata+csv/aspect_ratio_new_yanir/Split_3_classes/datasets/validation/labels/'

    output_dir = '/mnt/Data4/Summer2024/RNarasimha/All_Model_Outputs/model_output_5_9_24_VAE/VAE_8000_ie-3_64_oct5/'
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(latent_dim).to(device)
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

    # Training loop
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save the model
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f'vae_epoch_{epoch + 1}.pth'))

        # Early stopping
        #if val_loss < min_loss:
           # min_loss = val_loss
           # epochs_without_improvement = 0
      #  else:
           # epochs_without_improvement += 1

       # if epochs_without_improvement >= early_stopping_patience:
          #  print(f"Early stopping at epoch {epoch + 1}")
          #  break

    wandb.finish()


if __name__ == '__main__':
    main()
