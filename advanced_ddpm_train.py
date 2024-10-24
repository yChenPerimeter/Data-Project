# Advanced DDPM Training and Evaluation Script with Configurable Checkpointing
# Author: ychen
# Date: 2024-10-21
# Description: This script is designed for training a Denoising Diffusion Probabilistic Model (DDPM) using Ray Tune for hyperparameter tuning.
# It uses Unet as the backbone and logs experiment data with WandB. The script includes checkpointing, sampling images for visualization,
# and evaluating the best model based on the latest epoch for image generation purposes.

import os
import torch
import wandb
import tempfile
from PIL import Image
import numpy as np
import sys
import multiprocessing

from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image, make_grid

# import ray
from ray import train
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.air import session
from ray.train import RunConfig
from ray.tune import TuneConfig, Tuner
from ray.tune import ResultGrid

from denoising_diffusion_pytorch import Unet, GaussianDiffusion



#global environment setting else wont save the result to your desired path
storage_path = '/home/ychen/Documents/project/Data-Project/results/DDPM'
os.environ['TUNE_RESULT_DIR'] = storage_path

#You can change checkpoints saving  temp dir, default is /tmp/ray
# os.environ["RAY_TMPDIR"] = "/tmp/my_tmp_dir"
# ray.init(_temp_dir="/tmp/my_tmp_dir")

# Define the dataset class
class SingleClassImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Define the training function with Ray Tune and WandB logging, trainable
def train_ddpm(config, checkpoint_dir=None, user_config=None):
    # Initialize WandB with project name 'DDPM Trail' and add a label
    wandb_run = wandb.init(
        project="DDPM Trail", 
        name="DDPM trail Experiment",
        config=config
    ) if not wandb.run else wandb.run
    wandb_run._label(repo='DDPM')
    
    image_dir = '/home/ychen/Documents/project/Data-Project/datasets/0928_suspecious_equall_split/trainB'
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    dataset = SingleClassImageDataset(image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Model and Diffusion Setup
    model = Unet(dim=64, dim_mults=(1, 2, 4, 8), flash_attn=True).to(config["device"])
    diffusion = GaussianDiffusion(model, image_size=128, timesteps=config["timesteps"]).to(config["device"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Load checkpoint if available
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pth"))
            start_epoch = checkpoint_dict["epoch"] + 1
            model.load_state_dict(checkpoint_dict["model_state"])
            optimizer.load_state_dict(checkpoint_dict["optimizer_state"])
    else:
        start_epoch = 0

    # best_loss = float('inf')
    # best_model_state = None

    for epoch in range(start_epoch, config["epochs"]):
        total_loss = 0
        for i, images in enumerate(dataloader):
            images = images.to(config["device"])
            optimizer.zero_grad()
            loss = diffusion(images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Log metrics to WandB
            if i % 2 == 0:
                wandb.log({"Epoch": epoch + 1, "Step": i, "Loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"End of Epoch {epoch+1}/{config['epochs']} - Avg Loss: {avg_loss:.4f}")
        wandb.log({"Avg Loss": avg_loss})
        
        # Report the average loss to Ray Tune using session.report , abandoned since session not maintaing
        # session.report({'epoch': epoch + 1, 'avg_loss': avg_loss})

        metrics = {"avg_loss": avg_loss}
        # #Save checkpoint based on user configuration
        if user_config["checkpoint_strategy"] == "interval" and (epoch + 1) % user_config["checkpoint_interval"] == 0:
            with tempfile.TemporaryDirectory() as tempdir:
                
                torch.save(
                    {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()},
                    os.path.join(tempdir, "checkpoint.pth")
                )
                print("train ddpm checkpoint checker: tempdir is : ",tempdir)
                
                #TODO lead to significant CPU usage
                #use train.report() to report the checkpoint back to Ray Tune. and auto send to WANDB
                #That’s why you see the checkpoint being saved in /home/ychen/Documents/project/Data-Project/results/DDPM/DDPM_exps/....
                train.report(metrics, checkpoint=train.Checkpoint.from_directory(tempdir))
                #print(f"Model checkpoint saved at epoch {epoch+1}")

    # #Save the best model for each trail based on loss
    # if user_config["checkpoint_strategy"] == "best" and avg_loss < best_loss:
    #     best_loss = avg_loss
    #     best_model_state = {
    #         "epoch": epoch,
    #         "model_state": model.state_dict(),
    #         "optimizer_state": optimizer.state_dict()
    #     }

        # Sample images and log to WandB based on user configuration
        if (epoch + 1) % user_config["wandb_image_log_interval"] == 0:
            
            sampled_images = diffusion.sample(batch_size=4)  # Assuming this returns a batch of images (4, 3, H, W)
            # Check the shape of the sampled images to ensure it is correct
            print("Shape of sampled images:", sampled_images.shape)

            # Iterate through each image and log it individually to WandB
            for idx, img in enumerate(sampled_images):
                # Use wandb.Image() to log each image separately
                #TODO not logging to server there not sure why
                wandb.log({f"Sampled Image {idx+1} (Epoch {epoch+1})": wandb.Image(img, caption=f"Epoch {epoch+1}, Image {idx+1}")})
                
                save_image(img, os.path.join("/home/ychen/Documents/project/Data-Project/results/DDPM/dumy_wandb", f"inference_image_{idx + 1}.png"))

            print("Logged individual images")
            
    # #Save the best model per trail if applicable
    # if user_config["checkpoint_strategy"] == "best" and best_model_state is not None:
    #     with tempfile.TemporaryDirectory() as tempdir:
    #         torch.save(best_model_state, os.path.join(tempdir, "checkpoint.pth"))
    #         train.report(metrics={"epoch": best_model_state["epoch"] + 1}, checkpoint=train.Checkpoint.from_directory(tempdir))
    #         print(f"Best model checkpoint saved from epoch {best_model_state['epoch'] + 1}")

# Main function to run Ray Tune experiments
def main(num_samples=2, max_num_epochs=2, gpus_per_trial=2):
    params_config = {
        "epochs": max_num_epochs,
        "batch_size": tune.choice([24]), # set it to 16 for now, as we have two gpu now. but 32 is too large...
        "lr": tune.grid_search([1e-4, 1e-3]),
        "timesteps": tune.choice([1000]),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    user_config = {
        "checkpoint_strategy": "interval",  # Configurable checkpoint strategy
        "checkpoint_interval": 5,  # Configurable checkpoint interval
        "wandb_image_log_interval": 5,  # Configurable image log interval
        "exp_name": "DDPM_exps"  # Experiment name
    }
    
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
        metric='avg_loss',
        mode='min'
    )
    
    wandb_callback = WandbLoggerCallback(
        project="DDPM Training",
        log_config=True,
        reinit=True
    )
    
    # define your trainable function there
    # if more cpu avil can do: multiprocessing.cpu_count()/torch.cuda.device_count()
    trainable_with_resources = tune.with_resources(train_ddpm, {"cpu":multiprocessing.cpu_count()/torch.cuda.device_count(), "gpu": gpus_per_trial}) 
    
    tuner = Tuner(
        tune.with_parameters(trainable_with_resources, user_config=user_config),
        run_config=RunConfig(
            name=user_config["exp_name"],
            storage_path=storage_path,
            callbacks=[wandb_callback],
            #TODO better checkpoint_score_attribute to use, now it kept the latest checkpoint per trail
            #TODO num_to_keep=2, # keep 2 best checkpoints only
            
            # as checkpoint config it wont work with functional API, thus we do it manually
            checkpoint_config = train.CheckpointConfig(
                checkpoint_score_attribute="avg_loss", # what ever metric you want to use during trail
            )
        ),
        tune_config=TuneConfig(
            metric='avg_loss', # we kept same metric for now, but you can change it to other metric use average loss as way to pick best, but this is not optimal
            mode='min',
            num_samples=num_samples,
        ),
        param_space=params_config
    )


    result_grid: ResultGrid = tuner.fit()
    num_results = len(result_grid)
    print("Number of results:", num_results)
    for i, result in enumerate(result_grid):
        if not result.error:
            print(f"Trial {i + 1} finished successfully with metrics: {result.metrics}.")
            if result.checkpoint:
                    print(f"Checkpoint for trial {i + 1}: {result.checkpoint}")
        else:
            print(f"Trial {i + 1} failed with error: {result.error}.")

    # Get the best result based on the 'loss' metric
    best_result = result_grid.get_best_result(metric="avg_loss", mode="min")

    # Get the best checkpoint corresponding to the best result.
    best_checkpoint = best_result.checkpoint 
    if best_checkpoint:
        test_best_model(best_checkpoint)
    else:
        print("No best checkpoint found.")




# Define the function for testing the best model
def test_best_model(best_checkpoint):
    # Model setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Unet(dim=64, dim_mults=(1, 2, 4, 8), flash_attn=True).to(device)
    
    print("Testing the best model")
    with best_checkpoint.as_directory() as tmpdir:
        # Load model from directory        
        checkpoint_path = os.path.join(tmpdir, "checkpoint.pth")  # Get the path to the checkpoint file
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")

        checkpoint_dict = torch.load(checkpoint_path)  # Load the checkpoint file
        model.load_state_dict(checkpoint_dict["model_state"])
        
        # Generate and save 10 inference images
        result_folder = "/home/ychen/Documents/project/Data-Project/results/DDPM/BestModel_Inference"
        os.makedirs(result_folder, exist_ok=True)
        
        with torch.no_grad():
            model.eval()  # Set model to evaluation mode
            diffusion = GaussianDiffusion(model, image_size=256, timesteps=1000).to(device)
            sampled_images = diffusion.sample(batch_size=10)  # Assuming the model has a sample function to generate images
            
            for idx, img in enumerate(sampled_images):
                save_image(img, os.path.join(result_folder, f"inference_image_{idx + 1}.png"))

        print("Saved 10 inference local images for the best model.")


    
 

    # save the best model checkpoint to local directory #TODO change the path to your desired path assoicate with project name or something
    checkpoint_dir = best_checkpoint.to_directory("/home/ychen/Documents/project/Data-Project/checkpoints/ray_DDPM")
    print("local checkpoint_dir is: ",checkpoint_dir)
    




# Run main function
if __name__ == "__main__":
    # num_samples: Number of times to sample from the hyperparameter space. 
    main(num_samples=1, max_num_epochs=300, gpus_per_trial=1)
