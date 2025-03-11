import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import os

from config import get_config
config = get_config()

from models.ddpm import Denoiser, Diffusion
from data.mnist_dataloader import get_dataloaders
from torch.utils.data import DataLoader
from typing import Tuple


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    diffusion: Diffusion,
    denoising_loss: nn.Module,
    epochs: int = config["epochs"],
    device: torch.device = config["DEVICE"],
    save_epochs: list = config["save_epochs"]
) -> None:
    """
    Train the denoising diffusion probabilistic model (DDPM).

    Args:
        model (nn.Module): The denoising model to be trained.
        dataloader (DataLoader): DataLoader providing training batches.
        optimizer (optim.Optimizer): Optimizer for model training.
        diffusion (Diffusion): Diffusion process model.
        denoising_loss (nn.Module): Loss function for denoising.
        epochs (int, optional): Number of training epochs. Defaults to config.epochs.
        device (torch.device, optional): Device to use for training. Defaults to config.DEVICE.
        save_epochs (list, optional): List of epochs at which to save the model.
    """
    print("Start training DDPMs...")
    model.train()

    os.makedirs("checkpoints", exist_ok=True)  
    
    for epoch in range(epochs + 1):  
        noise_prediction_loss = 0
        
        if epoch in save_epochs:
            checkpoint_path = f"checkpoints/ddpm_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
        
        if epoch == 0:  
            continue  
        for batch_idx, (x, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
            optimizer.zero_grad()
            x = x.to(device)
            
            noisy_input, epsilon, pred_epsilon = diffusion(x)
            loss = denoising_loss(pred_epsilon, epsilon)
            
            noise_prediction_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        print(f"\tEpoch {epoch} complete! Denoising Loss: {noise_prediction_loss / batch_idx:.6f}")

    print("Training finished!")
    torch.save(model.state_dict(), "checkpoints/ddpm.pth")
    print("Model saved at checkpoints/ddpm.pth")
    


def get_perturbed_images(
    model: nn.Module,
    diffusion: Diffusion,
    dataloader: DataLoader,
    device: torch.device
) -> torch.Tensor:
    """
    Generate a batch of perturbed images using the diffusion process.

    Args:
        model (nn.Module): The trained model.
        diffusion (Diffusion): The diffusion process model.
        dataloader (DataLoader): DataLoader providing test data.
        device (torch.device): Device to perform computation.

    Returns:
        torch.Tensor: A batch of perturbed images.
    """
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(device)
            perturbed_images, _, _ = diffusion(x)
            return diffusion.reverse_scale_to_zero_to_one(perturbed_images)
    return torch.empty(0)


def generate_image(
    model: nn.Module, diffusion: Diffusion, device: torch.device
) -> torch.Tensor:
    """
    Generate new images using the trained diffusion model.

    Args:
        model (nn.Module): The trained model.
        diffusion (Diffusion): The diffusion process model.
        device (torch.device): Device to perform computation.

    Returns:
        torch.Tensor: A batch of generated images.
    """
    model.eval()
    with torch.no_grad():
        return diffusion.sample(N=config["inference_batch_size"])


def draw_sample_image(x: torch.Tensor, postfix: str, save_dir: str = "generated_images") -> None:
    """
    Save and display a sample grid of images.

    Args:
        x (torch.Tensor): Batch of images to display.
        postfix (str): Filename identifier.
        save_dir (str, optional): Directory to save images. Defaults to "generated_images".
    """
    os.makedirs(save_dir, exist_ok=True)

    x = x[:64]

    grid_img = make_grid(x.detach().cpu(), nrow=8, padding=2, normalize=True)

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(f"Visualization of {postfix}")
    plt.imshow(np.transpose(grid_img, (1, 2, 0)))

    save_path = os.path.join(save_dir, f"{postfix}.png")
    plt.savefig(save_path)
    print(f"Image saved at {save_path}")
    plt.show()

if __name__ == "__main__":    
    
    os.makedirs("checkpoints", exist_ok=True)
    
    dataloader, test_dataloader = get_dataloaders(
        config["paths"]["train_data"],
        config["paths"]["test_data"]
    )
    print("Dataloader loaded successfully!")
    
    model = Denoiser(
        image_resolution=config["img_size"],
        hidden_dims=config["hidden_dims"],
        diffusion_time_embedding_dim=config["timestep_embedding_dim"],
        n_times=config["n_timesteps"]
    ).to(config["DEVICE"])
    
    diffusion = Diffusion(
        model,
        image_resolution=config["img_size"],
        n_times=config["n_timesteps"],
        use_cosine_schedule=True,
        device=config["DEVICE"]
    ).to(config["DEVICE"])
    
    denoising_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    
    train_model(model, dataloader, optimizer, diffusion, denoising_loss, epochs=config["epochs"], save_epochs=config["save_epochs"])

    
    print("Generating perturbed images...")
    perturbed_images = get_perturbed_images(model, diffusion, test_dataloader, config["DEVICE"])
    draw_sample_image(perturbed_images, "Perturbed Images")
    
    print("Generating model images...")
    generated_images = generate_image(model, diffusion, config["DEVICE"])
    draw_sample_image(generated_images, "Generated Images")
