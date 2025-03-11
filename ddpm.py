import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
import math

class SinusoidalPosEmb(nn.Module):
    """
    Computes sinusoidal positional embeddings.
    Args:
        dim (int): Dimension of the embedding.
    Returns:
        Tensor: Sinusoidal embeddings.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ConvBlock(nn.Conv2d):
    """
    Convolutional Block with optional activation and normalization.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        activation_fn (bool): Whether to use activation function.
        drop_rate (float): Dropout rate.
        stride (int): Stride of the convolution.
        padding (str/int): Padding type or size.
        dilation (int): Dilation rate.
        groups (int): Number of groups in group convolution.
        bias (bool): Whether to use bias.
        gn (bool): Whether to use group normalization.
        gn_groups (int): Number of groups for group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, activation_fn=None, drop_rate=0.,
                    stride=1, padding='same', dilation=1, groups=1, bias=True, gn=False, gn_groups=8):
        
        if padding == 'same':
            padding = kernel_size // 2 * dilation
        
        super(ConvBlock, self).__init__(in_channels, out_channels, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation,
                                            groups=groups, bias=bias)

        self.activation_fn = nn.SiLU() if activation_fn else None
        self.group_norm = nn.GroupNorm(gn_groups, out_channels) if gn else None
        
    def forward(self, x, time_embedding=None, residual=False):
        """
        Forward pass for the convolutional block.
        Args:
            x (Tensor): Input tensor.
            time_embedding (Tensor, optional): Positional embedding for residual connection.
            residual (bool): Whether to use residual connection.
        Returns:
            Tensor: Output after convolution, normalization, and activation.
        """
        if residual:
            x = x + time_embedding  # Add time embedding for residual blocks
            y = x
            x = super(ConvBlock, self).forward(x)
            y = y + x  # Residual connection
        else:
            y = super(ConvBlock, self).forward(x)
        
        y = self.group_norm(y) if self.group_norm is not None else y
        y = self.activation_fn(y) if self.activation_fn is not None else y
        
        return y

class Denoiser(nn.Module):
    """
    Simplified U-Net model for denoising.
    Args:
        image_resolution (tuple): Image resolution (height, width, channels).
        hidden_dims (list): List of hidden dimensions.
        diffusion_time_embedding_dim (int): Dimension of time embedding.
        n_times (int): Number of diffusion steps.
    """
    def __init__(self, image_resolution, hidden_dims=[256, 256], diffusion_time_embedding_dim = 256, n_times=1000):
        super(Denoiser, self).__init__()
        
        _, _, img_C = image_resolution
        
        self.time_embedding = SinusoidalPosEmb(diffusion_time_embedding_dim) 
        self.in_project = ConvBlock(img_C, hidden_dims[0], kernel_size=7) 
        
        self.time_project = nn.Sequential( 
            ConvBlock(diffusion_time_embedding_dim, hidden_dims[0], kernel_size=1, activation_fn=True),
            ConvBlock(hidden_dims[0], hidden_dims[0], kernel_size=1))
        
        self.convs = nn.ModuleList([ConvBlock(in_channels=hidden_dims[0], out_channels=hidden_dims[0], kernel_size=3)]) 
        
        for idx in range(1, len(hidden_dims)):
            self.convs.append(ConvBlock(hidden_dims[idx-1], hidden_dims[idx], kernel_size=3, dilation=3**((idx-1)//2),
                                                    activation_fn=True, gn=True, gn_groups=8))                                
                                
        self.out_project = ConvBlock(hidden_dims[-1], out_channels=img_C, kernel_size=3) 
        
    def forward(self, perturbed_x, diffusion_timestep): 
        """
        Forward pass of the denoiser.
        Args:
            perturbed_x (Tensor): Noisy image.
            diffusion_timestep (Tensor): Diffusion time step.
        Returns:
            Tensor: Denoised image.
        """
        y = perturbed_x
        diffusion_embedding = self.time_embedding(diffusion_timestep)
        diffusion_embedding = self.time_project(diffusion_embedding.unsqueeze(-1).unsqueeze(-2))
        y = self.in_project(y)
        
        for i in range(len(self.convs)):
            y = self.convs[i](y, diffusion_embedding, residual = True)
        
        y = self.out_project(y) 
        return y 

class Diffusion(nn.Module):
    """
    Diffusion Model for noise perturbation and denoising.
    Args:
        model (nn.Module): Denoiser model.
        image_resolution (list): Image resolution [height, width, channels].
        n_times (int): Number of diffusion steps.
        use_cosine_schedule (bool): Whether to use a cosine noise schedule.
        device (str): Device ('cuda' or 'cpu').
    """
    def __init__(self, model, image_resolution=[32, 32, 3], n_times=1000, use_cosine_schedule=True, device='cuda'):
        super(Diffusion, self).__init__()
        
        self.n_times = n_times
        self.img_H, self.img_W, self.img_C = image_resolution
        self.model = model
        self.device = device
        
        if use_cosine_schedule:
            betas = self.cosine_schedule(n_times).to(device)
        else:
            beta_1, beta_T = 1e-4, 2e-2  
            betas = torch.linspace(start=beta_1, end=beta_T, steps=n_times).to(device)  
        
        self.sqrt_betas = torch.sqrt(betas)
        self.alphas = 1 - betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)

    def cosine_schedule(self, timesteps, s=0.008):
        """
        Defines a cosine-based scheduler.
        Returns:
            Tensor: Beta schedule.
        """
        t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float32) / timesteps
        f_t = torch.cos((t + s) / (1 + s) * (math.pi / 2)) ** 2
        alphas_cumprod = f_t / f_t[0]  
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)  
    
    def extract(self, a, t, x_shape): 
        """
            from lucidrains' implementation
                https://github.com/lucidrains/denoising-diffusion-pytorch/blob/beb2f2d8dd9b4f2bd5be4719f37082fe061ee450/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L376
        """
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    # Normalization
    def scale_to_minus_one_to_one(self, x): 
        """Scales input from [0,1] range to [-1,1] range."""
        return x * 2 - 1
    
    def reverse_scale_to_zero_to_one(self, x): 
        """Scales input from [-1,1] range back to [0,1] range."""
        return (x + 1) * 0.5

    # Perturbation
    def make_noisy(self, x_zeros, t): 
        """Adds noise to the input image based on the diffusion step."""
        epsilon = torch.randn_like(x_zeros).to(self.device)
        
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t, x_zeros.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t, x_zeros.shape)
        
        noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar
    
        return noisy_sample.detach(), epsilon 
    
    # FORWARD PASS
    def forward(self, x_zeros):
        """Executes the forward diffusion process."""
        x_zeros = self.scale_to_minus_one_to_one(x_zeros)
        
        B, _, _, _ = x_zeros.shape
        
        # (1) randomly choose diffusion time-step
        t = torch.randint(low=0, high=self.n_times, size=(B,)).long().to(self.device)
        
        # (2) forward diffusion process: perturb x_zeros with fixed variance schedule
        perturbed_images, epsilon = self.make_noisy(x_zeros, t) 
        
        # (3) predict epsilon(noise) given perturbed data at diffusion-timestep t.
        pred_epsilon = self.model(perturbed_images, t)
        
        return perturbed_images, epsilon, pred_epsilon
    
    # Denoising
    def denoise_at_t(self, x_t, timestep, t):   
        """Performs denoising at a given timestep."""
        B, _, _, _ = x_t.shape
        if t > 1: 
            z = torch.randn_like(x_t).to(self.device)
        else:
            z = torch.zeros_like(x_t).to(self.device)
        
        # at inference, we use predicted noise (epsilon) to restore perturbed data sample.
        epsilon_pred = self.model(x_t, timestep)
        
        alpha = self.extract(self.alphas, timestep, x_t.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas, timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, x_t.shape)
        sqrt_beta = self.extract(self.sqrt_betas, timestep, x_t.shape)
        
        # denoise at time t, utilizing predicted noise
        x_t_minus_1 = 1 / sqrt_alpha * (x_t - (1-alpha)/sqrt_one_minus_alpha_bar*epsilon_pred) + sqrt_beta*z
        
        return x_t_minus_1.clamp(-1., 1)

    # Sampling
    def sample(self, N): 
        """Generates samples from the noise distribution.""" 
        # start from random noise vector, x_0 (for simplicity, x_T declared as x_t instead of x_T)
        x_t = torch.randn((N, self.img_C, self.img_H, self.img_W)).to(self.device)
        
        # autoregressively denoise from x_T to x_0 i.e., generate image from noise, x_T
        for t in range(self.n_times-1, -1, -1):
            timestep = torch.tensor([t]).repeat_interleave(N, dim=0).long().to(self.device)
            x_t = self.denoise_at_t(x_t, timestep, t)
        
        # denormalize x_0 into 0 ~ 1 ranged values.
        x_0 = self.reverse_scale_to_zero_to_one(x_t)
        
        return x_0