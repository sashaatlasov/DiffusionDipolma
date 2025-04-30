from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from celluloid import Camera
import matplotlib.pyplot as plt
import numpy as np

from unet_small import UnetModel


def total_loss_fn(predicted, target, l_sparsity=0.01, l_energy=0.05, l_outer=0.05, 
                  center_x=15, center_y=15, r_cutoff=10):
    """
    predicted: model output, shape [batch, channels, height, width]
    target: ground truth energy deposit
    center_x, center_y: expected center of the shower (can be constant or dynamic)
    """
    diffusion_loss = F.mse_loss(predicted, target)

    sparsity_loss = torch.mean(predicted[predicted < np.log1p(5e-3)])  
    
    _, _, height, width = predicted.shape
    device = predicted.device

    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=device), 
        torch.arange(width, device=device), indexing="ij"
    )
    distance_from_center = torch.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    outer_mask = (distance_from_center > r_cutoff).float()

    outer_energy = (predicted.squeeze(1) * outer_mask).mean()

    predicted_total = predicted.sum(dim=[1,2,3])
    target_total = target.sum(dim=[1,2,3])
    energy_loss = F.l1_loss(predicted_total, target_total)

    total_loss = (diffusion_loss
                  + l_sparsity * sparsity_loss
                  + l_energy * energy_loss
                  + l_outer * outer_energy)

    return total_loss


class DiffusionModel(nn.Module):
    def __init__(
        self,
        num_timesteps: int,
        hidden_size: int,
        schedule: str = 'linear',
        loss: str = 'l2',
        betas: Tuple[float, float] = (1e-4, 0.2),
    ):
        super().__init__()
        self.eps_model = UnetModel(1, 1, hidden_size)

        if schedule == 'linear':
            for name, schedule in get_schedules(betas[0], betas[1], num_timesteps).items():
                self.register_buffer(name, schedule)
        elif schedule == 'cosine':
            for name, schedule in get_cosine_schedules(num_timesteps + 1).items():
                self.register_buffer(name, schedule)
        self.num_timesteps = num_timesteps

        if loss == 'l2':
            self.criterion = nn.MSELoss()
        elif loss == 'l1':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError('loss type is not supported')

    def forward(self, x: torch.Tensor, m: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        device = x.device

        timestep = torch.randint(
            1, self.num_timesteps + 1, (x.shape[0],), device=device)
        eps = torch.randn_like(x, device=device)

        x_t = (
            self.sqrt_alphas_cumprod[timestep, None, None, None] * x
            + self.sqrt_one_minus_alpha_prod[timestep, None, None, None] * eps
        )
        predicted = self.eps_model(x_t, m, p, timestep / self.num_timesteps)
        predicted_clean = (x_t - self.sqrt_one_minus_alpha_prod[timestep, None, None, None] * predicted) / self.sqrt_alphas_cumprod[timestep, None, None, None]
        # return self.criterion(eps, self.eps_model(x_t, m, p, timestep / self.num_timesteps))
        return total_loss_fn(predicted_clean, x)

    def sample(self, m: torch.Tensor, p: torch.Tensor, truncate: float = None) -> torch.Tensor:

        size = (1, 30, 30)
        num_samples = m.shape[0]
        device = m.device

        x_i = torch.randn(num_samples, *size, device=device)
        for i in tqdm(range(self.num_timesteps - 1, 0, -1), leave=False):
            z = torch.randn(num_samples, *size, device=device) if i > 1 else 0 
            eps = self.eps_model(x_i, m, p, torch.tensor(
                i / self.num_timesteps).repeat(num_samples, 1).to(device))
            x_i = self.inv_sqrt_alphas[i] * (
                x_i - eps * self.one_minus_alpha_over_prod[i]) + self.sqrt_betas[i] * z

        if truncate:
            x_i[x_i < np.log1p(truncate)] = 0

        return x_i

    def sample_single(self, m: torch.Tensor, p: torch.Tensor, plot=False, name='animation') -> torch.Tensor:
        device = m.device
        x_i = torch.randn(1, 1, 30, 30, device=device)

        m, p = m.unsqueeze(0), p.unsqueeze(0)
        if plot:
            fig = plt.figure(dpi=250)
            camera = Camera(fig)
            plt.imshow(np.transpose(x_i.squeeze(0).numpy(), (1, 2, 0)), cmap='inferno')
            plt.title(f"Point: {tuple(p[0].numpy())}, Momentum {torch.norm(m) ** 2:.2f}")
            plt.legend(f'Step №{0}', loc='lower left')
            camera.snap()

        for i in tqdm(range(self.num_timesteps, 0, -1), leave=False):
            z = torch.randn(1, 1, 30, 30, device=device) if i > 1 else 0
            eps = self.eps_model(x_i, m, p, torch.tensor(
                i / self.num_timesteps).repeat(1, 1).to(device))
            x_i = self.inv_sqrt_alphas[i] * (
                x_i - eps * self.one_minus_alpha_over_prod[i]) + self.sqrt_betas[i] * z
            if plot:
                plt.imshow(np.transpose(x_i.squeeze(0).numpy(), (1, 2, 0)), cmap='inferno')
                plt.legend(f'Step №{i}', loc='lower left')
                camera.snap()
        if plot:
            anim = camera.animate(interval=150, blit=True)
            anim.save(name + '.gif', writer='imagemagick')


def get_schedules(beta1: float, beta2: float, num_timesteps: int) -> Dict[str, torch.Tensor]:
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    betas = (beta2 - beta1) * torch.arange(0, num_timesteps +
                                           1, dtype=torch.float32) / num_timesteps + beta1
    sqrt_betas = torch.sqrt(betas)
    alphas = 1 - betas

    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    inv_sqrt_alphas = 1 / torch.sqrt(alphas)

    sqrt_one_minus_alpha_prod = torch.sqrt(1 - alphas_cumprod)
    one_minus_alpha_over_prod = (1 - alphas) / sqrt_one_minus_alpha_prod

    return {
        "alphas": alphas,
        "inv_sqrt_alphas": inv_sqrt_alphas,
        "sqrt_betas": sqrt_betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alpha_prod": sqrt_one_minus_alpha_prod,
        "one_minus_alpha_over_prod": one_minus_alpha_over_prod,
    }


def timestep_to_alpha(timesteps, T):
    return np.cos((timesteps / T + 0.008) * np.pi / ((1 + 0.008) * 2)) ** 2


def get_cosine_schedules(num_timesteps: int) -> Dict[str, torch.Tensor]:

    alphas_cumprod = torch.tensor(timestep_to_alpha(
        np.arange(0, num_timesteps + 1), num_timesteps + 1), dtype=torch.float32)
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = torch.clip(betas, min=1e-8, max=0.999)
    sqrt_betas = torch.sqrt(betas)

    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    inv_sqrt_alphas = 1 / torch.sqrt(alphas)

    sqrt_one_minus_alpha_prod = torch.sqrt(1 - alphas_cumprod)
    one_minus_alpha_over_prod = (1 - alphas) / sqrt_one_minus_alpha_prod
    sigmas = torch.sqrt(
        (1 - alphas_cumprod[:-1]) / (1 - alphas_cumprod[1:]) * (1 - alphas_cumprod[1:] / alphas_cumprod[:-1])
    )

    return {
        "alphas": alphas,
        "inv_sqrt_alphas": inv_sqrt_alphas,
        "sqrt_betas": sqrt_betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alpha_prod": sqrt_one_minus_alpha_prod,
        "one_minus_alpha_over_prod": one_minus_alpha_over_prod,
        "sigmas": sigmas
    }
