from tqdm.auto import tqdm
import torch
import torch.nn as nn
from typing import Dict, Tuple

from celluloid import Camera
import matplotlib.pyplot as plt
import numpy as np

from unet_small import UnetModel

class MaxWithValue(nn.Module):
    def __init__(self, min_value):
        super(MaxWithValue, self).__init__()
        self.min_value = min_value

    def forward(self, x):
        return torch.maximum(x, torch.tensor(self.min_value, dtype=x.dtype, device=x.device))


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
            for name, schedule in get_cosine_schedules(num_timesteps).items():
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
        return self.criterion(eps, self.eps_model(x_t, m, p, timestep / self.num_timesteps))

    def sample(self, m: torch.Tensor, p: torch.Tensor, truncate: float = None) -> torch.Tensor:

        size = (1, 30, 30)
        num_samples = m.shape[0]
        device = m.device

        x_i = torch.randn(num_samples, *size, device=device)

        for i in tqdm(range(self.num_timesteps, 0, -1), leave=False):
            z = torch.randn(num_samples, *size, device=device) if i > 1 else 0
            eps = self.eps_model(x_i, m, p, torch.tensor(
                i / self.num_timesteps).repeat(num_samples, 1).to(device))
            x_i = self.inv_sqrt_alphas[i] * (
                x_i - eps * self.one_minus_alpha_over_prod[i]) + self.sqrt_betas[i] * z
        
        if truncate:
            x_i[x_i < np.log1p(truncate)] = 0

        return x_i
    
    def implicit_sample(self, m: torch.Tensor, p: torch.Tensor, fast_sampling: int, eta: float):
        
        size = (1, 30, 30)
        num_samples = m.shape[0]
        device = m.device

        x_i = torch.randn(num_samples, *size, device=device)
        #steps = self.num_timesteps * torch.rand(fast_sampling, device=device)
        #steps = torch.round(steps).sort(descending=True).values.int()
        steps = torch.arange(0, self.num_timesteps, step=self.num_timesteps // fast_sampling)

        for i in tqdm(range(1, len(steps)), leave=False):
            current, prev = steps[i], steps[i - 1]
            sigma = eta * torch.sqrt((1 - self.alphas_cumprod[prev]) * (1 - self.alphas[prev]) / self.alphas_cumprod[current])
            z = torch.randn(num_samples, *size, device=device) if current > 1 else 0
            eps = self.eps_model(x_i, m, p, torch.tensor(
                i / self.num_timesteps).repeat(num_samples, 1).to(device))
            x_i = torch.sqrt(self.alphas_cumprod[current]) * (
                x_i - eps * torch.sqrt(1 - self.alphas_cumprod[prev])) / self.alphas_cumprod[prev] *  + torch.sqrt(1 - self.alphas_cumprod[current] - sigma ** 2) * eps + sigma * z
        
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
                plt.imshow(np.transpose(x_i.squeeze(
                    0).numpy(), (1, 2, 0)), cmap='inferno')
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
    return np.cos((timesteps / T + 0.008) * np.pi / ((1 + 0.008) * 2))

def get_cosine_schedules(num_timesteps: int) -> Dict[str, torch.Tensor]:

    alphas_cumprod = timestep_to_alpha(np.arange(0, num_timesteps + 1), num_timesteps + 1)
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    sqrt_betas = torch.sqrt(betas)
    alphas = 1 - betas
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

