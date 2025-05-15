from tqdm.auto import tqdm
import torch
import torch.nn as nn
from typing import Dict, Tuple
import numpy as np

from unet_small import UnetModel


class GammaDiffusionModel(nn.Module):
    def __init__(
        self,
        num_timesteps: int,
        hidden_size: int,
        theta0: float = 1e-3,
        betas: Tuple[float, float] = (1e-4, 0.02)
    ):
        super().__init__()
        self.eps_model = UnetModel(1, 1, hidden_size)

        for name, schedule in get_schedules(betas[0], betas[1], theta0, num_timesteps).items():
            self.register_buffer(name, schedule)

        self.num_timesteps = num_timesteps
        self.criterion = nn.L1Loss()

    def sample_gamma(self, timestep: torch.Tensor, device: str = 'cpu',
                     shape: Tuple[int] = (1, 30, 30)) -> Tuple[torch.Tensor]:
        k, theta = self.k_t_bar[timestep], self.theta_t[timestep]
        gamma = torch.distributions.Gamma(k, 1 / theta)
        eps = gamma.sample(shape).squeeze(-1).permute(3, 0, 1, 2).to(device)
        return eps, (eps - (k * theta)[:, None, None, None])

    def forward(self, x: torch.Tensor, m: torch.Tensor, p: torch.Tensor) -> torch.Tensor:

        device = x.device

        timestep = torch.randint(
            1, self.num_timesteps + 1, (x.shape[0],), device=device)

        _, centered_eps = self.sample_gamma(timestep, device)

        x_t = self.sqrt_alphas_cumprod[timestep,
                                       None, None, None] * x + centered_eps

        return self.criterion(centered_eps / self.sqrt_one_minus_alpha_prod[timestep, None, None, None], self.eps_model(x_t, m, p, timestep / self.num_timesteps))

    def sample(self, m: torch.Tensor, p: torch.Tensor, truncate=None) -> torch.Tensor:

        num_samples = m.shape[0]
        device = m.device

        T = torch.tensor([-1]).repeat(num_samples)
        x_i = self.sample_gamma(T, device)[1]

        for i in tqdm(range(self.num_timesteps, 0, -1), leave=False):
            i = torch.tensor([i]).repeat(num_samples)
            if i[0] > 1:
                z = self.sample_gamma(i, device)[1] / self.sqrt_one_minus_alpha_prod[i[0]]
            else:
                z = 0
            i = i[0]
            eps = self.eps_model(x_i, m, p, torch.tensor(
                i / self.num_timesteps).repeat(num_samples, 1).to(device))
            x_i = self.inv_sqrt_alphas[i] * (
                x_i - eps * self.one_minus_alpha_over_prod[i]) + self.sqrt_betas[i] * z
        
        if truncate:
            x_i[x_i < np.log1p(truncate / 5e-3)] = 0

        return x_i


def get_schedules(beta1: float, beta2: float, theta: float, num_timesteps: int) -> Dict[str, torch.Tensor]:

    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    betas = (beta2 - beta1) * torch.arange(0, num_timesteps +
                                           1, dtype=torch.float32) / num_timesteps + beta1
    sqrt_betas = torch.sqrt(betas)
    alphas = 1 - betas

    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

    k_t = (betas / theta ** 2) / alphas_cumprod
    k_t_bar = torch.cumsum(k_t, dim=0)
    theta_t = sqrt_alphas_cumprod * theta

    inv_sqrt_alphas = 1 / torch.sqrt(alphas)

    sqrt_one_minus_alpha_prod = torch.sqrt(1 - alphas_cumprod)
    one_minus_alpha_over_prod = (1 - alphas) / sqrt_one_minus_alpha_prod

    return {
        "betas": betas,
        "alphas": alphas,
        "inv_sqrt_alphas": inv_sqrt_alphas,
        "sqrt_betas": sqrt_betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alpha_prod": sqrt_one_minus_alpha_prod,
        "k_t": k_t,
        "k_t_bar": k_t_bar,
        "theta_t": theta_t,
        "one_minus_alpha_over_prod": one_minus_alpha_over_prod,
    }
