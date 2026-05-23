from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from celluloid import Camera
import matplotlib.pyplot as plt
import numpy as np

from unet_small import UnetModel


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
            for name, schedule in get_schedules(betas[0], betas[1], num_timesteps).items():
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

    @torch.no_grad()
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
            x_i[x_i < np.log1p(truncate / 5e-3)] = 0

        return x_i

    @torch.no_grad()
    def implicit_sample(self, m: torch.Tensor, p: torch.Tensor, num_steps: int = 10,
                    eta: float = 0.5, truncate: float = None) -> torch.Tensor:
        size = (1, 30, 30)
        num_samples = m.shape[0]
        device = m.device

        x_i = torch.randn(num_samples, *size, device=device)

        sub_seq = list(reversed(
            torch.linspace(1, self.num_timesteps, num_steps).long().tolist()
        ))  

        for i in tqdm(range(len(sub_seq) - 1), leave=False):
            t, t1 = sub_seq[i], sub_seq[i + 1]

            eps = self.eps_model(x_i, m, p,
                            torch.tensor([t / self.num_timesteps]).repeat(num_samples, 1).to(device))

            alpha_t  = self.alphas_cumprod[t]
            alpha_t1 = self.alphas_cumprod[t1]

            x0_pred = (x_i - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)

            sigma = eta * torch.sqrt((1 - alpha_t1) / (1 - alpha_t).clamp(min=1e-8)) \
                        * torch.sqrt((1 - alpha_t / alpha_t1).clamp(min=0.0))

            direction_coeff = torch.sqrt(
                torch.clamp(1 - alpha_t1 - sigma ** 2, min=0.0)
            )
            direction = direction_coeff * eps

            noise = sigma * torch.randn_like(x_i) if eta > 0 else 0

            x_i = torch.sqrt(alpha_t1) * x0_pred + direction + noise

        if truncate:
            x_i[x_i < np.log1p(truncate / 5e-3)] = 0

        return x_i


    def _dpm_get_alpha_sigma(self, t_continuous: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = self.num_timesteps
        idx = (t_continuous * T).long().clamp(1, T)
        ab = self.alphas_cumprod[idx]          
        alpha = torch.sqrt(ab)
        sigma = torch.sqrt(1.0 - ab)
        return alpha, sigma
 
    def _dpm_lambda(self, t_continuous: torch.Tensor) -> torch.Tensor:
        alpha, sigma = self._dpm_get_alpha_sigma(t_continuous)
        return torch.log(alpha / sigma.clamp(min=1e-8))
 
    def _dpm_eps(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        p: torch.Tensor,
        t_continuous: torch.Tensor,   
    ) -> torch.Tensor:
        num_samples = x.shape[0]
        t_ratio = t_continuous.reshape(1).repeat(num_samples, 1) 
        return self.eps_model(x, m, p, t_ratio)
 
    def _dpm_step1(
        self,
        x_s: torch.Tensor,
        m: torch.Tensor,
        p: torch.Tensor,
        t_s: torch.Tensor,
        t_t: torch.Tensor,
        eps_s: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    
        lam_s = self._dpm_lambda(t_s)
        lam_t = self._dpm_lambda(t_t)
        h = lam_t - lam_s                        
 
        alpha_t, sigma_t = self._dpm_get_alpha_sigma(t_t)
        alpha_s, _       = self._dpm_get_alpha_sigma(t_s)
 
        if eps_s is None:
            eps_s = self._dpm_eps(x_s, m, p, t_s)
 
        coeff = torch.exp(h) - 1.0               
        x_t = (alpha_t / alpha_s) * x_s - sigma_t * coeff * eps_s
        return x_t, eps_s
 
    def _dpm_step2(
        self,
        x_s: torch.Tensor,
        m: torch.Tensor,
        p: torch.Tensor,
        t_s: torch.Tensor,
        t_t: torch.Tensor,
    ) -> torch.Tensor:
        
        device = x_s.device
        t_mid = ((t_s + t_t) / 2.0).to(device)
 
        eps_s = self._dpm_eps(x_s, m, p, t_s)
        if torch.isnan(eps_s).any():
            print(f"eps_s nan: {torch.isnan(eps_s).any()}, max: {eps_s.abs().max():.2f}")

        x_mid, _ = self._dpm_step1(x_s, m, p, t_s, t_mid, eps_s=eps_s)
        if torch.isnan(x_mid).any():
            print(f"x_mid nan: {torch.isnan(x_mid).any()}, max: {x_mid.abs().max():.2f}")


        lam_s   = self._dpm_lambda(t_s)
        lam_t   = self._dpm_lambda(t_t)
        lam_mid = self._dpm_lambda(t_mid)
 
        h     = lam_t   - lam_s     
        h_mid = lam_mid - lam_s     
        if torch.isnan(h_mid).any():
            print(f"h={h:.5f}, h_mid={h_mid:.5f}")
 
        alpha_t, sigma_t = self._dpm_get_alpha_sigma(t_t)
        alpha_s, _       = self._dpm_get_alpha_sigma(t_s)
 
        eps_mid = self._dpm_eps(x_mid, m, p, t_mid)
        if torch.isnan(eps_mid).any():
            print(f"eps_mid nan: {torch.isnan(eps_mid).any()}, max: {eps_mid.abs().max():.2f}")

        coeff_1 = torch.exp(h) - 1.0
        coeff_2 = (h - 1.0) * torch.exp(h) + 1.0
 
        if h.abs() > 1e-3:
            r  = h_mid / h          
            D1 = (eps_mid - eps_s) / r
            if torch.isnan(D1).any():
                print(f"r={r:.4f}, D1 max: {D1.abs().max():.2f}, nan: {torch.isnan(D1).any()}")
        else:
            D1 = torch.zeros_like(eps_s)
 
        x_t = (alpha_t / alpha_s) * x_s \
              - sigma_t * coeff_1 * eps_s \
              - sigma_t * coeff_2 * D1
        return x_t
 
    @torch.no_grad()
    def dpm_sample(
        self,
        m: torch.Tensor,
        p: torch.Tensor,
        num_steps: int = 10,
        order: int = 2,
        truncate: float = None,
    ) -> torch.Tensor:

        assert order in (1, 2), "order должен быть 1 или 2"
 
        size = (1, 30, 30)
        num_samples = m.shape[0]
        device = m.device
 
        x = torch.randn(num_samples, *size, device=device)
 
        t_max = (self.num_timesteps - 1) / self.num_timesteps
        eps_t = 1.0 / self.num_timesteps
        t_seq = torch.linspace(t_max, eps_t, num_steps + 1).to(device)
 
        for i in tqdm(range(num_steps), desc=f'DPM-Solver-{order}', leave=False):
            t_s = t_seq[i]
            t_t = t_seq[i + 1]
 
            if order == 2:
                x = self._dpm_step2(x, m, p, t_s, t_t)
            else:
                x, _ = self._dpm_step1(x, m, p, t_s, t_t)
 
        if truncate:
            x[x < np.log1p(truncate / 5e-3)] = 0
 
        return x
    
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
