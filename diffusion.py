from tqdm.auto import tqdm
import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Tuple
import numpy as np

# from unet_small import UnetModel
from unet import UNet


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
        self.eps_model = UNet(1, hidden_size)

        if schedule == 'linear':
            schedules = get_schedules(betas[0], betas[1], num_timesteps)
            for name, schedule in schedules.items():
                setattr(self, name, schedule)
        elif schedule == 'cosine':
            schedules = get_cosine_schedules(num_timesteps + 1)
            for name, schedule in schedules.items():
                setattr(self, name, schedule)
        else:
            raise ValueError('schedule type is not supported')
        
        self.num_timesteps = num_timesteps

        if loss == 'l2':
            self.criterion = nn.losses.mse_loss
        elif loss == 'l1':
            self.criterion = nn.losses.mae_loss
        else:
            raise ValueError('loss type is not supported')

    def __call__(self, x: mx.array, m: mx.array, p: mx.array) -> mx.array:
        """Forward pass for training (predict noise)."""
        batch_size = x.shape[0]
        
        timestep = mx.random.randint(0, self.num_timesteps, (batch_size,))
        
        eps = mx.random.normal(shape=x.shape)
        
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timestep]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alpha_prod[timestep]
        
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.reshape(-1, 1, 1, 1)
        
        x_t = sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_prod * eps
  
        t_normalized = timestep.reshape(-1, 1) / self.num_timesteps
  
        eps_pred = self.eps_model(x_t, m, p, t_normalized)
        return self.criterion(eps, eps_pred).mean()

    def sample(self, m: mx.array, p: mx.array, truncate: float = None) -> mx.array:
        """Generate samples from noise."""
        size = (30, 30, 1)
        num_samples = m.shape[0]
        
        x_i = mx.random.normal(shape=(num_samples, *size))
        
        # Iterative denoising
        for i in tqdm(range(self.num_timesteps - 1, 0, -1), leave=False):

            z = mx.zeros_like(x_i) if i == 1 else mx.random.normal(shape=(num_samples, *size))
            
            t_normalized = mx.full((num_samples, 1), i / self.num_timesteps)
            eps = self.eps_model(x_i, m, p, t_normalized)

            x_i = self.inv_sqrt_alphas[i] * (
                x_i - eps * self.one_minus_alpha_over_prod[i]
            ) + self.sqrt_betas[i] * z
            
        # Apply truncation if specified
        if truncate:
            x_i = mx.where(x_i < np.log1p(truncate / 5e-3), 0, x_i)
        
        return x_i

    def sample_single(self, m: mx.array, p: mx.array, plot=False, name='animation') -> mx.array:
        """Generate a single sample with optional visualization."""
        device = m.device if hasattr(m, 'device') else None
        x_i = mx.random.normal(shape=(1, 1, 30, 30))
        
        m = m.reshape(1, -1)
        p = p.reshape(1, -1)
        
        if plot:
            # Note: MLX doesn't have direct numpy() conversion without explicit array
            # You'll need to convert to numpy for plotting
            try:
                import matplotlib.pyplot as plt
                from celluloid import Camera
                
                fig = plt.figure(dpi=250)
                camera = Camera(fig)
                
                # Convert to numpy for visualization
                x_np = np.array(x_i)[0, 0]  # Remove batch and channel dims
                plt.imshow(x_np, cmap='inferno')
                plt.title(f"Point: {np.array(p[0])}, Momentum {np.linalg.norm(np.array(m[0])) ** 2:.2f}")
                plt.legend(f'Step №{0}', loc='lower left')
                camera.snap()
            except ImportError:
                print("Warning: matplotlib or celluloid not installed, skipping plot")
                plot = False
        
        for i in tqdm(range(self.num_timesteps, 0, -1), leave=False):
            z = mx.random.normal(shape=(1, 1, 30, 30)) if i > 1 else 0
            t_normalized = mx.full((1, 1), i / self.num_timesteps)
            eps = self.eps_model(x_i, m, p, t_normalized)
            x_i = self.inv_sqrt_alphas[i] * (
                x_i - eps * self.one_minus_alpha_over_prod[i]
            ) + self.sqrt_betas[i] * z
            
            if plot:
                try:
                    x_np = np.array(x_i)[0, 0]
                    plt.imshow(x_np, cmap='inferno')
                    plt.legend(f'Step №{i}', loc='lower left')
                    camera.snap()
                except:
                    pass
        
        if plot:
            try:
                anim = camera.animate(interval=150, blit=True)
                anim.save(name + '.gif', writer='imagemagick')
            except:
                print(f"Warning: Could not save animation to {name}.gif")
        
        return x_i


def get_schedules(beta1: float, beta2: float, num_timesteps: int) -> Dict[str, mx.array]:
    """Generate linear noise schedule."""
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    
    # Create beta schedule
    betas = (beta2 - beta1) * mx.arange(0, num_timesteps + 1, dtype=mx.float32) / num_timesteps + beta1
    sqrt_betas = mx.sqrt(betas)
    alphas = 1 - betas
    
    # Compute cumulative products
    alphas_cumprod = mx.cumprod(alphas, axis=0)
    
    # Precompute all required terms
    sqrt_alphas_cumprod = mx.sqrt(alphas_cumprod)
    inv_sqrt_alphas = 1 / mx.sqrt(alphas)
    sqrt_one_minus_alpha_prod = mx.sqrt(1 - alphas_cumprod)
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


def timestep_to_alpha(timesteps: np.ndarray, T: int) -> np.ndarray:
    """Helper function for cosine schedule."""
    return np.cos((timesteps / T + 0.008) * np.pi / ((1 + 0.008) * 2)) ** 2


def get_cosine_schedules(num_timesteps: int) -> Dict[str, mx.array]:
    """Generate cosine noise schedule."""
    # Compute alphas using numpy (cleaner for this math)
    alphas_cumprod_np = timestep_to_alpha(
        np.arange(0, num_timesteps + 1), num_timesteps + 1
    )
    alphas_cumprod = mx.array(alphas_cumprod_np, dtype=mx.float32)
    
    # Compute betas
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = mx.clip(betas, 1e-8, 0.999)
    sqrt_betas = mx.sqrt(betas)
    
    alphas = 1 - betas
    alphas_cumprod = mx.cumprod(alphas, axis=0)
    
    sqrt_alphas_cumprod = mx.sqrt(alphas_cumprod)
    inv_sqrt_alphas = 1 / mx.sqrt(alphas)
    sqrt_one_minus_alpha_prod = mx.sqrt(1 - alphas_cumprod)
    one_minus_alpha_over_prod = (1 - alphas) / sqrt_one_minus_alpha_prod
    
    # Compute sigmas (for DDIM sampling if needed)
    sigmas = mx.sqrt(
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
        "sigmas": sigmas,
    }