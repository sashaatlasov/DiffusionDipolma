from tqdm.notebook import tqdm
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Tuple
from mlx.utils import tree_flatten

DEVICE = "cpu"  # Просто для совместимости со старым кодом, не используется

NAMES = [
    'Longitudial Cluster Asymmetry', 
    'Transverse Cluster Asymmetry',
    'Cluster Longitudual Width', 
    'Cluster Transverse Width'
]


def calc_grad_norm(grads) -> float:
    def recurse(g):
        if isinstance(g, dict):
            return sum(recurse(v) for v in g.values())
        else:
            return mx.sum(g ** 2).item()
    
    total = recurse(grads)
    return total ** 0.5


def calc_params(model: nn.Module) -> int:
    return sum(v.size for _, v in tree_flatten(model.parameters()))


def train_step_mlx(
    model: nn.Module,
    energy: mx.array,
    condition: Tuple[mx.array, mx.array],
    optimizer: nn.Optimizer
) -> mx.array:
    """
    Один шаг обучения для MLX модели.
    
    Args:
        model: MLX модель
        energy: Энергетические данные (batch, 1, 30, 30)
        condition: Кортеж (point, momentum)
        optimizer: MLX оптимизатор
    
    Returns:
        Значение функции потерь
    """
    point, momentum = condition
    
    def loss_fn(model, energy, momentum, point):
        return model(energy, momentum, point)
    
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model, energy, momentum, point)
    
    optimizer.update(model, grads)

    mx.eval(model.parameters(), optimizer.state, loss)
    
    return loss, grads


def train_epoch_mlx(
    model: nn.Module,
    train_dataloader,
    optimizer: nn.Optimizer,
    loss_ema_alpha: float = 0.9
) -> float:
    """
    Одна эпоха обучения для MLX модели.
    
    Args:
        model: MLX модель
        train_dataloader: DataLoader с данными в формате mx.array
        optimizer: MLX оптимизатор
        loss_ema_alpha: Коэффициент сглаживания для EMA loss
    
    Returns:
        Среднее значение потерь за эпоху (EMA)
    """
    model.train()
    pbar = tqdm(train_dataloader, leave=False)
    
    loss_ema = None
    total_loss = 0.0
    num_batches = 0
    
    for energy, condition in pbar:

        loss, grads = train_step_mlx(model, energy, condition, optimizer)
        loss_value = loss.item()
        
        # Обновляем EMA loss
        if loss_ema is None:
            loss_ema = loss_value
        else:
            loss_ema = loss_ema_alpha * loss_ema + (1 - loss_ema_alpha) * loss_value
        
        total_loss += loss_value
        num_batches += 1
        
        pbar.set_description(f"loss: {loss_ema:.6f}")
    
    return loss_ema if loss_ema is not None else 0.0, grads


def set_seed(seed: int = 42):
    """
    Устанавливает seed для воспроизводимости результатов.
    
    Args:
        seed: Значение seed
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)
    
    print(f"Random seed set to {seed}")


# Функции для сохранения и загрузки моделей
def save_model_weights(model: nn.Module, filepath: str):
    """
    Сохраняет веса MLX модели в формате NPZ.
    
    Args:
        model: MLX модель
        filepath: Путь для сохранения
    """
    weights_dict = {}
    for i, param in enumerate(model.trainable_parameters()):
        weights_dict[f"param_{i}"] = np.array(param)
    
    np.savez(filepath, **weights_dict)
    print(f"Model saved to {filepath}")


def load_model_weights(model: nn.Module, filepath: str):
    """
    Загружает веса MLX модели из NPZ файла.
    
    Args:
        model: MLX модель
        filepath: Путь к файлу с весами
    """
    weights_data = np.load(filepath, allow_pickle=True)
    
    # Загружаем параметры (нужно соответствие архитектуре)
    for i, param in enumerate(model.trainable_parameters()):
        if f"param_{i}" in weights_data:
            param[:] = mx.array(weights_data[f"param_{i}"])
    
    print(f"Model loaded from {filepath}")
