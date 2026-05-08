import torch
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import sqrtm

from metrics.calogan_metrics import get_physical_stats
from metrics.calogan_prd import plot_pr_aucs, calc_pr_rec_from_embeds, get_energy_embedding
from metrics.metrics import ConditionBinsMetric, AveragePRDAUCMetric
from data import log1p_inverse_transform
from utils import DEVICE, NAMES
import mlx.core as mx

import mlx.core as mx
from typing import Tuple, Optional, List


def plot_bins_prd(prds):
    dims_bins_cnt = [3, 3]
    x_lims = [0, 1]
    y_lims = [0, 1]
    x_bounds = np.linspace(*x_lims, num=dims_bins_cnt[0]+1)
    y_bounds = np.linspace(*y_lims, num=dims_bins_cnt[1]+1)

    fig, ax = plt.subplots(dpi=300)

    ax.axis('equal')
    ax.set_xlim(x_bounds[0], x_bounds[-1])
    ax.set_ylim(y_bounds[0], y_bounds[-1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    for x_bound in x_bounds:
        ax.axvline(x_bound)
    for y_bound in y_bounds:
        ax.axhline(y_bound)

    for k, val in enumerate(prds):
        i = (k % dims_bins_cnt[0])
        j = k // dims_bins_cnt[0]
        x = (x_bounds[i] + x_bounds[i+1]) / 2
        y = (y_bounds[j] + y_bounds[j+1]) / 2
        ax.text(x-0.05, y, f'{val:.3f}', fontsize=13)
        ax.text(x, y-0.1, k, color='blue', fontsize=10)

    ax.axis('off')
    ax.set_title(
        'Значение PRD-AUC в каждом из бинов при разбиении по эмбеддингам выходной матрицы')
    plt.show()
    return fig


def kl_div(true_probs, fake_probs):
    calc_indices = true_probs != 0
    if (fake_probs[calc_indices] == 0.).any():
        return np.inf
    else:
        return (true_probs[calc_indices] * np.log(true_probs[calc_indices] / fake_probs[calc_indices])).sum()


def calculate_fid(real, gen):
    mean1 = np.mean(real, axis=0)
    mean2 = np.mean(gen, axis=0)

    cov1 = np.cov(real, rowvar=False)
    cov2 = np.cov(gen, rowvar=False)

    sqrt_matrix = sqrtm(cov1 @ cov2).real
    return np.sum((mean1 - mean2) ** 2) + np.trace(cov1 + cov2 - 2 * sqrt_matrix)


def plot_stat_distribution(real, sampled, name, range=None):
    fig = plt.figure(dpi=300)
    hist1 = plt.hist(real, alpha=0.5, bins=75, density=True, color='orange',
                     edgecolor='black', label='Geant', range=range)
    hist2 = plt.hist(sampled, alpha=0.5, bins=75, density=True, color='steelblue',
                     edgecolor='black', label='Diffusion', range=range)
    plt.plot(
        [], [], ' ', label=f'KL: {kl_div(hist1[0] / len(real), hist2[0] / len(sampled)):.5f}')
    plt.title(name)
    plt.grid(axis='y')
    plt.legend()
    plt.show()
    return fig


def sample_energy_mlx(
    model, 
    val_data, 
    num_batches: int, 
    t: Optional[float] = None
) -> Tuple[Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]], 
           Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """
    Генерирует сэмплы энергии с помощью MLX модели и собирает эмбеддинги.
    
    Args:
        model: MLX модель (DiffusionModel)
        val_data: DataLoader с валидационными данными
        num_batches: Количество батчей для обработки
        t: Параметр truncate для sample()
    
    Returns:
        ((real_embeds, extra_embeds_real, (points, momentums)),
         (sampled_embeds, extra_embeds_sampled, (points, momentums)))
    """
    all_sampled_embeds = []
    all_real_embeds = []
    all_point = []
    all_momentum = []
    all_extra_embeds_real = []
    all_extra_embeds_sampled = []
    
    for i, (energy, (point, momentum)) in enumerate(tqdm(val_data, desc="Sampling energy")):
        if i >= num_batches:
            break
        
        # Данные уже должны быть в формате mx.array
        # Если нет - конвертируем
        if not isinstance(energy, mx.array):
            energy = mx.array(energy)
        if not isinstance(point, mx.array):
            point = mx.array(point)
        if not isinstance(momentum, mx.array):
            momentum = mx.array(momentum)
        
        # Генерация сэмплов (без градиентов)
        samples = model.sample(momentum, point, truncate=t)
        
        # Получаем эмбеддинги (эти функции нужно адаптировать под MLX)
        sampled_embeds = get_energy_embedding_mlx(samples)
        real_embeds = get_energy_embedding_mlx(energy)
        
        # Применяем обратную трансформацию
        energy_inv = log1p_inverse_transform(energy)
        samples_inv = log1p_inverse_transform(samples)
        
        # Убираем лишние размерности
        energy_inv = mx.squeeze(energy_inv)
        samples_inv = mx.squeeze(samples_inv)
        
        # Конвертируем в numpy для физических статистик
        energy_np = np.array(energy_inv)
        samples_np = np.array(samples_inv)
        point_np = np.array(point)
        momentum_np = np.array(momentum)
        
        # Вычисляем физические статистики (работают с numpy)
        extra_embeds_sampled = get_physical_stats(samples_np, momentum_np, point_np)
        extra_embeds_real = get_physical_stats(energy_np, momentum_np, point_np)
        
        # Конвертируем эмбеддинги в numpy
        all_sampled_embeds.append(np.array(sampled_embeds))
        all_real_embeds.append(np.array(real_embeds))
        all_extra_embeds_sampled.append(extra_embeds_sampled)
        all_extra_embeds_real.append(extra_embeds_real)
        all_point.append(point_np)
        all_momentum.append(momentum_np)
    
    # Объединяем все результаты
    return (
        (
            np.concatenate(all_real_embeds, axis=0),
            np.concatenate(all_extra_embeds_real, axis=0),
            (np.concatenate(all_point, axis=0), np.concatenate(all_momentum, axis=0))
        ),
        (
            np.concatenate(all_sampled_embeds, axis=0),
            np.concatenate(all_extra_embeds_sampled, axis=0),
            (np.concatenate(all_point, axis=0), np.concatenate(all_momentum, axis=0))
        )
    )


def calc_metrics_mlx(
    model, 
    val_data, 
    num_batches: Optional[int] = None, 
    t: Optional[float] = None
) -> Tuple[
    Tuple[float, float, float, float, float, float],
    Tuple,
    List
]:
    """
    Вычисляет метрики для сгенерированных данных.
    
    Args:
        model: MLX модель
        val_data: DataLoader с валидационными данными
        num_batches: Количество батчей для обработки (None = все)
        t: Параметр truncate для sample()
    
    Returns:
        ((E-PRD, P-PRD, Conditional-E-PRD, Conditional-P-PRD, E-FID, P-FID),
         (fig1, fig2, fig3, fig4),
         stat_dists)
    """
    # Определяем диапазоны для графиков
    ranges = [None, None, (0, 15), (0, 7)]
    
    if num_batches is None:
        num_batches = len(val_data)
    
    # Генерируем сэмплы и собираем эмбеддинги
    val_data_tuple, gen_data_tuple = sample_energy_mlx(model, val_data, num_batches, t=t)
    
    val_embeds, val_extra_embeds, (val_points, val_momentums) = val_data_tuple
    gen_embeds, gen_extra_embeds, (gen_points, gen_momentums) = gen_data_tuple
    
    # 1. Статистические распределения для физических параметров
    stat_dists = []
    for i in range(len(NAMES)):
        fig = plot_stat_distribution(
            val_extra_embeds[:, i], 
            gen_extra_embeds[:, i], 
            NAMES[i], 
            range=ranges[i]
        )
        stat_dists.append(fig)
    
    # 2. PRD для основных эмбеддингов
    prec, rec = calc_pr_rec_from_embeds(val_embeds, gen_embeds)
    result, fig1 = plot_pr_aucs(prec, rec)
    total_prd = np.mean(result)
    
    # 3. PRD для физических эмбеддингов
    prec, rec = calc_pr_rec_from_embeds(val_extra_embeds, gen_extra_embeds)
    result, fig2 = plot_pr_aucs(prec, rec)
    prd_phys = np.mean(result)
    
    # 4. Conditional PRD (требует torch tensors)
    # Конвертируем в torch тензоры для метрик
    val_cond = torch.tensor(val_points)
    gen_cond = torch.tensor(gen_points)
    val_embeds_torch = torch.tensor(val_embeds)
    gen_embeds_torch = torch.tensor(gen_embeds)
    val_extra_embeds_torch = torch.tensor(val_extra_embeds)
    gen_extra_embeds_torch = torch.tensor(gen_extra_embeds)
    
    calculated_metric = AveragePRDAUCMetric(num_clusters=20, num_runs=10, enforce_balance=True)
    metric = ConditionBinsMetric(
        calculated_metric,
        dim_bins=torch.Tensor([3, 3]),
        condition_index=0
    )
    
    # Conditional PRD для основных эмбеддингов
    result = metric.evaluate((val_embeds_torch, val_cond), (gen_embeds_torch, gen_cond))
    cond_prd = np.mean(result)
    fig3 = plot_bins_prd(result)
    
    # Conditional PRD для физических эмбеддингов
    result = metric.evaluate((val_extra_embeds_torch, val_cond), (gen_extra_embeds_torch, gen_cond))
    cond_prd_phys = np.mean(result)
    fig4 = plot_bins_prd(result)
    
    # 5. FID метрики
    efid = calculate_fid(val_embeds, gen_embeds)
    pfid = calculate_fid(val_extra_embeds, gen_extra_embeds)
    
    return (
        (total_prd, prd_phys, cond_prd, cond_prd_phys, efid, pfid),
        (fig1, fig2, fig3, fig4),
        stat_dists
    )


# Вспомогательные функции для работы с эмбеддингами в MLX
def get_energy_embedding_mlx(energy: mx.array) -> mx.array:
    """
    Получает эмбеддинги энергии (нужно реализовать под вашу задачу).
    
    Args:
        energy: Тензор энергии (batch, 1, H, W)
    
    Returns:
        Эмбеддинги (batch, embedding_dim)
    """
    # Здесь должна быть ваша реализация получения эмбеддингов
    # Например, flatten или использование предобученной модели
    batch_size = energy.shape[0]
    return energy.reshape(batch_size, -1)  # Простой flatten как пример


def get_physical_stats_np(
    energy: np.ndarray, 
    momentum: np.ndarray, 
    point: np.ndarray
) -> np.ndarray:
    """
    Вычисляет физические статистики для numpy массивов.
    
    Args:
        energy: Энергия (batch, H, W)
        momentum: Импульс (batch, 2)
        point: Точка (batch, 2)
    
    Returns:
        Массив физических статистик (batch, 4)
    """
    # Здесь должна быть ваша реализация
    # Пример: вычисляем асимметрии и ширины кластеров
    batch_size = energy.shape[0]
    stats = np.zeros((batch_size, 4))
    
    for i in range(batch_size):
        e = energy[i]
        
        # Пример вычислений (замените на ваши)
        # Longitudial Cluster Asymmetry
        h, w = e.shape
        left_sum = np.sum(e[:, :w//2])
        right_sum = np.sum(e[:, w//2:])
        stats[i, 0] = (left_sum - right_sum) / (left_sum + right_sum + 1e-8)
        
        # Transverse Cluster Asymmetry
        top_sum = np.sum(e[:h//2, :])
        bottom_sum = np.sum(e[h//2:, :])
        stats[i, 1] = (top_sum - bottom_sum) / (top_sum + bottom_sum + 1e-8)
        
        # Cluster Longitudial Width
        x_proj = np.sum(e, axis=0)
        x_center = np.sum(x_proj * np.arange(w)) / (np.sum(x_proj) + 1e-8)
        stats[i, 2] = np.sqrt(np.sum(x_proj * (np.arange(w) - x_center)**2) / (np.sum(x_proj) + 1e-8))
        
        # Cluster Transverse Width
        y_proj = np.sum(e, axis=1)
        y_center = np.sum(y_proj * np.arange(h)) / (np.sum(y_proj) + 1e-8)
        stats[i, 3] = np.sqrt(np.sum(y_proj * (np.arange(h) - y_center)**2) / (np.sum(y_proj) + 1e-8))
    
    return stats


# Если у вас уже есть функция get_physical_stats, работающая с numpy,
# используйте её вместо написанной выше:
# def get_physical_stats(energy, momentum, point):
#     # ваша реализация
#     pass


# Для совместимости со старым кодом (если нужно)
def sample_energy(model, val_data, num_batches, t):
    """Обертка для совместимости с PyTorch (не рекомендуется)."""
    print("Warning: Using deprecated sample_energy. Use sample_energy_mlx instead.")
    return sample_energy_mlx(model, val_data, num_batches, t)

