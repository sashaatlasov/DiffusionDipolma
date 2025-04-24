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
        return (true_probs[calc_indices] * np.log(true_probs[calc_indices] / fake_probs[calc_indices])).mean()


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
        [], [], ' ', label=f'KL: {kl_div(hist1[0] / len(real), hist2[0] / len(sampled))}')
    plt.title(name)
    plt.grid(axis='y')
    plt.legend()
    plt.show()
    return fig


def sample_energy(model, val_data, num_batches, t):
    model.eval()
    all_sampled_embeds, all_real_embeds, all_point, all_momentum = [], [], [], []
    all_extra_embeds_real, all_extra_embeds_sampled = [], []

    for i, (energy, (point, momentum)) in enumerate(tqdm(val_data)):
        if i >= num_batches:
            break

        energy, point, momentum = map(
            lambda x: x.to(DEVICE), (energy, point, momentum))
        with torch.no_grad():
            samples = model.sample(momentum, point, truncate=t)
            sampled_embeds = get_energy_embedding(samples)
            real_embeds = get_energy_embedding(energy)

        energy, samples = torch.squeeze(log1p_inverse_transform(
            energy)), torch.squeeze(log1p_inverse_transform(samples))
        energy, samples, point, momentum = map(
            lambda x: x.detach().cpu().numpy(), (energy, samples, point, momentum))
        extra_embeds_sampled = get_physical_stats(samples, momentum, point)
        extra_embeds_real = get_physical_stats(energy, momentum, point)

        all_sampled_embeds.append(sampled_embeds)
        all_real_embeds.append(real_embeds)
        all_extra_embeds_sampled.append(extra_embeds_sampled)
        all_extra_embeds_real.append(extra_embeds_real)
        all_point.append(point)
        all_momentum.append(momentum)

    return (
        (
            np.concatenate(all_real_embeds),
            np.concatenate(all_extra_embeds_real),
            (np.concatenate(all_point), np.concatenate(all_momentum))
        ),
        (
            np.concatenate(all_sampled_embeds),
            np.concatenate(all_extra_embeds_sampled),
            (np.concatenate(all_point), np.concatenate(all_momentum))
        )
    )


def calc_metrics(model, val_data, num_batches=None, t=None):
    ranges = [None, None, (0, 15), (0, 7)]

    if num_batches is None:
        num_batches = len(val_data)

    val_data, gen_data = sample_energy(model, val_data, num_batches, t=t)

    stat_dists = []
    for i in range(len(NAMES)):
        fig = plot_stat_distribution(
            val_data[1][:, i], gen_data[1][:, i], NAMES[i], range=ranges[i])
        stat_dists.append(fig)

    prec, rec = calc_pr_rec_from_embeds(val_data[0], gen_data[0])
    result, fig1 = plot_pr_aucs(prec, rec)
    total_prd = np.mean(result)

    prec, rec = calc_pr_rec_from_embeds(val_data[1], gen_data[1])
    result, fig2 = plot_pr_aucs(prec, rec)
    prd_phys = np.mean(result)

    calculated_metric = AveragePRDAUCMetric(num_clusters=20, num_runs=10,
                                            enforce_balance=True)
    metric = ConditionBinsMetric(
        calculated_metric,
        dim_bins=torch.Tensor([3, 3]),
        condition_index=0
    )
    val_cond = torch.tensor(val_data[2][0])
    gen_cond = torch.tensor(gen_data[2][0])

    result = metric.evaluate((val_data[0], val_cond), (gen_data[0], gen_cond))
    cond_prd = np.mean(result)
    fig3 = plot_bins_prd(result)

    result = metric.evaluate((val_data[1], val_cond), (gen_data[1], gen_cond))
    cond_prd_phys = np.mean(result)
    fig4 = plot_bins_prd(result)

    efid = calculate_fid(val_data[0], gen_data[0])
    pfid = calculate_fid(val_data[1], gen_data[1])

    return (total_prd, prd_phys, cond_prd, cond_prd_phys, efid, pfid), (fig1, fig2, fig3, fig4), stat_dists
