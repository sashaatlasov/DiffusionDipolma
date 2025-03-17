import torch
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt

from metrics.calogan_prd import plot_pr_aucs, calc_pr_rec_from_embeds, get_energy_embedding
from metrics.metrics import ConditionBinsMetric, AveragePRDAUCMetric

from device import get_local_device
DEVICE = get_local_device()

def plot_bins_prd(prds):
    dims_bins_cnt = [3, 3]
    x_lims = [0, 1]
    y_lims = [0, 1]
    x_bounds = np.linspace(*x_lims, num=dims_bins_cnt[0]+1)
    y_bounds = np.linspace(*y_lims, num=dims_bins_cnt[1]+1)

    fig, ax = plt.subplots()

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
    ax.set_title('Значение PRD-AUC в каждом из бинов при разбиении по эмбеддингам выходной матрицы')
    plt.show()

def sample_energy(model, val_data, num_batches):   

    model.eval()

    all_sampled_embeds = None
    all_real_embeds = None
    all_point = None
    all_momentum = None

    cnt = 0
    for batch in tqdm(val_data):
        energy, point, momentum = batch[0], batch[1][0], batch[1][1]
        energy, point, momentum = energy.to(DEVICE), point.to(DEVICE), momentum.to(DEVICE)
        shape = energy[0].shape
        with torch.no_grad():
            samples = model.sample(momentum, point, shape)
        cnt += 1 
        samples = get_energy_embedding(samples)
        real = get_energy_embedding(energy)

        if all_sampled_embeds is None:
            all_sampled_embeds = samples
            all_real_embeds = real
            all_point = point
            all_momentum = momentum
        else:
            all_sampled_embeds = np.concatenate((all_sampled_embeds, samples), 0)
            all_real_embeds = np.concatenate((all_real_embeds, real), 0)
            all_point = torch.concatenate((all_point, point), 0)
            all_momentum = torch.concatenate((all_momentum, momentum), 0) 
        
        if cnt == num_batches:
            break

    gen_data = (all_sampled_embeds, (all_point, all_momentum))
    val_data = (all_real_embeds, (all_point, all_momentum))

    return val_data, gen_data


def calc_metrics(model, val_data, num_batches=None):

    if num_batches is None:
        num_batches = len(val_data) 

    val_data, gen_data = sample_energy(model, val_data, num_batches)

    prec, rec = calc_pr_rec_from_embeds(val_data[0], gen_data[0])
    result = plot_pr_aucs(prec, rec)
    total_prd = np.mean(result)

    calculated_metric = AveragePRDAUCMetric(num_clusters=20, num_runs=10,
                                            enforce_balance=True)
    metric = ConditionBinsMetric(
            calculated_metric,
            dim_bins=torch.Tensor([3, 3]),
            condition_index=0
        )
    val_data[1][0], gen_data[1][0] = val_data[1][0].detach().cpu().numpy(), gen_data[1][0].detach().cpu().numpy()

    result = metric.evaluate((val_data[0], val_data[1][0]), (gen_data[0], gen_data[1][0]))
    cond_prd = np.mean(result)
    plot_bins_prd(result)

    return total_prd, cond_prd





    
    