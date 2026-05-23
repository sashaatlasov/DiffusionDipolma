import tempfile

import mlflow
import torch
from matplotlib import pyplot as plt

from utils import *
from diffusion import DiffusionModel
from gamma_diffusion import GammaDiffusionModel
from data import get_dataloaders
from evaluate import calc_metrics


def _log_sample_images(samples, step: int):
    samples_np = samples.detach().cpu().numpy()
    fig, axes = plt.subplots(1, len(samples_np), figsize=(3 * len(samples_np), 3))
    if len(samples_np) == 1:
        axes = [axes]
    for ax, img in zip(axes, samples_np):
        ax.imshow(img.squeeze(), cmap='inferno')
        ax.axis('off')
    plt.tight_layout()
    mlflow.log_figure(fig, f"samples/epoch_{step:04d}.png")
    plt.close(fig)


def _log_metric_figures(prd_curves, stats):
    for fig, curve_name in zip(prd_curves, ["E", "P", "Cond-E", "Cond-P"]):
        mlflow.log_figure(fig, f"prd_curves/{curve_name}.png")
        plt.close(fig)
    for fig, stat_name in zip(stats, NAMES):
        mlflow.log_figure(fig, f"stats/{stat_name}.png")
        plt.close(fig)


def run_experiment(config, datapath, savepath, name='classic diffusion', gamma=False, checkpoint=None):
    mlflow.set_experiment("FinalExperiments")

    with mlflow.start_run(run_name=name):
        mlflow.log_params(config)

        if gamma:
            model = GammaDiffusionModel(
                config['timesteps'], config['hidden_size'], config['theta0'])
        else:
            model = DiffusionModel(
                config['timesteps'], config['hidden_size'], config['schedule'])

        model = model.to(DEVICE)
        n_params = calc_params(model)
        print("trainable_params", n_params)
        mlflow.log_param("trainable_params", n_params)

        if checkpoint:
            model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))

        train_dataloader, val_dataloader = get_dataloaders(datapath, config['batch_size'])
        energy, cond = next(iter(train_dataloader))
        point, momentum = cond[0][:5], cond[1][:5]

        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

        for i in range(config['epochs']):
            loss = train_epoch(model, train_dataloader, optimizer)
            grad_norm = calc_grad_norm(model)
            loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
            mlflow.log_metrics({"train_loss": loss_val, "grad_norm": grad_norm}, step=i)
            print(f"Epoch {i + 1} | Loss {loss_val:.6f}")
            if i % 10 == 0:
                with torch.no_grad():
                    samples = model.sample(momentum.to(DEVICE), point.to(DEVICE), truncate=0)
                _log_sample_images(samples, step=i)

        torch.save(model.state_dict(), savepath)
        mlflow.log_artifact(savepath, artifact_path="checkpoints")

        prds, prd_curves, stats = calc_metrics(model, val_dataloader, t=config['threshold'])
        e_prd, p_prd, cond_e_prd, cond_p_prd, e_fid, p_fid = prds
        mlflow.log_metrics({
            "E-PRD": e_prd,
            "P-PRD": p_prd,
            "Conditional-E-PRD": cond_e_prd,
            "Conditional-P-PRD": cond_p_prd,
            "E-FID": e_fid,
            "P-FID": p_fid,
        })
        _log_metric_figures(prd_curves, stats)


def run_evaluation(name, config, datapath, checkpoint, gamma=False, speed=None):
    mlflow.set_experiment("Metrics")

    with mlflow.start_run(run_name=name):
        mlflow.log_params(config)
        mlflow.log_param("checkpoint", checkpoint)

        _, val_dataloader = get_dataloaders(datapath, config['batch_size'])

        if gamma:
            model = GammaDiffusionModel(
                config['timesteps'], config['hidden_size'], config['theta0'])
        else:
            model = DiffusionModel(config['timesteps'], config['hidden_size'], config['schedule'])

        model = model.to(DEVICE)
        mlflow.log_param("trainable_params", calc_params(model))
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))

        prds, prd_curves, stats = calc_metrics(model, val_dataloader, t=config['threshold'], speed=speed)
        e_prd, p_prd, cond_e_prd, cond_p_prd, e_fid, p_fid = prds
        mlflow.log_metrics({
            "E-PRD": e_prd,
            "P-PRD": p_prd,
            "Conditional-E-PRD": cond_e_prd,
            "Conditional-P-PRD": cond_p_prd,
            "E-FID": e_fid,
            "P-FID": p_fid,
        })
        _log_metric_figures(prd_curves, stats)
