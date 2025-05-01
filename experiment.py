import torch
import wandb

from utils import *
from diffusion import DiffusionModel
from gamma_diffusion import GammaDiffusionModel
from data import get_dataloaders
from evaluate import calc_metrics


def run_experiment(config, datapath, savepath, name='classic diffusion', gamma=False, checkpoint=None):

    run = wandb.init(project="FinalExperiments", config=config,
                     name=name, reinit=True)

    if gamma:
        model = GammaDiffusionModel(
            config['timesteps'], config['hidden_size'], config['theta0'])
    else:
        model = DiffusionModel(
            config['timesteps'], config['hidden_size'], config['schedule'])

    model = model.to(DEVICE)
    wandb.log({"trainable_params": calc_params(model)}, step=0)

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))

    train_dataloader, val_dataloader = get_dataloaders(
        datapath, config['batch_size'])
    energy, cond = next(iter(train_dataloader))
    point, momentum = cond[0][:5], cond[1][:5]
    wandb.log({"inputs": wandb.Image(energy[:5])}, step=0)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['learning_rate'])

    for i in range(config['epochs']):
        loss = train_epoch(model, train_dataloader, optimizer)
        grad_norm = calc_grad_norm(model)
        wandb.log({"train_loss": loss, "grad_norm": grad_norm}, step=i)
        print(f"Epoch {i + 1} | Loss {loss}")
        if i % 10 == 0:
            with torch.no_grad():
                samples = model.sample(momentum.to(DEVICE), point.to(DEVICE), truncate=0)
                wandb.log({"examples": wandb.Image(samples)}, step=i)

    torch.save(model.state_dict(), savepath)
    wandb.save(savepath)
    
    prds, prd_curves, stats = calc_metrics(
        model, val_dataloader, t=config['threshold'])
    wandb.log({"E-PRD": prds[0], "P-PRD": prds[1],
              "Conditional-E-PRD": prds[2], "Conditional-P-PRD": prds[3], "E-FID": prds[4], "P-FID": prds[5]})
    for i, name in enumerate(["E", "P", "Cond-E", "Cond-P"]):
        wandb.log({name + "-curve": wandb.Image(prd_curves[i])})
    for i, name in enumerate(NAMES):
        wandb.log({name: wandb.Image(stats[i])})
    run.finish()


def run_evaluation(config, datapath, checkpoint, gamma=False):
    _, val_dataloader = get_dataloaders(datapath, config['batch_size'])

    run = wandb.init(project="Metrics", config=config,
                     name='evaluated', reinit=True)

    if gamma:
        model = GammaDiffusionModel(
            config['timesteps'], config['hidden_size'], config['theta0'])
    else:
        model = DiffusionModel(config['timesteps'], config['hidden_size'])

    model = model.to(DEVICE)
    wandb.log({"trainable_params": calc_params(model)}, step=0)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))

    prds, prd_curves, stats = calc_metrics(
        model, val_dataloader, t=config['threshold'])
    wandb.log({"E-PRD": prds[0], "P-PRD": prds[1],
              "Conditional-E-PRD": prds[2], "Conditional-P-PRD": prds[3], "E-FID": prds[4], "P-FID": prds[5]})
    for i, name in enumerate(["E", "P", "Cond-E", "Cond-P"]):
        wandb.log({name + "-curve": wandb.Image(prd_curves[i])})
    for i, name in enumerate(NAMES):
        wandb.log({name: wandb.Image(stats[i])})
    run.finish()
