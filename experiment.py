import torch 
import wandb

from utils import *
from diffusion import DiffusionModel
from data import get_dataloaders
from evaluate import calc_metrics


def run_experiment(config, datapath, savepath, checkpoint=None):

    wandb.init(project="Diploma", config=config, name='classic diffusion')

    model = DiffusionModel(config['timesteps'], config['hidden_size'])
    model = model.to(DEVICE)

    wandb.log({"trainable_params": calc_params(model)}, step=0)

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))

    train_dataloader, val_dataloader = get_dataloaders(datapath, config['batch_size'])
    energy, cond = next(iter(train_dataloader))
    point, momentum = cond[0][:5], cond[1][:5]
    wandb.log({"inputs": wandb.Image(energy[:5])}, step=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    for i in range(config['epochs']):
        loss = train_epoch(model, train_dataloader, optimizer)
        grad_norm = calc_grad_norm(model)
        wandb.log({"train_loss": loss, "grad_norm": grad_norm}, step=i)
        print(f"Epoch {i + 1} | Loss {loss}")
        if i % 10 == 0:
            with torch.no_grad():
                samples = model.sample(momentum.to(DEVICE), point.to(DEVICE))
                wandb.log({"examples": wandb.Image(samples)}, step=i)
    
    total_prd, cond_prd, fig1, fig2 = calc_metrics(model, val_dataloader)
    print(f"FINAL -- PRD-AUC: {total_prd}, Conditional PRD-AUC: {cond_prd}")
    wandb.log({"prd-auc": total_prd, "conditional-prd": cond_prd,
               "prd-auc plot": wandb.Image(fig1), "prd bins plot": wandb.Image(fig2)})

    torch.save(model.state_dict(), savepath)