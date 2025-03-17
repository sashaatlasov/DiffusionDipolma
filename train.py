import torch 
from tqdm.auto import tqdm
import wandb

from device import get_local_device
from diffusion import DiffusionModel
from unet_small import UnetModel
from data import get_dataloaders
from evaluate import calc_metrics

DEVICE = get_local_device()

def train_step(model, energy, condition, optimizer):
    optimizer.zero_grad()
    energy = energy.to(DEVICE)
    point, momentum = condition[0].to(DEVICE), condition[1].to(DEVICE)
    loss = model(energy, momentum, point)
    loss.backward()
    optimizer.step()
    return loss

def train_epoch(model, train_dataloader, optimizer):
    model.train()
    pbar = tqdm(train_dataloader, leave=False)
    loss_ema = None
    for energy, condition in pbar:
        train_loss = train_step(model, energy, condition, optimizer)
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss
        pbar.set_description(f"loss: {loss_ema:.4f}")
    return loss_ema

# Вынести отсюда
def calc_grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def train_model(config, datapath, savepath, checkpoint=None):

    wandb.init(project="Diploma", config=config, name='classic')

    model = UnetModel(1, 1, config['hidden_dim'])
    dm = DiffusionModel(model, config['timesteps'])
    dm = dm.to(DEVICE)

    wandb.log({"trainable_params": sum(param.numel() for param in dm.parameters() if param.requires_grad)})

    if checkpoint:
        dm.load_state_dict(torch.load(checkpoint, map_location=DEVICE))

    train_dataloader, val_dataloader = get_dataloaders(datapath, config['batch_size'])
    energy, point, momentum = next(iter(train_dataloader))
    wandb.log({"inputs": wandb.Image(energy)})
    

    optimizer = torch.optim.Adam(dm.parameters(), lr=config['learning_rate'])

    for i in range(config['epochs']):
        loss = train_epoch(dm, train_dataloader, optimizer)
        grad_norm = calc_grad_norm(dm)
        wandb.log({"train_loss": loss, "grad_norm": grad_norm})
        print(f"Epoch {i + 1} | Loss {loss}")
        if i % 10 == 0:
            with torch.no_grad():
                samples = dm.sample(momentum.to(DEVICE), point.to(DEVICE))
                wandb.log({"examples": wandb.Image(samples)})
    
    total_prd, cond_prd = calc_metrics(dm, val_dataloader)
    print(f"FINAL -- PRD-AUC: {total_prd}, Conditional PRD-AUC: {cond_prd}")
    wandb.log({"prd-auc": total_prd, "conditional-prd": cond_prd})

    torch.save(dm.state_dict(), savepath)







