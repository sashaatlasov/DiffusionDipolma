import torch 
from tqdm.auto import tqdm

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

def train_model(config, datapath, savepath, checkpoint=None):
    model = UnetModel(1, 1, config['hidden_dim'])
    dm = DiffusionModel(model, config['timesteps'])
    dm = dm.to(DEVICE)
    if checkpoint:
        dm.load_state_dict(torch.load(checkpoint, map_location=DEVICE))

    train_dataloader, val_dataloader = get_dataloaders(datapath, config['batch_size'])
    optimizer = torch.optim.Adam(dm.parameters(), lr=config['learning_rate'])

    for i in range(config['epochs']):
        loss = train_epoch(dm, train_dataloader, optimizer)
        print(f"Epoch {i + 1} | Loss {loss}")
        if i % 10 == 0:
            total_prd, cond_prd = calc_metrics(dm, val_dataloader)
            print(f"PRD-AUC: {total_prd}, Conditional PRD-AUC: {cond_prd}")

    torch.save(dm.state_dict(), savepath)







