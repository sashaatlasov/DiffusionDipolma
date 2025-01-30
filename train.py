import torch 
from tqdm.auto import tqdm
from device import get_local_device

def train_step(model, energy, condition, optimizer, device):
    optimizer.zero_grad()
    energy = energy.to(device)
    point, momentum = condition[0].to(device), condition[1].to(device)
    loss = model(energy, momentum, point)
    loss.backward()
    optimizer.step()
    return loss

def train_epoch(model, train_dataloader, optimizer, device):
    model.train()
    pbar = tqdm(train_dataloader)
    loss_ema = None
    for energy, condition in pbar:
        train_loss = train_step(model, energy, condition, optimizer, device)
        loss_ema = train_loss if loss_ema is None else 0.9 * loss_ema + 0.1 * train_loss
        pbar.set_description(f"loss: {loss_ema:.4f}")
    return loss_ema


