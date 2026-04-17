from tqdm.auto import tqdm
import torch

def get_local_device() -> torch.device:
    return torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

DEVICE = get_local_device()
NAMES = ['Longitudual Cluster Asymmetry', 'Transverse Cluster Asymmetry',
             'Cluster Longitudual Width', 'Cluster Transverse Width']

def calc_grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def calc_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

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
