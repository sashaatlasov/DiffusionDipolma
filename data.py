from typing import Tuple, Any

import numpy as np
import torch
import torch.utils.data


class UnifiedDatasetWrapper(torch.utils.data.Dataset):
    """
    Обёртка для поддержки датасетов обоих типов
    """
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.dataset = dataset
        self.inverse_transform = getattr(dataset, 'inverse_transform', None)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, n: int) -> Tuple[Any, Any]:
        element = self.dataset[n]
        if isinstance(element, tuple):
            assert len(element) == 2
            x, y = element
        else:
            x, y = element, None
        return x, y


class PhysicsDataset(torch.utils.data.Dataset):
    """
    one element: (energy deposit, (point, momentum))
    """
    def __init__(self, energy: torch.Tensor, point: torch.Tensor, momentum: torch.Tensor,
                 transform=None, inverse_transform=None) -> None:
        """
        TODO: указать сигнатуры transform и inverse_transform через typing
        transform(energy, point, momentum) - tensors as batches or single
        inverse_transform(energy, point, momentum)
        """
        self.transform = transform
        self.inverse_transform = inverse_transform  # for outer use

        if transform is not None:
            energy = self.transform(energy)

        self.energy = energy
        self.point = point
        self.momentum = momentum

    def __getitem__(self, idx: int) -> tuple:
        return self.energy[idx], (self.point[idx], self.momentum[idx])

    def __len__(self) -> int:
        return self.energy.shape[0]


# принимают batch-и x-ов
def log1p_transform(x: torch.Tensor):
    return torch.log1p(x)


def log1p_inverse_transform(x: torch.Tensor):
    return torch.expm1(x)


def get_physics_dataset(path: str, train: bool = True, val_ratio: float = 0.5,
                        log1p_energy: bool = True) -> torch.utils.data.Dataset:
    TRAIN_VAL_SPLIT_SEED = 0x3df3fa

    data_train = np.load(path)

    np.random.seed(TRAIN_VAL_SPLIT_SEED)
    dataset_size = len(data_train['EnergyDeposit'])
    val_size = int(dataset_size * val_ratio)

    all_indices = np.arange(dataset_size)
    val_indices = np.random.choice(all_indices, size=val_size, replace=False)
    val_mask = np.zeros(dataset_size, dtype=bool)
    val_mask[val_indices] = True
    train_indices = all_indices[~val_mask]
    indices = train_indices if train else val_indices

    energy = torch.tensor(data_train['EnergyDeposit'][indices]).float()
    energy = torch.permute(energy, dims=(0, 3, 1, 2))
    point = torch.tensor(data_train['ParticlePoint'][:, :2][indices]).float()
    momentum = torch.tensor(data_train['ParticleMomentum'][indices]).float()

    transform, inverse_transform = None, None
    if log1p_energy:
        transform = log1p_transform
        inverse_transform = log1p_inverse_transform

    return PhysicsDataset(energy, point, momentum,
                          transform=transform, inverse_transform=inverse_transform)


def get_dataloaders(datapath, batch_size):
    train = UnifiedDatasetWrapper(get_physics_dataset(datapath, train=True))
    val = UnifiedDatasetWrapper(get_physics_dataset(datapath, train=False))
    return torch.utils.data.DataLoader(train, batch_size), torch.utils.data.DataLoader(val, batch_size=batch_size)

