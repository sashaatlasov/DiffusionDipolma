from typing import Tuple, List, Optional, Union, Any, Sequence, Iterator
import numpy as np
from itertools import cycle
import mlx.core as mx
from pathlib import Path


class MLXBatchSampler:
    """Batch sampler со случайной выборкой."""
    def __init__(self, dataset_size: int, batch_size: int, drop_last: bool = False, shuffle: bool = True):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
    
    def __iter__(self):
        if self.shuffle:
            # Перемешиваем индексы
            indices = np.random.permutation(self.dataset_size)
        else:
            indices = np.arange(self.dataset_size)
        
        batch = []
        for idx in indices:
            batch.append(int(idx))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        if not self.drop_last and batch:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return self.dataset_size // self.batch_size
        else:
            return (self.dataset_size + self.batch_size - 1) // self.batch_size


def default_collate_mlx(batch):
    """
    Стандартная функция для объединения списка элементов в батч.
    Работает с mx.array, numpy arrays, числами и кортежами.
    """
    if not batch:
        return None
    
    # Проверяем тип первого элемента
    first_elem = batch[0]
    
    if isinstance(first_elem, mx.array):
        # Для MLX массивов - просто стек
        return mx.stack(batch)
    elif isinstance(first_elem, np.ndarray):
        # Для numpy массивов - конвертируем в mx.array и стек
        return mx.stack([mx.array(item) for item in batch])
    elif isinstance(first_elem, (int, float, np.number)):
        # Для чисел - конвертируем в массив
        return mx.array(batch)
    elif isinstance(first_elem, tuple):
        # Для кортежей - рекурсивно обрабатываем каждый элемент
        # Транспонируем список кортежей в кортеж списков
        transposed = zip(*batch)
        return tuple(default_collate_mlx(list(sublist)) for sublist in transposed)
    elif isinstance(first_elem, dict):
        # Для словарей - рекурсивно обрабатываем значения
        return {key: default_collate_mlx([item[key] for item in batch]) for key in first_elem}
    elif first_elem is None:
        return None
    else:
        raise TypeError(f"default_collate: batch must contain mx.array, numpy arrays, numbers, dicts or lists; found {type(first_elem)}")


class MLXDataLoader:
    def __init__(self, dataset, batch_size: int, shuffle: bool = True, 
                 drop_last: bool = False, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn if collate_fn is not None else default_collate_mlx
        
        # Создаем батч семплер
        self.batch_sampler = MLXBatchSampler(
            len(dataset), batch_size, drop_last, shuffle
        )
    
    def __iter__(self):
        for batch_indices in self.batch_sampler:
            # Собираем элементы по индексам
            batch = [self.dataset[idx] for idx in batch_indices]
            # Применяем collate функцию
            yield self.collate_fn(batch)
    
    def __len__(self):
        return len(self.batch_sampler)



class MLXDataset:
    """Базовый класс для MLX датасетов."""
    def __init__(self):
        self.inverse_transform = None
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        raise NotImplementedError
    
    def __len__(self) -> int:
        raise NotImplementedError


class UnifiedDatasetWrapper(MLXDataset):
    """
    Обёртка для поддержки датасетов разных типов.
    """
    def __init__(self, dataset: MLXDataset):
        super().__init__()
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


class PhysicsDataset(MLXDataset):
    """
    Датасет для физических данных.
    Элемент: (energy deposit, (point, momentum))
    """
    def __init__(
        self, 
        energy: mx.array, 
        point: mx.array, 
        momentum: mx.array,
        transform=None, 
        inverse_transform=None
    ):
        """
        Args:
            energy: Тензор энергии (N, 1, 30, 30)
            point: Тензор точек (N, 2) или (N, 3)
            momentum: Тензор импульсов (N, 2)
            transform: Функция трансформации (применяется к energy)
            inverse_transform: Обратная трансформация (для внешнего использования)
        """
        super().__init__()
        self.transform = transform
        self.inverse_transform = inverse_transform
        
        if transform is not None:
            energy = transform(energy)
        
        self.energy = energy
        self.point = point
        self.momentum = momentum
    
    def __getitem__(self, idx: int) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        # Возвращаем отдельные элементы
        # Важно: возвращаем именно mx.array, а не срезы с сохранением размерности
        return self.energy[idx], (self.point[idx], self.momentum[idx])
    
    def __len__(self) -> int:
        return self.energy.shape[0]


def log1p_transform(x: mx.array) -> mx.array:
    """
    Логарифмическая трансформация для энергии.
    
    Args:
        x: Входной массив
    
    Returns:
        log(1 + x/5e-3)
    """
    return mx.log1p(x / 5e-3)


def log1p_inverse_transform(x: mx.array) -> mx.array:
    """
    Обратная логарифмическая трансформация.
    
    Args:
        x: Входной массив
    
    Returns:
        (exp(x) - 1) * 5e-3
    """
    return mx.expm1(x) * 5e-3


def load_physics_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Загружает физические данные из NPZ файла.
    
    Args:
        path: Путь к NPZ файлу
    
    Returns:
        Кортеж (energy, point, momentum) в виде numpy массивов
    """
    data = np.load(path)
    
    energy = data['EnergyDeposit']
    point = data['ParticlePoint']
    momentum = data['ParticleMomentum']
    
    # Переставляем размерности энергии: (N, 30, 30, 1) -> (N, 1, 30, 30)
    #energy = np.transpose(energy, (0, 3, 1, 2))
    
    # Берем только первые 2 компоненты точки (x, y)
    point = point[:, :2]
    
    return energy, point, momentum


def split_train_val(
    energy: np.ndarray, 
    point: np.ndarray, 
    momentum: np.ndarray, 
    val_ratio: float = 0.5,
    seed: int = 0x3df3fa
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Разделяет данные на train и val.
    
    Args:
        energy: Массив энергии
        point: Массив точек
        momentum: Массив импульсов
        val_ratio: Доля валидационной выборки
        seed: Seed для воспроизводимости
    
    Returns:
        ((train_energy, train_point, train_momentum), (val_energy, val_point, val_momentum))
    """
    np.random.seed(seed)
    dataset_size = len(energy)
    val_size = int(dataset_size * val_ratio)
    
    # Случайный выбор валидационных индексов
    all_indices = np.arange(dataset_size)
    val_indices = np.random.choice(all_indices, size=val_size, replace=False)
    val_mask = np.zeros(dataset_size, dtype=bool)
    val_mask[val_indices] = True
    
    train_indices = all_indices[~val_mask]
    
    train_energy = energy[train_indices]
    train_point = point[train_indices]
    train_momentum = momentum[train_indices]
    
    val_energy = energy[val_indices]
    val_point = point[val_indices]
    val_momentum = momentum[val_indices]
    
    return (train_energy, train_point, train_momentum), (val_energy, val_point, val_momentum)


def get_physics_dataset_mlx(
    path: str, 
    train: bool = True, 
    val_ratio: float = 0.5,
    log1p_energy: bool = True
) -> PhysicsDataset:
    """
    Создает PhysicsDataset для MLX.
    
    Args:
        path: Путь к NPZ файлу с данными
        train: True для тренировочного датасета, False для валидационного
        val_ratio: Доля валидационной выборки
        log1p_energy: Применять ли log1p трансформацию к энергии
    
    Returns:
        PhysicsDataset в формате MLX
    """
    # Загружаем данные
    energy, point, momentum = load_physics_data(path)
    
    # Разделяем на train/val
    (train_energy, train_point, train_momentum), (val_energy, val_point, val_momentum) = \
        split_train_val(energy, point, momentum, val_ratio)
    
    # Выбираем нужную часть
    if train:
        energy, point, momentum = train_energy, train_point, train_momentum
    else:
        energy, point, momentum = val_energy, val_point, val_momentum
    
    # Конвертируем в mx.array
    energy = mx.array(energy.astype(np.float32))
    point = mx.array(point.astype(np.float32))
    momentum = mx.array(momentum.astype(np.float32))
    
    # Применяем трансформации
    transform, inverse_transform = None, None
    if log1p_energy:
        transform = log1p_transform
        inverse_transform = log1p_inverse_transform
        #energy = transform(energy)
    
    return PhysicsDataset(energy, point, momentum, 
                         transform=transform, inverse_transform=inverse_transform)


def get_dataloaders_mlx(
    datapath: str, 
    batch_size: int,
    val_ratio: float = 0.5,
    log1p_energy: bool = True,
    shuffle_train: bool = True
) -> Tuple[MLXDataLoader, MLXDataLoader]:
    """
    Создает train и val DataLoader для MLX.
    
    Args:
        datapath: Путь к NPZ файлу с данными
        batch_size: Размер батча
        val_ratio: Доля валидационной выборки
        log1p_energy: Применять ли log1p трансформацию
        shuffle_train: Перемешивать ли тренировочные данные
    
    Returns:
        (train_loader, val_loader)
    """
    # Создаем датасеты
    train_dataset = UnifiedDatasetWrapper(
        get_physics_dataset_mlx(datapath, train=True, val_ratio=val_ratio, 
                                log1p_energy=log1p_energy)
    )
    
    val_dataset = UnifiedDatasetWrapper(
        get_physics_dataset_mlx(datapath, train=False, val_ratio=val_ratio,
                                log1p_energy=log1p_energy)
    )
    
    # Создаем DataLoader'ы
    train_loader = MLXDataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = MLXDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
