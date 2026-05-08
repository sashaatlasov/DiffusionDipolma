import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple
import mlflow
import mlflow.pytorch  # для совместимости, но будем логировать артефакты
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt

from utils import *
from diffusion import DiffusionModel
from gamma_diffusion import GammaDiffusionModel
from data import get_dataloaders_mlx
from evaluate import calc_metrics_mlx  


def setup_mlflow_logging(experiment_name: str = "Experiments", run_name: str = None):
    """Setup MLflow logging with local tracking."""
    Path("./mlruns").mkdir(exist_ok=True)
    
    mlflow.set_tracking_uri("./mlruns")
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    if run_name:
        return mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
    else:
        return mlflow.start_run(experiment_id=experiment_id)


def log_model_summary(model: nn.Module, config: Dict, step: int = 0):
    """Log model architecture and parameters to MLflow."""

    total_params = calc_params(model)
    
    mlflow.log_param("total_params", total_params)
    mlflow.log_param("model_config", json.dumps(config))
    
    with open("model_architecture.txt", "w") as f:
        f.write(str(model))
    mlflow.log_artifact("model_architecture.txt")
    
    return total_params


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, path: str):
    """Сохраняет чекпоинт модели и оптимизатора."""
    checkpoint = {
        'epoch': epoch,
        'model_state': model.parameters(),
        'optimizer_state': optimizer.state
    }
    
    np.savez(
        path,
        epoch=epoch,
        **{name: np.array(param) for name, param in checkpoint['model_state'].items()}
    )
    mlflow.log_artifact(path)


def run_experiment(
    config: Dict[str, Any],
    datapath: str,
    savepath: str,
    name: str = 'classic diffusion',
    gamma: bool = False,
    checkpoint: Optional[str] = None
):
    """
    Запуск эксперимента с логированием в MLflow.
    
    Args:
        config: Конфигурация эксперимента
        datapath: Путь к данным
        savepath: Путь для сохранения модели
        name: Название эксперимента
        gamma: Использовать GammaDiffusionModel или нет
        checkpoint: Путь к чекпоинту для возобновления
    """
    
    with setup_mlflow_logging("FinalExperiments", name):
        
        if gamma:
            model = GammaDiffusionModel(
                config['timesteps'], config['hidden_size'], config['theta0']
            )
        else:
            model = DiffusionModel(
                config['timesteps'], config['hidden_size'], config['schedule']
            )
        print(model.trainable_parameters().keys())
        total_params = log_model_summary(model, config)
        print(f"Trainable parameters: {total_params}")
        mlflow.log_metric("trainable_params", total_params, step=0)
        
        start_epoch = 0
        optimizer_state = None
        
        if checkpoint:
            checkpoint_data = np.load(checkpoint, allow_pickle=True)
            start_epoch = checkpoint_data['epoch'].item()
            # Загрузка параметров модели (нужно адаптировать под вашу архитектуру)
            # model.load_state_dict(checkpoint_data)
            print(f"Resumed from checkpoint at epoch {start_epoch}")
        
        train_dataloader, val_dataloader = get_dataloaders_mlx(datapath, config['batch_size'])
        
        energy, cond = next(iter(train_dataloader))
        point, momentum = cond[0][:5], cond[1][:5]
  
        for i in range(min(5, energy.shape[0])):
            energy_img = np.transpose(energy, (0, 3, 1, 2))
            energy_img = np.array(energy_img[i, 0])  # (30, 30)
            fig, ax = plt.subplots()
            im = ax.imshow(energy_img, cmap='inferno')
            plt.colorbar(im)
            ax.set_title(f"Energy sample {i}")
            mlflow.log_figure(fig, f"inputs/energy_sample_{i}.png")
            plt.close(fig)
        
        optimizer = optim.Adam(learning_rate=config['learning_rate'])
        
        for epoch in range(start_epoch, config['epochs']):
            epoch_start_time = time.time()
            
            loss, grads = train_epoch_mlx(model, train_dataloader, optimizer)
            
            grad_norm = calc_grad_norm(grads)
            
            epoch_time = time.time() - epoch_start_time
            
            mlflow.log_metrics({
                "train_loss": loss,
                "grad_norm": grad_norm,
                "epoch_time": epoch_time
            }, step=epoch)
            
            print(f"Epoch {epoch + 1}/{config['epochs']} | Loss: {loss:.6f} | "
                  f"Grad Norm: {grad_norm:.6f} | Time: {epoch_time:.2f}s")
            
            
            if (epoch + 1) % config.get('log_interval', 10) == 0:
                samples = model.sample(
                        momentum, point, truncate=config.get('truncate', 0)
                    )
                
                for i in range(min(5, samples.shape[0])):
                    sample_img = np.transpose(samples, (0, 3, 1, 2))
                    sample_img = np.array(sample_img[i, 0])  # (30, 30)
                    fig, ax = plt.subplots()
                    im = ax.imshow(sample_img, cmap='inferno')
                    plt.colorbar(im)
                    ax.set_title(f"Generated sample {i} at epoch {epoch+1}")
                    mlflow.log_figure(fig, f"samples/epoch_{epoch+1}_sample_{i}.png")
                    plt.close(fig)
                
                # Сохранение чекпоинта
                checkpoint_path = f"{savepath}_epoch_{epoch+1}.npz"
                save_checkpoint(model, optimizer, epoch, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
            break
        
        final_checkpoint_path = f"{savepath}_final.npz"
        save_checkpoint(model, optimizer, config['epochs'] - 1, final_checkpoint_path)
        mlflow.log_artifact(final_checkpoint_path)
        print(f"Final model saved to {final_checkpoint_path}")
        
        # Вычисление метрик на валидационной выборке
        print("Calculating metrics on validation set...")
        prds, prd_curves, stats = calc_metrics_mlx(
            model, val_dataloader, t=config.get('threshold', 0.5)
        )
        
        # Логирование метрик
        metrics_names = [
            "E-PRD", "P-PRD", "Conditional-E-PRD", 
            "Conditional-P-PRD", "E-FID", "P-FID"
        ]
        for name, value in zip(metrics_names, prds):
            mlflow.log_metric(name, value)
            print(f"{name}: {value:.4f}")
        
        # Логирование PRD кривых
        for i, name in enumerate(["E", "P", "Cond-E", "Cond-P"]):
            if i < len(prd_curves):
                fig, ax = plt.subplots()
                ax.plot(prd_curves[i][0], prd_curves[i][1])
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title(f'{name} PRD Curve')
                ax.grid(True)
                mlflow.log_figure(fig, f"prd_curves/{name}_curve.png")
                plt.close(fig)
        
        # Логирование статистик
        for i, name in enumerate(['Energy', 'Momentum', 'Point', 'Conditional']):
            if i < len(stats):
                fig, axes = plt.subplots(1, len(stats[i]), figsize=(15, 3))
                for j, stat in enumerate(stats[i]):
                    axes[j].hist(np.array(stat).flatten(), bins=50)
                    axes[j].set_title(f'{name} Stat {j}')
                mlflow.log_figure(fig, f"stats/{name}_distribution.png")
                plt.close(fig)


def run_evaluation(
    config: Dict[str, Any],
    datapath: str,
    checkpoint: str,
    gamma: bool = False
):
    """
    Запуск оценки модели с логированием в MLflow.
    
    Args:
        config: Конфигурация эксперимента
        datapath: Путь к данным
        checkpoint: Путь к чекпоинту модели
        gamma: Использовать GammaDiffusionModel или нет
    """
    
    with setup_mlflow_logging("Metrics", "evaluated"):
        
        # Логируем конфигурацию
        mlflow.log_params({f"config.{k}": v for k, v in config.items()})
        
        # Загружаем данные
        _, val_dataloader = get_dataloaders_mlx(datapath, config['batch_size'])
        
        # Создаем модель
        if gamma:
            model = GammaDiffusionModel(
                config['timesteps'], config['hidden_size'], config['theta0']
            )
        else:
            model = DiffusionModel(
                config['timesteps'], config['hidden_size'], config['schedule']
            )
        
        # Логируем информацию о модели
        total_params = log_model_summary(model, config)
        mlflow.log_metric("trainable_params", total_params, step=0)
        
        # Загружаем чекпоинт
        checkpoint_data = np.load(checkpoint, allow_pickle=True)
        # Загрузка параметров (нужно адаптировать)
        # model.load_state_dict(checkpoint_data)
        print(f"Loaded checkpoint from {checkpoint}")
        mlflow.log_artifact(checkpoint)
        
        # Вычисляем метрики
        print("Calculating metrics...")
        prds, prd_curves, stats = calc_metrics_mlx(
            model, val_dataloader, t=config.get('threshold', 0.5)
        )
        
        # Логируем метрики
        metrics_names = [
            "E-PRD", "P-PRD", "Conditional-E-PRD", 
            "Conditional-P-PRD", "E-FID", "P-FID"
        ]
        for name, value in zip(metrics_names, prds):
            mlflow.log_metric(name, value)
            print(f"{name}: {value:.4f}")
        
        # Логируем PRD кривые
        for i, name in enumerate(["E", "P", "Cond-E", "Cond-P"]):
            if i < len(prd_curves):
                fig, ax = plt.subplots()
                ax.plot(prd_curves[i][0], prd_curves[i][1])
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title(f'{name} PRD Curve')
                ax.grid(True)
                mlflow.log_figure(fig, f"prd_curves/{name}_curve.png")
                plt.close(fig)
        
        # Логируем статистики
        for i, name in enumerate(['Energy', 'Momentum', 'Point', 'Conditional']):
            if i < len(stats):
                fig, axes = plt.subplots(1, len(stats[i]), figsize=(15, 3))
                for j, stat in enumerate(stats[i]):
                    axes[j].hist(np.array(stat).flatten(), bins=50)
                    axes[j].set_title(f'{name} Stat {j}')
                mlflow.log_figure(fig, f"stats/{name}_distribution.png")
                plt.close(fig)
        
        print("Evaluation completed!")


# Вспомогательная функция для просмотра логов MLflow
def view_mlflow_logs():
    """Запускает MLflow UI для просмотра логов."""
    import subprocess
    import webbrowser
    
    print("Starting MLflow UI...")
    print("Logs are stored in ./mlflow_logs")
    print("Open http://localhost:5000 in your browser")
    
    # Запускаем MLflow UI
    subprocess.Popen(["mlflow", "ui", "--backend-store-uri", "./mlflow_logs"])
    webbrowser.open("http://localhost:5000")


# Пример использования
if __name__ == "__main__":
    # Конфигурация эксперимента
    config = {
        'timesteps': 1000,
        'hidden_size': 128,
        'schedule': 'linear',
        'batch_size': 32,
        'learning_rate': 1e-3,
        'epochs': 100,
        'log_interval': 10,
        'threshold': 0.5,
        'truncate': 0
    }
    
    # Запуск эксперимента
    run_experiment(
        config=config,
        datapath="./data",
        savepath="./models/model",
        name="my_experiment",
        gamma=False
    )
    
    # Для просмотра логов выполните:
    # view_mlflow_logs()