"""
Comprehensive metrics logging system for Multi-Perspective SDF-VAE

Tracks all metrics needed for research questions:
- Training dynamics (loss curves, overfitting)
- Latent space statistics (norms, variance, SDF correlation)
- Observer analysis (attention, diversity, specialization)
- Computational metrics (time, memory)
"""

import torch
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
import time


class MetricsLogger:
    """
    Centralized metrics logging for experiments

    Automatically saves metrics to disk and provides analysis utilities
    """

    def __init__(self, log_dir: str, experiment_name: str):
        """
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of this experiment
        """
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Storage for different metric types
        self.training_metrics = defaultdict(list)
        self.validation_metrics = defaultdict(list)
        self.test_metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(list)
        self.observer_metrics = defaultdict(list)
        self.latent_metrics = defaultdict(list)
        self.timing_metrics = defaultdict(list)

        # Metadata
        self.experiment_name = experiment_name
        self.start_time = time.time()
        self.current_epoch = 0
        self.total_steps = 0

    def log_step(self, phase: str, step: int, metrics: Dict[str, float]):
        """
        Log metrics for a single training/validation step

        Args:
            phase: 'train', 'val', or 'test'
            step: Global step number
            metrics: Dictionary of metric values
        """
        metrics['step'] = step
        metrics['timestamp'] = time.time() - self.start_time

        if phase == 'train':
            for key, value in metrics.items():
                self.training_metrics[key].append(value)
        elif phase == 'val':
            for key, value in metrics.items():
                self.validation_metrics[key].append(value)
        elif phase == 'test':
            for key, value in metrics.items():
                self.test_metrics[key].append(value)

        self.total_steps = max(self.total_steps, step)

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """
        Log end-of-epoch metrics

        Args:
            epoch: Epoch number
            metrics: Dictionary of epoch-level metrics
        """
        metrics['epoch'] = epoch
        metrics['timestamp'] = time.time() - self.start_time

        for key, value in metrics.items():
            self.epoch_metrics[key].append(value)

        self.current_epoch = epoch

    def log_observer_analysis(self, epoch: int, observer_data: Dict[str, Any]):
        """
        Log observer-specific analysis

        Args:
            epoch: Current epoch
            observer_data: Dictionary containing observer analysis
                Expected keys:
                - attention_weights: [B, num_observers]
                - sdf_values: [B, num_observers]
                - feature_norms: [num_observers]
                - specialization_scores: [num_observers]
        """
        observer_data['epoch'] = epoch
        observer_data['timestamp'] = time.time() - self.start_time

        for key, value in observer_data.items():
            self.observer_metrics[key].append(value)

    def log_latent_analysis(self, epoch: int, latent_data: Dict[str, Any]):
        """
        Log latent space analysis

        Args:
            epoch: Current epoch
            latent_data: Dictionary containing latent space metrics
                Expected keys:
                - latent_codes: [N, latent_dim]
                - sdf_values: [N, num_observers]
                - reconstruction_errors: [N]
        """
        latent_data['epoch'] = epoch
        latent_data['timestamp'] = time.time() - self.start_time

        for key, value in latent_data.items():
            self.latent_metrics[key].append(value)

    def log_timing(self, phase: str, duration: float, **kwargs):
        """
        Log timing information

        Args:
            phase: What was timed (e.g., 'forward', 'backward', 'epoch')
            duration: Duration in seconds
            **kwargs: Additional timing metadata
        """
        timing_data = {
            'phase': phase,
            'duration': duration,
            'timestamp': time.time() - self.start_time,
            **kwargs
        }

        for key, value in timing_data.items():
            self.timing_metrics[key].append(value)

    def get_generalization_gap(self) -> Dict[str, float]:
        """
        Compute train-test generalization gap

        Returns:
            Dictionary with gap statistics
        """
        if not self.training_metrics or not self.validation_metrics:
            return {}

        # Get final losses
        train_loss = self.training_metrics.get('total_loss', [])
        val_loss = self.validation_metrics.get('total_loss', [])

        if not train_loss or not val_loss:
            return {}

        # Compute gaps
        final_gap = val_loss[-1] - train_loss[-1] if len(train_loss) > 0 and len(val_loss) > 0 else 0.0

        # Mean gap over last 10% of training
        n_recent = max(1, len(train_loss) // 10)
        recent_train = np.mean(train_loss[-n_recent:])
        recent_val = np.mean(val_loss[-n_recent:])
        recent_gap = recent_val - recent_train

        # Maximum gap
        min_len = min(len(train_loss), len(val_loss))
        if min_len > 0:
            gaps = [val_loss[i] - train_loss[i] for i in range(min_len)]
            max_gap = max(gaps)
            mean_gap = np.mean(gaps)
        else:
            max_gap = 0.0
            mean_gap = 0.0

        return {
            'final_gap': final_gap,
            'recent_gap': recent_gap,
            'max_gap': max_gap,
            'mean_gap': mean_gap
        }

    def get_observer_statistics(self) -> Dict[str, Any]:
        """
        Compute observer specialization statistics

        Returns:
            Dictionary with observer analysis
        """
        if not self.observer_metrics.get('attention_weights'):
            return {}

        # Get attention weights over time
        attention_history = self.observer_metrics['attention_weights']

        # Most recent attention distribution
        if len(attention_history) > 0:
            recent_attention = attention_history[-1]  # [B, num_observers] or similar

            # Compute statistics
            if isinstance(recent_attention, torch.Tensor):
                recent_attention = recent_attention.cpu().numpy()

            mean_attention = np.mean(recent_attention, axis=0)
            std_attention = np.std(recent_attention, axis=0)

            # Compute entropy (lower = more specialized)
            entropy = -np.sum(mean_attention * np.log(mean_attention + 1e-10))

            # Dominant observer
            dominant_observer = int(np.argmax(mean_attention))

            return {
                'mean_attention_per_observer': mean_attention.tolist(),
                'std_attention_per_observer': std_attention.tolist(),
                'attention_entropy': float(entropy),
                'dominant_observer': dominant_observer
            }

        return {}

    def save(self, filename: str = 'metrics.pkl'):
        """
        Save all metrics to disk

        Args:
            filename: Filename for saved metrics
        """
        save_path = self.log_dir / filename

        metrics_data = {
            'training_metrics': dict(self.training_metrics),
            'validation_metrics': dict(self.validation_metrics),
            'test_metrics': dict(self.test_metrics),
            'epoch_metrics': dict(self.epoch_metrics),
            'observer_metrics': dict(self.observer_metrics),
            'latent_metrics': dict(self.latent_metrics),
            'timing_metrics': dict(self.timing_metrics),
            'metadata': {
                'experiment_name': self.experiment_name,
                'total_epochs': self.current_epoch,
                'total_steps': self.total_steps,
                'total_time': time.time() - self.start_time
            }
        }

        with open(save_path, 'wb') as f:
            pickle.dump(metrics_data, f)

        print(f"✓ Metrics saved to {save_path}")

    def save_json(self, filename: str = 'metrics.json'):
        """
        Save metrics summary as JSON (for easy viewing)

        Args:
            filename: Filename for JSON summary
        """
        save_path = self.log_dir / filename

        # Create JSON-serializable summary
        summary = {
            'experiment_name': self.experiment_name,
            'total_epochs': self.current_epoch,
            'total_steps': self.total_steps,
            'total_time': time.time() - self.start_time,
            'generalization_gap': self.get_generalization_gap(),
            'observer_statistics': self.get_observer_statistics(),
            'final_metrics': {
                'train_loss': float(self.training_metrics['total_loss'][-1]) if self.training_metrics.get('total_loss') else None,
                'val_loss': float(self.validation_metrics['total_loss'][-1]) if self.validation_metrics.get('total_loss') else None,
            }
        }

        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✓ JSON summary saved to {save_path}")

    @staticmethod
    def load(log_dir: str, experiment_name: str, filename: str = 'metrics.pkl') -> 'MetricsLogger':
        """
        Load previously saved metrics

        Args:
            log_dir: Directory containing logs
            experiment_name: Name of experiment
            filename: Filename of saved metrics

        Returns:
            MetricsLogger instance with loaded data
        """
        logger = MetricsLogger(log_dir, experiment_name)
        load_path = logger.log_dir / filename

        if not load_path.exists():
            print(f"WARNING: No saved metrics found at {load_path}")
            return logger

        with open(load_path, 'rb') as f:
            metrics_data = pickle.load(f)

        # Restore metrics
        logger.training_metrics = defaultdict(list, metrics_data['training_metrics'])
        logger.validation_metrics = defaultdict(list, metrics_data['validation_metrics'])
        logger.test_metrics = defaultdict(list, metrics_data['test_metrics'])
        logger.epoch_metrics = defaultdict(list, metrics_data['epoch_metrics'])
        logger.observer_metrics = defaultdict(list, metrics_data['observer_metrics'])
        logger.latent_metrics = defaultdict(list, metrics_data['latent_metrics'])
        logger.timing_metrics = defaultdict(list, metrics_data['timing_metrics'])

        # Restore metadata
        metadata = metrics_data['metadata']
        logger.current_epoch = metadata['total_epochs']
        logger.total_steps = metadata['total_steps']

        print(f"✓ Metrics loaded from {load_path}")
        return logger


class AblationLogger:
    """
    Special logger for ablation studies

    Tracks multiple experiments (e.g., with/without components) and
    facilitates comparison
    """

    def __init__(self, log_dir: str, ablation_name: str):
        """
        Args:
            log_dir: Base directory for logs
            ablation_name: Name of this ablation study
        """
        self.log_dir = Path(log_dir) / 'ablations' / ablation_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.ablation_name = ablation_name
        self.experiments = {}  # experiment_name -> MetricsLogger

    def add_experiment(self, experiment_name: str, logger: MetricsLogger):
        """
        Add an experiment to this ablation study

        Args:
            experiment_name: Name of the experiment variant
            logger: MetricsLogger for this experiment
        """
        self.experiments[experiment_name] = logger

    def compare(self, metric_name: str = 'total_loss', phase: str = 'val') -> Dict[str, List[float]]:
        """
        Compare a specific metric across all experiments

        Args:
            metric_name: Name of metric to compare
            phase: 'train', 'val', or 'test'

        Returns:
            Dictionary mapping experiment names to metric values
        """
        comparison = {}

        for exp_name, logger in self.experiments.items():
            if phase == 'train':
                metrics = logger.training_metrics
            elif phase == 'val':
                metrics = logger.validation_metrics
            else:
                metrics = logger.test_metrics

            if metric_name in metrics:
                comparison[exp_name] = metrics[metric_name]

        return comparison

    def save_comparison(self, filename: str = 'ablation_comparison.json'):
        """
        Save comparison of all experiments

        Args:
            filename: Filename for comparison
        """
        save_path = self.log_dir / filename

        comparison_data = {
            'ablation_name': self.ablation_name,
            'experiments': {}
        }

        for exp_name, logger in self.experiments.items():
            comparison_data['experiments'][exp_name] = {
                'generalization_gap': logger.get_generalization_gap(),
                'observer_statistics': logger.get_observer_statistics(),
                'final_train_loss': float(logger.training_metrics['total_loss'][-1]) if logger.training_metrics.get('total_loss') else None,
                'final_val_loss': float(logger.validation_metrics['total_loss'][-1]) if logger.validation_metrics.get('total_loss') else None,
            }

        with open(save_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)

        print(f"✓ Ablation comparison saved to {save_path}")
