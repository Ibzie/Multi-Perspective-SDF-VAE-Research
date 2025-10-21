"""
Training framework for Multi-Perspective SDF-VAE

Handles:
- Training loop with curriculum learning
- Validation and checkpointing
- Metric logging and visualization
- Early stopping and learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
from tqdm import tqdm

from ..models import MultiPerspectiveSDFVAE, MultiPerspectiveSDFVAELoss
from ..utils import MetricsLogger
from .curriculum import CurriculumScheduler


class Trainer:
    """
    Comprehensive trainer for Multi-Perspective SDF-VAE

    Features:
    - Curriculum learning with gradual loss component introduction
    - Comprehensive metric logging
    - Automatic checkpointing
    - Early stopping
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: MultiPerspectiveSDFVAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = None,
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs',
        experiment_name: str = 'sdf_vae_experiment'
    ):
        """
        Args:
            model: Multi-Perspective SDF-VAE model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            optimizer: Optimizer (default: Adam)
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            experiment_name: Name of this experiment
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        # Loss function (weights will be updated by curriculum)
        self.loss_fn = MultiPerspectiveSDFVAELoss()

        # Curriculum scheduler
        self.curriculum = None  # Will be set in train()

        # Metrics logger
        self.metrics_logger = MetricsLogger(log_dir, experiment_name)

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

    def train(
        self,
        num_epochs: int,
        warmup_epochs: int = 5,
        log_interval: int = 100,
        save_interval: int = 5,
        early_stopping_patience: int = 10
    ):
        """
        Main training loop

        Args:
            num_epochs: Number of epochs to train
            warmup_epochs: Number of warmup epochs for curriculum
            log_interval: Log metrics every N steps
            save_interval: Save checkpoint every N epochs
            early_stopping_patience: Stop if no improvement for N epochs
        """
        # Initialize curriculum
        self.curriculum = CurriculumScheduler(
            num_epochs=num_epochs,
            warmup_epochs=warmup_epochs
        )

        # Print curriculum schedule
        self.curriculum.print_schedule()

        # Early stopping
        epochs_without_improvement = 0

        print(f"\n{'='*70}")
        print(f"Starting training: {self.experiment_name}")
        print(f"Device: {self.device}")
        print(f"Total epochs: {num_epochs}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print(f"{'='*70}\n")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Get curriculum weights for this epoch
            loss_weights = self.curriculum.get_loss_weights(epoch)
            self.loss_fn.update_weights(**loss_weights)

            # Get learning rate scale
            lr_scale = self.curriculum.get_learning_rate_scale(epoch)
            self._update_learning_rate(lr_scale)

            print(f"\n{'='*70}")
            print(f"EPOCH {epoch+1}/{num_epochs} - {self.curriculum.get_stage_name(epoch)}")
            print(f"Loss weights: α={loss_weights['alpha']:.3f}, β={loss_weights['beta']:.3f}, "
                  f"γ={loss_weights['gamma']:.3f}, δ={loss_weights['delta']:.3f}, ε={loss_weights['epsilon']:.3f}")
            print(f"{'='*70}")

            # Train one epoch
            train_metrics = self._train_epoch(log_interval)

            # Validate
            val_metrics = self._validate_epoch()

            # Log epoch metrics
            epoch_metrics = {
                'train_loss': train_metrics['total_loss'],
                'val_loss': val_metrics['total_loss'],
                'train_test_gap': val_metrics['total_loss'] - train_metrics['total_loss'],
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }
            self.metrics_logger.log_epoch(epoch, epoch_metrics)

            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_metrics['total_loss']:.6f}")
            print(f"  Val Loss:   {val_metrics['total_loss']:.6f}")
            print(f"  Gap:        {epoch_metrics['train_test_gap']:.6f}")

            # Save checkpoint
            is_best = val_metrics['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total_loss']
                epochs_without_improvement = 0
                print(f"  ✓ New best validation loss!")
            else:
                epochs_without_improvement += 1

            if (epoch + 1) % save_interval == 0 or is_best:
                self._save_checkpoint(epoch, val_metrics['total_loss'], is_best=is_best)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'])
                else:
                    self.scheduler.step()

            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n{'='*70}")
                print(f"Early stopping triggered - no improvement for {early_stopping_patience} epochs")
                print(f"Best validation loss: {self.best_val_loss:.6f}")
                print(f"{'='*70}")
                break

        # Final evaluation on test set
        if self.test_loader is not None:
            print(f"\n{'='*70}")
            print("Evaluating on test set...")
            print(f"{'='*70}")
            test_metrics = self._test_epoch()
            print(f"\nTest Loss: {test_metrics['total_loss']:.6f}")

        # Save final metrics
        self.metrics_logger.save()
        self.metrics_logger.save_json()

        print(f"\n{'='*70}")
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Metrics saved to: {self.metrics_logger.log_dir}")
        print(f"{'='*70}\n")

    def _train_epoch(self, log_interval: int) -> Dict[str, float]:
        """
        Train for one epoch

        Args:
            log_interval: Log metrics every N steps

        Returns:
            Dictionary of average metrics for the epoch
        """
        self.model.train()

        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction': 0.0,
            'kl': 0.0,
            'sdf_consistency': 0.0,
            'eikonal': 0.0,
            'diversity': 0.0
        }

        pbar = tqdm(self.train_loader, desc=f"Training")

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)

            # Forward pass
            output = self.model(images, return_metrics=False)

            # Compute loss
            losses = self.loss_fn(
                x=images,
                reconstruction=output['reconstruction'],
                mean=output['mean'],
                logvar=output['logvar'],
                sdf_values=output['sdf_values'],
                feature_projections=output['feature_projections'],
                attention_weights=output['attention_weights'],
                encoder_features=output.get('encoder_features')
            )

            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate losses
            for key in epoch_losses.keys():
                epoch_losses[key] += losses[key].item()

            # Update progress bar
            pbar.set_postfix({'loss': losses['total'].item()})

            # Log step metrics
            if (batch_idx + 1) % log_interval == 0:
                step_metrics = {k: v.item() for k, v in losses.items()}
                self.metrics_logger.log_step('train', self.global_step, step_metrics)

            self.global_step += 1

        # Average losses
        num_batches = len(self.train_loader)
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches

        return epoch_losses

    def _validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch

        Returns:
            Dictionary of average validation metrics
        """
        self.model.eval()

        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction': 0.0,
            'kl': 0.0,
            'sdf_consistency': 0.0,
            'eikonal': 0.0,
            'diversity': 0.0
        }

        with torch.no_grad():
            for images, _ in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)

                # Forward pass
                output = self.model(images, return_metrics=False)

                # Compute loss
                losses = self.loss_fn(
                    x=images,
                    reconstruction=output['reconstruction'],
                    mean=output['mean'],
                    logvar=output['logvar'],
                    sdf_values=output['sdf_values'],
                    feature_projections=output['feature_projections'],
                    attention_weights=output['attention_weights'],
                    encoder_features=output.get('encoder_features')
                )

                # Accumulate losses
                for key in epoch_losses.keys():
                    epoch_losses[key] += losses[key].item()

        # Average losses
        num_batches = len(self.val_loader)
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches

        return epoch_losses

    def _test_epoch(self) -> Dict[str, float]:
        """
        Test for one epoch

        Returns:
            Dictionary of average test metrics
        """
        self.model.eval()

        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction': 0.0,
            'kl': 0.0,
            'sdf_consistency': 0.0,
            'eikonal': 0.0,
            'diversity': 0.0
        }

        with torch.no_grad():
            for images, _ in tqdm(self.test_loader, desc="Testing"):
                images = images.to(self.device)

                # Forward pass
                output = self.model(images, return_metrics=False)

                # Compute loss
                losses = self.loss_fn(
                    x=images,
                    reconstruction=output['reconstruction'],
                    mean=output['mean'],
                    logvar=output['logvar'],
                    sdf_values=output['sdf_values'],
                    feature_projections=output['feature_projections'],
                    attention_weights=output['attention_weights'],
                    encoder_features=output.get('encoder_features')
                )

                # Accumulate losses
                for key in epoch_losses.keys():
                    epoch_losses[key] += losses[key].item()

        # Average losses
        num_batches = len(self.test_loader)
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches

        return epoch_losses

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """
        Save model checkpoint

        Args:
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path.name}")

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"  Best model saved: {best_path.name}")

    def _update_learning_rate(self, scale: float):
        """
        Update learning rate based on curriculum schedule

        Args:
            scale: Learning rate scale factor
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
        print(f"  Best validation loss: {self.best_val_loss:.6f}")
