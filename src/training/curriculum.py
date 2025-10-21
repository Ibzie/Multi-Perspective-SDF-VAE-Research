"""
Curriculum Learning for Multi-Perspective SDF-VAE

Gradually introduces loss components during training to prevent instability
and ensure stable convergence. This is critical for the multi-objective
optimization problem.

Loss component introduction schedule:
- Epochs 0-3: Reconstruction only (learn basic image structure)
- Epochs 3-10: Gradually introduce KL (learn latent space)
- Epochs 10-20: Introduce SDF consistency (align observers)
- Epochs 20+: Full model with all components
"""

import numpy as np
from typing import Dict, Tuple


class CurriculumScheduler:
    """
    Manages curriculum learning schedule for loss weights

    Key insight from original work: Starting with all losses at once causes
    training instability. Curriculum learning ensures smooth convergence.
    """

    def __init__(
        self,
        num_epochs: int,
        warmup_epochs: int = 5,
        full_curriculum_epochs: int = 20
    ):
        """
        Args:
            num_epochs: Total number of training epochs
            warmup_epochs: Number of epochs for initial warmup
            full_curriculum_epochs: Epoch when all losses are fully active
        """
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.full_curriculum_epochs = full_curriculum_epochs
        self.current_epoch = 0

        # Base loss weights (from original successful configuration)
        self.base_weights = {
            'alpha': 1.0,      # Reconstruction
            'beta': 0.2,       # KL divergence
            'gamma': 0.1,      # SDF consistency
            'delta': 0.05,     # Eikonal
            'epsilon': 0.1     # Diversity
        }

    def get_loss_weights(self, epoch: int) -> Dict[str, float]:
        """
        Get loss weights for current epoch based on curriculum schedule

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of loss weights
        """
        self.current_epoch = epoch

        # Stage 1 (Epochs 0-3): Reconstruction only
        if epoch < 3:
            return {
                'alpha': 1.0,
                'beta': 0.0,
                'gamma': 0.0,
                'delta': 0.0,
                'epsilon': 0.0
            }

        # Stage 2 (Epochs 3-10): Gradually introduce KL and diversity
        elif epoch < 10:
            progress = (epoch - 3) / 7.0  # 0 to 1 over epochs 3-10

            return {
                'alpha': 1.0,
                'beta': self.base_weights['beta'] * progress,
                'gamma': 0.0,
                'delta': 0.0,
                'epsilon': self.base_weights['epsilon'] * progress
            }

        # Stage 3 (Epochs 10-20): Introduce SDF consistency
        elif epoch < 20:
            progress = (epoch - 10) / 10.0  # 0 to 1 over epochs 10-20

            return {
                'alpha': 1.0,
                'beta': self.base_weights['beta'],
                'gamma': self.base_weights['gamma'] * progress,
                'delta': 0.0,  # Still no Eikonal (expensive)
                'epsilon': self.base_weights['epsilon']
            }

        # Stage 4 (Epoch 20+): Full model with all components
        else:
            # Optionally introduce Eikonal very gradually
            if epoch < 30:
                eikonal_progress = (epoch - 20) / 10.0
                delta = self.base_weights['delta'] * eikonal_progress
            else:
                delta = self.base_weights['delta']

            return {
                'alpha': 1.0,
                'beta': self.base_weights['beta'],
                'gamma': self.base_weights['gamma'],
                'delta': delta,
                'epsilon': self.base_weights['epsilon']
            }

    def get_learning_rate_scale(self, epoch: int) -> float:
        """
        Get learning rate scale factor for current epoch

        Implements warmup and optional decay

        Args:
            epoch: Current epoch number

        Returns:
            Learning rate scale factor (multiply with base LR)
        """
        # Linear warmup for first warmup_epochs
        if epoch < self.warmup_epochs:
            return (epoch + 1) / self.warmup_epochs

        # Cosine decay after warmup (optional)
        # For now, keep constant after warmup
        return 1.0

    def should_enable_eikonal(self, epoch: int) -> bool:
        """
        Check if Eikonal loss should be computed

        Eikonal is expensive (O(n²)), so we only enable it late in training

        Args:
            epoch: Current epoch number

        Returns:
            Whether to compute Eikonal loss
        """
        return epoch >= 20

    def get_stage_name(self, epoch: int) -> str:
        """
        Get human-readable name of current training stage

        Args:
            epoch: Current epoch number

        Returns:
            Stage name string
        """
        if epoch < 3:
            return "Stage 1: Reconstruction Only"
        elif epoch < 10:
            return "Stage 2: Introducing KL + Diversity"
        elif epoch < 20:
            return "Stage 3: Adding SDF Consistency"
        else:
            return "Stage 4: Full Model"

    def print_schedule(self):
        """
        Print the complete curriculum schedule
        """
        print("=" * 70)
        print("CURRICULUM LEARNING SCHEDULE")
        print("=" * 70)
        print(f"\nTotal epochs: {self.num_epochs}")
        print(f"Warmup epochs: {self.warmup_epochs}")
        print(f"Full curriculum by epoch: {self.full_curriculum_epochs}")
        print("\nStage breakdown:")
        print("-" * 70)

        for stage_epoch in [0, 3, 10, 20, 30]:
            if stage_epoch < self.num_epochs:
                weights = self.get_loss_weights(stage_epoch)
                stage = self.get_stage_name(stage_epoch)
                print(f"\nEpoch {stage_epoch}: {stage}")
                print(f"  Reconstruction (α): {weights['alpha']:.3f}")
                print(f"  KL Divergence (β): {weights['beta']:.3f}")
                print(f"  SDF Consistency (γ): {weights['gamma']:.3f}")
                print(f"  Eikonal (δ): {weights['delta']:.3f}")
                print(f"  Diversity (ε): {weights['epsilon']:.3f}")

        print("=" * 70)


class AdaptiveCurriculumScheduler(CurriculumScheduler):
    """
    Adaptive curriculum that adjusts based on training dynamics

    If validation loss increases, slow down curriculum progression
    If training is stable, can accelerate
    """

    def __init__(
        self,
        num_epochs: int,
        warmup_epochs: int = 5,
        full_curriculum_epochs: int = 20,
        patience: int = 3
    ):
        super().__init__(num_epochs, warmup_epochs, full_curriculum_epochs)
        self.patience = patience
        self.val_loss_history = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

    def update(self, val_loss: float) -> bool:
        """
        Update curriculum based on validation loss

        Args:
            val_loss: Current validation loss

        Returns:
            Whether to proceed to next stage (True) or stay in current (False)
        """
        self.val_loss_history.append(val_loss)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            return True  # Proceed normally
        else:
            self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                print(f"WARNING: Validation loss hasn't improved for {self.patience} epochs")
                print(f"  Slowing down curriculum progression")
                return False  # Hold at current stage

            return True

    def get_loss_weights(self, epoch: int) -> Dict[str, float]:
        """
        Get loss weights with adaptive adjustment

        If training is unstable, hold at current stage longer

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of loss weights
        """
        # If training is unstable, use weights from earlier epoch
        if self.epochs_without_improvement >= self.patience:
            effective_epoch = max(0, epoch - self.patience)
        else:
            effective_epoch = epoch

        return super().get_loss_weights(effective_epoch)
