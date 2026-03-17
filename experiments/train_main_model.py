#!/usr/bin/env python3
"""
Train Multi-Perspective SDF-VAE on Fashion-MNIST or Medical MNIST

This is the main experiment script for training the Multi-Perspective SDF-VAE
with curriculum learning on either Fashion-MNIST or Medical MNIST datasets.

Example usage:
    python experiments/train_main_model.py --dataset fashion --epochs 100
    python experiments/train_main_model.py --dataset medical --epochs 50
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
import argparse

from src.models import MultiPerspectiveSDFVAE
from src.training import Trainer
from src.data import get_fashion_mnist_loaders, get_medical_mnist_loaders, get_celeba_loaders


def parse_args():
    parser = argparse.ArgumentParser(description='Train Multi-Perspective SDF-VAE')

    # Dataset
    parser.add_argument('--dataset', type=str, default='fashion',
                        choices=['fashion', 'medical', 'celeba'],
                        help='Dataset to use (fashion, medical, or celeba)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for datasets')

    # Model architecture
    parser.add_argument('--image_size', type=int, default=64,
                        help='Image size (default: 64)')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent space dimension (default: 128)')
    parser.add_argument('--light_dim', type=int, default=256,
                        help='Light source dimension (default: 256)')
    parser.add_argument('--projection_dim', type=int, default=256,
                        help='Observer projection dimension (default: 256)')
    parser.add_argument('--num_observers', type=int, default=5,
                        help='Number of observers (default: 5)')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Curriculum warmup epochs (default: 5)')

    # Logging & checkpointing
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (default: auto-generated)')
    parser.add_argument('--checkpoint_dir', type=str, default='./results/checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./results/logs',
                        help='Log directory')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log metrics every N steps')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')

    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Auto-generate experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"sdf_vae_{args.dataset}_latent{args.latent_dim}_obs{args.num_observers}"

    print("=" * 70)
    print("Multi-Perspective SDF-VAE Training")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Num observers: {args.num_observers}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    if args.dataset == 'fashion':
        train_loader, val_loader, test_loader = get_fashion_mnist_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            augment_train=True
        )
    elif args.dataset == 'medical':
        train_loader, val_loader, test_loader = get_medical_mnist_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            augment_train=True
        )
    else:  # celeba
        train_loader, val_loader, test_loader = get_celeba_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            augment_train=True
        )

    # Create model
    print("\nCreating model...")
    model = MultiPerspectiveSDFVAE(
        image_size=args.image_size,
        in_channels=3,
        latent_dim=args.latent_dim,
        light_dim=args.light_dim,
        projection_dim=args.projection_dim,
        num_observers=args.num_observers
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name
    )

    # Train
    print("\nStarting training...\n")
    trainer.train(
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        early_stopping_patience=20
    )

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")
    print(f"Logs saved to: {trainer.metrics_logger.log_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
