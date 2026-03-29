#!/usr/bin/env python3
"""
Train baseline VAE models (Vanilla VAE and β-VAE)

This script trains baseline models for comparison with Multi-Perspective SDF-VAE.

Example usage:
    python experiments/train_baselines.py --model vanilla --dataset fashion
    python experiments/train_baselines.py --model beta --beta 4.0 --dataset medical
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from src.models import VanillaVAE, BetaVAE, BaselineVAELoss
from src.data import get_fashion_mnist_loaders, get_medical_mnist_loaders, get_celeba_loaders
from src.utils import MetricsLogger


def parse_args():
    parser = argparse.ArgumentParser(description='Train Baseline VAE Models')

    # Model
    parser.add_argument('--model', type=str, default='vanilla',
                        choices=['vanilla', 'beta'],
                        help='Model type (vanilla or beta)')
    parser.add_argument('--beta', type=float, default=4.0,
                        help='Beta value for β-VAE (default: 4.0)')

    # Dataset
    parser.add_argument('--dataset', type=str, default='fashion',
                        choices=['fashion', 'medical', 'celeba'],
                        help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for datasets')

    # Architecture
    parser.add_argument('--image_size', type=int, default=64,
                        help='Image size')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent dimension')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')

    # Logging
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--checkpoint_dir', type=str, default='./results/checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./results/logs',
                        help='Log directory')

    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """Train for one epoch"""
    model.train()

    epoch_losses = {'total': 0.0, 'reconstruction': 0.0, 'kl': 0.0}

    for images, _ in tqdm(train_loader, desc="Training"):
        images = images.to(device)

        # Forward
        output = model(images)

        # Loss
        losses = loss_fn(
            x=images,
            reconstruction=output['reconstruction'],
            mean=output['mean'],
            logvar=output['logvar']
        )

        # Backward
        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate (map 'total' key)
        epoch_losses['total'] += losses['total'].item()
        epoch_losses['reconstruction'] += losses['reconstruction'].item()
        epoch_losses['kl'] += losses['kl'].item()

    # Average
    num_batches = len(train_loader)
    for key in epoch_losses.keys():
        epoch_losses[key] /= num_batches

    return epoch_losses


def validate_epoch(model, val_loader, loss_fn, device):
    """Validate for one epoch"""
    model.eval()

    epoch_losses = {'total': 0.0, 'reconstruction': 0.0, 'kl': 0.0}

    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc="Validating"):
            images = images.to(device)

            # Forward
            output = model(images)

            # Loss
            losses = loss_fn(
                x=images,
                reconstruction=output['reconstruction'],
                mean=output['mean'],
                logvar=output['logvar']
            )

            # Accumulate (map 'total' key)
            epoch_losses['total'] += losses['total'].item()
            epoch_losses['reconstruction'] += losses['reconstruction'].item()
            epoch_losses['kl'] += losses['kl'].item()

    # Average
    num_batches = len(val_loader)
    for key in epoch_losses.keys():
        epoch_losses[key] /= num_batches

    return epoch_losses


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Auto-generate experiment name
    if args.experiment_name is None:
        if args.model == 'beta':
            args.experiment_name = f"beta_vae_{args.dataset}_beta{args.beta}_latent{args.latent_dim}"
        else:
            args.experiment_name = f"vanilla_vae_{args.dataset}_latent{args.latent_dim}"

    print("=" * 70)
    print(f"Training {args.model.upper()} VAE")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Experiment: {args.experiment_name}")
    if args.model == 'beta':
        print(f"Beta: {args.beta}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    if args.dataset == 'fashion':
        train_loader, val_loader, test_loader = get_fashion_mnist_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size
        )
    elif args.dataset == 'celeba':
        train_loader, val_loader, test_loader = get_celeba_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size
        )
    else:
        train_loader, val_loader, test_loader = get_medical_mnist_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size
        )

    # Create model
    print("\nCreating model...")
    if args.model == 'beta':
        model = BetaVAE(
            image_size=args.image_size,
            in_channels=3,
            latent_dim=args.latent_dim,
            beta=args.beta
        )
        loss_fn = BaselineVAELoss(beta=args.beta)
    else:
        model = VanillaVAE(
            image_size=args.image_size,
            in_channels=3,
            latent_dim=args.latent_dim
        )
        loss_fn = BaselineVAELoss(beta=1.0)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Metrics logger
    metrics_logger = MetricsLogger(args.log_dir, args.experiment_name)

    # Checkpoint dir
    checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device)

        # Validate
        val_metrics = validate_epoch(model, val_loader, loss_fn, device)

        # Log
        metrics_logger.log_epoch(epoch, {
            'train_loss': train_metrics['total'],
            'val_loss': val_metrics['total'],
            'train_recon': train_metrics['reconstruction'],
            'val_recon': val_metrics['reconstruction'],
            'train_kl': train_metrics['kl'],
            'val_kl': val_metrics['kl']
        })

        print(f"Train Loss: {train_metrics['total']:.6f}")
        print(f"Val Loss:   {val_metrics['total']:.6f}")
        print(f"Gap:        {val_metrics['total'] - train_metrics['total']:.6f}")

        # Save best
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total']
            }, checkpoint_dir / 'best.pt')
            print("✓ New best model saved!")

        # LR schedule
        scheduler.step(val_metrics['total'])

    # Save metrics
    metrics_logger.save()
    metrics_logger.save_json()

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
