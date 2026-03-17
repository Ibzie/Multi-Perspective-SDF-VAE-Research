#!/usr/bin/env python3
"""
Quantitative Model Comparison

Computes comprehensive metrics for all three models:
- Reconstruction Loss (MSE)
- Train/Test Loss Gap (Generalization)
- Latent Space Statistics
- Per-sample reconstruction quality

Usage:
    python experiments/quantitative_comparison.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json

from src.models import MultiPerspectiveSDFVAE, VanillaVAE, BetaVAE
from src.data import get_celeba_loaders


def load_model(model_type, checkpoint_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model_type == 'sdf_vae':
        model = MultiPerspectiveSDFVAE(
            image_size=64,
            in_channels=3,
            latent_dim=128,
            num_observers=5
        )
    elif model_type == 'vanilla':
        model = VanillaVAE(
            image_size=64,
            in_channels=3,
            latent_dim=128
        )
    elif model_type == 'beta':
        model = BetaVAE(
            image_size=64,
            in_channels=3,
            latent_dim=128,
            beta=4.0
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint


def compute_reconstruction_loss(model, loader, device, desc="Computing"):
    """Compute average reconstruction loss on dataset"""
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(loader, desc=desc, leave=False)):
            images = images.to(device)
            batch_size = images.size(0)

            # Forward pass
            output = model(images)
            reconstruction = output['reconstruction']

            # MSE loss
            loss = F.mse_loss(reconstruction, images, reduction='sum')
            total_loss += loss.item()
            total_samples += batch_size

    return total_loss / total_samples


def compute_latent_statistics(model, loader, device, desc="Computing latent stats"):
    """Compute latent space statistics"""
    all_means = []
    all_logvars = []

    with torch.no_grad():
        for images, _ in tqdm(loader, desc=desc, leave=False):
            images = images.to(device)

            # Encode
            output = model(images)
            all_means.append(output['mean'].cpu())
            all_logvars.append(output['logvar'].cpu())

    all_means = torch.cat(all_means, dim=0)
    all_logvars = torch.cat(all_logvars, dim=0)

    stats = {
        'mean_norm': torch.norm(all_means, dim=1).mean().item(),
        'mean_std': all_means.std(dim=0).mean().item(),
        'logvar_mean': all_logvars.mean().item(),
        'logvar_std': all_logvars.std().item(),
    }

    return stats


def compute_sdf_statistics(model, loader, device):
    """Compute SDF-specific statistics (only for SDF-VAE)"""
    all_sdf_values = []
    all_attention_weights = []

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Computing SDF stats", leave=False):
            images = images.to(device)

            output = model(images)
            all_sdf_values.append(output['sdf_values'].cpu())
            all_attention_weights.append(output['attention_weights'].cpu())

    all_sdf_values = torch.cat(all_sdf_values, dim=0)  # [N, num_observers, 1]
    all_attention_weights = torch.cat(all_attention_weights, dim=0)  # [N, num_observers, 1]

    stats = {
        'sdf_mean': all_sdf_values.mean().item(),
        'sdf_std': all_sdf_values.std().item(),
        'sdf_abs_mean': all_sdf_values.abs().mean().item(),
        'attention_entropy': compute_attention_entropy(all_attention_weights),
        'attention_max_mean': all_attention_weights.max(dim=1)[0].mean().item(),
        'attention_min_mean': all_attention_weights.min(dim=1)[0].mean().item(),
    }

    return stats


def compute_attention_entropy(attention_weights):
    """Compute average entropy of attention distributions"""
    # attention_weights: [N, num_observers, 1]
    probs = attention_weights.squeeze(-1) + 1e-10  # [N, num_observers]
    entropy = -(probs * torch.log(probs)).sum(dim=1).mean()
    return entropy.item()


def evaluate_model(model, model_type, train_loader, val_loader, test_loader, device):
    """Comprehensive evaluation of a single model"""
    print(f"\nEvaluating model...")

    results = {}

    # 1. Reconstruction losses
    print("  Computing reconstruction losses...")
    results['train_recon_loss'] = compute_reconstruction_loss(
        model, train_loader, device, desc="  Train set"
    )
    results['val_recon_loss'] = compute_reconstruction_loss(
        model, val_loader, device, desc="  Val set"
    )
    results['test_recon_loss'] = compute_reconstruction_loss(
        model, test_loader, device, desc="  Test set"
    )

    # Generalization gap
    results['train_test_gap'] = abs(results['test_recon_loss'] - results['train_recon_loss'])
    results['generalization_ratio'] = results['test_recon_loss'] / results['train_recon_loss']

    # 2. Latent space statistics
    print("  Computing latent statistics...")
    latent_stats = compute_latent_statistics(model, test_loader, device)
    results.update(latent_stats)

    # 3. SDF-specific statistics (only for SDF-VAE)
    if model_type == 'sdf_vae':
        print("  Computing SDF statistics...")
        sdf_stats = compute_sdf_statistics(model, test_loader, device)
        results.update(sdf_stats)

    return results


def format_results_table(all_results):
    """Format results as a nice table"""
    lines = []
    lines.append("=" * 100)
    lines.append("QUANTITATIVE MODEL COMPARISON")
    lines.append("=" * 100)
    lines.append("")

    # Reconstruction Quality
    lines.append("-" * 100)
    lines.append("RECONSTRUCTION QUALITY (MSE Loss - Lower is Better)")
    lines.append("-" * 100)
    lines.append(f"{'Metric':<30} {'SDF-VAE':>20} {'Vanilla VAE':>20} {'β-VAE':>20}")
    lines.append("-" * 100)

    metrics = ['train_recon_loss', 'val_recon_loss', 'test_recon_loss']
    labels = ['Training Set', 'Validation Set', 'Test Set']

    for metric, label in zip(metrics, labels):
        sdf = all_results['sdf_vae'].get(metric, 0)
        vanilla = all_results['vanilla'].get(metric, 0)
        beta = all_results['beta'].get(metric, 0)
        lines.append(f"{label:<30} {sdf:>20.6f} {vanilla:>20.6f} {beta:>20.6f}")

    lines.append("")

    # Generalization
    lines.append("-" * 100)
    lines.append("GENERALIZATION (Lower is Better)")
    lines.append("-" * 100)
    lines.append(f"{'Metric':<30} {'SDF-VAE':>20} {'Vanilla VAE':>20} {'β-VAE':>20}")
    lines.append("-" * 100)

    for metric, label in [('train_test_gap', 'Train-Test Gap (Absolute)'),
                          ('generalization_ratio', 'Test/Train Ratio')]:
        sdf = all_results['sdf_vae'].get(metric, 0)
        vanilla = all_results['vanilla'].get(metric, 0)
        beta = all_results['beta'].get(metric, 0)
        lines.append(f"{label:<30} {sdf:>20.6f} {vanilla:>20.6f} {beta:>20.6f}")

    lines.append("")

    # Latent Space Statistics
    lines.append("-" * 100)
    lines.append("LATENT SPACE STATISTICS")
    lines.append("-" * 100)
    lines.append(f"{'Metric':<30} {'SDF-VAE':>20} {'Vanilla VAE':>20} {'β-VAE':>20}")
    lines.append("-" * 100)

    latent_metrics = [
        ('mean_norm', 'Mean Latent Norm'),
        ('mean_std', 'Latent Std Dev'),
        ('logvar_mean', 'Avg Log-Variance'),
        ('logvar_std', 'Log-Variance Std'),
    ]

    for metric, label in latent_metrics:
        sdf = all_results['sdf_vae'].get(metric, 0)
        vanilla = all_results['vanilla'].get(metric, 0)
        beta = all_results['beta'].get(metric, 0)
        lines.append(f"{label:<30} {sdf:>20.6f} {vanilla:>20.6f} {beta:>20.6f}")

    lines.append("")

    # SDF-specific statistics
    if 'sdf_mean' in all_results['sdf_vae']:
        lines.append("-" * 100)
        lines.append("SDF-VAE SPECIFIC STATISTICS")
        lines.append("-" * 100)

        sdf_metrics = [
            ('sdf_mean', 'SDF Mean Value'),
            ('sdf_std', 'SDF Std Dev'),
            ('sdf_abs_mean', 'SDF Absolute Mean'),
            ('attention_entropy', 'Attention Entropy'),
            ('attention_max_mean', 'Max Attention Weight'),
            ('attention_min_mean', 'Min Attention Weight'),
        ]

        for metric, label in sdf_metrics:
            value = all_results['sdf_vae'].get(metric, 0)
            lines.append(f"{label:<50} {value:>20.6f}")

        lines.append("")

    lines.append("=" * 100)
    return "\n".join(lines)


def main():
    print("=" * 100)
    print("QUANTITATIVE MODEL COMPARISON")
    print("=" * 100)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data
    print("Loading data loaders...")
    train_loader, val_loader, test_loader = get_celeba_loaders(
        data_root='./data',
        batch_size=32,
        num_workers=4,
        image_size=64
    )
    print()

    # Model configurations
    models_config = [
        ('SDF-VAE', 'sdf_vae',
         'results/checkpoints/sdf_vae_celeba/best.pt'),
        ('Vanilla VAE', 'vanilla',
         'results/checkpoints/vanilla_vae_celeba/best.pt'),
        ('β-VAE', 'beta',
         'results/checkpoints/beta_vae_celeba/best.pt'),
    ]

    # Evaluate all models
    all_results = {}

    for name, model_type, checkpoint_path in models_config:
        print("=" * 100)
        print(f"Evaluating: {name}")
        print("=" * 100)

        # Load model
        print(f"Loading checkpoint: {checkpoint_path}")
        model, checkpoint = load_model(model_type, checkpoint_path, device)

        # Show training info
        if 'epoch' in checkpoint:
            print(f"  Trained for {checkpoint['epoch']} epochs")
        if 'best_val_loss' in checkpoint:
            print(f"  Best validation loss: {checkpoint['best_val_loss']:.6f}")

        # Evaluate
        results = evaluate_model(
            model, model_type, train_loader, val_loader, test_loader, device
        )
        all_results[model_type] = results

        print(f"\n✓ {name} evaluation complete")

    # Format and display results
    print("\n")
    results_table = format_results_table(all_results)
    print(results_table)

    # Save results
    output_dir = Path('experiments/results/comparisons')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / 'quantitative_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved JSON results: {json_path}")

    # Save text report
    txt_path = output_dir / 'quantitative_results.txt'
    with open(txt_path, 'w') as f:
        f.write(results_table)
    print(f"✓ Saved text report: {txt_path}")

    print("\n" + "=" * 100)
    print("COMPARISON COMPLETE!")
    print("=" * 100)


if __name__ == '__main__':
    main()
