#!/usr/bin/env python3
"""
Evaluate Uncertainty Calibration for VAE Models

This script evaluates how well the predicted uncertainty (latent variance)
correlates with actual reconstruction error. Well-calibrated models should
have high uncertainty when reconstruction error is high, and vice versa.

Metrics:
- Calibration curves (binned uncertainty vs error)
- Expected Calibration Error (ECE)
- Correlation between uncertainty and error
- Reliability diagrams
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.models import MultiPerspectiveSDFVAE, VanillaVAE, BetaVAE
from src.data import get_fashion_mnist_loaders


def load_model(checkpoint_path, model_type='sdf_vae'):
    """Load a trained model from checkpoint"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == 'sdf_vae':
        model = MultiPerspectiveSDFVAE(
            image_size=64,
            in_channels=3,
            latent_dim=128,
            light_dim=256,
            projection_dim=256,
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
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, device


def compute_calibration_metrics(model, test_loader, device, model_type='sdf_vae'):
    """
    Compute calibration metrics for a model.

    Returns:
        uncertainties: Predicted uncertainties (mean of latent variance)
        errors: Actual reconstruction errors (MSE per sample)
        latent_vars: Full latent variance vectors
    """
    uncertainties = []
    errors = []
    latent_vars = []

    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc=f"Evaluating {model_type}"):
            images = images.to(device)
            batch_size = images.size(0)

            # Forward pass
            output = model(images)
            recon = output['reconstruction']
            logvar = output['logvar']

            # Compute per-sample reconstruction error (MSE)
            recon_error = F.mse_loss(recon, images, reduction='none')
            recon_error = recon_error.view(batch_size, -1).mean(dim=1)  # [B]

            # Compute uncertainty (mean of latent variance)
            variance = torch.exp(logvar)  # [B, latent_dim]
            mean_variance = variance.mean(dim=1)  # [B]

            uncertainties.append(mean_variance.cpu())
            errors.append(recon_error.cpu())
            latent_vars.append(variance.cpu())

    uncertainties = torch.cat(uncertainties).numpy()
    errors = torch.cat(errors).numpy()
    latent_vars = torch.cat(latent_vars).numpy()

    return uncertainties, errors, latent_vars


def compute_ece(uncertainties, errors, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).

    Bins samples by predicted uncertainty and compares predicted uncertainty
    to actual average error in each bin.
    """
    # Normalize uncertainties to [0, 1] for binning
    unc_min, unc_max = uncertainties.min(), uncertainties.max()
    uncertainties_norm = (uncertainties - unc_min) / (unc_max - unc_min + 1e-8)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (uncertainties_norm > bin_lower) & (uncertainties_norm <= bin_upper)
        bin_count = in_bin.sum()

        if bin_count > 0:
            # Average predicted uncertainty in bin
            avg_confidence = uncertainties[in_bin].mean()
            # Average actual error in bin
            avg_accuracy = errors[in_bin].mean()

            # ECE contribution (weighted by bin size)
            ece += (bin_count / len(uncertainties)) * abs(avg_confidence - avg_accuracy)

            bin_accs.append(avg_accuracy)
            bin_confs.append(avg_confidence)
            bin_counts.append(bin_count)
        else:
            bin_accs.append(0)
            bin_confs.append(0)
            bin_counts.append(0)

    return ece, bin_accs, bin_confs, bin_counts, bin_boundaries


def main():
    print("=" * 60)
    print("UNCERTAINTY CALIBRATION EVALUATION")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load test data
    print("Loading Fashion-MNIST test set...")
    _, _, test_loader = get_fashion_mnist_loaders(
        batch_size=64,
        data_root='./data',
        image_size=64,
        num_workers=4
    )

    # Model configurations
    models_config = {
        'SDF-VAE': {
            'path': 'experiments/results/checkpoints/sdf_vae_fashion_latent128_obs5/checkpoint_epoch_0099.pt',
            'type': 'sdf_vae'
        },
        'Vanilla VAE': {
            'path': 'experiments/results/checkpoints/vanilla_vae_fashion_latent128/best.pt',
            'type': 'vanilla'
        },
        'β-VAE': {
            'path': 'experiments/results/checkpoints/beta_vae_fashion_beta4.0_latent128/best.pt',
            'type': 'beta'
        }
    }

    results = {}

    for model_name, config in models_config.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating {model_name}")
        print(f"{'=' * 60}")

        # Load model
        model, device = load_model(config['path'], config['type'])

        # Compute calibration metrics
        uncertainties, errors, latent_vars = compute_calibration_metrics(
            model, test_loader, device, config['type']
        )

        # Compute ECE
        ece, bin_accs, bin_confs, bin_counts, bin_boundaries = compute_ece(
            uncertainties, errors, n_bins=10
        )

        # Compute correlation
        correlation, p_value = stats.spearmanr(uncertainties, errors)

        # Compute statistics
        results[model_name] = {
            'ece': float(ece),
            'correlation': float(correlation),
            'correlation_pvalue': float(p_value),
            'mean_uncertainty': float(uncertainties.mean()),
            'std_uncertainty': float(uncertainties.std()),
            'mean_error': float(errors.mean()),
            'std_error': float(errors.std()),
            'bin_accuracies': [float(x) for x in bin_accs],
            'bin_confidences': [float(x) for x in bin_confs],
            'bin_counts': [int(x) for x in bin_counts],
            'bin_boundaries': [float(x) for x in bin_boundaries],
            'uncertainties': uncertainties.tolist(),
            'errors': errors.tolist()
        }

        print(f"\nResults for {model_name}:")
        print(f"  Expected Calibration Error (ECE): {ece:.6f}")
        print(f"  Correlation (Spearman):            {correlation:.4f} (p={p_value:.2e})")
        print(f"  Mean Uncertainty:                  {uncertainties.mean():.6f}")
        print(f"  Std Uncertainty:                   {uncertainties.std():.6f}")
        print(f"  Mean Error:                        {errors.mean():.6f}")
        print(f"  Std Error:                         {errors.std():.6f}")

    # Save results
    output_dir = Path('experiments/results/calibration')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results (with data arrays)
    with open(output_dir / 'calibration_full.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save summary (without large arrays)
    summary = {}
    for model_name, data in results.items():
        summary[model_name] = {
            k: v for k, v in data.items()
            if k not in ['uncertainties', 'errors']
        }

    with open(output_dir / 'calibration_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary comparison
    print("\n" + "=" * 60)
    print("CALIBRATION COMPARISON")
    print("=" * 60)
    print(f"{'Model':<15} {'ECE':<12} {'Correlation':<12} {'Mean Unc.':<12}")
    print("-" * 60)
    for model_name, data in results.items():
        print(f"{model_name:<15} {data['ece']:<12.6f} {data['correlation']:<12.4f} {data['mean_uncertainty']:<12.6f}")

    print("\n" + "=" * 60)
    print("Results saved to:")
    print(f"  - {output_dir / 'calibration_summary.json'}")
    print(f"  - {output_dir / 'calibration_full.json'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
