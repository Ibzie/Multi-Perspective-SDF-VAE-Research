#!/usr/bin/env python3
"""
Evaluate Out-of-Distribution (OOD) Detection using SDF Scores

This script evaluates how well different VAE models can detect OOD samples.
We use Fashion-MNIST as in-distribution and MNIST as out-of-distribution.

For SDF-VAE, we use:
1. Mean absolute SDF value (|SDF| should be high for OOD)
2. Latent reconstruction error
3. Latent variance (uncertainty)

For baselines, we use:
1. Reconstruction error
2. Latent variance
3. KL divergence

Metrics:
- AUROC (Area Under ROC Curve)
- AUPR (Area Under Precision-Recall Curve)
- FPR@95TPR (False Positive Rate at 95% True Positive Rate)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import json
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt

from src.models import MultiPerspectiveSDFVAE, VanillaVAE, BetaVAE
from src.data import get_fashion_mnist_loaders
import torchvision
import torchvision.transforms as transforms


def get_mnist_loader(batch_size=64, image_size=64, num_workers=4):
    """Load MNIST dataset (OOD for Fashion-MNIST trained models)"""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader


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


def compute_ood_scores(model, loader, device, model_type='sdf_vae'):
    """
    Compute OOD detection scores for a dataset.

    Returns dict with different scoring methods.
    """
    recon_errors = []
    latent_vars = []
    kl_divs = []
    sdf_scores = []

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Computing scores"):
            images = images.to(device)
            batch_size = images.size(0)

            # Forward pass
            output = model(images)
            recon = output['reconstruction']
            mu = output['mean']
            logvar = output['logvar']

            # 1. Reconstruction error (higher = more likely OOD)
            recon_error = F.mse_loss(recon, images, reduction='none')
            recon_error = recon_error.view(batch_size, -1).mean(dim=1)

            # 2. Latent variance (uncertainty - higher = more likely OOD)
            variance = torch.exp(logvar)
            mean_variance = variance.mean(dim=1)

            # 3. KL divergence (higher = more likely OOD)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

            recon_errors.append(recon_error.cpu())
            latent_vars.append(mean_variance.cpu())
            kl_divs.append(kl.cpu())

            # 4. SDF-based score (only for SDF-VAE)
            if model_type == 'sdf_vae' and 'sdf_values' in output:
                sdf_values = output['sdf_values']  # [B, num_observers, 1]
                # Use mean absolute SDF across observers
                mean_abs_sdf = torch.abs(sdf_values).mean(dim=1).squeeze()  # [B]
                sdf_scores.append(mean_abs_sdf.cpu())

    recon_errors = torch.cat(recon_errors).numpy()
    latent_vars = torch.cat(latent_vars).numpy()
    kl_divs = torch.cat(kl_divs).numpy()

    scores = {
        'recon_error': recon_errors,
        'latent_var': latent_vars,
        'kl_div': kl_divs,
    }

    if sdf_scores:
        scores['sdf_score'] = torch.cat(sdf_scores).numpy()

    return scores


def compute_metrics(in_scores, ood_scores):
    """
    Compute OOD detection metrics.

    Args:
        in_scores: Scores for in-distribution samples (lower = more in-dist)
        ood_scores: Scores for OOD samples (higher = more OOD)

    Returns:
        Dictionary of metrics
    """
    # Create labels (0 = in-dist, 1 = OOD)
    y_true = np.concatenate([
        np.zeros(len(in_scores)),
        np.ones(len(ood_scores))
    ])

    # Concatenate scores
    y_scores = np.concatenate([in_scores, ood_scores])

    # Compute AUROC
    auroc = roc_auc_score(y_true, y_scores)

    # Compute AUPR
    aupr = average_precision_score(y_true, y_scores)

    # Compute FPR@95TPR
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fpr_at_95tpr = fpr[np.argmax(tpr >= 0.95)]

    return {
        'auroc': float(auroc),
        'aupr': float(aupr),
        'fpr_at_95tpr': float(fpr_at_95tpr)
    }


def main():
    print("=" * 70)
    print("OUT-OF-DISTRIBUTION DETECTION EVALUATION")
    print("=" * 70)
    print("In-Distribution:  Fashion-MNIST")
    print("Out-of-Distribution: MNIST")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")

    # Load data
    print("Loading datasets...")
    _, _, fashion_test_loader = get_fashion_mnist_loaders(
        batch_size=64,
        data_root='./data',
        image_size=64,
        num_workers=4
    )
    mnist_loader = get_mnist_loader(batch_size=64, image_size=64, num_workers=4)

    print(f"Fashion-MNIST test samples: {len(fashion_test_loader.dataset)}")
    print(f"MNIST test samples: {len(mnist_loader.dataset)}")

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
        print(f"\n{'=' * 70}")
        print(f"Evaluating {model_name}")
        print(f"{'=' * 70}")

        # Load model
        model, device = load_model(config['path'], config['type'])

        # Compute scores for in-distribution (Fashion-MNIST)
        print("\nComputing scores for in-distribution (Fashion-MNIST)...")
        in_scores = compute_ood_scores(model, fashion_test_loader, device, config['type'])

        # Compute scores for OOD (MNIST)
        print("Computing scores for OOD (MNIST)...")
        ood_scores = compute_ood_scores(model, mnist_loader, device, config['type'])

        # Compute metrics for each scoring method
        model_results = {}

        for score_name in in_scores.keys():
            print(f"\n  Scoring method: {score_name}")
            metrics = compute_metrics(in_scores[score_name], ood_scores[score_name])

            print(f"    AUROC:       {metrics['auroc']:.4f}")
            print(f"    AUPR:        {metrics['aupr']:.4f}")
            print(f"    FPR@95TPR:   {metrics['fpr_at_95tpr']:.4f}")

            model_results[score_name] = metrics

        # Store scores for visualization
        model_results['in_scores'] = {k: v.tolist() for k, v in in_scores.items()}
        model_results['ood_scores'] = {k: v.tolist() for k, v in ood_scores.items()}

        results[model_name] = model_results

    # Save results
    output_dir = Path('experiments/results/ood_detection')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results (with score arrays)
    with open(output_dir / 'ood_detection_full.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save summary (without large arrays)
    summary = {}
    for model_name, data in results.items():
        summary[model_name] = {
            k: v for k, v in data.items()
            if k not in ['in_scores', 'ood_scores']
        }

    with open(output_dir / 'ood_detection_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 70)
    print("OOD DETECTION COMPARISON (AUROC)")
    print("=" * 70)

    # Determine which scores to compare
    all_score_types = set()
    for model_results in results.values():
        all_score_types.update([k for k in model_results.keys() if k not in ['in_scores', 'ood_scores']])

    for score_type in sorted(all_score_types):
        print(f"\n{score_type.upper()}:")
        print(f"{'Model':<15} {'AUROC':<10} {'AUPR':<10} {'FPR@95TPR':<10}")
        print("-" * 70)

        for model_name in results.keys():
            if score_type in results[model_name]:
                metrics = results[model_name][score_type]
                print(f"{model_name:<15} {metrics['auroc']:<10.4f} {metrics['aupr']:<10.4f} {metrics['fpr_at_95tpr']:<10.4f}")

    print("\n" + "=" * 70)
    print("Results saved to:")
    print(f"  - {output_dir / 'ood_detection_summary.json'}")
    print(f"  - {output_dir / 'ood_detection_full.json'}")
    print("=" * 70)


if __name__ == '__main__':
    main()
