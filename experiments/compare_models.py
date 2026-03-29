#!/usr/bin/env python3
"""
Compare model outputs visually

Runs identical inference on all three models:
- Multi-Perspective SDF-VAE
- Vanilla VAE
- β-VAE

Generates:
1. Reconstruction comparison (same test images)
2. Random sampling comparison
3. Latent space interpolation comparison

Usage:
    python experiments/compare_models.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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

    return model


def get_test_images(test_loader, num_images=8, device='cuda'):
    """Get fixed set of test images"""
    images, labels = next(iter(test_loader))
    return images[:num_images].to(device), labels[:num_images]


def reconstruct_images(model, images):
    """Reconstruct images with model"""
    with torch.no_grad():
        output = model(images)
        return output['reconstruction']


def sample_images(model, num_samples, device):
    """Generate random samples"""
    with torch.no_grad():
        return model.sample(num_samples, device)


def interpolate_latent(model, images, num_steps=8):
    """Interpolate between two images in latent space"""
    with torch.no_grad():
        # Encode two images
        if isinstance(model, MultiPerspectiveSDFVAE):
            output = model(images[:2])
            z1, z2 = output['z'][0], output['z'][1]
        else:
            mean, logvar = model.encode(images[:2])
            z1, z2 = mean[0], mean[1]

        # Interpolate
        alphas = torch.linspace(0, 1, num_steps, device=images.device)
        interpolations = []

        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2

            if isinstance(model, MultiPerspectiveSDFVAE):
                img = model.decode(z_interp.unsqueeze(0))
            else:
                img = model.decode(z_interp.unsqueeze(0))

            interpolations.append(img[0])

        return torch.stack(interpolations)


def tensor_to_image(tensor):
    """Convert tensor to displayable image"""
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return img


def create_comparison_figure(original, reconstructions, model_names, save_path):
    """Create side-by-side reconstruction comparison"""
    num_images = original.shape[0]
    num_models = len(reconstructions)

    fig, axes = plt.subplots(num_images, num_models + 1, figsize=(3 * (num_models + 1), 3 * num_images))

    if num_images == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_images):
        # Original
        axes[i, 0].imshow(tensor_to_image(original[i]))
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=14, fontweight='bold')

        # Reconstructions
        for j, (recon, name) in enumerate(zip(reconstructions, model_names)):
            axes[i, j + 1].imshow(tensor_to_image(recon[i]))
            axes[i, j + 1].axis('off')
            if i == 0:
                axes[i, j + 1].set_title(name, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_sampling_figure(samples, model_names, save_path):
    """Create random sampling comparison"""
    num_samples = samples[0].shape[0]
    num_models = len(samples)

    fig, axes = plt.subplots(num_models, num_samples, figsize=(2 * num_samples, 2 * num_models))

    if num_models == 1:
        axes = axes.reshape(1, -1)

    for i, (sample, name) in enumerate(zip(samples, model_names)):
        for j in range(num_samples):
            axes[i, j].imshow(tensor_to_image(sample[j]))
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(name, fontsize=12, fontweight='bold', rotation=0,
                                      ha='right', va='center', labelpad=40)

    plt.suptitle('Random Samples from Latent Space', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_interpolation_figure(interpolations, model_names, save_path):
    """Create latent interpolation comparison"""
    num_steps = interpolations[0].shape[0]
    num_models = len(interpolations)

    fig, axes = plt.subplots(num_models, num_steps, figsize=(1.5 * num_steps, 2 * num_models))

    if num_models == 1:
        axes = axes.reshape(1, -1)

    for i, (interp, name) in enumerate(zip(interpolations, model_names)):
        for j in range(num_steps):
            axes[i, j].imshow(tensor_to_image(interp[j]))
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(name, fontsize=12, fontweight='bold', rotation=0,
                                      ha='right', va='center', labelpad=40)

    plt.suptitle('Latent Space Interpolation', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("Model Comparison - Visual Inference")
    print("=" * 70)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data
    print("Loading test data...")
    _, _, test_loader = get_celeba_loaders(
        data_root='./data',
        batch_size=16,
        num_workers=4,
        image_size=64
    )

    # Get fixed test images
    test_images, test_labels = get_test_images(test_loader, num_images=8, device=device)
    print(f"✓ Loaded {test_images.shape[0]} test images\n")

    # Model configurations
    models_config = [
        ('Multi-Perspective SDF-VAE', 'sdf_vae',
         'results/checkpoints/sdf_vae_celeba/best.pt'),
        ('Vanilla VAE', 'vanilla',
         'results/checkpoints/vanilla_vae_celeba/best.pt'),
        ('β-VAE (β=4.0)', 'beta',
         'results/checkpoints/beta_vae_celeba/best.pt'),
    ]

    # Load all models
    print("Loading trained models...")
    models = []
    model_names = []

    for name, model_type, checkpoint_path in models_config:
        print(f"  Loading {name}...")
        model = load_model(model_type, checkpoint_path, device)
        models.append(model)
        model_names.append(name)

    print("✓ All models loaded\n")

    # Create output directory
    output_dir = Path('experiments/results/comparisons')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reconstruction comparison
    print("=" * 70)
    print("1. Generating reconstructions...")
    print("=" * 70)

    reconstructions = []
    for model, name in zip(models, model_names):
        print(f"  {name}...")
        recon = reconstruct_images(model, test_images)
        reconstructions.append(recon)

    create_comparison_figure(
        test_images,
        reconstructions,
        model_names,
        output_dir / 'reconstruction_comparison.png'
    )
    print()

    # 2. Random sampling comparison
    print("=" * 70)
    print("2. Generating random samples...")
    print("=" * 70)

    samples = []
    num_samples = 8
    for model, name in zip(models, model_names):
        print(f"  {name}...")
        sample = sample_images(model, num_samples, device)
        samples.append(sample)

    create_sampling_figure(
        samples,
        model_names,
        output_dir / 'sampling_comparison.png'
    )
    print()

    # 3. Latent interpolation comparison
    print("=" * 70)
    print("3. Generating latent interpolations...")
    print("=" * 70)

    interpolations = []
    num_steps = 8
    for model, name in zip(models, model_names):
        print(f"  {name}...")
        interp = interpolate_latent(model, test_images, num_steps)
        interpolations.append(interp)

    create_interpolation_figure(
        interpolations,
        model_names,
        output_dir / 'interpolation_comparison.png'
    )
    print()

    # Summary
    print("=" * 70)
    print("COMPARISON COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print(f"  1. reconstruction_comparison.png - Side-by-side reconstructions")
    print(f"  2. sampling_comparison.png - Random samples from latent space")
    print(f"  3. interpolation_comparison.png - Latent space interpolations")
    print("=" * 70)


if __name__ == '__main__':
    main()
