#!/usr/bin/env python3
"""
Test Generalization Capabilities - SDF Geometry Advantage

Tests that leverage the SDF geometry to demonstrate superior generalization:
1. Robustness to noise (Gaussian, salt-and-pepper)
2. Robustness to occlusion (random masking)
3. Latent space interpolation quality (dissimilar classes)
4. Manifold distance estimation (SDF as confidence measure)
5. Extrapolation capability

The hypothesis: Multi-Perspective SDF-VAE should outperform baselines on these
tests because the SDF geometry provides:
- Robust distance metrics to the data manifold
- Multi-perspective views that capture different aspects
- Geometric constraints that regularize the latent space

Usage:
    python experiments/test_generalization.py
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
from src.data import get_fashion_mnist_loaders


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


def add_gaussian_noise(images, noise_levels=[0.1, 0.3, 0.5, 0.7]):
    """Add Gaussian noise at different levels"""
    noisy_images = []
    for level in noise_levels:
        noise = torch.randn_like(images) * level
        noisy = torch.clamp(images + noise, 0, 1)
        noisy_images.append(noisy)
    return noisy_images, noise_levels


def add_occlusion(images, occlusion_ratios=[0.2, 0.4, 0.6, 0.8]):
    """Add random occlusion (masking)"""
    occluded_images = []
    for ratio in occlusion_ratios:
        occluded = images.clone()
        B, C, H, W = images.shape

        # Random rectangular occlusion
        for i in range(B):
            mask_h = int(H * ratio)
            mask_w = int(W * ratio)
            y = torch.randint(0, H - mask_h + 1, (1,)).item()
            x = torch.randint(0, W - mask_w + 1, (1,)).item()
            occluded[i, :, y:y+mask_h, x:x+mask_w] = 0

        occluded_images.append(occluded)
    return occluded_images, occlusion_ratios


def compute_reconstruction_error(model, images):
    """Compute reconstruction error (MSE)"""
    with torch.no_grad():
        output = model(images)
        recon = output['reconstruction']
        mse = F.mse_loss(recon, images, reduction='none').mean(dim=[1, 2, 3])
        return mse.mean().item(), mse.std().item()


def test_noise_robustness(models, model_names, test_images):
    """Test 1: Robustness to Gaussian noise"""
    print("\n" + "=" * 70)
    print("TEST 1: ROBUSTNESS TO GAUSSIAN NOISE")
    print("=" * 70)
    print("Hypothesis: SDF geometry should provide more robust reconstructions")
    print("under noise corruption.\n")

    noisy_images, noise_levels = add_gaussian_noise(test_images)

    results = {name: {'mean': [], 'std': []} for name in model_names}

    # Clean images baseline
    print("Noise Level    " + "  ".join([f"{name:>20}" for name in model_names]))
    print("-" * 70)

    for name, model in zip(model_names, models):
        mean_err, std_err = compute_reconstruction_error(model, test_images)
        results[name]['mean'].append(mean_err)
        results[name]['std'].append(std_err)

    print(f"Clean (0.0)    " + "  ".join([f"{results[name]['mean'][0]:>20.2f}" for name in model_names]))

    # Noisy images
    for i, (noisy, level) in enumerate(zip(noisy_images, noise_levels)):
        for name, model in zip(model_names, models):
            mean_err, std_err = compute_reconstruction_error(model, noisy)
            results[name]['mean'].append(mean_err)
            results[name]['std'].append(std_err)

        print(f"Noise {level:.1f}      " + "  ".join([f"{results[name]['mean'][i+1]:>20.2f}" for name in model_names]))

    print("\n✓ Lower reconstruction error = Better noise robustness")
    return results, noisy_images


def test_occlusion_robustness(models, model_names, test_images):
    """Test 2: Robustness to occlusion"""
    print("\n" + "=" * 70)
    print("TEST 2: ROBUSTNESS TO OCCLUSION")
    print("=" * 70)
    print("Hypothesis: SDF geometry should enable better reconstruction from")
    print("partial observations.\n")

    occluded_images, occlusion_ratios = add_occlusion(test_images)

    results = {name: {'mean': [], 'std': []} for name in model_names}

    print("Occlusion %    " + "  ".join([f"{name:>20}" for name in model_names]))
    print("-" * 70)

    # Clean images baseline
    for name, model in zip(model_names, models):
        mean_err, std_err = compute_reconstruction_error(model, test_images)
        results[name]['mean'].append(mean_err)
        results[name]['std'].append(std_err)

    print(f"None (0%)      " + "  ".join([f"{results[name]['mean'][0]:>20.2f}" for name in model_names]))

    # Occluded images
    for i, (occluded, ratio) in enumerate(zip(occluded_images, occlusion_ratios)):
        for name, model in zip(model_names, models):
            mean_err, std_err = compute_reconstruction_error(model, occluded)
            results[name]['mean'].append(mean_err)
            results[name]['std'].append(std_err)

        print(f"{ratio*100:>4.0f}%          " + "  ".join([f"{results[name]['mean'][i+1]:>20.2f}" for name in model_names]))

    print("\n✓ Lower reconstruction error = Better occlusion robustness")
    return results, occluded_images


def test_cross_class_interpolation(models, model_names, test_loader, device):
    """Test 3: Interpolation between dissimilar classes"""
    print("\n" + "=" * 70)
    print("TEST 3: CROSS-CLASS INTERPOLATION QUALITY")
    print("=" * 70)
    print("Hypothesis: SDF geometry should enable smoother interpolation")
    print("between dissimilar classes.\n")

    # Find images from different classes
    all_images = []
    all_labels = []
    for images, labels in test_loader:
        all_images.append(images)
        all_labels.append(labels)
        if len(all_images) * images.shape[0] > 1000:
            break

    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)

    # Select pairs of dissimilar classes
    # Fashion-MNIST: 0=T-shirt, 1=Trouser, 2=Pullover, 3=Dress, 4=Coat,
    #                5=Sandal, 6=Shirt, 7=Sneaker, 8=Bag, 9=Ankle boot
    dissimilar_pairs = [
        (0, 1),  # T-shirt to Trouser
        (5, 2),  # Sandal to Pullover
        (7, 8),  # Sneaker to Bag
        (3, 9),  # Dress to Ankle boot
    ]

    interpolations = {name: [] for name in model_names}

    for class1, class2 in dissimilar_pairs:
        # Get one image from each class
        idx1 = (all_labels == class1).nonzero()[0].item()
        idx2 = (all_labels == class2).nonzero()[0].item()

        img1 = all_images[idx1:idx1+1].to(device)
        img2 = all_images[idx2:idx2+1].to(device)

        for name, model in zip(model_names, models):
            with torch.no_grad():
                # Encode
                if isinstance(model, MultiPerspectiveSDFVAE):
                    out1 = model(img1)
                    out2 = model(img2)
                    z1, z2 = out1['z'], out2['z']
                else:
                    mean1, _ = model.encode(img1)
                    mean2, _ = model.encode(img2)
                    z1, z2 = mean1, mean2

                # Interpolate
                alphas = torch.linspace(0, 1, 8, device=device)
                interp = []
                for alpha in alphas:
                    z = (1 - alpha) * z1 + alpha * z2
                    img = model.decode(z)
                    interp.append(img[0])

                interpolations[name].append(torch.stack(interp))

    print(f"Generated interpolations for {len(dissimilar_pairs)} dissimilar class pairs")
    print("Visual inspection required - check generated images for smoothness")

    return interpolations, dissimilar_pairs


def test_sdf_confidence(model, test_images, device):
    """Test 4: SDF as confidence/anomaly score (SDF-VAE only)"""
    print("\n" + "=" * 70)
    print("TEST 4: SDF-BASED CONFIDENCE ESTIMATION")
    print("=" * 70)
    print("Hypothesis: SDF values should correlate with distance to data manifold")
    print("(lower SDF = closer to manifold = higher confidence)\n")

    if not isinstance(model, MultiPerspectiveSDFVAE):
        print("⚠️  Only applicable to Multi-Perspective SDF-VAE\n")
        return None

    with torch.no_grad():
        output = model(test_images, return_metrics=True)
        sdf_values = output['sdf_values']  # [B, num_observers, 1]

        # Average SDF across observers
        mean_sdf = sdf_values.mean(dim=1).squeeze()  # [B]
        min_sdf = sdf_values.min(dim=1)[0].squeeze()  # [B]
        max_sdf = sdf_values.max(dim=1)[0].squeeze()  # [B]

        # Reconstruction error
        recon = output['reconstruction']
        recon_err = F.mse_loss(recon, test_images, reduction='none').mean(dim=[1, 2, 3])

        # Correlation between SDF and reconstruction error
        correlation = torch.corrcoef(torch.stack([mean_sdf, recon_err]))[0, 1].item()

        print(f"Mean SDF value: {mean_sdf.mean().item():.4f} ± {mean_sdf.std().item():.4f}")
        print(f"Min SDF value:  {min_sdf.mean().item():.4f} ± {min_sdf.std().item():.4f}")
        print(f"Max SDF value:  {max_sdf.mean().item():.4f} ± {max_sdf.std().item():.4f}")
        print(f"\nCorrelation (SDF vs Recon Error): {correlation:.4f}")
        print("✓ Positive correlation = SDF predicts reconstruction difficulty")

    return {
        'mean_sdf': mean_sdf.cpu(),
        'min_sdf': min_sdf.cpu(),
        'max_sdf': max_sdf.cpu(),
        'recon_err': recon_err.cpu(),
        'correlation': correlation
    }


def visualize_noise_robustness(original, noisy_images, reconstructions, noise_levels, save_path):
    """Visualize noise robustness results"""
    num_noise = len(noise_levels)
    num_models = len(reconstructions)

    fig, axes = plt.subplots(num_noise + 1, num_models + 2, figsize=(3 * (num_models + 2), 2 * (num_noise + 1)))

    model_names = list(reconstructions.keys())

    # Clean image row
    img = original[0].cpu().permute(1, 2, 0).numpy()
    axes[0, 0].imshow(np.clip(img, 0, 1))
    axes[0, 0].set_title('Original', fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].text(0.5, 0.5, 'Clean\nImage', ha='center', va='center', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    for j, name in enumerate(model_names):
        # reconstructions[name][0] is the batch tensor [B, C, H, W]
        recon = reconstructions[name][0][0].cpu().permute(1, 2, 0).numpy()
        axes[0, j + 2].imshow(np.clip(recon, 0, 1))
        axes[0, j + 2].set_title(name, fontweight='bold', fontsize=10)
        axes[0, j + 2].axis('off')

    # Noisy image rows
    for i, (noisy, level) in enumerate(zip(noisy_images, noise_levels), start=1):
        img = noisy[0].cpu().permute(1, 2, 0).numpy()
        axes[i, 0].imshow(np.clip(img, 0, 1))
        axes[i, 0].axis('off')

        axes[i, 1].text(0.5, 0.5, f'Noise\n{level:.1f}', ha='center', va='center', fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')

        for j, name in enumerate(model_names):
            # reconstructions[name][i] is the batch tensor [B, C, H, W]
            recon = reconstructions[name][i][0].cpu().permute(1, 2, 0).numpy()
            axes[i, j + 2].imshow(np.clip(recon, 0, 1))
            axes[i, j + 2].axis('off')

    plt.suptitle('Noise Robustness Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("GENERALIZATION TESTS - SDF Geometry Advantage")
    print("=" * 70)
    print("Testing hypothesis: Multi-Perspective SDF-VAE generalizes better")
    print("due to geometric constraints from SDF learning.")
    print()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data
    print("Loading test data...")
    _, _, test_loader = get_fashion_mnist_loaders(
        data_root='./data',
        batch_size=32,
        num_workers=4,
        image_size=64
    )

    # Get test images
    test_images, test_labels = next(iter(test_loader))
    test_images = test_images.to(device)
    print(f"✓ Loaded {test_images.shape[0]} test images\n")

    # Model configurations
    models_config = [
        ('Multi-Perspective SDF-VAE', 'sdf_vae',
         'results/checkpoints/sdf_vae_fashion_latent128_obs5/best.pt'),
        ('Vanilla VAE', 'vanilla',
         'results/checkpoints/vanilla_vae_fashion_latent128/best.pt'),
        ('β-VAE (β=4.0)', 'beta',
         'results/checkpoints/beta_vae_fashion_beta4.0_latent128/best.pt'),
    ]

    # Load models
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
    output_dir = Path('results/generalization_tests')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run tests
    test_results = {}

    # Test 1: Noise robustness
    noise_results, noisy_images = test_noise_robustness(models, model_names, test_images)
    test_results['noise'] = noise_results

    # Get reconstructions for visualization
    noise_reconstructions = {name: [] for name in model_names}

    # Clean reconstructions
    with torch.no_grad():
        for name, model in zip(model_names, models):
            recon = model(test_images)['reconstruction']
            noise_reconstructions[name].append(recon)

    # Noisy reconstructions
    for noisy in noisy_images:
        with torch.no_grad():
            for name, model in zip(model_names, models):
                recon = model(noisy)['reconstruction']
                noise_reconstructions[name].append(recon)

    visualize_noise_robustness(test_images, noisy_images, noise_reconstructions,
                               [0.1, 0.3, 0.5, 0.7], output_dir / 'noise_robustness.png')

    # Test 2: Occlusion robustness
    occlusion_results, occluded_images = test_occlusion_robustness(models, model_names, test_images)
    test_results['occlusion'] = occlusion_results

    # Test 3: Cross-class interpolation
    interpolations, pairs = test_cross_class_interpolation(models, model_names, test_loader, device)
    test_results['interpolation'] = interpolations

    # Test 4: SDF confidence (SDF-VAE only)
    sdf_results = test_sdf_confidence(models[0], test_images, device)
    test_results['sdf_confidence'] = sdf_results

    # Summary
    print("\n" + "=" * 70)
    print("GENERALIZATION TESTS COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nKey Findings:")
    print("  1. Noise robustness: Check reconstruction errors under noise")
    print("  2. Occlusion robustness: Check reconstruction from partial views")
    print("  3. Interpolation quality: Visual inspection required")
    print("  4. SDF confidence: Correlation with reconstruction error")
    print("=" * 70)


if __name__ == '__main__':
    main()
