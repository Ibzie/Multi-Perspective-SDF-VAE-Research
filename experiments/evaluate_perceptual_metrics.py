"""
Perceptual Metrics Evaluation for Multi-Perspective SDF-VAE

Computes advanced perceptual quality metrics:
- FID (Fréchet Inception Distance): Measures distribution similarity using Inception features
- LPIPS (Learned Perceptual Image Patch Similarity): Neural perceptual distance
- SSIM (Structural Similarity Index): Structural similarity measure

Also generates individual and combined visualization figures.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.sdf_vae import MultiPerspectiveSDFVAE
from src.models.baselines import VanillaVAE, BetaVAE
from src.data.datasets import get_fashion_mnist_loaders


class InceptionV3FeatureExtractor(torch.nn.Module):
    """Extracts features from InceptionV3 for FID computation."""

    def __init__(self, device='cuda'):
        super().__init__()
        # Load pretrained InceptionV3
        inception = torchvision.models.inception_v3(pretrained=True, transform_input=False)
        inception.eval()

        # Extract feature layers (before final classification)
        self.feature_extractor = torch.nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            torch.nn.MaxPool2d(3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            torch.nn.MaxPool2d(3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )

        self.to(device)
        self.device = device

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Extract features from images.

        Args:
            x: Tensor of shape (B, 3, H, W) in range [0, 1]

        Returns:
            features: Tensor of shape (B, 2048)
        """
        # Resize to 299x299 (InceptionV3 input size)
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        # Normalize to [-1, 1] (ImageNet normalization)
        x = 2 * x - 1

        features = self.feature_extractor(x)
        return features.squeeze(-1).squeeze(-1)


class LPIPSMetric(torch.nn.Module):
    """LPIPS perceptual similarity metric using VGG features."""

    def __init__(self, device='cuda'):
        super().__init__()
        # Use VGG16 as perceptual feature extractor
        vgg = torchvision.models.vgg16(pretrained=True).features

        # Use layers: conv1_2, conv2_2, conv3_3, conv4_3, conv5_3
        self.slice1 = torch.nn.Sequential(*list(vgg[:4]))   # conv1_2
        self.slice2 = torch.nn.Sequential(*list(vgg[4:9]))  # conv2_2
        self.slice3 = torch.nn.Sequential(*list(vgg[9:16])) # conv3_3
        self.slice4 = torch.nn.Sequential(*list(vgg[16:23])) # conv4_3
        self.slice5 = torch.nn.Sequential(*list(vgg[23:30])) # conv5_3

        self.to(device)
        self.device = device

        for param in self.parameters():
            param.requires_grad = False

        # Learned linear weights (initialized to equal weighting)
        self.weights = [0.125, 0.25, 0.25, 0.25, 0.125]

    def normalize_features(self, x, eps=1e-10):
        """Normalize feature maps across spatial dimensions."""
        norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        return x / (norm_factor + eps)

    def forward(self, x, y):
        """Compute LPIPS distance between x and y.

        Args:
            x, y: Tensors of shape (B, 3, H, W) in range [0, 1]

        Returns:
            lpips_dist: Scalar LPIPS distance
        """
        # Normalize to ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        x = (x - mean) / std
        y = (y - mean) / std

        # Extract features from multiple layers
        h_x = x
        h_y = y

        dists = []

        for i, (slice_fn, weight) in enumerate(zip(
            [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5],
            self.weights
        )):
            h_x = slice_fn(h_x)
            h_y = slice_fn(h_y)

            # Normalize features
            h_x_norm = self.normalize_features(h_x)
            h_y_norm = self.normalize_features(h_y)

            # Compute L2 distance
            diff = (h_x_norm - h_y_norm) ** 2
            dist = weight * diff.mean(dim=[1, 2, 3])
            dists.append(dist)

        return torch.stack(dists).sum(dim=0).mean()


def calculate_fid(real_features, generated_features):
    """Calculate Fréchet Inception Distance between real and generated features.

    Args:
        real_features: (N, 2048) array of features from real images
        generated_features: (N, 2048) array of features from generated images

    Returns:
        fid: FID score (lower is better)
    """
    # Calculate mean and covariance
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)

    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2)

    # Calculate sqrt of product of covariances
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    # Check for imaginary numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)

    return fid


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """Calculate SSIM between two images.

    Args:
        img1, img2: Tensors of shape (B, C, H, W) in range [0, 1]
        window_size: Size of Gaussian window
        size_average: If True, return mean SSIM, else return per-image SSIM

    Returns:
        ssim: SSIM score (higher is better, max 1.0)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
                          for x in range(window_size)])
    window = gauss / gauss.sum()
    window = window.unsqueeze(1)
    window = window.mm(window.t()).float().unsqueeze(0).unsqueeze(0)
    window = window.to(img1.device)

    channel = img1.size(1)
    window = window.repeat(channel, 1, 1, 1)

    # Calculate means
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Calculate variances
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=[1, 2, 3])


def evaluate_model_perceptual(model, test_loader, device='cuda', model_name='Model'):
    """Evaluate perceptual metrics for a single model.

    Args:
        model: VAE model
        test_loader: DataLoader for test set
        device: Device to run on
        model_name: Name for logging

    Returns:
        results: Dict with FID, LPIPS, SSIM scores
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")

    model.eval()

    # Initialize feature extractors
    inception = InceptionV3FeatureExtractor(device=device)
    lpips_metric = LPIPSMetric(device=device)

    real_features_list = []
    recon_features_list = []
    lpips_scores = []
    ssim_scores = []

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= 100:  # Limit to 100 batches for speed
                break

            images = images.to(device)

            # Get reconstructions - all models return dictionaries
            output = model(images)
            recon = output['reconstruction']

            # Clamp to [0, 1]
            images_clamped = torch.clamp(images, 0, 1)
            recon_clamped = torch.clamp(recon, 0, 1)

            # Extract Inception features for FID
            real_feat = inception(images_clamped)
            recon_feat = inception(recon_clamped)

            real_features_list.append(real_feat.cpu().numpy())
            recon_features_list.append(recon_feat.cpu().numpy())

            # Compute LPIPS
            lpips_score = lpips_metric(images_clamped, recon_clamped)
            lpips_scores.append(lpips_score.item())

            # Compute SSIM
            ssim_score = calculate_ssim(images_clamped, recon_clamped)
            ssim_scores.append(ssim_score.item())

            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1} batches...")

    # Compute FID
    real_features = np.concatenate(real_features_list, axis=0)
    recon_features = np.concatenate(recon_features_list, axis=0)
    fid_score = calculate_fid(real_features, recon_features)

    # Average LPIPS and SSIM
    lpips_avg = np.mean(lpips_scores)
    ssim_avg = np.mean(ssim_scores)

    results = {
        'FID': fid_score,
        'LPIPS': lpips_avg,
        'SSIM': ssim_avg
    }

    print(f"\nResults for {model_name}:")
    print(f"  FID:   {fid_score:.4f} (lower is better)")
    print(f"  LPIPS: {lpips_avg:.4f} (lower is better)")
    print(f"  SSIM:  {ssim_avg:.4f} (higher is better, max 1.0)")

    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load test data
    print("\nLoading Fashion-MNIST test set...")
    _, _, test_loader = get_fashion_mnist_loaders(
        batch_size=64,
        data_root='./data',
        image_size=64,
        num_workers=4
    )

    # Model configurations
    models_config = {
        'SDF-VAE': {
            'class': MultiPerspectiveSDFVAE,
            'kwargs': {'image_size': 64, 'in_channels': 3, 'latent_dim': 128, 'num_observers': 5},
            'checkpoint': 'experiments/results/checkpoints/sdf_vae_fashion_latent128_obs5/checkpoint_epoch_0099.pt'
        },
        'Vanilla VAE': {
            'class': VanillaVAE,
            'kwargs': {'image_size': 64, 'in_channels': 3, 'latent_dim': 128},
            'checkpoint': 'experiments/results/checkpoints/vanilla_vae_fashion_latent128/best.pt'
        },
        'β-VAE': {
            'class': BetaVAE,
            'kwargs': {'image_size': 64, 'in_channels': 3, 'latent_dim': 128, 'beta': 4.0},
            'checkpoint': 'experiments/results/checkpoints/beta_vae_fashion_beta4.0_latent128/best.pt'
        }
    }

    # Evaluate all models
    all_results = {}

    for model_name, config in models_config.items():
        # Load model
        model = config['class'](**config['kwargs']).to(device)
        checkpoint_path = Path(__file__).parent.parent / config['checkpoint']

        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"WARNING: Checkpoint not found: {checkpoint_path}")
            continue

        # Evaluate
        results = evaluate_model_perceptual(model, test_loader, device, model_name)
        all_results[model_name] = results

    # Save results
    output_dir = Path(__file__).parent / 'results' / 'perceptual_metrics'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    with open(output_dir / 'perceptual_metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Save formatted text
    with open(output_dir / 'perceptual_metrics.txt', 'w') as f:
        f.write("Perceptual Quality Metrics Comparison\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"{'Model':<15} {'FID':<12} {'LPIPS':<12} {'SSIM':<12}\n")
        f.write("-" * 60 + "\n")

        for model_name, results in all_results.items():
            f.write(f"{model_name:<15} "
                   f"{results['FID']:<12.4f} "
                   f"{results['LPIPS']:<12.4f} "
                   f"{results['SSIM']:<12.4f}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Lower is better for FID and LPIPS\n")
        f.write("Higher is better for SSIM (max 1.0)\n")

    print(f"\n{'='*60}")
    print("Results saved to:")
    print(f"  - {output_dir / 'perceptual_metrics.json'}")
    print(f"  - {output_dir / 'perceptual_metrics.txt'}")
    print(f"{'='*60}\n")

    return all_results


if __name__ == '__main__':
    results = main()
