"""
Baseline VAE models for comparison

Implements:
1. Vanilla VAE (standard variational autoencoder)
2. β-VAE (with adjustable β for disentanglement)

These baselines use the same encoder/decoder architecture as Multi-Perspective SDF-VAE
for fair comparison, but without the multi-observer mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .components import CNNEncoder, CNNDecoder


class VanillaVAE(nn.Module):
    """
    Standard Variational Autoencoder

    Simple baseline with encoder-decoder architecture and KL divergence regularization.
    No multi-perspective observers, no SDF, just basic VAE.

    Args:
        image_size: Size of input images
        in_channels: Number of input channels
        latent_dim: Dimension of latent space
    """

    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 3,
        latent_dim: int = 128
    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = CNNEncoder(in_channels, image_size)
        encoder_output_dim = self.encoder.feature_size

        # Latent projection
        self.fc_mean = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, latent_dim)

        # Decoder
        self.decoder = CNNDecoder(latent_dim, in_channels, image_size)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images to latent distribution parameters

        Args:
            x: Input images [B, C, H, W]

        Returns:
            mean: Latent mean [B, latent_dim]
            logvar: Latent log-variance [B, latent_dim]
        """
        features, _ = self.encoder(x)
        mean = self.fc_mean(features)
        logvar = self.fc_logvar(features)
        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick

        Args:
            mean: Latent mean [B, latent_dim]
            logvar: Latent log-variance [B, latent_dim]

        Returns:
            Sampled latent [B, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes to images

        Args:
            z: Latent codes [B, latent_dim]

        Returns:
            Reconstructed images [B, C, H, W]
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Dictionary containing reconstruction, mean, logvar, z
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decode(z)

        return {
            'reconstruction': reconstruction,
            'mean': mean,
            'logvar': logvar,
            'z': z
        }

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample random images

        Args:
            num_samples: Number of samples
            device: Device to generate on

        Returns:
            Generated images [num_samples, C, H, W]
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        with torch.no_grad():
            images = self.decode(z)
        return images


class BetaVAE(VanillaVAE):
    """
    β-VAE for disentangled representations

    Same as Vanilla VAE but with adjustable β weight on KL divergence.
    Higher β encourages more disentangled latent representations.

    Higgins et al. 2017: "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"

    Args:
        image_size: Size of input images
        in_channels: Number of input channels
        latent_dim: Dimension of latent space
        beta: Weight on KL divergence (default: 4.0, as in original paper)
    """

    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 3,
        latent_dim: int = 128,
        beta: float = 4.0
    ):
        super().__init__(image_size, in_channels, latent_dim)
        self.beta = beta

    def get_beta(self) -> float:
        """Get current beta value"""
        return self.beta

    def set_beta(self, beta: float):
        """Update beta value"""
        self.beta = beta


class BaselineVAELoss(nn.Module):
    """
    Loss function for baseline VAE models

    Loss = Reconstruction + β * KL

    Args:
        beta: Weight on KL divergence (1.0 for Vanilla VAE, >1.0 for β-VAE)
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss

        Args:
            x: Original images [B, C, H, W]
            reconstruction: Reconstructed images [B, C, H, W]
            mean: Latent mean [B, latent_dim]
            logvar: Latent log-variance [B, latent_dim]

        Returns:
            Dictionary of loss components
        """
        batch_size = x.size(0)

        # Reconstruction loss (BCE)
        x_flat = x.view(batch_size, -1)
        recon_flat = reconstruction.view(batch_size, -1)

        # Clamp for numerical stability
        recon_flat = torch.clamp(recon_flat, min=1e-7, max=1-1e-7)
        recon_loss = F.binary_cross_entropy(recon_flat, x_flat, reduction='sum')
        recon_loss = recon_loss / batch_size

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        kl_loss = kl_loss / batch_size

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'kl': kl_loss
        }

    def update_beta(self, beta: float):
        """Update beta weight"""
        self.beta = beta
