"""
Multi-Perspective SDF-VAE Model

Main model class that combines all components and provides the complete
Multi-Perspective SDF-VAE architecture with metric collection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from .components import (
    CNNEncoder,
    LightSource,
    SDFObserver,
    ManifoldAggregator,
    CNNDecoder
)


class MultiPerspectiveSDFVAE(nn.Module):
    """
    Multi-Perspective SDF-VAE

    A variational autoencoder that uses multiple "observer" networks to learn
    different perspectives on the data manifold. Each observer:
    1. Learns a signed distance function (SDF) to the manifold
    2. Produces a unique feature representation
    3. Contributes to the final latent code through attention-weighted aggregation

    Key innovations:
    - SDF-based confidence weighting
    - Multi-perspective learning with diversity enforcement
    - Interpretable latent space through geometric structure

    Args:
        image_size: Size of input images (assumes square)
        in_channels: Number of input channels (3 for RGB)
        latent_dim: Dimension of latent space
        light_dim: Dimension of light source projection
        projection_dim: Dimension of observer feature projections
        num_observers: Number of observer networks
        observer_hidden_dims: Hidden layer dimensions for observers
    """

    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 3,
        latent_dim: int = 128,
        light_dim: int = 256,
        projection_dim: int = 256,
        num_observers: int = 5,
        observer_hidden_dims: list[int] = None
    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_observers = num_observers

        # Default observer hidden dimensions if not specified
        if observer_hidden_dims is None:
            observer_hidden_dims = [512, 256, 128]

        # 1. CNN Encoder - Extracts features from images
        self.cnn_encoder = CNNEncoder(in_channels, image_size)
        encoder_output_dim = self.cnn_encoder.feature_size

        # 2. Light Source - Projects features to consistent space
        self.light_source = LightSource(encoder_output_dim, light_dim)

        # 3. Multi-Perspective Observers - Each learns unique SDF and features
        self.observers = nn.ModuleList([
            SDFObserver(
                input_dim=light_dim,
                hidden_dims=observer_hidden_dims,
                projection_dim=projection_dim,
                observer_id=i
            )
            for i in range(num_observers)
        ])

        # 4. Manifold Aggregator - Combines observer perspectives
        self.aggregator = ManifoldAggregator(
            num_observers=num_observers,
            projection_dim=projection_dim,
            latent_dim=latent_dim
        )

        # 5. CNN Decoder - Reconstructs images from latent codes
        self.decoder = CNNDecoder(
            latent_dim=latent_dim,
            output_channels=in_channels,
            image_size=image_size
        )

    def forward(
        self,
        x: torch.Tensor,
        return_metrics: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Multi-Perspective SDF-VAE

        Args:
            x: Input images [B, C, H, W]
            return_metrics: Whether to return detailed metrics for analysis

        Returns:
            Dictionary containing:
                - reconstruction: Reconstructed images [B, C, H, W]
                - mean: Latent mean [B, latent_dim]
                - logvar: Latent log-variance [B, latent_dim]
                - z: Sampled latent code [B, latent_dim]
                - sdf_values: Per-observer SDF values [B, num_observers, 1]
                - feature_projections: Per-observer features [B, num_observers, projection_dim]
                - attention_weights: Observer attention weights [B, num_observers, 1]
                - metrics: (if return_metrics=True) Detailed analysis metrics
        """
        batch_size = x.size(0)

        # 1. Encode image to features
        encoder_features, spatial_features = self.cnn_encoder(x)

        # 2. Project through light source
        light_projection = self.light_source(encoder_features)

        # 3. Forward through all observers
        sdf_values = []
        feature_projections = []

        for observer in self.observers:
            sdf, features = observer(light_projection)
            sdf_values.append(sdf)
            feature_projections.append(features)

        # Stack observer outputs
        sdf_values = torch.stack(sdf_values, dim=1)  # [B, num_observers, 1]
        feature_projections = torch.stack(feature_projections, dim=1)  # [B, num_observers, projection_dim]

        # 4. Aggregate observer perspectives
        mean, logvar, attention_weights = self.aggregator(sdf_values, feature_projections)

        # 5. Sample from latent distribution (reparameterization trick)
        z = self.aggregator.reparameterize(mean, logvar)

        # 6. Decode to reconstruction
        reconstruction = self.decoder(z)

        # Prepare output dictionary
        output = {
            'reconstruction': reconstruction,
            'mean': mean,
            'logvar': logvar,
            'z': z,
            'sdf_values': sdf_values,
            'feature_projections': feature_projections,
            'attention_weights': attention_weights,
            'encoder_features': encoder_features,
            'spatial_features': spatial_features
        }

        # Add detailed metrics if requested
        if return_metrics:
            output['metrics'] = self._compute_metrics(output, x)

        return output

    def _compute_metrics(
        self,
        forward_output: Dict[str, torch.Tensor],
        x: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute detailed metrics for analysis

        Args:
            forward_output: Output from forward pass
            x: Input images

        Returns:
            Dictionary of metrics
        """
        with torch.no_grad():
            # Extract values
            sdf_values = forward_output['sdf_values']
            attention_weights = forward_output['attention_weights']
            mean = forward_output['mean']
            logvar = forward_output['logvar']
            feature_projections = forward_output['feature_projections']

            batch_size = x.size(0)

            # SDF statistics
            sdf_mean = sdf_values.mean().item()
            sdf_std = sdf_values.std().item()
            sdf_abs_mean = sdf_values.abs().mean().item()

            # Attention statistics
            attention_entropy = self._compute_entropy(attention_weights.squeeze(-1))

            # Latent space statistics
            latent_norm = torch.norm(mean, dim=1).mean().item()
            latent_std = logvar.exp().sqrt().mean().item()

            # Observer diversity (pairwise cosine similarity)
            diversity_scores = []
            for i in range(self.num_observers):
                for j in range(i+1, self.num_observers):
                    feat_i = F.normalize(feature_projections[:, i, :], dim=1)
                    feat_j = F.normalize(feature_projections[:, j, :], dim=1)
                    sim = (feat_i * feat_j).sum(dim=1).abs().mean().item()
                    diversity_scores.append(sim)

            avg_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0

            return {
                'sdf_mean': sdf_mean,
                'sdf_std': sdf_std,
                'sdf_abs_mean': sdf_abs_mean,
                'attention_entropy': attention_entropy,
                'latent_norm': latent_norm,
                'latent_std': latent_std,
                'observer_diversity': 1.0 - avg_diversity,  # Higher = more diverse
            }

    def _compute_entropy(self, probabilities: torch.Tensor) -> float:
        """
        Compute entropy of probability distribution

        Args:
            probabilities: Probability distribution [B, num_observers]

        Returns:
            Mean entropy across batch
        """
        probs = probabilities + 1e-10  # Numerical stability
        entropy = -(probs * torch.log(probs)).sum(dim=1).mean()
        return entropy.item()

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images to latent distribution parameters

        Args:
            x: Input images [B, C, H, W]

        Returns:
            mean: Latent mean [B, latent_dim]
            logvar: Latent log-variance [B, latent_dim]
        """
        output = self.forward(x, return_metrics=False)
        return output['mean'], output['logvar']

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes to images

        Args:
            z: Latent codes [B, latent_dim]

        Returns:
            Reconstructed images [B, C, H, W]
        """
        return self.decoder(z)

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample random images from the model

        Args:
            num_samples: Number of images to generate
            device: Device to generate on

        Returns:
            Generated images [num_samples, C, H, W]
        """
        # Sample from standard normal distribution
        z = torch.randn(num_samples, self.latent_dim, device=device)

        # Decode to images
        with torch.no_grad():
            images = self.decode(z)

        return images

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct images (encode then decode)

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Reconstructed images [B, C, H, W]
        """
        with torch.no_grad():
            output = self.forward(x, return_metrics=False)
            return output['reconstruction']

    def get_observer_contributions(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze individual observer contributions

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Dictionary containing per-observer analysis
        """
        with torch.no_grad():
            output = self.forward(x, return_metrics=False)

            return {
                'attention_weights': output['attention_weights'],  # [B, num_observers, 1]
                'sdf_values': output['sdf_values'],  # [B, num_observers, 1]
                'feature_projections': output['feature_projections'],  # [B, num_observers, projection_dim]
                'observer_ids': list(range(self.num_observers))
            }

    def interpolate_latent(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        num_steps: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interpolate between two images in latent space

        Args:
            x1: First image [1, C, H, W]
            x2: Second image [1, C, H, W]
            num_steps: Number of interpolation steps

        Returns:
            interpolated_images: Interpolated images [num_steps, C, H, W]
            interpolated_latents: Interpolated latent codes [num_steps, latent_dim]
        """
        with torch.no_grad():
            # Encode both images
            mean1, _ = self.encode(x1)
            mean2, _ = self.encode(x2)

            # Create interpolation alphas
            alphas = torch.linspace(0, 1, num_steps, device=x1.device)

            # Interpolate in latent space
            interpolated_latents = []
            for alpha in alphas:
                z_interp = (1 - alpha) * mean1 + alpha * mean2
                interpolated_latents.append(z_interp)

            interpolated_latents = torch.cat(interpolated_latents, dim=0)

            # Decode interpolated latents
            interpolated_images = self.decode(interpolated_latents)

        return interpolated_images, interpolated_latents
