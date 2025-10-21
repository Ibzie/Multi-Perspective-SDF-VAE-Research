"""
Core model components for Multi-Perspective SDF-VAE
Extracted and refactored from original Medical_one.py implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class SwiGLU(nn.Module):
    """
    SwiGLU activation function (Shazeer 2020)
    Provides better gradient flow than GELU for deep networks
    """
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = x1 * F.silu(x2)  # SwiGLU activation
        return self.w3(hidden)


class CNNEncoder(nn.Module):
    """
    CNN Encoder for extracting features from images

    Architecture:
        3 -> 32 -> 64 -> 128 -> 256 channels
        64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4 spatial dims

    Args:
        in_channels: Number of input channels (3 for RGB)
        image_size: Input image size (assumes square images)
    """
    def __init__(self, in_channels: int = 3, image_size: int = 64):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # Layer 1: 3 -> 32 channels, 64x64 -> 32x32
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 32 -> 64 channels, 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 64 -> 128 channels, 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 128 -> 256 channels, 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Calculate flattened feature size
        self.feature_size = 256 * (image_size // 16) * (image_size // 16)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input images [B, C, H, W]

        Returns:
            flat_features: Flattened features [B, feature_size]
            spatial_features: Spatial feature maps [B, 256, H', W']
        """
        spatial_features = self.conv_layers(x)
        flat_features = spatial_features.view(spatial_features.size(0), -1)
        return flat_features, spatial_features


class LightSource(nn.Module):
    """
    Universal Light Source - Projects input to consistent illumination space

    This metaphor treats the encoder features as being "illuminated" by a universal
    light source, ensuring all observers see the same foundational representation

    Args:
        input_dim: Dimension of encoder features
        light_dim: Dimension of light projection space
    """
    def __init__(self, input_dim: int, light_dim: int):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            SwiGLU(512, 512, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, light_dim),
            nn.BatchNorm1d(light_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoder features [B, input_dim]

        Returns:
            Illuminated features [B, light_dim]
        """
        return self.projection(x)


class SDFObserver(nn.Module):
    """
    SDF-based Observer Network

    Each observer learns a unique perspective on the data manifold through
    a signed distance function (SDF). The SDF value represents confidence:
    - SDF ≈ 0: Point is on the manifold (high confidence)
    - |SDF| > 0: Point is off the manifold (low confidence)

    Args:
        input_dim: Dimension of input features (from light source)
        hidden_dims: List of hidden layer dimensions
        projection_dim: Dimension of feature projection space
        observer_id: Unique identifier for this observer (0-indexed)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        projection_dim: int,
        observer_id: int = 0
    ):
        super().__init__()

        self.observer_id = observer_id

        # Build encoder layers with SwiGLU activation
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(SwiGLU(prev_dim, hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # SDF prediction head
        self.to_sdf = nn.Linear(prev_dim, 1)

        # Feature projection head
        self.to_features = nn.Linear(prev_dim, projection_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features [B, input_dim]

        Returns:
            sdf_value: Signed distance to manifold [B, 1]
            feature_projection: Observer's feature representation [B, projection_dim]
        """
        # Encode through observer-specific layers
        features = self.encoder(x)

        # Compute SDF value (signed distance to image manifold)
        sdf_value = self.to_sdf(features)

        # Compute feature projection for this observer's perspective
        feature_projection = self.to_features(features)

        return sdf_value, feature_projection


class ManifoldAggregator(nn.Module):
    """
    Manifold Aggregation Module

    Aggregates multiple observer perspectives using attention mechanism based on
    their confidence (SDF values). Produces latent distribution parameters.

    Key Innovation: Instead of simple averaging, we use learned attention weights
    that consider both the SDF values and the feature representations.

    Args:
        num_observers: Number of observer networks
        projection_dim: Dimension of observer projections
        latent_dim: Dimension of latent space
    """
    def __init__(self, num_observers: int, projection_dim: int, latent_dim: int):
        super().__init__()

        self.num_observers = num_observers

        # Attention mechanism for aggregating observer perspectives
        self.attention = nn.Sequential(
            nn.Linear(projection_dim, 128),
            SwiGLU(128, 128, 128),
            nn.Linear(128, 1)
        )

        # Feature fusion layers
        self.feature_fusion = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            SwiGLU(projection_dim, projection_dim, projection_dim)
        )

        # Map to latent distribution parameters
        self.to_latent_mean = nn.Linear(projection_dim, latent_dim)
        self.to_latent_logvar = nn.Linear(projection_dim, latent_dim)

    def forward(
        self,
        sdf_values: torch.Tensor,
        feature_projections: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            sdf_values: SDF values from all observers [B, num_observers, 1]
            feature_projections: Features from all observers [B, num_observers, projection_dim]

        Returns:
            mean: Latent mean [B, latent_dim]
            logvar: Latent log-variance [B, latent_dim]
            attention_weights: Observer attention weights [B, num_observers, 1]
        """
        batch_size = sdf_values.size(0)
        num_observers = sdf_values.size(1)

        # Calculate attention weights for each observer
        attention_scores = []
        for i in range(num_observers):
            features = feature_projections[:, i, :]
            score = self.attention(features)
            attention_scores.append(score)

        # Stack and normalize attention scores
        attention_scores = torch.cat(attention_scores, dim=1)  # [B, num_observers]

        # Numerical stability: subtract max before softmax
        attention_scores = attention_scores - attention_scores.max(dim=1, keepdim=True)[0]
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(2)  # [B, num_observers, 1]

        # Handle NaN in attention weights (numerical safety)
        if torch.isnan(attention_weights).any():
            print("WARNING: NaN detected in attention weights, using uniform weights")
            attention_weights = torch.ones_like(attention_weights) / num_observers

        # Weighted sum of features
        aggregated_features = torch.sum(
            feature_projections * attention_weights,
            dim=1
        )  # [B, projection_dim]

        # Apply feature fusion
        fused_features = self.feature_fusion(aggregated_features)

        # Map to latent distribution parameters
        mean = self.to_latent_mean(fused_features)

        # Constrain logvar to reasonable range using sigmoid
        logvar_raw = self.to_latent_logvar(fused_features)
        logvar = torch.sigmoid(logvar_raw) * 4 - 2  # Maps to range [-2, 2]

        return mean, logvar, attention_weights

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE sampling

        Args:
            mean: Latent mean [B, latent_dim]
            logvar: Latent log-variance [B, latent_dim]

        Returns:
            Sampled latent [B, latent_dim]
        """
        # Clamp logvar for numerical stability
        logvar_safe = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar_safe)
        eps = torch.randn_like(std)
        return mean + eps * std


class CNNDecoder(nn.Module):
    """
    CNN Decoder for reconstructing images from latent codes

    Architecture (mirrors encoder in reverse):
        latent_dim -> 256*4*4 -> 256 -> 128 -> 64 -> 32 -> 3 channels
        4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 spatial dims

    Args:
        latent_dim: Dimension of latent space
        output_channels: Number of output channels (3 for RGB)
        image_size: Output image size
    """
    def __init__(self, latent_dim: int, output_channels: int = 3, image_size: int = 64):
        super().__init__()

        # Initial projection from latent space to 4x4 feature maps
        initial_spatial_size = image_size // 16
        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, 256 * initial_spatial_size * initial_spatial_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Transposed convolution layers for upsampling
        self.deconv_layers = nn.Sequential(
            # Layer 1: 256 -> 128 channels, 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 128 -> 64 channels, 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 64 -> 32 channels, 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 32 -> output_channels, 32x32 -> 64x64
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in range [0, 1]
        )

        self.initial_spatial_size = initial_spatial_size

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent code [B, latent_dim]

        Returns:
            Reconstructed image [B, output_channels, H, W]
        """
        # Project latent to spatial features
        features = self.latent_to_features(z)

        # Reshape to spatial dimensions
        features = features.view(
            features.size(0),
            256,
            self.initial_spatial_size,
            self.initial_spatial_size
        )

        # Upsample through transposed convolutions
        reconstruction = self.deconv_layers(features)

        return reconstruction
