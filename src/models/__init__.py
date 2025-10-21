"""
Multi-Perspective SDF-VAE Models

This package contains all model components:
- components: Individual building blocks (encoder, decoder, observers, etc.)
- sdf_vae: Main Multi-Perspective SDF-VAE model
- baselines: Baseline VAE models for comparison
- losses: Loss functions for training
"""

from .components import (
    SwiGLU,
    CNNEncoder,
    LightSource,
    SDFObserver,
    ManifoldAggregator,
    CNNDecoder
)

from .sdf_vae import MultiPerspectiveSDFVAE
from .baselines import VanillaVAE, BetaVAE, BaselineVAELoss
from .losses import MultiPerspectiveSDFVAELoss, safe_bce

__all__ = [
    # Components
    'SwiGLU',
    'CNNEncoder',
    'LightSource',
    'SDFObserver',
    'ManifoldAggregator',
    'CNNDecoder',
    # Main model
    'MultiPerspectiveSDFVAE',
    # Baselines
    'VanillaVAE',
    'BetaVAE',
    'BaselineVAELoss',
    # Losses
    'MultiPerspectiveSDFVAELoss',
    'safe_bce',
]
