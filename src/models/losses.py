"""
Loss functions for Multi-Perspective SDF-VAE

Implements the complete loss formulation with all components:
1. Reconstruction Loss (BCE)
2. KL Divergence
3. SDF Consistency Loss
4. Eikonal Regularization
5. Observer Diversity Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def safe_bce(input: torch.Tensor, target: torch.Tensor, reduction: str = 'sum') -> torch.Tensor:
    """
    Numerically stable binary cross entropy

    Args:
        input: Predictions [B, C, H, W] in range [0, 1]
        target: Targets [B, C, H, W] in range [0, 1]
        reduction: 'sum', 'mean', or 'none'

    Returns:
        BCE loss
    """
    # Clamp input to prevent numerical issues
    input_safe = torch.clamp(input, min=1e-7, max=1-1e-7)
    target_safe = torch.clamp(target, min=0.0, max=1.0)

    # Check for NaN/Inf
    if torch.isnan(input_safe).any() or torch.isinf(input_safe).any():
        print("WARNING: NaN or Inf detected in BCE input, replacing with safe values")
        input_safe = torch.where(
            torch.isnan(input_safe) | torch.isinf(input_safe),
            torch.ones_like(input_safe) * 0.5,
            input_safe
        )

    return F.binary_cross_entropy(input_safe, target_safe, reduction=reduction)


class MultiPerspectiveSDFVAELoss(nn.Module):
    """
    Complete loss function for Multi-Perspective SDF-VAE

    Loss = α·Recon + β·KL + γ·SDF_Consistency + δ·Eikonal - ε·Diversity

    Args:
        alpha: Reconstruction loss weight
        beta: KL divergence weight
        gamma: SDF consistency weight
        delta: Eikonal loss weight
        epsilon: Observer diversity weight (negative to maximize diversity)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.2,
        gamma: float = 1.0,  # Increased from 0.1
        delta: float = 0.5,  # Increased from 0.05
        epsilon: float = 0.1,
        zeta: float = 0.5    # NEW: SDF supervision weight
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.zeta = zeta

    def forward(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        sdf_values: torch.Tensor,
        feature_projections: torch.Tensor,
        attention_weights: torch.Tensor,
        encoder_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components

        Args:
            x: Original images [B, C, H, W]
            reconstruction: Reconstructed images [B, C, H, W]
            mean: Latent mean [B, latent_dim]
            logvar: Latent log-variance [B, latent_dim]
            sdf_values: Per-observer SDF values [B, num_observers, 1]
            feature_projections: Per-observer features [B, num_observers, projection_dim]
            attention_weights: Observer attention weights [B, num_observers, 1]
            encoder_features: Optional encoder features for Eikonal loss [B, feature_dim]

        Returns:
            Dictionary of all loss components
        """
        batch_size = x.size(0)
        device = x.device

        # 1. RECONSTRUCTION LOSS (Binary Cross Entropy)
        x_flat = x.view(batch_size, -1)
        recon_flat = reconstruction.view(batch_size, -1)

        recon_loss = safe_bce(recon_flat, x_flat, reduction='sum')

        # Scale by number of elements for consistent loss scaling
        num_elements = x.numel()
        recon_loss = recon_loss / num_elements * 10000

        # 2. KL DIVERGENCE LOSS
        kl_div = 1 + logvar - mean.pow(2) - logvar.exp()
        kl_div_clamped = torch.clamp(kl_div, min=-100, max=100)
        kl_loss = -0.5 * torch.sum(kl_div_clamped) / batch_size

        # Check for NaN
        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            print("WARNING: KL loss is NaN or Inf, setting to 1.0")
            kl_loss = torch.tensor(1.0, device=device)

        # 3. SDF CONSISTENCY LOSS
        # All observers should agree on points near the manifold
        sdf_consistency_loss = self._compute_sdf_consistency(sdf_values)

        # 4. EIKONAL REGULARIZATION
        # SDF gradients should have unit norm (Lipschitz constraint)
        eikonal_loss = self._compute_eikonal_loss(
            x, sdf_values, encoder_features, batch_size, device
        )

        # 5. OBSERVER DIVERSITY LOSS
        # Observers should learn different features (minimize similarity)
        diversity_loss = self._compute_diversity_loss(feature_projections)

        # 6. SDF SUPERVISION LOSS
        # SDF should correlate with reconstruction quality
        # Good reconstruction → low SDF, bad reconstruction → high SDF
        sdf_supervision_loss = self._compute_sdf_supervision(
            sdf_values, reconstruction, x, batch_size, device
        )

        # TOTAL LOSS
        total_loss = (
            self.alpha * recon_loss +
            self.beta * kl_loss +
            self.gamma * sdf_consistency_loss +
            self.delta * eikonal_loss -
            self.epsilon * diversity_loss +  # Negative to maximize diversity
            self.zeta * sdf_supervision_loss  # NEW: Supervise SDF with recon error
        )

        # Final safety check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("WARNING: Total loss is NaN or Inf, setting to 100.0")
            print(f"Components: recon={recon_loss.item():.4f}, kl={kl_loss.item():.4f}, "
                  f"sdf={sdf_consistency_loss.item():.4f}, eikonal={eikonal_loss.item():.4f}, "
                  f"diversity={diversity_loss.item():.4f}")
            total_loss = torch.tensor(100.0, device=device)

        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'kl': kl_loss,
            'sdf_consistency': sdf_consistency_loss,
            'eikonal': eikonal_loss,
            'diversity': diversity_loss,
            'sdf_supervision': sdf_supervision_loss
        }

    def _compute_sdf_consistency(self, sdf_values: torch.Tensor) -> torch.Tensor:
        """
        Compute SDF consistency loss

        All observers should agree on the SDF value, especially near the manifold.
        Points with SDF ≈ 0 (on manifold) should have higher agreement weight.

        Args:
            sdf_values: [B, num_observers, 1]

        Returns:
            SDF consistency loss (scalar)
        """
        # Compute mean SDF across observers
        sdf_mean = torch.mean(sdf_values, dim=1, keepdim=True)  # [B, 1, 1]

        # Compute deviation from mean
        sdf_diff = torch.abs(sdf_values - sdf_mean)  # [B, num_observers, 1]

        # Weight by proximity to manifold: exp(-|sdf_mean|)
        # Points closer to manifold (smaller |sdf|) get higher weight
        abs_sdf_mean = torch.abs(sdf_mean)
        manifold_weight = torch.exp(-torch.clamp(abs_sdf_mean * 5.0, max=10.0))

        # Weighted consistency loss
        consistency_loss = torch.mean(sdf_diff * manifold_weight) * 100

        return consistency_loss

    def _compute_eikonal_loss(
        self,
        x: torch.Tensor,
        sdf_values: torch.Tensor,
        encoder_features: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute Eikonal regularization loss

        For a valid SDF, the gradient norm should be 1: ||∇SDF|| = 1
        This is the Lipschitz constraint for signed distance functions.

        We approximate this by checking SDF differences in feature space.

        Args:
            x: Input images [B, C, H, W]
            sdf_values: Per-observer SDF values [B, num_observers, 1]
            encoder_features: Encoder features [B, feature_dim]
            batch_size: Batch size
            device: Device

        Returns:
            Eikonal loss (scalar)
        """
        if batch_size <= 1 or encoder_features is None:
            return torch.tensor(0.0, device=device)

        try:
            # Compute mean SDF across observers
            sdf_mean = sdf_values.mean(dim=1)  # [B, 1]

            # Limit number of points for efficiency
            max_points = min(batch_size, 32)
            indices = torch.randperm(batch_size, device=device)[:max_points]

            # Extract features and SDFs for selected points
            features = encoder_features[indices]  # [N, feature_dim]
            sdf_subset = sdf_mean[indices]  # [N, 1]

            # Compute pairwise distances in feature space
            feat_expanded = features.unsqueeze(1)  # [N, 1, D]
            feat_transposed = features.unsqueeze(0)  # [1, N, D]
            pairwise_diff = feat_expanded - feat_transposed  # [N, N, D]
            pairwise_dist = torch.norm(pairwise_diff, dim=2) + 1e-8  # [N, N]

            # Compute pairwise SDF differences
            sdf_expanded = sdf_subset.view(-1).unsqueeze(1)  # [N, 1]
            sdf_transposed = sdf_subset.view(-1).unsqueeze(0)  # [1, N]
            sdf_diff = torch.abs(sdf_expanded - sdf_transposed)  # [N, N]

            # Eikonal constraint: |sdf_diff - pairwise_dist| should be small
            # Only apply to nearby points for stability
            mask = (pairwise_dist < 0.1) & (pairwise_dist > 1e-6)

            if torch.sum(mask) > 0:
                eikonal_loss = torch.sum(
                    torch.abs(sdf_diff[mask] - pairwise_dist[mask])
                ) / (torch.sum(mask) + 1e-8)
            else:
                eikonal_loss = torch.tensor(0.0, device=device)

            return eikonal_loss

        except Exception as e:
            print(f"WARNING: Error computing eikonal loss: {e}")
            return torch.tensor(0.0, device=device)

    def _compute_diversity_loss(self, feature_projections: torch.Tensor) -> torch.Tensor:
        """
        Compute observer diversity loss

        Observers should learn different features to avoid redundancy.
        Measured as the average pairwise cosine similarity between observers.

        Args:
            feature_projections: [B, num_observers, projection_dim]

        Returns:
            Diversity loss (scalar) - higher means more similar (less diverse)
        """
        batch_size, num_observers, projection_dim = feature_projections.shape
        device = feature_projections.device

        diversity_loss = 0.0
        valid_pairs = 0

        for i in range(num_observers):
            for j in range(i+1, num_observers):
                try:
                    # Normalize features for cosine similarity
                    feat_i = F.normalize(feature_projections[:, i, :], dim=1)
                    feat_j = F.normalize(feature_projections[:, j, :], dim=1)

                    # Cosine similarity
                    similarity = torch.sum(feat_i * feat_j, dim=1).abs()

                    # Filter out NaN/Inf
                    valid_sim = similarity[~torch.isnan(similarity) & ~torch.isinf(similarity)]

                    if len(valid_sim) > 0:
                        diversity_loss += torch.mean(valid_sim)
                        valid_pairs += 1

                except Exception as e:
                    print(f"WARNING: Error computing diversity for observers {i} and {j}: {e}")
                    continue

        # Average over all valid pairs
        if valid_pairs > 0:
            diversity_loss = diversity_loss / valid_pairs
        else:
            diversity_loss = torch.tensor(0.0, device=device)

        return diversity_loss

    def _compute_sdf_supervision(
        self,
        sdf_values: torch.Tensor,
        reconstruction: torch.Tensor,
        x: torch.Tensor,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute SDF supervision loss based on reconstruction quality

        The SDF should predict reconstruction difficulty:
        - Good reconstruction (low error) → low |SDF| (close to manifold)
        - Bad reconstruction (high error) → high |SDF| (far from manifold)

        Args:
            sdf_values: [B, num_observers, 1]
            reconstruction: [B, C, H, W]
            x: [B, C, H, W]
            batch_size: Batch size
            device: Device

        Returns:
            SDF supervision loss (scalar)
        """
        try:
            # Compute per-sample reconstruction error
            recon_error = F.mse_loss(reconstruction, x, reduction='none')
            recon_error_per_sample = recon_error.mean(dim=[1, 2, 3])  # [B]

            # Normalize error to [0, 1] range and scale to SDF range [-0.1, 0.1]
            # Use tanh to bound the target
            target_sdf = torch.tanh(recon_error_per_sample * 10) * 0.1  # [B]
            target_sdf = target_sdf.unsqueeze(1)  # [B, 1]

            # Mean SDF across observers
            mean_sdf = sdf_values.mean(dim=1)  # [B, 1]

            # MSE loss between predicted SDF and target
            supervision_loss = F.mse_loss(mean_sdf, target_sdf) * 1000

            return supervision_loss

        except Exception as e:
            print(f"WARNING: Error computing SDF supervision loss: {e}")
            return torch.tensor(0.0, device=device)

    def update_weights(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        delta: Optional[float] = None,
        epsilon: Optional[float] = None,
        zeta: Optional[float] = None
    ):
        """
        Update loss weights (useful for curriculum learning)

        Args:
            alpha: New reconstruction weight
            beta: New KL weight
            gamma: New SDF consistency weight
            delta: New Eikonal weight
            epsilon: New diversity weight
            zeta: New SDF supervision weight
        """
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        if delta is not None:
            self.delta = delta
        if epsilon is not None:
            self.epsilon = epsilon
        if zeta is not None:
            self.zeta = zeta
