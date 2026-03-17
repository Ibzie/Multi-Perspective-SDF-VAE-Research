#!/usr/bin/env python3
"""
Diagnose Multi-Perspective SDF-VAE Issues

Analyzes the trained model to identify problems:
1. Observer attention distribution (are all observers being used?)
2. Observer specialization (are they learning different things?)
3. SDF value distributions (are they meaningful?)
4. Diversity metrics (is diversity loss working?)
5. Feature analysis (what is each observer learning?)

Usage:
    python experiments/diagnose_sdf_vae.py
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

from src.models import MultiPerspectiveSDFVAE
from src.data import get_fashion_mnist_loaders


def load_model(checkpoint_path, device):
    """Load trained SDF-VAE model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = MultiPerspectiveSDFVAE(
        image_size=64,
        in_channels=3,
        latent_dim=128,
        num_observers=5
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint


def analyze_observer_attention(model, test_loader, device, num_batches=50):
    """Analyze observer attention weights across dataset"""
    print("\n" + "=" * 70)
    print("DIAGNOSIS 1: OBSERVER ATTENTION ANALYSIS")
    print("=" * 70)
    print("Question: Are all 5 observers being used, or is one dominating?\n")

    all_attention = []

    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(test_loader, desc="Analyzing attention")):
            if i >= num_batches:
                break

            images = images.to(device)
            output = model(images, return_metrics=True)

            # Attention weights: [B, num_observers, 1]
            attention = output['attention_weights'].squeeze(-1)  # [B, num_observers]
            all_attention.append(attention.cpu())

    all_attention = torch.cat(all_attention, dim=0)  # [N, num_observers]

    # Statistics
    mean_attention = all_attention.mean(dim=0)
    std_attention = all_attention.std(dim=0)

    print("Observer Attention Statistics:")
    print("-" * 70)
    print(f"{'Observer':<12} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 70)

    for i in range(5):
        print(f"Observer {i+1:<3} {mean_attention[i]:>12.4f} {std_attention[i]:>12.4f} "
              f"{all_attention[:, i].min():>12.4f} {all_attention[:, i].max():>12.4f}")

    print()
    print(f"Expected (uniform): {1/5:.4f} for each observer")

    # Entropy (measure of diversity in attention)
    # High entropy = attention spread across observers
    # Low entropy = attention concentrated on few observers
    entropy = -(all_attention * torch.log(all_attention + 1e-10)).sum(dim=1).mean()
    max_entropy = np.log(5)  # Maximum possible entropy for 5 observers

    print(f"\nAttention Entropy: {entropy:.4f} (max: {max_entropy:.4f})")
    print(f"Entropy ratio: {entropy/max_entropy:.2%}")

    if entropy/max_entropy < 0.5:
        print("⚠️  LOW ENTROPY - Observers are NOT being used equally!")
    else:
        print("✓ Good entropy - Observers are being used")

    return all_attention


def analyze_sdf_values(model, test_loader, device, num_batches=50):
    """Analyze SDF value distributions"""
    print("\n" + "=" * 70)
    print("DIAGNOSIS 2: SDF VALUE ANALYSIS")
    print("=" * 70)
    print("Question: Are SDF values meaningful distance measures?\n")

    all_sdf = []
    all_recon_error = []

    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(test_loader, desc="Analyzing SDF")):
            if i >= num_batches:
                break

            images = images.to(device)
            output = model(images, return_metrics=True)

            # SDF values: [B, num_observers, 1]
            sdf = output['sdf_values'].squeeze(-1)  # [B, num_observers]
            all_sdf.append(sdf.cpu())

            # Reconstruction error
            recon = output['reconstruction']
            recon_err = F.mse_loss(recon, images, reduction='none').mean(dim=[1, 2, 3])
            all_recon_error.append(recon_err.cpu())

    all_sdf = torch.cat(all_sdf, dim=0)  # [N, num_observers]
    all_recon_error = torch.cat(all_recon_error, dim=0)  # [N]

    # Per-observer statistics
    print("SDF Value Statistics (per observer):")
    print("-" * 70)
    print(f"{'Observer':<12} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 70)

    for i in range(5):
        print(f"Observer {i+1:<3} {all_sdf[:, i].mean():>12.6f} {all_sdf[:, i].std():>12.6f} "
              f"{all_sdf[:, i].min():>12.6f} {all_sdf[:, i].max():>12.6f}")

    # Overall statistics
    mean_sdf = all_sdf.mean(dim=1)  # Average across observers

    print()
    print("Aggregated SDF Statistics:")
    print(f"  Mean: {mean_sdf.mean():.6f}")
    print(f"  Std:  {mean_sdf.std():.6f}")
    print(f"  Min:  {mean_sdf.min():.6f}")
    print(f"  Max:  {mean_sdf.max():.6f}")
    print(f"  Range: {mean_sdf.max() - mean_sdf.min():.6f}")

    if mean_sdf.std() < 0.01:
        print("⚠️  VERY LOW VARIANCE - SDF values are almost constant!")
        print("    This means SDF is NOT learning meaningful distances.")

    # Correlation with reconstruction error
    correlation = torch.corrcoef(torch.stack([mean_sdf, all_recon_error]))[0, 1].item()
    print(f"\nCorrelation (SDF vs Reconstruction Error): {correlation:.4f}")

    if correlation < 0:
        print("⚠️  NEGATIVE CORRELATION - SDF is backwards!")
        print("    Expected: Higher SDF → Worse reconstruction")
        print("    Actual:   Higher SDF → Better reconstruction")
    elif correlation < 0.3:
        print("⚠️  WEAK CORRELATION - SDF doesn't predict reconstruction quality well")
    else:
        print("✓ Good correlation - SDF is working as expected")

    return all_sdf, all_recon_error


def analyze_observer_diversity(model, test_loader, device, num_batches=50):
    """Analyze observer specialization and diversity"""
    print("\n" + "=" * 70)
    print("DIAGNOSIS 3: OBSERVER DIVERSITY ANALYSIS")
    print("=" * 70)
    print("Question: Are observers learning different representations?\n")

    all_projections = []

    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(test_loader, desc="Analyzing diversity")):
            if i >= num_batches:
                break

            images = images.to(device)
            output = model(images, return_metrics=True)

            # Feature projections: [B, num_observers, projection_dim]
            projections = output['feature_projections']
            all_projections.append(projections.cpu())

    all_projections = torch.cat(all_projections, dim=0)  # [N, num_observers, projection_dim]

    # Compute pairwise cosine similarity between observers
    print("Pairwise Cosine Similarity Between Observers:")
    print("(1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)")
    print("-" * 70)

    # Average projections across samples for each observer
    mean_projections = all_projections.mean(dim=0)  # [num_observers, projection_dim]

    similarity_matrix = torch.zeros(5, 5)

    for i in range(5):
        for j in range(5):
            similarity = F.cosine_similarity(
                mean_projections[i:i+1],
                mean_projections[j:j+1],
                dim=1
            ).item()
            similarity_matrix[i, j] = similarity

    # Print similarity matrix
    print("     ", end="")
    for i in range(5):
        print(f"  Obs{i+1}", end="")
    print()

    for i in range(5):
        print(f"Obs{i+1}", end="")
        for j in range(5):
            if i == j:
                print(f"  1.000", end="")
            else:
                print(f"  {similarity_matrix[i, j]:>5.3f}", end="")
        print()

    # Average off-diagonal similarity (how similar observers are to each other)
    off_diag = []
    for i in range(5):
        for j in range(i+1, 5):
            off_diag.append(similarity_matrix[i, j].item())

    avg_similarity = np.mean(off_diag)

    print()
    print(f"Average inter-observer similarity: {avg_similarity:.4f}")

    if avg_similarity > 0.9:
        print("⚠️  VERY HIGH SIMILARITY - Observers are learning nearly identical features!")
        print("    Diversity loss is NOT working effectively.")
    elif avg_similarity > 0.7:
        print("⚠️  HIGH SIMILARITY - Limited observer diversity")
    else:
        print("✓ Good diversity - Observers are learning different features")

    return similarity_matrix


def analyze_loss_components(checkpoint):
    """Analyze what the model optimized during training"""
    print("\n" + "=" * 70)
    print("DIAGNOSIS 4: TRAINING DYNAMICS ANALYSIS")
    print("=" * 70)
    print("Question: What did the model actually optimize?\n")

    epoch = checkpoint.get('epoch', 'Unknown')
    val_loss = checkpoint.get('val_loss', 'Unknown')

    print(f"Checkpoint Information:")
    print(f"  Epoch: {epoch}")
    print(f"  Validation Loss: {val_loss}")
    print()

    # We can't get full training history from checkpoint, but we can
    # make recommendations based on what we've seen
    print("Recommendations based on diagnoses:")
    print("-" * 70)


def visualize_diagnostics(attention, sdf_values, similarity_matrix, save_dir):
    """Create diagnostic visualizations"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Attention distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Violin plot of attention
    attention_np = attention.numpy()
    parts = axes[0].violinplot([attention_np[:, i] for i in range(5)],
                               positions=range(1, 6),
                               showmeans=True,
                               showmedians=True)
    axes[0].set_xlabel('Observer', fontweight='bold')
    axes[0].set_ylabel('Attention Weight', fontweight='bold')
    axes[0].set_title('Observer Attention Distribution', fontweight='bold')
    axes[0].axhline(y=0.2, color='r', linestyle='--', label='Uniform (1/5)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bar plot of mean attention
    mean_attention = attention.mean(dim=0).numpy()
    std_attention = attention.std(dim=0).numpy()
    axes[1].bar(range(1, 6), mean_attention, yerr=std_attention, capsize=5)
    axes[1].axhline(y=0.2, color='r', linestyle='--', label='Uniform (1/5)')
    axes[1].set_xlabel('Observer', fontweight='bold')
    axes[1].set_ylabel('Mean Attention Weight', fontweight='bold')
    axes[1].set_title('Mean Observer Attention', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'attention_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {save_dir / 'attention_analysis.png'}")
    plt.close()

    # 2. SDF distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of SDF values per observer
    sdf_np = sdf_values.numpy()
    for i in range(5):
        axes[0].hist(sdf_np[:, i], bins=50, alpha=0.5, label=f'Observer {i+1}')
    axes[0].set_xlabel('SDF Value', fontweight='bold')
    axes[0].set_ylabel('Frequency', fontweight='bold')
    axes[0].set_title('SDF Value Distributions', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot
    axes[1].boxplot([sdf_np[:, i] for i in range(5)], labels=[f'Obs{i+1}' for i in range(5)])
    axes[1].set_xlabel('Observer', fontweight='bold')
    axes[1].set_ylabel('SDF Value', fontweight='bold')
    axes[1].set_title('SDF Value Distribution by Observer', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'sdf_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_dir / 'sdf_analysis.png'}")
    plt.close()

    # 3. Similarity heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(similarity_matrix.numpy(), cmap='RdYlGn_r', vmin=-1, vmax=1)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([f'Obs{i+1}' for i in range(5)])
    ax.set_yticklabels([f'Obs{i+1}' for i in range(5)])

    # Add text annotations
    for i in range(5):
        for j in range(5):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')

    ax.set_title('Observer Feature Similarity Matrix', fontweight='bold', fontsize=14)
    plt.colorbar(im, ax=ax, label='Cosine Similarity')

    plt.tight_layout()
    plt.savefig(save_dir / 'diversity_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_dir / 'diversity_analysis.png'}")
    plt.close()


def main():
    print("=" * 70)
    print("DIAGNOSING MULTI-PERSPECTIVE SDF-VAE")
    print("=" * 70)
    print("Identifying issues with observer specialization, SDF learning,")
    print("and diversity enforcement.")
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
    print("✓ Data loaded\n")

    # Load model
    print("Loading trained Multi-Perspective SDF-VAE...")
    checkpoint_path = 'results/checkpoints/sdf_vae_fashion_latent128_obs5/best.pt'
    model, checkpoint = load_model(checkpoint_path, device)
    print("✓ Model loaded\n")

    # Run diagnostics
    attention = analyze_observer_attention(model, test_loader, device)
    sdf_values, recon_errors = analyze_sdf_values(model, test_loader, device)
    similarity_matrix = analyze_observer_diversity(model, test_loader, device)
    analyze_loss_components(checkpoint)

    # Generate visualizations
    print("\nGenerating diagnostic visualizations...")
    output_dir = Path('results/diagnostics')
    visualize_diagnostics(attention, sdf_values, similarity_matrix, output_dir)

    # Summary and recommendations
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    # Check issues
    issues = []
    fixes = []

    # Check 1: Attention entropy
    entropy = -(attention * torch.log(attention + 1e-10)).sum(dim=1).mean()
    max_entropy = np.log(5)
    if entropy/max_entropy < 0.5:
        issues.append("⚠️  Low attention entropy - observers not used equally")
        fixes.append("→ Increase temperature in Gumbel-Softmax")
        fixes.append("→ Add attention entropy regularization")

    # Check 2: SDF variance
    mean_sdf = sdf_values.mean(dim=1)
    if mean_sdf.std() < 0.01:
        issues.append("⚠️  SDF values have very low variance")
        fixes.append("→ Adjust SDF initialization scale")
        fixes.append("→ Increase SDF consistency loss weight")

    # Check 3: Observer similarity
    off_diag = []
    for i in range(5):
        for j in range(i+1, 5):
            off_diag.append(similarity_matrix[i, j].item())
    avg_similarity = np.mean(off_diag)

    if avg_similarity > 0.9:
        issues.append("⚠️  Observers are nearly identical (>0.9 similarity)")
        fixes.append("→ Increase diversity loss weight (epsilon)")
        fixes.append("→ Use contrastive loss instead of negative cosine similarity")
        fixes.append("→ Initialize observers with different random seeds")

    # Print issues
    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print("\nRECOMMENDED FIXES:")
        for fix in fixes:
            print(f"  {fix}")
    else:
        print("\n✓ No major issues detected!")

    print()
    print(f"Diagnostic visualizations saved to: {output_dir.absolute()}")
    print("=" * 70)


if __name__ == '__main__':
    main()
