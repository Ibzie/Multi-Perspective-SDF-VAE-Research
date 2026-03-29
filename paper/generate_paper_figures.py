#!/usr/bin/env python3
"""
Generate publication-quality figures for the research paper

Uses existing comparison results and model outputs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from umap import UMAP
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil

from src.models import MultiPerspectiveSDFVAE, VanillaVAE, BetaVAE
from src.data import get_celeba_loaders

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'


def copy_existing_figures():
    """Copy already generated comparison figures"""
    source_dir = Path('experiments/results/comparisons')
    dest_dir = Path('paper/figures')
    dest_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        'reconstruction_comparison.png',
        'sampling_comparison.png',
        'interpolation_comparison.png'
    ]

    print("Copying existing comparison figures...")
    for fig in figures:
        src = source_dir / fig
        if src.exists():
            shutil.copy(src, dest_dir / fig)
            print(f"  ✓ {fig}")


def create_quantitative_comparison_table():
    """Create publication table from quantitative results"""
    print("\nCreating quantitative comparison table...")

    with open('experiments/results/comparisons/quantitative_results.json', 'r') as f:
        results = json.load(f)

    # Create LaTeX table (using %-formatting to avoid {} conflicts)
    latex_table = r"""\begin{table}[t]
\centering
\caption{Quantitative Comparison of VAE Models on Fashion-MNIST}
\label{tab:quantitative_results}
\begin{tabular}{lrrr}
\toprule
\textbf{Metric} & \textbf{SDF-VAE} & \textbf{Vanilla VAE} & \textbf{$\beta$-VAE} \\
\midrule
\multicolumn{4}{l}{\textit{Reconstruction Quality (MSE, lower is better)}} \\
Training Loss & %.2f & \textbf{%.2f} & %.2f \\
Validation Loss & %.2f & \textbf{%.2f} & %.2f \\
Test Loss & %.2f & \textbf{%.2f} & %.2f \\
\midrule
\multicolumn{4}{l}{\textit{Generalization (closer to 1.0 is better)}} \\
Train-Test Gap & %.3f & %.3f & %.3f \\
Test/Train Ratio & \textbf{%.3f} & %.3f & %.3f \\
\midrule
\multicolumn{4}{l}{\textit{Latent Space Statistics}} \\
Mean Latent Norm & \textbf{%.2f} & %.2f & %.2f \\
Latent Std Dev & \textbf{%.2f} & %.2f & %.2f \\
Avg Log-Variance & %.2f & %.2f & %.2f \\
LogVar Std & \textbf{%.4f} & %.2f & %.2f \\
\bottomrule
\end{tabular}
\end{table}
""" % (
        results['sdf_vae']['train_recon_loss'],
        results['vanilla']['train_recon_loss'],
        results['beta']['train_recon_loss'],
        results['sdf_vae']['val_recon_loss'],
        results['vanilla']['val_recon_loss'],
        results['beta']['val_recon_loss'],
        results['sdf_vae']['test_recon_loss'],
        results['vanilla']['test_recon_loss'],
        results['beta']['test_recon_loss'],
        results['sdf_vae']['train_test_gap'],
        results['vanilla']['train_test_gap'],
        results['beta']['train_test_gap'],
        results['sdf_vae']['generalization_ratio'],
        results['vanilla']['generalization_ratio'],
        results['beta']['generalization_ratio'],
        results['sdf_vae']['mean_norm'],
        results['vanilla']['mean_norm'],
        results['beta']['mean_norm'],
        results['sdf_vae']['mean_std'],
        results['vanilla']['mean_std'],
        results['beta']['mean_std'],
        results['sdf_vae']['logvar_mean'],
        results['vanilla']['logvar_mean'],
        results['beta']['logvar_mean'],
        results['sdf_vae']['logvar_std'],
        results['vanilla']['logvar_std'],
        results['beta']['logvar_std'],
    )

    # Save LaTeX table
    with open('paper/data/quantitative_comparison.tex', 'w') as f:
        f.write(latex_table)
    print("  ✓ quantitative_comparison.tex")

    # Also create a visual bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Comparison Summary', fontsize=16, fontweight='bold')

    models = ['SDF-VAE', 'Vanilla VAE', 'β-VAE']
    colors = sns.color_palette("husl", 3)

    # Plot 1: Reconstruction Loss
    ax = axes[0, 0]
    test_losses = [
        results['sdf_vae']['test_recon_loss'],
        results['vanilla']['test_recon_loss'],
        results['beta']['test_recon_loss']
    ]
    bars = ax.bar(models, test_losses, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Reconstruction Quality (Test Set)\nLower is Better')
    ax.grid(True, alpha=0.3, axis='y')
    # Highlight winner
    bars[1].set_edgecolor('green')
    bars[1].set_linewidth(3)

    # Plot 2: Generalization Ratio
    ax = axes[0, 1]
    gen_ratios = [
        results['sdf_vae']['generalization_ratio'],
        results['vanilla']['generalization_ratio'],
        results['beta']['generalization_ratio']
    ]
    bars = ax.bar(models, gen_ratios, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Perfect (1.0)')
    ax.set_ylabel('Test/Train Ratio')
    ax.set_title('Generalization Performance\nCloser to 1.0 is Better')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    # Highlight winner
    bars[0].set_edgecolor('green')
    bars[0].set_linewidth(3)

    # Plot 3: Latent Variance Stability
    ax = axes[1, 0]
    logvar_stds = [
        results['sdf_vae']['logvar_std'],
        results['vanilla']['logvar_std'],
        results['beta']['logvar_std']
    ]
    bars = ax.bar(models, logvar_stds, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Log-Variance Std Dev')
    ax.set_title('Latent Variance Stability\nLower is More Stable')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    # Highlight winner
    bars[0].set_edgecolor('green')
    bars[0].set_linewidth(3)

    # Plot 4: Latent Expressiveness
    ax = axes[1, 1]
    latent_norms = [
        results['sdf_vae']['mean_norm'],
        results['vanilla']['mean_norm'],
        results['beta']['mean_norm']
    ]
    bars = ax.bar(models, latent_norms, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Mean Latent Norm')
    ax.set_title('Latent Space Expressiveness\nHigher = More Expressive')
    ax.grid(True, alpha=0.3, axis='y')
    # Highlight winner
    bars[0].set_edgecolor('green')
    bars[0].set_linewidth(3)

    plt.tight_layout()
    plt.savefig('paper/figures/comparison_summary.pdf', bbox_inches='tight')
    plt.savefig('paper/figures/comparison_summary.png', bbox_inches='tight')
    print("  ✓ comparison_summary.pdf")
    plt.close()


def plot_latent_space_tsne():
    """Generate t-SNE visualization"""
    print("\nGenerating latent space t-SNE visualization...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load SDF-VAE model
    model = MultiPerspectiveSDFVAE(
        image_size=64, in_channels=3, latent_dim=128, num_observers=5
    )
    checkpoint = torch.load(
        'results/checkpoints/sdf_vae_celeba/best.pt',
        map_location=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load data
    _, _, test_loader = get_celeba_loaders(
        data_root='./data',
        batch_size=128,
        num_workers=4,
        image_size=64
    )

    # Extract latent codes
    latent_codes = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in tqdm(test_loader, desc="  Encoding", leave=False):
            images = images.to(device)
            output = model(images)
            latent_codes.append(output['mean'].cpu().numpy())
            labels.append(batch_labels.numpy())

    latent_codes = np.vstack(latent_codes)
    labels = np.concatenate(labels)

    # Compute t-SNE
    print("  Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(latent_codes[:5000])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embedded[:, 0], embedded[:, 1],
                        c=labels[:5000], cmap='tab10',
                        alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    ax.set_title('Latent Space Visualization (t-SNE)', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    plt.colorbar(scatter, ax=ax, label='Class', ticks=range(10))
    plt.tight_layout()
    plt.savefig('paper/figures/latent_tsne.pdf', bbox_inches='tight')
    plt.savefig('paper/figures/latent_tsne.png', bbox_inches='tight')
    print("  ✓ latent_tsne.pdf")
    plt.close()


def plot_observer_analysis():
    """Generate observer analysis plots"""
    print("\nGenerating observer analysis...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = MultiPerspectiveSDFVAE(
        image_size=64, in_channels=3, latent_dim=128, num_observers=5
    )
    checkpoint = torch.load(
        'results/checkpoints/sdf_vae_celeba/best.pt',
        map_location=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load data
    _, _, test_loader = get_celeba_loaders(
        data_root='./data',
        batch_size=128,
        num_workers=4,
        image_size=64
    )

    # Collect statistics
    all_sdf_values = []
    all_attention_weights = []

    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="  Collecting stats", leave=False):
            images = images.to(device)
            output = model(images)
            all_sdf_values.append(output['sdf_values'].cpu().numpy())
            all_attention_weights.append(output['attention_weights'].cpu().numpy())

    sdf_values = np.vstack(all_sdf_values).squeeze(-1)  # [N, 5]
    attention_weights = np.vstack(all_attention_weights).squeeze(-1)  # [N, 5]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Multi-Perspective Observer Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Attention weights per observer
    ax = axes[0, 0]
    mean_attention = attention_weights.mean(axis=0)
    std_attention = attention_weights.std(axis=0)
    observers = np.arange(1, 6)
    ax.bar(observers, mean_attention, yerr=std_attention,
           capsize=5, color=sns.color_palette("husl", 5), alpha=0.7,
           edgecolor='black')
    ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.7,
               label='Uniform (1/5)', linewidth=2)
    ax.set_xlabel('Observer ID')
    ax.set_ylabel('Mean Attention Weight')
    ax.set_title('Observer Attention Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(observers)

    # Plot 2: SDF distribution per observer
    ax = axes[0, 1]
    bp = ax.boxplot([sdf_values[:, i] for i in range(5)],
                     labels=[f'O{i+1}' for i in range(5)],
                     patch_artist=True)
    for patch, color in zip(bp['boxes'], sns.color_palette("husl", 5)):
        patch.set_facecolor(color)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7,
               label='Manifold (SDF=0)', linewidth=2)
    ax.set_xlabel('Observer ID')
    ax.set_ylabel('SDF Value')
    ax.set_title('SDF Distribution per Observer')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Attention entropy
    ax = axes[1, 0]
    entropy = -(attention_weights * np.log(attention_weights + 1e-10)).sum(axis=1)
    ax.hist(entropy, bins=50, alpha=0.7, edgecolor='black', color='purple')
    ax.axvline(x=np.log(5), color='red', linestyle='--',
               label=f'Max Entropy ({np.log(5):.2f})', linewidth=2)
    ax.axvline(x=entropy.mean(), color='green', linestyle='--',
               label=f'Mean ({entropy.mean():.2f})', linewidth=2)
    ax.set_xlabel('Attention Entropy')
    ax.set_ylabel('Frequency')
    ax.set_title('Attention Distribution Entropy')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Observer diversity
    ax = axes[1, 1]
    diversity = 1 - attention_weights.max(axis=1)
    ax.hist(diversity, bins=50, alpha=0.7, edgecolor='black', color='green')
    ax.axvline(x=diversity.mean(), color='red', linestyle='--',
               label=f'Mean: {diversity.mean():.3f}', linewidth=2)
    ax.set_xlabel('Diversity Score (1 - max attention)')
    ax.set_ylabel('Frequency')
    ax.set_title('Observer Diversity Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('paper/figures/observer_analysis.pdf', bbox_inches='tight')
    plt.savefig('paper/figures/observer_analysis.png', bbox_inches='tight')
    print("  ✓ observer_analysis.pdf")
    plt.close()


def plot_variance_stability():
    """Plot variance stability comparison"""
    print("\nGenerating variance stability comparison...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = {
        'SDF-VAE': (MultiPerspectiveSDFVAE(image_size=64, in_channels=3, latent_dim=128, num_observers=5),
                   'results/checkpoints/sdf_vae_celeba/best.pt'),
        'Vanilla VAE': (VanillaVAE(image_size=64, in_channels=3, latent_dim=128),
                       'results/checkpoints/vanilla_vae_celeba/best.pt'),
        'β-VAE': (BetaVAE(image_size=64, in_channels=3, latent_dim=128, beta=4.0),
                 'results/checkpoints/beta_vae_celeba/best.pt'),
    }

    # Load data
    _, _, test_loader = get_celeba_loaders(
        data_root='./data',
        batch_size=128,
        num_workers=4,
        image_size=64
    )

    # Collect logvars
    all_logvars = {}
    for name, (model, checkpoint_path) in models.items():
        print(f"  Processing {name}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        logvars = []
        with torch.no_grad():
            for images, _ in tqdm(test_loader, desc=f"    Encoding", leave=False):
                images = images.to(device)
                output = model(images)
                logvars.append(output['logvar'].cpu().numpy())

        all_logvars[name] = np.vstack(logvars)

    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Latent Space Variance Stability Comparison', fontsize=16, fontweight='bold')

    for idx, (name, logvar) in enumerate(all_logvars.items()):
        ax = axes[idx]
        ax.hist(logvar.flatten(), bins=100, alpha=0.7, edgecolor='black')
        mean_logvar = logvar.mean()
        std_logvar = logvar.std()
        ax.axvline(x=mean_logvar, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_logvar:.3f}')
        ax.set_xlabel('Log-Variance')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name}\nStd: {std_logvar:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('paper/figures/variance_stability.pdf', bbox_inches='tight')
    plt.savefig('paper/figures/variance_stability.png', bbox_inches='tight')
    print("  ✓ variance_stability.pdf")
    plt.close()


def main():
    print("="*70)
    print("GENERATING PAPER FIGURES")
    print("="*70)

    # Create directories
    Path('paper/figures').mkdir(parents=True, exist_ok=True)
    Path('paper/data').mkdir(parents=True, exist_ok=True)

    # Copy existing figures
    copy_existing_figures()

    # Generate new figures
    create_quantitative_comparison_table()
    plot_latent_space_tsne()
    plot_observer_analysis()
    plot_variance_stability()

    print("\n" + "="*70)
    print("PAPER FIGURES COMPLETE!")
    print("="*70)
    print(f"\nFigures saved to: paper/figures/")
    print("\nGenerated files:")
    for file in sorted(Path('paper/figures').glob('*.pdf')):
        print(f"  - {file.name}")


if __name__ == '__main__':
    main()
