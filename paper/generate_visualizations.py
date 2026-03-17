#!/usr/bin/env python3
"""
Generate all visualizations for the research paper

Creates publication-quality figures for:
1. Training curves (all models)
2. Latent space visualizations (t-SNE, UMAP)
3. Observer attention analysis
4. SDF distribution analysis
5. Reconstruction quality comparisons
6. Architecture diagrams
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

from src.models import MultiPerspectiveSDFVAE, VanillaVAE, BetaVAE
from src.data import get_fashion_mnist_loaders

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize'] = (8, 6)


class VisualizationGenerator:
    def __init__(self, output_dir='paper/figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def load_metrics(self, log_path):
        """Load training metrics from JSON"""
        with open(log_path, 'r') as f:
            return json.load(f)

    def plot_training_curves(self):
        """Generate training curve comparisons"""
        print("\n" + "="*70)
        print("Generating Training Curves")
        print("="*70)

        # Load metrics for all models
        metrics_paths = {
            'SDF-VAE': 'experiments/results/logs/sdf_vae_fashion_latent128_obs5/metrics.json',
            'Vanilla VAE': 'experiments/results/logs/vanilla_vae_fashion_latent128/metrics.json',
            'β-VAE': 'experiments/results/logs/beta_vae_fashion_beta4.0_latent128/metrics.json',
        }

        all_metrics = {}
        for name, path in metrics_paths.items():
            all_metrics[name] = self.load_metrics(path)

        # Create multi-panel figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Dynamics Comparison', fontsize=16, fontweight='bold')

        # Plot 1: Total Loss
        ax = axes[0, 0]
        for name, metrics in all_metrics.items():
            epochs = range(len(metrics['train_loss']))
            ax.plot(epochs, metrics['train_loss'], label=f'{name} (Train)', linewidth=2)
            ax.plot(epochs, metrics['val_loss'], label=f'{name} (Val)', linewidth=2, linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss Evolution')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        # Plot 2: Reconstruction Loss
        ax = axes[0, 1]
        for name, metrics in all_metrics.items():
            epochs = range(len(metrics['train_recon_loss']))
            ax.plot(epochs, metrics['train_recon_loss'], label=f'{name}', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reconstruction Loss')
        ax.set_title('Reconstruction Loss (Train Set)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: KL Divergence
        ax = axes[1, 0]
        for name, metrics in all_metrics.items():
            epochs = range(len(metrics['train_kl_loss']))
            ax.plot(epochs, metrics['train_kl_loss'], label=f'{name}', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('KL Divergence')
        ax.set_title('KL Divergence (Train Set)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Train-Val Gap
        ax = axes[1, 1]
        for name, metrics in all_metrics.items():
            epochs = range(len(metrics['train_loss']))
            gap = np.array(metrics['val_loss']) - np.array(metrics['train_loss'])
            ax.plot(epochs, gap, label=f'{name}', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation - Training Loss')
        ax.set_title('Generalization Gap (Lower is Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / 'training_curves.pdf'
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.png'), bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

    def plot_sdf_specific_metrics(self):
        """Plot SDF-VAE specific training metrics"""
        print("\nGenerating SDF-Specific Metrics...")

        metrics = self.load_metrics('experiments/results/logs/sdf_vae_fashion_latent128_obs5/metrics.json')

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('SDF-VAE Specific Training Metrics', fontsize=16, fontweight='bold')

        epochs = range(len(metrics['train_sdf_loss']))

        # Plot 1: SDF Loss
        ax = axes[0, 0]
        ax.plot(epochs, metrics['train_sdf_loss'], label='Train', linewidth=2)
        ax.plot(epochs, metrics['val_sdf_loss'], label='Validation', linewidth=2, linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('SDF Loss')
        ax.set_title('SDF Consistency Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Eikonal Loss
        ax = axes[0, 1]
        ax.plot(epochs, metrics['train_eikonal_loss'], label='Train', linewidth=2)
        ax.plot(epochs, metrics['val_eikonal_loss'], label='Validation', linewidth=2, linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Eikonal Loss')
        ax.set_title('Eikonal Constraint |∇SDF| = 1')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Diversity Loss
        ax = axes[1, 0]
        ax.plot(epochs, metrics['train_diversity_loss'], label='Train', linewidth=2)
        ax.plot(epochs, metrics['val_diversity_loss'], label='Validation', linewidth=2, linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Diversity Loss')
        ax.set_title('Observer Diversity Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: All SDF losses stacked
        ax = axes[1, 1]
        ax.plot(epochs, metrics['train_sdf_loss'], label='SDF', linewidth=2)
        ax.plot(epochs, metrics['train_eikonal_loss'], label='Eikonal', linewidth=2)
        ax.plot(epochs, metrics['train_diversity_loss'], label='Diversity', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Value')
        ax.set_title('SDF Loss Components (Train)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        save_path = self.output_dir / 'sdf_training_metrics.pdf'
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.png'), bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

    def plot_latent_space_visualization(self):
        """Generate t-SNE and UMAP visualizations of latent space"""
        print("\nGenerating Latent Space Visualizations...")

        # Load models
        models = {
            'SDF-VAE': (MultiPerspectiveSDFVAE(image_size=64, in_channels=3, latent_dim=128, num_observers=5),
                       'experiments/results/checkpoints/sdf_vae_fashion_latent128_obs5/best.pt'),
            'Vanilla VAE': (VanillaVAE(image_size=64, in_channels=3, latent_dim=128),
                           'experiments/results/checkpoints/vanilla_vae_fashion_latent128/best.pt'),
            'β-VAE': (BetaVAE(image_size=64, in_channels=3, latent_dim=128, beta=4.0),
                     'experiments/results/checkpoints/beta_vae_fashion_beta4.0_latent128/best.pt'),
        }

        # Load data
        _, _, test_loader = get_fashion_mnist_loaders(
            data_root='./data',
            batch_size=128,
            num_workers=4,
            image_size=64
        )

        # Collect latent codes and labels
        latent_codes = {}
        labels = None

        for name, (model, checkpoint_path) in models.items():
            print(f"  Processing {name}...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()

            codes = []
            lbls = []

            with torch.no_grad():
                for images, batch_labels in tqdm(test_loader, desc=f"  Encoding {name}", leave=False):
                    images = images.to(self.device)
                    output = model(images)
                    codes.append(output['mean'].cpu().numpy())
                    lbls.append(batch_labels.numpy())

            latent_codes[name] = np.vstack(codes)
            if labels is None:
                labels = np.concatenate(lbls)

        # Generate t-SNE plot
        print("  Computing t-SNE embeddings...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Latent Space Visualization (t-SNE)', fontsize=16, fontweight='bold')

        for idx, (name, codes) in enumerate(latent_codes.items()):
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embedded = tsne.fit_transform(codes[:5000])  # Use subset for speed

            ax = axes[idx]
            scatter = ax.scatter(embedded[:, 0], embedded[:, 1],
                               c=labels[:5000], cmap='tab10',
                               alpha=0.6, s=10)
            ax.set_title(name)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')

            if idx == 2:
                plt.colorbar(scatter, ax=ax, label='Class')

        plt.tight_layout()
        save_path = self.output_dir / 'latent_space_tsne.pdf'
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.png'), bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

        # Generate UMAP plot
        print("  Computing UMAP embeddings...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Latent Space Visualization (UMAP)', fontsize=16, fontweight='bold')

        for idx, (name, codes) in enumerate(latent_codes.items()):
            umap = UMAP(n_components=2, random_state=42, n_neighbors=15)
            embedded = umap.fit_transform(codes[:5000])

            ax = axes[idx]
            scatter = ax.scatter(embedded[:, 0], embedded[:, 1],
                               c=labels[:5000], cmap='tab10',
                               alpha=0.6, s=10)
            ax.set_title(name)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')

            if idx == 2:
                plt.colorbar(scatter, ax=ax, label='Class')

        plt.tight_layout()
        save_path = self.output_dir / 'latent_space_umap.pdf'
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.png'), bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

    def plot_observer_analysis(self):
        """Analyze and visualize observer behavior"""
        print("\nGenerating Observer Analysis...")

        # Load SDF-VAE model
        model = MultiPerspectiveSDFVAE(
            image_size=64, in_channels=3, latent_dim=128, num_observers=5
        )
        checkpoint = torch.load(
            'experiments/results/checkpoints/sdf_vae_fashion_latent128_obs5/best.pt',
            map_location=self.device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        # Load data
        _, _, test_loader = get_fashion_mnist_loaders(
            data_root='./data',
            batch_size=128,
            num_workers=4,
            image_size=64
        )

        # Collect observer statistics
        all_sdf_values = []
        all_attention_weights = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="  Collecting observer stats", leave=False):
                images = images.to(self.device)
                output = model(images)

                all_sdf_values.append(output['sdf_values'].cpu().numpy())
                all_attention_weights.append(output['attention_weights'].cpu().numpy())
                all_labels.append(labels.numpy())

        sdf_values = np.vstack(all_sdf_values)  # [N, num_observers, 1]
        attention_weights = np.vstack(all_attention_weights)  # [N, num_observers, 1]
        labels = np.concatenate(all_labels)

        # Create comprehensive observer analysis figure
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle('Multi-Perspective Observer Analysis', fontsize=16, fontweight='bold')

        # Plot 1: SDF distribution per observer
        ax1 = fig.add_subplot(gs[0, :2])
        sdf_data = sdf_values.squeeze(-1)  # [N, num_observers]
        positions = np.arange(1, 6)
        bp = ax1.boxplot([sdf_data[:, i] for i in range(5)],
                         positions=positions,
                         labels=[f'Obs {i+1}' for i in range(5)],
                         patch_artist=True)
        for patch, color in zip(bp['boxes'], sns.color_palette("husl", 5)):
            patch.set_facecolor(color)
        ax1.set_ylabel('SDF Value')
        ax1.set_title('SDF Value Distribution per Observer')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Manifold (SDF=0)')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Attention weight distribution
        ax2 = fig.add_subplot(gs[0, 2])
        attention_data = attention_weights.squeeze(-1)  # [N, num_observers]
        mean_attention = attention_data.mean(axis=0)
        std_attention = attention_data.std(axis=0)
        observers = np.arange(1, 6)
        ax2.bar(observers, mean_attention, yerr=std_attention,
               capsize=5, color=sns.color_palette("husl", 5), alpha=0.7)
        ax2.set_xlabel('Observer')
        ax2.set_ylabel('Mean Attention Weight')
        ax2.set_title('Average Attention Weights')
        ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Uniform (1/5)')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Attention entropy histogram
        ax3 = fig.add_subplot(gs[1, 0])
        entropy = -(attention_data * np.log(attention_data + 1e-10)).sum(axis=1)
        ax3.hist(entropy, bins=50, alpha=0.7, edgecolor='black')
        ax3.axvline(x=np.log(5), color='red', linestyle='--',
                   label=f'Max Entropy ({np.log(5):.2f})', linewidth=2)
        ax3.set_xlabel('Attention Entropy')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Attention Distribution Entropy')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Correlation matrix of observer attention
        ax4 = fig.add_subplot(gs[1, 1])
        corr_matrix = np.corrcoef(attention_data.T)
        im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax4.set_xticks(range(5))
        ax4.set_yticks(range(5))
        ax4.set_xticklabels([f'O{i+1}' for i in range(5)])
        ax4.set_yticklabels([f'O{i+1}' for i in range(5)])
        ax4.set_title('Observer Attention Correlation')
        plt.colorbar(im, ax=ax4, label='Correlation')

        # Add correlation values
        for i in range(5):
            for j in range(5):
                text = ax4.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)

        # Plot 5: Per-class attention specialization
        ax5 = fig.add_subplot(gs[1, 2])
        class_attention = np.zeros((10, 5))
        for class_id in range(10):
            mask = labels == class_id
            class_attention[class_id] = attention_data[mask].mean(axis=0)

        im = ax5.imshow(class_attention.T, cmap='viridis', aspect='auto')
        ax5.set_xlabel('Class')
        ax5.set_ylabel('Observer')
        ax5.set_xticks(range(10))
        ax5.set_yticks(range(5))
        ax5.set_yticklabels([f'O{i+1}' for i in range(5)])
        ax5.set_title('Observer-Class Attention Pattern')
        plt.colorbar(im, ax=ax5, label='Attention Weight')

        # Plot 6: SDF vs Attention scatter
        ax6 = fig.add_subplot(gs[2, 0])
        # Sample 1000 points for clarity
        sample_idx = np.random.choice(len(sdf_data), 1000, replace=False)
        for obs_id in range(5):
            ax6.scatter(sdf_data[sample_idx, obs_id],
                       attention_data[sample_idx, obs_id],
                       alpha=0.3, s=10, label=f'Observer {obs_id+1}')
        ax6.set_xlabel('SDF Value')
        ax6.set_ylabel('Attention Weight')
        ax6.set_title('SDF vs Attention Relationship')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)

        # Plot 7: Cumulative attention distribution
        ax7 = fig.add_subplot(gs[2, 1])
        sorted_attention = np.sort(attention_data, axis=1)[:, ::-1]
        cumulative = np.cumsum(sorted_attention, axis=1).mean(axis=0)
        ax7.plot(range(1, 6), cumulative, marker='o', linewidth=2, markersize=8)
        ax7.set_xlabel('Top K Observers')
        ax7.set_ylabel('Cumulative Attention')
        ax7.set_title('Cumulative Attention Coverage')
        ax7.grid(True, alpha=0.3)
        ax7.set_xticks(range(1, 6))

        # Plot 8: Observer diversity over samples
        ax8 = fig.add_subplot(gs[2, 2])
        # Compute diversity as 1 - max_attention (higher = more diverse)
        diversity = 1 - attention_data.max(axis=1)
        ax8.hist(diversity, bins=50, alpha=0.7, edgecolor='black', color='green')
        ax8.set_xlabel('Diversity Score (1 - max attention)')
        ax8.set_ylabel('Frequency')
        ax8.set_title('Observer Diversity Distribution')
        ax8.axvline(x=diversity.mean(), color='red', linestyle='--',
                   label=f'Mean: {diversity.mean():.3f}', linewidth=2)
        ax8.legend()
        ax8.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = self.output_dir / 'observer_analysis.pdf'
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.png'), bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

    def plot_sdf_distribution_analysis(self):
        """Analyze SDF value distributions in detail"""
        print("\nGenerating SDF Distribution Analysis...")

        # Load model
        model = MultiPerspectiveSDFVAE(
            image_size=64, in_channels=3, latent_dim=128, num_observers=5
        )
        checkpoint = torch.load(
            'experiments/results/checkpoints/sdf_vae_fashion_latent128_obs5/best.pt',
            map_location=self.device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        # Load data
        _, _, test_loader = get_fashion_mnist_loaders(
            data_root='./data',
            batch_size=128,
            num_workers=4,
            image_size=64
        )

        # Collect SDF values
        all_sdf_values = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="  Collecting SDF values", leave=False):
                images = images.to(self.device)
                output = model(images)
                all_sdf_values.append(output['sdf_values'].cpu().numpy())
                all_labels.append(labels.numpy())

        sdf_values = np.vstack(all_sdf_values).squeeze(-1)  # [N, num_observers]
        labels = np.concatenate(all_labels)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Signed Distance Function (SDF) Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Overall SDF distribution
        ax = axes[0, 0]
        ax.hist(sdf_values.flatten(), bins=100, alpha=0.7, edgecolor='black', color='blue')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Manifold (SDF=0)')
        ax.set_xlabel('SDF Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Overall SDF Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 2: SDF per class
        ax = axes[0, 1]
        for class_id in range(10):
            mask = labels == class_id
            class_sdf = sdf_values[mask].flatten()
            ax.hist(class_sdf, bins=50, alpha=0.4, label=f'Class {class_id}')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('SDF Value')
        ax.set_ylabel('Frequency')
        ax.set_title('SDF Distribution per Class')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Mean SDF per class
        ax = axes[1, 0]
        mean_sdf_per_class = [sdf_values[labels == i].mean() for i in range(10)]
        std_sdf_per_class = [sdf_values[labels == i].std() for i in range(10)]
        ax.bar(range(10), mean_sdf_per_class, yerr=std_sdf_per_class,
              capsize=5, alpha=0.7, color=sns.color_palette("husl", 10))
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.set_xlabel('Class')
        ax.set_ylabel('Mean SDF Value')
        ax.set_title('Mean SDF per Class')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 4: SDF absolute value (distance to manifold)
        ax = axes[1, 1]
        abs_sdf = np.abs(sdf_values)
        ax.hist(abs_sdf.flatten(), bins=100, alpha=0.7, edgecolor='black', color='orange')
        mean_dist = abs_sdf.mean()
        ax.axvline(x=mean_dist, color='red', linestyle='--', linewidth=2,
                  label=f'Mean Distance: {mean_dist:.4f}')
        ax.set_xlabel('|SDF| (Distance to Manifold)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distance to Learned Manifold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = self.output_dir / 'sdf_distribution_analysis.pdf'
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.png'), bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

    def plot_latent_variance_stability(self):
        """Visualize the exceptional latent variance stability"""
        print("\nGenerating Latent Variance Stability Analysis...")

        models = {
            'SDF-VAE': (MultiPerspectiveSDFVAE(image_size=64, in_channels=3, latent_dim=128, num_observers=5),
                       'experiments/results/checkpoints/sdf_vae_fashion_latent128_obs5/best.pt'),
            'Vanilla VAE': (VanillaVAE(image_size=64, in_channels=3, latent_dim=128),
                           'experiments/results/checkpoints/vanilla_vae_fashion_latent128/best.pt'),
            'β-VAE': (BetaVAE(image_size=64, in_channels=3, latent_dim=128, beta=4.0),
                     'experiments/results/checkpoints/beta_vae_fashion_beta4.0_latent128/best.pt'),
        }

        # Load data
        _, _, test_loader = get_fashion_mnist_loaders(
            data_root='./data',
            batch_size=128,
            num_workers=4,
            image_size=64
        )

        # Collect latent statistics
        all_logvars = {}

        for name, (model, checkpoint_path) in models.items():
            print(f"  Processing {name}...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()

            logvars = []
            with torch.no_grad():
                for images, _ in tqdm(test_loader, desc=f"  Encoding {name}", leave=False):
                    images = images.to(self.device)
                    output = model(images)
                    logvars.append(output['logvar'].cpu().numpy())

            all_logvars[name] = np.vstack(logvars)

        # Create comparison figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Latent Space Variance Stability Analysis', fontsize=16, fontweight='bold')

        for idx, (name, logvar) in enumerate(all_logvars.items()):
            # Plot 1: Variance per dimension
            ax = axes[0, idx]
            var_per_dim = np.exp(logvar).mean(axis=0)
            ax.plot(var_per_dim, linewidth=2)
            ax.set_xlabel('Latent Dimension')
            ax.set_ylabel('Variance')
            ax.set_title(f'{name}\nVariance per Dimension')
            ax.grid(True, alpha=0.3)

            # Plot 2: LogVar distribution
            ax = axes[1, idx]
            ax.hist(logvar.flatten(), bins=100, alpha=0.7, edgecolor='black')
            mean_logvar = logvar.mean()
            std_logvar = logvar.std()
            ax.axvline(x=mean_logvar, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_logvar:.3f}\nStd: {std_logvar:.3f}')
            ax.set_xlabel('Log-Variance')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name}\nLog-Variance Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = self.output_dir / 'latent_variance_stability.pdf'
        plt.savefig(save_path, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.png'), bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

    def generate_all(self):
        """Generate all visualizations"""
        print("\n" + "="*70)
        print("GENERATING ALL PAPER VISUALIZATIONS")
        print("="*70)

        self.plot_training_curves()
        self.plot_sdf_specific_metrics()
        self.plot_latent_space_visualization()
        self.plot_observer_analysis()
        self.plot_sdf_distribution_analysis()
        self.plot_latent_variance_stability()

        print("\n" + "="*70)
        print("ALL VISUALIZATIONS COMPLETE!")
        print("="*70)
        print(f"\nFigures saved to: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob('*.pdf')):
            print(f"  - {file.name}")


if __name__ == '__main__':
    generator = VisualizationGenerator()
    generator.generate_all()
