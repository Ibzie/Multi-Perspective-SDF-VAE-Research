#!/usr/bin/env python3
"""
Visualize Calibration and OOD Detection Results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

# Colors
COLORS = {
    'SDF-VAE': '#2E86AB',
    'Vanilla VAE': '#A23B72',
    'β-VAE': '#F18F01'
}


def plot_calibration_curves():
    """Plot calibration curves showing binned uncertainty vs error"""
    # Load data
    with open('experiments/results/calibration/calibration_summary.json') as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (model_name, model_data) in enumerate(data.items()):
        ax = axes[idx]

        bin_confs = np.array(model_data['bin_confidences'])
        bin_accs = np.array(model_data['bin_accuracies'])
        bin_counts = np.array(model_data['bin_counts'])

        # Filter out empty bins
        mask = bin_counts > 0
        bin_confs = bin_confs[mask]
        bin_accs = bin_accs[mask]
        bin_counts = bin_counts[mask]

        # Normalize bin sizes for visualization
        max_count = bin_counts.max()
        sizes = (bin_counts / max_count) * 500

        # Plot calibration curve
        ax.scatter(bin_confs, bin_accs, s=sizes, alpha=0.6,
                   color=COLORS[model_name], label=model_name, edgecolors='black', linewidth=0.5)

        # Plot perfect calibration line
        min_val = min(bin_confs.min(), bin_accs.min())
        max_val = max(bin_confs.max(), bin_accs.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'k--', alpha=0.5, linewidth=1.5, label='Perfect Calibration')

        # Set labels
        ax.set_xlabel('Predicted Uncertainty', fontsize=11)
        ax.set_ylabel('Actual Error', fontsize=11)
        ax.set_title(f'{model_name}\nECE: {model_data["ece"]:.4f}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiments/results/calibration/calibration_curves.pdf', bbox_inches='tight')
    plt.savefig('experiments/results/calibration/calibration_curves.png', bbox_inches='tight')
    plt.close()

    print("✓ Generated: calibration_curves.pdf/png")


def plot_uncertainty_error_scatter():
    """Plot scatter plots of uncertainty vs error for all models"""
    # Load full data
    with open('experiments/results/calibration/calibration_full.json') as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (model_name, model_data) in enumerate(data.items()):
        ax = axes[idx]

        uncertainties = np.array(model_data['uncertainties'])
        errors = np.array(model_data['errors'])

        # Sample for visualization (too many points)
        n_samples = min(5000, len(uncertainties))
        indices = np.random.choice(len(uncertainties), n_samples, replace=False)

        # Create 2D histogram
        h = ax.hist2d(uncertainties[indices], errors[indices],
                      bins=50, cmap='Blues', cmin=1)
        plt.colorbar(h[3], ax=ax, label='Count')

        # Add correlation text
        corr = model_data['correlation']
        p_val = model_data['correlation_pvalue']
        ax.text(0.05, 0.95, f'Spearman ρ: {corr:.3f}\np < {p_val:.1e}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Predicted Uncertainty', fontsize=11)
        ax.set_ylabel('Reconstruction Error', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('experiments/results/calibration/uncertainty_error_scatter.pdf', bbox_inches='tight')
    plt.savefig('experiments/results/calibration/uncertainty_error_scatter.png', bbox_inches='tight')
    plt.close()

    print("✓ Generated: uncertainty_error_scatter.pdf/png")


def plot_ood_roc_curves():
    """Plot ROC curves for OOD detection"""
    # Load data
    with open('experiments/results/ood_detection/ood_detection_full.json') as f:
        data = json.load(f)

    # Create figure with subplots for each scoring method
    score_types = ['recon_error', 'kl_div', 'sdf_score']
    score_names = {'recon_error': 'Reconstruction Error',
                   'kl_div': 'KL Divergence',
                   'sdf_score': 'SDF Score (SDF-VAE only)'}

    fig, axes = plt.subplots(1, len(score_types), figsize=(15, 4))

    for idx, score_type in enumerate(score_types):
        ax = axes[idx]

        for model_name, model_data in data.items():
            if score_type not in model_data['in_scores']:
                continue

            # Get scores
            in_scores = np.array(model_data['in_scores'][score_type])
            ood_scores = np.array(model_data['ood_scores'][score_type])

            # Create labels
            y_true = np.concatenate([np.zeros(len(in_scores)), np.ones(len(ood_scores))])
            y_scores = np.concatenate([in_scores, ood_scores])

            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_scores)

            # Get AUROC
            auroc = model_data[score_type]['auroc']

            # Plot
            ax.plot(fpr, tpr, label=f'{model_name} (AUROC: {auroc:.3f})',
                    color=COLORS[model_name], linewidth=2)

        # Plot random baseline
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5, label='Random')

        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(score_names[score_type], fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiments/results/ood_detection/ood_roc_curves.pdf', bbox_inches='tight')
    plt.savefig('experiments/results/ood_detection/ood_roc_curves.png', bbox_inches='tight')
    plt.close()

    print("✓ Generated: ood_roc_curves.pdf/png")


def plot_ood_comparison_bar():
    """Create bar chart comparing OOD detection performance"""
    # Load summary
    with open('experiments/results/ood_detection/ood_detection_summary.json') as f:
        data = json.load(f)

    # Prepare data
    models = list(data.keys())
    score_types = ['recon_error', 'kl_div', 'sdf_score']
    score_names_short = {'recon_error': 'Recon. Error',
                         'kl_div': 'KL Div.',
                         'sdf_score': 'SDF Score'}

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(models))
    width = 0.25

    for idx, score_type in enumerate(score_types):
        aurocs = []
        for model in models:
            if score_type in data[model]:
                aurocs.append(data[model][score_type]['auroc'])
            else:
                aurocs.append(0)

        offset = (idx - 1) * width
        bars = ax.bar(x + offset, aurocs, width, label=score_names_short[score_type])

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, aurocs)):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax.set_title('Out-of-Distribution Detection Performance', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('experiments/results/ood_detection/ood_comparison_bar.pdf', bbox_inches='tight')
    plt.savefig('experiments/results/ood_detection/ood_comparison_bar.png', bbox_inches='tight')
    plt.close()

    print("✓ Generated: ood_comparison_bar.pdf/png")


def plot_combined_summary():
    """Create combined summary figure for paper"""
    # Load data
    with open('experiments/results/calibration/calibration_summary.json') as f:
        calib_data = json.load(f)

    with open('experiments/results/ood_detection/ood_detection_summary.json') as f:
        ood_data = json.load(f)

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.3)

    # (a) Calibration ECE comparison
    ax1 = fig.add_subplot(gs[0, 0])
    models = list(calib_data.keys())
    eces = [calib_data[m]['ece'] for m in models]
    colors = [COLORS[m] for m in models]

    bars = ax1.bar(models, eces, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, eces):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax1.set_ylabel('Expected Calibration Error', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Uncertainty Calibration', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(eces) * 1.2)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticklabels(models, rotation=0, fontsize=10)

    # (b) OOD Detection - Reconstruction Error
    ax2 = fig.add_subplot(gs[0, 1])
    aurocs_recon = [ood_data[m]['recon_error']['auroc'] for m in models]

    bars = ax2.bar(models, aurocs_recon, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, aurocs_recon):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax2.set_ylabel('AUROC', fontsize=11, fontweight='bold')
    ax2.set_title('(b) OOD Detection (Recon. Error)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticklabels(models, rotation=0, fontsize=10)

    # (c) SDF-based OOD Detection
    ax3 = fig.add_subplot(gs[0, 2])

    # Only SDF-VAE has SDF score
    sdf_auroc = ood_data['SDF-VAE']['sdf_score']['auroc']
    sdf_recon = ood_data['SDF-VAE']['recon_error']['auroc']

    methods = ['SDF Score', 'Recon. Error']
    aurocs = [sdf_auroc, sdf_recon]
    method_colors = ['#2E86AB', '#64B5CD']

    bars = ax3.bar(methods, aurocs, color=method_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, aurocs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax3.set_ylabel('AUROC', fontsize=11, fontweight='bold')
    ax3.set_title('(c) SDF-VAE OOD Methods', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticklabels(methods, rotation=0, fontsize=10)

    plt.savefig('experiments/results/calibration_ood_summary.pdf', bbox_inches='tight')
    plt.savefig('experiments/results/calibration_ood_summary.png', bbox_inches='tight')
    plt.close()

    print("✓ Generated: calibration_ood_summary.pdf/png")


def main():
    print("=" * 60)
    print("Generating Calibration and OOD Visualizations")
    print("=" * 60)

    plot_calibration_curves()
    plot_uncertainty_error_scatter()
    plot_ood_roc_curves()
    plot_ood_comparison_bar()
    plot_combined_summary()

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
