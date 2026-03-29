"""
Generate visualizations for perceptual metrics comparison

Creates both individual metric figures and combined comparison figures
for inclusion in the research paper.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 13

# Color palette (consistent with existing figures)
colors = {
    'SDF-VAE': '#2E86AB',      # Blue
    'Vanilla VAE': '#A23B72',  # Purple
    'β-VAE': '#F18F01'         # Orange
}


def load_results():
    """Load perceptual metrics results from JSON file."""
    results_file = Path(__file__).parent / 'results' / 'perceptual_metrics' / 'perceptual_metrics.json'

    with open(results_file, 'r') as f:
        results = json.load(f)

    return results


def plot_fid_comparison(results, output_dir):
    """Generate individual FID comparison figure."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    models = list(results.keys())
    fid_scores = [results[model]['FID'] for model in models]
    model_colors = [colors[model] for model in models]

    bars = ax.bar(models, fid_scores, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, score in zip(bars, fid_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.2f}',
               ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('FID Score', fontweight='bold')
    ax.set_title('Fréchet Inception Distance (FID)\nLower is Better', fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add annotation
    ax.text(0.02, 0.98, 'Measures distribution similarity\nusing Inception features',
           transform=ax.transAxes, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_dir / 'fid_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fid_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Generated: fid_comparison.pdf/png")


def plot_lpips_comparison(results, output_dir):
    """Generate individual LPIPS comparison figure."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    models = list(results.keys())
    lpips_scores = [results[model]['LPIPS'] for model in models]
    model_colors = [colors[model] for model in models]

    bars = ax.bar(models, lpips_scores, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, score in zip(bars, lpips_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.4f}',
               ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('LPIPS Distance', fontweight='bold')
    ax.set_title('Learned Perceptual Image Patch Similarity (LPIPS)\nLower is Better', fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add annotation
    ax.text(0.02, 0.98, 'Neural perceptual distance\nusing VGG features',
           transform=ax.transAxes, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_dir / 'lpips_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'lpips_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Generated: lpips_comparison.pdf/png")


def plot_ssim_comparison(results, output_dir):
    """Generate individual SSIM comparison figure."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    models = list(results.keys())
    ssim_scores = [results[model]['SSIM'] for model in models]
    model_colors = [colors[model] for model in models]

    bars = ax.bar(models, ssim_scores, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, score in zip(bars, ssim_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.4f}',
               ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('SSIM Score', fontweight='bold')
    ax.set_title('Structural Similarity Index (SSIM)\nHigher is Better (max 1.0)', fontweight='bold', pad=15)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add reference line at 1.0 (perfect similarity)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Perfect')
    ax.legend(loc='upper right')

    # Add annotation
    ax.text(0.02, 0.02, 'Measures structural similarity\nbetween images',
           transform=ax.transAxes, fontsize=9, va='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_dir / 'ssim_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'ssim_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Generated: ssim_comparison.pdf/png")


def plot_combined_metrics(results, output_dir):
    """Generate combined 3-panel comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    models = list(results.keys())
    model_colors = [colors[model] for model in models]

    # Panel (a): FID
    ax = axes[0]
    fid_scores = [results[model]['FID'] for model in models]
    bars = ax.bar(models, fid_scores, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar, score in zip(bars, fid_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.2f}',
               ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel('FID Score', fontweight='bold')
    ax.set_title('(a) FID\n(Lower is Better)', fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Panel (b): LPIPS
    ax = axes[1]
    lpips_scores = [results[model]['LPIPS'] for model in models]
    bars = ax.bar(models, lpips_scores, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar, score in zip(bars, lpips_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.4f}',
               ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel('LPIPS Distance', fontweight='bold')
    ax.set_title('(b) LPIPS\n(Lower is Better)', fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Panel (c): SSIM
    ax = axes[2]
    ssim_scores = [results[model]['SSIM'] for model in models]
    bars = ax.bar(models, ssim_scores, color=model_colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar, score in zip(bars, ssim_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.4f}',
               ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel('SSIM Score', fontweight='bold')
    ax.set_title('(c) SSIM\n(Higher is Better)', fontweight='bold', pad=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    plt.suptitle('Perceptual Quality Metrics Comparison', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'perceptual_metrics_combined.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'perceptual_metrics_combined.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Generated: perceptual_metrics_combined.pdf/png")


def plot_normalized_radar(results, output_dir):
    """Generate radar chart comparing all metrics (normalized)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection='polar'))

    models = list(results.keys())
    metrics = ['FID\n(inv)', 'LPIPS\n(inv)', 'SSIM']

    # Normalize metrics to [0, 1] where 1 is best
    # For FID and LPIPS: lower is better, so invert
    # For SSIM: higher is better

    fid_values = np.array([results[model]['FID'] for model in models])
    lpips_values = np.array([results[model]['LPIPS'] for model in models])
    ssim_values = np.array([results[model]['SSIM'] for model in models])

    # Normalize: map to [0, 1] where 1 is best
    # For metrics where lower is better, use 1 - normalized value
    fid_norm = 1 - (fid_values - fid_values.min()) / (fid_values.max() - fid_values.min() + 1e-10)
    lpips_norm = 1 - (lpips_values - lpips_values.min()) / (lpips_values.max() - lpips_values.min() + 1e-10)
    ssim_norm = (ssim_values - ssim_values.min()) / (ssim_values.max() - ssim_values.min() + 1e-10)

    # Angles for each metric
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the plot

    # Plot for each model
    for i, model in enumerate(models):
        values = [fid_norm[i], lpips_norm[i], ssim_norm[i]]
        values += values[:1]  # Close the plot

        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[model])
        ax.fill(angles, values, alpha=0.15, color=colors[model])

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)

    ax.set_title('Normalized Perceptual Metrics\n(1.0 = Best Performance)',
                fontweight='bold', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(output_dir / 'perceptual_metrics_radar.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'perceptual_metrics_radar.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Generated: perceptual_metrics_radar.pdf/png")


def plot_all_metrics_table_viz(results, output_dir):
    """Generate table-style visualization of all metrics."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')

    models = list(results.keys())

    # Create table data
    table_data = []
    table_data.append(['Metric', 'SDF-VAE', 'Vanilla VAE', 'β-VAE', 'Best'])

    # FID row
    fid_vals = [results[m]['FID'] for m in models]
    best_fid_idx = np.argmin(fid_vals)
    fid_row = ['FID ↓'] + [f"{v:.2f}" for v in fid_vals] + [models[best_fid_idx]]
    table_data.append(fid_row)

    # LPIPS row
    lpips_vals = [results[m]['LPIPS'] for m in models]
    best_lpips_idx = np.argmin(lpips_vals)
    lpips_row = ['LPIPS ↓'] + [f"{v:.4f}" for v in lpips_vals] + [models[best_lpips_idx]]
    table_data.append(lpips_row)

    # SSIM row
    ssim_vals = [results[m]['SSIM'] for m in models]
    best_ssim_idx = np.argmax(ssim_vals)
    ssim_row = ['SSIM ↑'] + [f"{v:.4f}" for v in ssim_vals] + [models[best_ssim_idx]]
    table_data.append(ssim_row)

    # Color cells
    cell_colors = []

    # Header row
    cell_colors.append(['lightgray'] * 5)

    # Data rows - highlight best values
    for row_idx, (metric_vals, best_idx) in enumerate([
        (fid_vals, best_fid_idx),
        (lpips_vals, best_lpips_idx),
        (ssim_vals, best_ssim_idx)
    ]):
        row_colors = ['white']  # Metric name
        for i in range(3):
            if i == best_idx:
                row_colors.append('#90EE90')  # Light green for best
            else:
                row_colors.append('white')
        row_colors.append('lightyellow')  # Best column
        cell_colors.append(row_colors)

    table = ax.table(cellText=table_data, cellColours=cell_colors,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.18, 0.18, 0.18, 0.18])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Make header row bold
    for i in range(5):
        table[(0, i)].set_text_props(weight='bold')

    ax.set_title('Perceptual Quality Metrics Summary\n↓ = Lower is Better, ↑ = Higher is Better',
                fontweight='bold', fontsize=13, pad=20)

    plt.savefig(output_dir / 'perceptual_metrics_table.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'perceptual_metrics_table.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Generated: perceptual_metrics_table.pdf/png")


def main():
    """Generate all perceptual metrics visualizations."""
    print("="*60)
    print("Generating Perceptual Metrics Visualizations")
    print("="*60 + "\n")

    # Load results
    results = load_results()
    print(f"Loaded results for {len(results)} models\n")

    # Output directory
    output_dir = Path(__file__).parent / 'results' / 'perceptual_metrics'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate individual metric figures
    print("Generating individual metric figures...")
    plot_fid_comparison(results, output_dir)
    plot_lpips_comparison(results, output_dir)
    plot_ssim_comparison(results, output_dir)

    # Generate combined figures
    print("\nGenerating combined figures...")
    plot_combined_metrics(results, output_dir)
    plot_normalized_radar(results, output_dir)
    plot_all_metrics_table_viz(results, output_dir)

    print("\n" + "="*60)
    print("All visualizations generated successfully!")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
