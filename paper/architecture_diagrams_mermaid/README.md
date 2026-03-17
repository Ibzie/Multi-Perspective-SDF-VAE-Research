# Architecture Diagrams - Mermaid Source Files

This directory contains Mermaid diagram source code for all architecture diagrams in the Multi-Perspective SDF-VAE paper.

## Files

1. **01_architecture_overview.md** - Complete end-to-end architecture showing all components from input to reconstruction
2. **02_light_source_mechanism.md** - Light source projection creating universal reference space
3. **03_observer_architecture.md** - Internal structure of a single observer (feature + SDF branches)
4. **04_sdf_confidence_aggregation.md** - How SDF values convert to confidence weights and aggregate features
5. **05_vae_comparison.md** - Side-by-side comparison of Vanilla VAE vs Multi-Perspective SDF-VAE
6. **06_sdf_concept.md** - Conceptual explanation of signed distance functions (inside/outside/on-manifold)
7. **07_reparameterization_trick.md** - Reparameterization trick for latent sampling (z = μ + σ⊙ε)
8. **08_weighting_comparison.md** - Learned attention vs geometric confidence comparison
9. **09_curriculum_learning.md** - 4-stage curriculum learning schedule (Gantt chart)
10. **10_loss_components.md** - All 5 loss function components and their purposes
11. **11_kl_divergence_concept.md** - KL divergence explanation and its role in VAE training
12. **12_eikonal_constraint.md** - Eikonal constraint (||∇SDF|| = 1) and its importance

## Usage

Each markdown file contains ONLY the Mermaid diagram code block. To convert to PNG:

1. Copy the Mermaid code from each `.md` file
2. Use a Mermaid rendering tool:
   - [Mermaid Live Editor](https://mermaid.live/)
   - [mermaid-cli](https://github.com/mermaid-js/mermaid-cli) (mmdc command)
   - VS Code Mermaid extension
3. Export as PNG at 300 DPI for publication quality
4. Save PNG with corresponding name in `paper/figures/` directory

## Color Scheme

- **Light Blue (#e1f5ff)**: Input/Features
- **Light Red (#ffe1e1)**: SDF-related components
- **Light Green (#e1ffe1)**: Observers/Outputs
- **Light Yellow (#fff4e1)**: Light source/Special components
- **Light Purple (#f4e1ff)**: Aggregation/Latent space
- **Light Gray (#f9f9f9)**: Notes/Explanations

## Recommended Export Settings

- **Format**: PNG
- **DPI**: 300 (for publication)
- **Background**: White (or transparent)
- **Width**: 1200-1600 pixels (will scale down appropriately)
