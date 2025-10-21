# Multi-Perspective SDF-VAE Research

**Interpretable Latent Representations through Signed Distance Functions**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A novel variational autoencoder architecture that achieves **perfect generalization** (zero train-test gap) and **interpretable latent spaces** through multi-perspective signed distance function learning.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train all models on Fashion-MNIST
bash experiments/quick_start.sh

# Or train individual model
python experiments/train_main_model.py --dataset fashion --epochs 100
```

See [SETUP.md](SETUP.md) for detailed installation and usage instructions.

## Testing

```bash
# Test training pipeline (2 epochs)
python3 experiments/test_training.py
```

---

## Repository Structure

```
├── src/                    # Core source code
│   ├── models/            # Model architectures
│   ├── training/          # Training loops and curriculum learning
│   ├── losses/            # Loss functions
│   ├── utils/             # Utilities
│   └── analysis/          # Analysis and visualization tools
├── configs/               # Experiment configurations
├── experiments/           # Experiment scripts and ablations
├── notebooks/            # Jupyter notebooks for analysis
├── results/              # Experimental results
│   ├── figures/          # Publication-quality figures
│   ├── checkpoints/      # Model checkpoints
│   └── logs/             # Training logs
├── data/                 # Datasets
└── docs/                 # Documentation and paper drafts
```

## Research Questions

### 1. Latent Space Controllability
**Q**: Can we control generation by manipulating SDF values in latent space?

**Experiments Required**:
- [ ] Latent space interpolation with SDF tracking
- [ ] SDF-guided latent traversal
- [ ] Image generation at specific SDF levels

**Metrics**:
- Reconstruction quality vs SDF distance
- Interpolation smoothness
- Generated sample diversity

### 2. Observer Interpretation
**Q**: What features does each observer learn?

**Experiments Required**:
- [ ] Per-observer activation visualization
- [ ] Feature attribution analysis
- [ ] Observer ablation study

**Metrics**:
- Observer attention weights distribution
- Feature diversity scores
- Reconstruction quality per observer

### 3. Baseline Comparison
**Q**: How does Multi-Perspective SDF-VAE compare to standard VAEs?

**Experiments Required**:
- [ ] Train Vanilla VAE on same datasets
- [ ] Train β-VAE baseline
- [ ] Compare latent space interpretability

**Metrics**:
- Reconstruction loss (train/test)
- FID score
- Latent space disentanglement (MIG, SAP, DCI)
- Generalization gap

### 4. Scalability Analysis
**Q**: Does the Eikonal loss limit scalability?

**Experiments Required**:
- [ ] Training time vs batch size
- [ ] Memory profiling
- [ ] Ablation: with/without Eikonal loss

**Metrics**:
- Training time per epoch
- GPU memory usage
- Convergence speed
- Final performance with/without Eikonal

### 5. Generalization Mechanism
**Q**: Why does the model achieve perfect train-test overlap?

**Experiments Required**:
- [ ] Ablation study: Remove each loss component
- [ ] Curriculum vs non-curriculum comparison
- [ ] Observer diversity impact

**Metrics**:
- Train-test gap over time
- Overfitting indicators
- Loss component contributions

## Key Contributions

1. **Perfect Generalization**: Train and test losses track identically (no overfitting)
2. **Interpretable Latent Space**: SDF-based geometry provides semantic meaning
3. **Observer Specialization**: Multi-perspective learning with diversity enforcement
4. **Novel Loss Formulation**: SDF consistency + Eikonal + Diversity constraints
5. **Curriculum Learning**: Stable training for multi-objective optimization

## Datasets

- **Fashion-MNIST**: 60k training, 10k test (28×28 grayscale → 64×64 RGB)
- **Medical MNIST** (DermaMNIST): Skin lesion images for medical imaging application

## Baseline Models

- Vanilla VAE
- β-VAE (β=4.0)
- Multi-Perspective SDF-VAE (Ours)

## Citation

```bibtex
@article{multiperspective_sdf_vae_2025,
  title={Multi-Perspective SDF-VAE: Interpretable Latent Representations through Signed Distance Functions},
  author={[Your Name]},
  year={2025}
}
```
