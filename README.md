# Multi-Perspective SDF-VAE

**Interpretable Latent Representations through Signed Distance Functions**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A novel variational autoencoder architecture that learns interpretable latent representations through multi-perspective signed distance function learning. Trained and evaluated on CelebA (162k face images), with calibration/OOD validation on Fashion-MNIST.

---

## Key Results (CelebA, 64×64 RGB)

| Model | Test Recon Loss | Train-Test Ratio | LogVar Std | Attention Entropy |
|-------|----------------|-----------------|------------|-------------------|
| **SDF-VAE (Ours)** | 118.68 | **0.9996** | **0.004** | **1.609/1.609** |
| Vanilla VAE | **73.87** | 1.004 | 1.421 | — |
| β-VAE (β=4.0) | 126.74 | 1.003 | 1.198 | — |

**358× variance stability improvement** over Vanilla VAE (LogVar Std 0.004 vs 1.421).

Cross-dataset validation on Fashion-MNIST: ECE=0.129 (calibration), AUROC=0.992 (OOD detection).

---

## Architecture

The Multi-Perspective SDF-VAE uses:

1. **5 Observer Encoders** — each learns different geometric perspectives of the input
2. **SDF-based Latent Space** — represents features as signed distance functions
3. **Confidence-Weighted Aggregation** — combines observer outputs via learned attention
4. **Curriculum Learning** — 4-stage training schedule for stable multi-objective optimization

**Loss components:** Reconstruction + KL divergence + SDF consistency + Eikonal (‖∇SDF‖=1) + Diversity

See [`paper/architecture_diagrams_mermaid/`](paper/architecture_diagrams_mermaid/) for diagrams.

---

## Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/Multi-Perspective-SDF-VAE.git
cd Multi-Perspective-SDF-VAE

# Create venv with uv (recommended)
uv venv --python 3.11
source .venv/bin/activate

# Or with standard venv
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Dataset

Download CelebA and place images in `data/celeba/img_align_celeba/`:

```bash
# ~1.4GB — download from official source or Kaggle
# https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
```

The loader expects 202,599 `.jpg` files in that directory (standard CelebA aligned crop release).

### Training

```bash
# Train SDF-VAE on CelebA (100 epochs, ~681s/epoch on GPU)
python experiments/train_main_model.py --dataset celeba --epochs 100

# Train baselines
python experiments/train_baselines.py --model vanilla --dataset celeba --epochs 100
python experiments/train_baselines.py --model beta --beta 4.0 --dataset celeba --epochs 100
```

### Evaluation & Figures

```bash
# Compare all models
python experiments/compare_models.py

# Generate paper figures
python paper/generate_paper_figures.py
```

---

## Repository Structure

```
├── src/                          # Core source code
│   ├── models/                   # Model architectures
│   │   ├── sdf_vae.py           # Multi-Perspective SDF-VAE (42.3M params)
│   │   ├── vanilla_vae.py       # Vanilla VAE baseline (8.5M params)
│   │   └── beta_vae.py          # β-VAE baseline (8.5M params)
│   ├── training/                 # Training loops and curriculum learning
│   ├── data/                     # Dataset loaders (CelebA, Fashion-MNIST)
│   └── utils/                    # Utilities
│
├── experiments/                  # Experiment scripts
│   ├── train_main_model.py      # Train SDF-VAE
│   ├── train_baselines.py       # Train Vanilla/β-VAE
│   ├── compare_models.py        # Quantitative comparison
│   ├── evaluate_calibration.py  # Calibration (ECE)
│   ├── evaluate_ood_detection.py# OOD detection (AUROC)
│   └── results/                 # JSON results, calibration, OOD outputs
│
├── paper/                        # Research paper
│   ├── paper.pdf                # Compiled paper (32 pages)
│   ├── paper.tex                # LaTeX source
│   ├── references.bib           # Bibliography
│   ├── figures/                 # Publication figures (CelebA)
│   ├── architecture_diagrams_mermaid/  # Architecture diagrams
│   ├── generate_paper_figures.py
│   └── generate_visualizations.py
│
├── results/                      # Training outputs (gitignored)
│   ├── checkpoints/             # best.pt for each model
│   └── logs/                    # metrics.json per run
│
├── docs/                         # Documentation
│   └── data_requirements.md
│
├── README.md
├── SETUP.md                      # Detailed setup instructions
├── CONTRIBUTING.md
├── CITATION.cff
├── LICENSE
└── requirements.txt
```

---

## Datasets

- **CelebA** (primary): 162,770 train / 19,867 val / ~19,962 test, 64×64 RGB, center-cropped
- **Fashion-MNIST** (calibration/OOD cross-validation): 60k/10k, 28×28 → 64×64 RGB

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
tqdm>=4.65.0
Pillow>=9.0.0
```

See [`requirements.txt`](requirements.txt) for the full list.

---

## Citation

```bibtex
@misc{multiperspective_sdf_vae_2026,
  title={Multi-Perspective SDF-VAE: Interpretable Latent Representations through Signed Distance Functions},
  author={[Author]},
  year={2026},
  howpublished={\url{https://github.com/YOUR_USERNAME/Multi-Perspective-SDF-VAE}},
  note={Independent Research Project}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable citation.

---

## License

MIT — see [`LICENSE`](LICENSE).
