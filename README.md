# Multi-Perspective SDF-VAE

**Interpretable Latent Representations through Signed Distance Functions**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An exploratory research project investigating whether signed distance functions can serve as a geometric inductive bias in variational autoencoders to improve latent space interpretability and uncertainty calibration. Trained on CelebA (162k face images), with calibration/OOD cross-validation on Fashion-MNIST.

> **Status:** Independent research / exploratory project. The architecture achieves its primary goals around variance stability and calibration, but does not improve reconstruction quality over simpler baselines — see [Limitations](#limitations).

---

## What This Is

Standard VAEs produce latent spaces where the estimated variance (`log σ²`) is often unreliable. This project explores using **signed distance functions** as a structured prior: K independent observer networks each produce an SDF value representing how close the input is to their learned manifold, and these SDF-based confidence scores drive a weighted aggregation of observer features into the latent parameters.

The hypothesis was that geometric constraints (SDF consistency + Eikonal regularization) would produce more reliable uncertainty estimates and more interpretable latent spaces. The variance stability and calibration results support this; the reconstruction quality result does not.

---

## Results

### CelebA (64×64 RGB)

| Model | Test Recon Loss ↓ | Train/Test Ratio | LogVar Std ↓ | Attn Entropy |
|-------|:-----------------:|:----------------:|:------------:|:------------:|
| **SDF-VAE (Ours)** | 118.68 | **0.9996** | **0.004** | **1.609/1.609** |
| Vanilla VAE | **73.87** | 1.004 | 1.421 | — |
| β-VAE (β=4.0) | 126.74 | 1.003 | 1.198 | — |

**358× variance stability improvement** (LogVar Std 0.004 vs 1.421). Reconstruction loss is 60% higher than Vanilla VAE.

### Fashion-MNIST Cross-Dataset Validation

| Metric | SDF-VAE | Vanilla VAE | β-VAE |
|--------|:-------:|:-----------:|:-----:|
| ECE (calibration) ↓ | **0.129** | 0.461 | 0.325 |
| OOD AUROC ↑ | **0.992** | 0.682 | 0.745 |

Result figures: [`experiments/results/`](experiments/results/)

---

## Architecture

```
Input [B, 3, 64, 64]
  └─ CNNEncoder → flat_features [B, 4096]
       ├─ LightSource MLP → light_projection [B, 256]
       └─ concat → observer_input [B, 4352]
            └─ × 5 SDFObserver (independent)
                 ├─ sdf_value [B, 1]  ∈ [-0.1, 0.1]
                 └─ feature_projection [B, 256]
            └─ ManifoldAggregator
                 ├─ confidence = exp(-|sdf| × 10)
                 ├─ weights = softmax(confidence)     [B, 5, 1]
                 ├─ aggregated = Σ weights × features [B, 256]
                 └─ → μ [B, 128],  log σ² [B, 128]  (bounded [-1, 1])
  └─ CNNDecoder(z) → reconstruction [B, 3, 64, 64]
```

**Loss:** `L = Lrecon + 0.2·Lkl + 1.0·Lconsistency + 0.5·Leikonal − 0.1·Ldiversity + 0.5·Lsupervision`

Trained with a 4-stage curriculum over 20 warm-up epochs (reconstruction only → +KL/diversity → +SDF geometry → full objective).

Full write-up: [`paper/paper.pdf`](paper/paper.pdf) · Architecture diagrams: [`paper/architecture_diagrams_mermaid/`](paper/architecture_diagrams_mermaid/)

---

## Limitations

- **Reconstruction quality is worse than Vanilla VAE.** The geometric constraints smooth the latent manifold in a way that trades pixel fidelity for stability.
- **The bounded log-variance `[-1, 1]`** is an architectural constraint baked into `ManifoldAggregator`, not an emergent property of the SDF training.
- **All 5 observers see identical input.** Diversity is enforced only by the diversity loss and random initialization — there are no genuinely different views.
- **Eikonal is approximated in feature space**, not pixel/spatial space. It's a soft pairwise Lipschitz penalty on observer feature vectors, not a true PDE constraint.
- The SDF range `[-0.1, 0.1]` is very tight — the geometric interpretation is qualitative.

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/Multi-Perspective-SDF-VAE.git
cd Multi-Perspective-SDF-VAE

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download CelebA and place the 202,599 `.jpg` files at `data/celeba/img_align_celeba/`. See [`SETUP.md`](SETUP.md) for details.

### Training

```bash
# Train SDF-VAE on CelebA (~100 epochs)
python experiments/train_main_model.py --dataset celeba --epochs 100

# Train baselines
python experiments/train_baselines.py --model vanilla --dataset celeba --epochs 100
python experiments/train_baselines.py --model beta --beta 4.0 --dataset celeba --epochs 100
```

### Evaluation

```bash
python experiments/compare_models.py                # visual reconstruction/sampling/interpolation
python experiments/quantitative_comparison.py       # metrics table
python experiments/evaluate_calibration.py          # ECE / reliability diagrams
python experiments/evaluate_ood_detection.py        # AUROC / ROC curves
python experiments/diagnose_sdf_vae.py              # observer attention & SDF analysis
python experiments/evaluate_perceptual_metrics.py   # FID, LPIPS, SSIM
```

---

## Repository Structure

```
├── src/
│   ├── models/
│   │   ├── components.py        # CNNEncoder, LightSource, SDFObserver,
│   │   │                        #   ManifoldAggregator, CNNDecoder, SwiGLU
│   │   ├── sdf_vae.py           # MultiPerspectiveSDFVAE  (~42M params)
│   │   ├── baselines.py         # VanillaVAE, BetaVAE, BaselineVAELoss (~8.5M each)
│   │   └── losses.py            # MultiPerspectiveSDFVAELoss (6 terms)
│   ├── training/
│   │   ├── trainer.py           # Training loop, checkpointing, early stopping
│   │   └── curriculum.py        # 4-stage curriculum + adaptive variant
│   ├── data/
│   │   └── datasets.py          # FashionMNISTRGB, MedicalMNIST, CelebA loaders
│   └── utils/
│       └── metrics_logger.py    # Per-step and per-epoch metrics tracking
│
├── experiments/
│   ├── train_main_model.py
│   ├── train_baselines.py
│   ├── compare_models.py
│   ├── quantitative_comparison.py
│   ├── evaluate_calibration.py
│   ├── evaluate_ood_detection.py
│   ├── evaluate_perceptual_metrics.py
│   ├── diagnose_sdf_vae.py
│   ├── test_generalization.py
│   └── results/                 # Output figures and JSON metrics
│
├── paper/
│   ├── paper.pdf                # Full write-up (39 pages)
│   ├── paper.tex                # LaTeX source
│   ├── references.bib
│   ├── figures/                 # Publication figures (PDFs + PNGs)
│   ├── architecture_diagrams_mermaid/  # Architecture PNGs
│   ├── generate_paper_figures.py
│   └── generate_visualizations.py
│
├── data/                        # Gitignored — download separately (see SETUP.md)
├── results/                     # Gitignored — generated at training time
├── README.md
├── SETUP.md
├── CONTRIBUTING.md
├── CITATION.cff
├── LICENSE
└── requirements.txt
```

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
tqdm>=4.65.0
Pillow>=9.0.0
medmnist>=2.2.3
```

See [`requirements.txt`](requirements.txt) for the full pinned list.

---

## Citation

```bibtex
@misc{akhtar2026sdfvae,
  title   = {Multi-Perspective {SDF}-{VAE}: Interpretable Latent Representations
             through Signed Distance Functions},
  author  = {Akhtar, Ibrahim},
  year    = {2026},
  note    = {Independent research project},
  url     = {https://github.com/YOUR_USERNAME/Multi-Perspective-SDF-VAE}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable citation.

---

## License

MIT — see [`LICENSE`](LICENSE).
