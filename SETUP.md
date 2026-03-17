# Setup Instructions

## Requirements

- Python 3.11+
- CUDA-capable GPU recommended (RTX 3070 or better for CelebA training)
- ~2GB disk space for CelebA images (not included in repo)

## Installation

### 1. Create Virtual Environment

**Using uv (recommended):**
```bash
uv venv --python 3.11
source .venv/bin/activate
```

**Using standard venv:**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Dataset Setup

### CelebA (Primary Dataset)

1. Download the aligned & cropped CelebA images from the [official source](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or Kaggle
2. Place the 202,599 `.jpg` files at:
   ```
   data/celeba/img_align_celeba/
   ```

The data loader handles train/val/test splits (162,770 / 19,867 / ~19,962) automatically using the standard CelebA partition file, or falls back to an 80/10/10 random split.

### Fashion-MNIST (Calibration / OOD validation)

Downloaded automatically on first run via torchvision.

## Training

### CelebA (Full Run)

```bash
# SDF-VAE (~681s/epoch, ~19 hours total on GPU)
python experiments/train_main_model.py --dataset celeba --epochs 100 --batch_size 64

# Vanilla VAE baseline (~43s/epoch)
python experiments/train_baselines.py --model vanilla --dataset celeba --epochs 100

# beta-VAE baseline (~38s/epoch)
python experiments/train_baselines.py --model beta --beta 4.0 --dataset celeba --epochs 100
```

Checkpoints and logs save to `results/checkpoints/` and `results/logs/`.

### Key Training Arguments

```
--dataset celeba|fashion      Dataset to use
--epochs INT                  Training epochs (default: 100)
--batch_size INT              Batch size (default: 64)
--latent_dim INT              Latent dimension (default: 128)
--lr FLOAT                    Learning rate (default: 1e-4)
--device cuda|cpu             Device (auto-detected if omitted)
--num_observers INT           Number of SDF observers (SDF-VAE only, default: 5)
```

## Evaluation

```bash
# Quantitative comparison across all models
python experiments/compare_models.py

# Calibration (ECE) -- uses Fashion-MNIST
python experiments/evaluate_calibration.py

# OOD detection (AUROC) -- Fashion-MNIST vs MNIST
python experiments/evaluate_ood_detection.py
```

## Paper Figures

```bash
# Requires trained CelebA checkpoints in results/checkpoints/
python paper/generate_paper_figures.py
# Outputs to paper/figures/
```

## Hardware Notes

| Configuration | SDF-VAE/epoch | Baselines/epoch |
|--------------|--------------|----------------|
| RTX 4060 8GB | ~681s | ~40s |
| CPU only | ~60-90 min | ~5 min |

For GPU with less than 8GB VRAM, reduce batch size: `--batch_size 32`

## Troubleshooting

**CUDA out of memory**: Use `--batch_size 32` or `--batch_size 16`

**CelebA images not found**: Verify `data/celeba/img_align_celeba/` contains `.jpg` files

**Slow training**: Increase workers with `--num_workers 8`
