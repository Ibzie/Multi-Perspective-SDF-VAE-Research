# Setup Instructions

## Installation

### 1. Create Virtual Environment
```bash
cd Multi-Perspective-SDF-VAE-Research
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### Train All Models (Fashion-MNIST)
```bash
bash experiments/quick_start.sh
```

This will train:
- Multi-Perspective SDF-VAE
- Vanilla VAE (baseline)
- β-VAE (baseline, β=4.0)

### Train Individual Models

**Multi-Perspective SDF-VAE:**
```bash
python experiments/train_main_model.py \
    --dataset fashion \
    --epochs 100 \
    --batch_size 64 \
    --latent_dim 128 \
    --num_observers 5
```

**Vanilla VAE:**
```bash
python experiments/train_baselines.py \
    --model vanilla \
    --dataset fashion \
    --epochs 100 \
    --batch_size 64 \
    --latent_dim 128
```

**β-VAE:**
```bash
python experiments/train_baselines.py \
    --model beta \
    --beta 4.0 \
    --dataset fashion \
    --epochs 100 \
    --batch_size 64 \
    --latent_dim 128
```

### Medical MNIST

Replace `--dataset fashion` with `--dataset medical` in any command:

```bash
python experiments/train_main_model.py \
    --dataset medical \
    --epochs 50 \
    --batch_size 64
```

## Directory Structure After Training

```
Multi-Perspective-SDF-VAE-Research/
├── data/                          # Downloaded datasets
│   ├── FashionMNIST/
│   └── medmnist/
├── results/
│   ├── checkpoints/               # Model checkpoints
│   │   ├── sdf_vae_fashion/
│   │   ├── vanilla_vae_fashion/
│   │   └── beta_vae_fashion/
│   └── logs/                      # Training logs & metrics
│       ├── sdf_vae_fashion/
│       │   ├── metrics.pkl
│       │   └── metrics.json
│       └── ...
└── notebooks/                     # Analysis notebooks (coming soon)
```

## Analysis

After training, use the notebooks to analyze results:

```bash
jupyter notebook notebooks/
```

Available notebooks:
1. `01_latent_space_exploration.ipynb` - Latent space controllability
2. `02_observer_specialization.ipynb` - Observer interpretation
3. `03_baseline_comparisons.ipynb` - Model comparison
4. `04_scalability_analysis.ipynb` - Computational analysis
5. `05_generalization_study.ipynb` - Perfect generalization analysis
6. `06_paper_figures.ipynb` - Publication figures

## Command-Line Arguments

### Common Arguments

```
--dataset fashion|medical    Dataset to use
--epochs INT                 Number of training epochs
--batch_size INT            Batch size
--latent_dim INT            Latent space dimension
--lr FLOAT                  Learning rate
--device cuda|cpu           Device to use
--experiment_name STR       Custom experiment name
```

### Multi-Perspective SDF-VAE Specific

```
--num_observers INT         Number of observers (default: 5)
--light_dim INT            Light source dimension (default: 256)
--projection_dim INT       Observer projection dimension (default: 256)
--warmup_epochs INT        Curriculum warmup epochs (default: 5)
```

### β-VAE Specific

```
--beta FLOAT               Beta weight on KL divergence (default: 4.0)
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python experiments/train_main_model.py --batch_size 32
```

Or reduce image size:
```bash
python experiments/train_main_model.py --image_size 32
```

### Slow Training

Increase number of workers:
```bash
python experiments/train_main_model.py --num_workers 8
```

### medmnist Installation Issues

If automatic download fails:
```bash
pip install --upgrade medmnist
```

## Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- GPU: Optional (CUDA-capable with 4GB+ VRAM)

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB+
- GPU: CUDA-capable with 8GB+ VRAM (e.g., RTX 3070, RTX 4060)

**Training Times** (Fashion-MNIST, 100 epochs):
- GPU (RTX 4060): ~2-3 hours
- CPU only: ~12-15 hours

## Next Steps

1. Train models using quick_start.sh
2. Analyze results in notebooks
3. Compare with baselines
4. Generate publication figures
5. Write research paper

## Support

For issues or questions:
1. Check this setup guide
2. Review code documentation
3. Check GitHub issues (if applicable)
