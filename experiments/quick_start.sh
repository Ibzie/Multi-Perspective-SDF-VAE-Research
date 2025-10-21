#!/bin/bash
# Quick start script to train all models on Fashion-MNIST

echo "=========================================="
echo "Multi-Perspective SDF-VAE Quick Start"
echo "=========================================="
echo ""
echo "This script will train:"
echo "1. Multi-Perspective SDF-VAE"
echo "2. Vanilla VAE (baseline)"
echo "3. β-VAE (baseline, β=4.0)"
echo ""
echo "Dataset: Fashion-MNIST"
echo "Epochs: 100"
echo "=========================================="
echo ""

# Change to project root
cd "$(dirname "$0")/.."

# Create directories
mkdir -p data results/checkpoints results/logs

# Train Multi-Perspective SDF-VAE
echo "Training Multi-Perspective SDF-VAE..."
python experiments/train_main_model.py \
    --dataset fashion \
    --epochs 100 \
    --batch_size 64 \
    --latent_dim 128 \
    --num_observers 5 \
    --experiment_name sdf_vae_fashion_quick

echo ""
echo "=========================================="
echo "Training Vanilla VAE baseline..."
python experiments/train_baselines.py \
    --model vanilla \
    --dataset fashion \
    --epochs 100 \
    --batch_size 64 \
    --latent_dim 128 \
    --experiment_name vanilla_vae_fashion_quick

echo ""
echo "=========================================="
echo "Training β-VAE baseline (β=4.0)..."
python experiments/train_baselines.py \
    --model beta \
    --beta 4.0 \
    --dataset fashion \
    --epochs 100 \
    --batch_size 64 \
    --latent_dim 128 \
    --experiment_name beta_vae_fashion_quick

echo ""
echo "=========================================="
echo "All training complete!"
echo "Results saved to: ./results/"
echo "=========================================="
