#!/usr/bin/env python3
"""
Quick test script to verify training pipeline works

Runs for just 2 epochs with small batch size to catch errors quickly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim

from src.models import MultiPerspectiveSDFVAE
from src.training import Trainer
from src.data import get_fashion_mnist_loaders


def test_training():
    """Test that training pipeline works end-to-end"""

    print("=" * 70)
    print("Testing Multi-Perspective SDF-VAE Training Pipeline")
    print("=" * 70)

    # Load data with small batch size
    print("\n1. Loading data...")
    train_loader, val_loader, test_loader = get_fashion_mnist_loaders(
        data_root='./data',
        batch_size=16,  # Small batch for quick test
        num_workers=2,
        image_size=64,
        augment_train=False  # Faster without augmentation
    )
    print("✓ Data loaded successfully")

    # Create model
    print("\n2. Creating model...")
    model = MultiPerspectiveSDFVAE(
        image_size=64,
        in_channels=3,
        latent_dim=64,  # Smaller for speed
        light_dim=128,
        projection_dim=128,
        num_observers=3  # Fewer observers for speed
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters")

    # Create optimizer
    print("\n3. Setting up optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    print("✓ Optimizer created")

    # Create trainer
    print("\n4. Creating trainer...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        checkpoint_dir='./test_checkpoints',
        log_dir='./test_logs',
        experiment_name='pipeline_test'
    )
    print("✓ Trainer created")

    # Run training for 2 epochs
    print("\n5. Running training test (2 epochs)...")
    print("-" * 70)

    try:
        trainer.train(
            num_epochs=2,
            warmup_epochs=1,
            log_interval=50,
            save_interval=1,
            early_stopping_patience=10
        )
        print("-" * 70)
        print("✓ Training completed successfully!")

    except Exception as e:
        print("-" * 70)
        print(f"✗ Training failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test inference
    print("\n6. Testing inference...")
    try:
        model.eval()
        with torch.no_grad():
            # Get a test batch
            test_images, _ = next(iter(test_loader))
            test_images = test_images[:4].to(device)  # Just 4 images

            # Encode
            mean, logvar = model.encode(test_images)
            print(f"   ✓ Encoding works (latent shape: {mean.shape})")

            # Decode
            reconstruction = model.decode(mean)
            print(f"   ✓ Decoding works (recon shape: {reconstruction.shape})")

            # Sample
            samples = model.sample(4, device)
            print(f"   ✓ Sampling works (samples shape: {samples.shape})")

            # Interpolation
            interp_images, interp_latents = model.interpolate_latent(
                test_images[0:1],
                test_images[1:2],
                num_steps=5
            )
            print(f"   ✓ Interpolation works (steps: {interp_images.shape[0]})")

    except Exception as e:
        print(f"   ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("SUCCESS! All tests passed!")
    print("=" * 70)
    print("\nYou can now run full training with:")
    print("  bash experiments/quick_start.sh")
    print("  OR")
    print("  python experiments/train_main_model.py --dataset fashion --epochs 100")
    print("=" * 70)

    return True


if __name__ == '__main__':
    success = test_training()
    sys.exit(0 if success else 1)
