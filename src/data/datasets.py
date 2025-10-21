"""
Dataset loaders for Multi-Perspective SDF-VAE experiments

Provides consistent data loading for:
- Fashion-MNIST (60k training, 10k test)
- Medical MNIST (DermaMNIST - skin lesion images)

All datasets are preprocessed to 64x64 RGB for consistency
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Callable


class FashionMNISTRGB(Dataset):
    """
    Fashion-MNIST dataset converted to RGB and resized to 64x64

    Original: 28x28 grayscale
    Output: 64x64 RGB (grayscale replicated to 3 channels)

    This matches the preprocessing from the original successful experiments.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = True,
        transform: Optional[Callable] = None,
        target_size: int = 64
    ):
        """
        Args:
            root: Root directory for data
            train: If True, load training set, else test set
            download: If True, download dataset
            transform: Optional transform to apply
            target_size: Target image size (default: 64)
        """
        self.root = Path(root)
        self.target_size = target_size
        self.transform = transform

        # Load Fashion-MNIST
        self.fashion_mnist = datasets.FashionMNIST(
            root=str(self.root),
            train=train,
            download=download,
            transform=None  # We'll apply transforms manually
        )

        # Class names
        self.classes = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]

    def __len__(self) -> int:
        return len(self.fashion_mnist)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            idx: Index

        Returns:
            image: RGB tensor [3, target_size, target_size] in range [0, 1]
            label: Class label (0-9)
        """
        # Get original Fashion-MNIST image and label
        image, label = self.fashion_mnist[idx]

        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Convert to PIL for transforms
        image = Image.fromarray((image * 255).astype(np.uint8))

        # Convert grayscale to RGB (replicate across 3 channels)
        image = image.convert('RGB')

        # Resize to target size
        image = image.resize((self.target_size, self.target_size), Image.BILINEAR)

        # Apply additional transforms if provided
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            image = transforms.ToTensor()(image)

        return image, label


class MedicalMNIST(Dataset):
    """
    Medical MNIST (DermaMNIST) dataset

    Skin lesion classification dataset from MedMNIST collection.
    Images are RGB and resized to 64x64.

    Yang et al. 2021: "MedMNIST: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification"
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        download: bool = True,
        transform: Optional[Callable] = None,
        target_size: int = 64
    ):
        """
        Args:
            root: Root directory for data
            split: 'train', 'val', or 'test'
            download: If True, download dataset
            transform: Optional transform to apply
            target_size: Target image size (default: 64)
        """
        self.root = Path(root)
        self.split = split
        self.target_size = target_size
        self.transform = transform

        # Try to import medmnist
        try:
            import medmnist
            from medmnist import INFO
        except ImportError:
            raise ImportError(
                "medmnist package not found. Install with: pip install medmnist"
            )

        # Get DermaMNIST dataset
        dataset_name = 'dermamnist'
        info = INFO[dataset_name]
        DataClass = getattr(medmnist, info['python_class'])

        # Load dataset
        self.dataset = DataClass(
            root=str(self.root),
            split=split,
            download=download,
            transform=None  # We'll apply transforms manually
        )

        self.n_channels = info['n_channels']
        self.n_classes = len(info['label'])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            idx: Index

        Returns:
            image: RGB tensor [3, target_size, target_size] in range [0, 1]
            label: Class label
        """
        # Get image and label from MedMNIST
        image, label = self.dataset[idx]

        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            # MedMNIST images are in range [0, 1] with shape [H, W, C]
            if image.ndim == 3:
                # Convert from [H, W, C] to PIL Image
                if image.shape[2] == 1:
                    # Grayscale -> RGB
                    image = (image * 255).astype(np.uint8).squeeze()
                    image = Image.fromarray(image).convert('RGB')
                else:
                    # Already RGB
                    image = (image * 255).astype(np.uint8)
                    image = Image.fromarray(image)
            else:
                # 2D grayscale
                image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image).convert('RGB')

        # Resize to target size
        image = image.resize((self.target_size, self.target_size), Image.BILINEAR)

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            image = transforms.ToTensor()(image)

        # Extract scalar label if it's an array
        if isinstance(label, np.ndarray):
            label = int(label[0])

        return image, label


def get_fashion_mnist_loaders(
    data_root: str = './data',
    batch_size: int = 64,
    num_workers: int = 4,
    val_split: float = 0.1,
    image_size: int = 64,
    augment_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get Fashion-MNIST data loaders

    Args:
        data_root: Root directory for data
        batch_size: Batch size
        num_workers: Number of data loading workers
        val_split: Fraction of training data to use for validation
        image_size: Target image size
        augment_train: Whether to apply data augmentation to training set

    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
    """
    # Training transforms (with augmentation)
    if augment_train:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load full training set
    full_train_dataset = FashionMNISTRGB(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
        target_size=image_size
    )

    # Split into train and validation
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Load test set
    test_dataset = FashionMNISTRGB(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform,
        target_size=image_size
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f"Fashion-MNIST Data Loaders:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_size}x{image_size}")

    return train_loader, val_loader, test_loader


def get_medical_mnist_loaders(
    data_root: str = './data',
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 64,
    augment_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get Medical MNIST (DermaMNIST) data loaders

    Args:
        data_root: Root directory for data
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Target image size
        augment_train: Whether to apply data augmentation

    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
    """
    # Training transforms (with augmentation)
    if augment_train:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load datasets
    train_dataset = MedicalMNIST(
        root=data_root,
        split='train',
        download=True,
        transform=train_transform,
        target_size=image_size
    )

    val_dataset = MedicalMNIST(
        root=data_root,
        split='val',
        download=True,
        transform=test_transform,
        target_size=image_size
    )

    test_dataset = MedicalMNIST(
        root=data_root,
        split='test',
        download=True,
        transform=test_transform,
        target_size=image_size
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f"Medical MNIST Data Loaders:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_size}x{image_size}")

    return train_loader, val_loader, test_loader
