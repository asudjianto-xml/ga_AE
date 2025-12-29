"""
CelebA dataset with rare attribute combination filtering.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA as TorchCelebA
import numpy as np
from pathlib import Path


class CelebARare(Dataset):
    """
    CelebA dataset with rare attribute combinations.

    Creates imbalanced dataset where specific attribute combinations are rare.
    """
    def __init__(
        self,
        root,
        split='train',
        rare_attributes=None,
        rare_ratio=0.02,
        transform=None,
        download=False,
        image_size=64
    ):
        """
        Args:
            root: Root directory for CelebA dataset
            split: 'train', 'valid', or 'test'
            rare_attributes: List of attribute names for rare combination
                           e.g., ['Male', 'Eyeglasses', 'Bald']
            rare_ratio: Target ratio for rare samples (default: 0.02 = 2%)
            transform: Optional transform to apply to images
            download: Whether to download dataset
            image_size: Size to resize images (default: 64x64)
        """
        self.root = Path(root)
        self.split = split
        self.rare_ratio = rare_ratio
        self.image_size = image_size

        # Default rare attributes if not specified
        if rare_attributes is None:
            rare_attributes = ['Male', 'Eyeglasses', 'Bald']
        self.rare_attributes = rare_attributes

        # Default transform if not provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.CenterCrop(178),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

        # Load base CelebA dataset
        print(f"Loading CelebA {split} split...")
        self.celeba = TorchCelebA(
            root=root,
            split=split,
            target_type='attr',
            transform=self.transform,
            download=download
        )

        # Get attribute names
        self.attr_names = self.celeba.attr_names

        # Find rare attribute indices
        self.rare_attr_indices = [
            self.attr_names.index(attr) for attr in rare_attributes
        ]

        # Create balanced dataset with rare samples
        self._create_balanced_dataset()

        print(f"Created {split} dataset:")
        print(f"  Total samples: {len(self.indices)}")
        print(f"  Rare samples: {self.n_rare} ({self.n_rare/len(self.indices)*100:.2f}%)")
        print(f"  Common samples: {self.n_common} ({self.n_common/len(self.indices)*100:.2f}%)")
        print(f"  Rare attributes: {self.rare_attributes}")

    def _create_balanced_dataset(self):
        """Create dataset with specified rare ratio."""
        # Get all attributes
        all_attrs = []
        for i in range(len(self.celeba)):
            _, attrs = self.celeba[i]
            all_attrs.append(attrs)
        all_attrs = torch.stack(all_attrs)

        # Find rare combination (all rare attributes = 1)
        rare_mask = torch.ones(len(all_attrs), dtype=torch.bool)
        for idx in self.rare_attr_indices:
            rare_mask &= (all_attrs[:, idx] == 1)

        rare_indices = torch.where(rare_mask)[0].tolist()
        common_indices = torch.where(~rare_mask)[0].tolist()

        print(f"  Natural rare frequency: {len(rare_indices)}/{len(all_attrs)} "
              f"= {len(rare_indices)/len(all_attrs)*100:.2f}%")

        # Determine target counts
        if self.split == 'train':
            # Create imbalanced training set
            n_total = 50000  # Use subset for faster training
            n_rare = int(n_total * self.rare_ratio)
            n_common = n_total - n_rare
        else:
            # Use all samples for validation/test
            n_rare = min(len(rare_indices), 1000)
            n_common = min(len(common_indices), 10000)

        # Sample indices
        if len(rare_indices) < n_rare:
            # Oversample if not enough rare samples
            rare_sample = np.random.choice(rare_indices, n_rare, replace=True)
        else:
            rare_sample = np.random.choice(rare_indices, n_rare, replace=False)

        common_sample = np.random.choice(common_indices, n_common, replace=False)

        # Combine and shuffle
        self.indices = np.concatenate([rare_sample, common_sample])
        np.random.shuffle(self.indices)

        # Store labels for evaluation
        self.is_rare = np.zeros(len(self.indices), dtype=bool)
        for i, idx in enumerate(self.indices):
            self.is_rare[i] = rare_mask[idx].item()

        self.n_rare = int(self.is_rare.sum())
        self.n_common = len(self.indices) - self.n_rare

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor of shape (3, image_size, image_size)
            attributes: Tensor of shape (40,) with all attributes
            is_rare: Boolean indicating if sample has rare combination
        """
        celeba_idx = self.indices[idx]
        image, attrs = self.celeba[celeba_idx]
        is_rare = self.is_rare[idx]

        return image, attrs, torch.tensor(is_rare, dtype=torch.float32)

    def get_rare_attributes(self):
        """Return rare attribute indices and names."""
        return self.rare_attr_indices, self.rare_attributes


def get_celeba_dataloaders(
    root,
    rare_attributes=None,
    rare_ratio=0.02,
    batch_size=128,
    image_size=64,
    num_workers=4,
    download=False
):
    """
    Create CelebA dataloaders for train/val/test.

    Args:
        root: Root directory for CelebA dataset
        rare_attributes: List of attribute names for rare combination
        rare_ratio: Target ratio for rare samples
        batch_size: Batch size
        image_size: Image size
        num_workers: Number of data loading workers
        download: Whether to download dataset

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = CelebARare(
        root=root,
        split='train',
        rare_attributes=rare_attributes,
        rare_ratio=rare_ratio,
        image_size=image_size,
        download=download
    )

    val_dataset = CelebARare(
        root=root,
        split='valid',
        rare_attributes=rare_attributes,
        rare_ratio=0.5,  # Higher ratio for better evaluation
        image_size=image_size,
        download=False
    )

    test_dataset = CelebARare(
        root=root,
        split='test',
        rare_attributes=rare_attributes,
        rare_ratio=0.5,
        image_size=image_size,
        download=False
    )

    # Create dataloaders
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
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test dataset creation
    root = '~/data/celeba'

    print("Testing CelebA rare dataset creation...")
    train_loader, val_loader, test_loader = get_celeba_dataloaders(
        root=root,
        rare_attributes=['Male', 'Eyeglasses', 'Bald'],
        rare_ratio=0.02,
        batch_size=32,
        download=True
    )

    # Check first batch
    images, attrs, is_rare = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Attributes: {attrs.shape}")
    print(f"  Is rare: {is_rare.shape}")
    print(f"  Rare in batch: {is_rare.sum().item()}/{len(is_rare)}")
