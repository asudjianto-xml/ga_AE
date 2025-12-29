"""
MNIST dataset with class imbalance for rare mode experiments.
"""
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path


class MNISTImbalanced(Dataset):
    """
    MNIST dataset with class imbalance.

    Creates imbalanced dataset where specific digit(s) are rare.
    """
    def __init__(
        self,
        root,
        train=True,
        rare_class=9,
        rare_ratio=0.02,
        transform=None,
        download=True
    ):
        """
        Args:
            root: Root directory for MNIST dataset
            train: If True, creates from training set, else test set
            rare_class: Which digit is rare (default: 9)
            rare_ratio: Target ratio for rare class (default: 0.02 = 2%)
            transform: Optional transform to apply
            download: Whether to download dataset
        """
        self.root = Path(root)
        self.train = train
        self.rare_class = rare_class
        self.rare_ratio = rare_ratio

        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # [-1, 1] range
            ])
        else:
            self.transform = transform

        # Load base MNIST dataset
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            transform=self.transform,
            download=download
        )

        # Create imbalanced dataset
        self._create_imbalanced_dataset()

        split_name = 'train' if train else 'test'
        print(f"Created MNIST {split_name} dataset:")
        print(f"  Total samples: {len(self.indices)}")
        print(f"  Rare class ({rare_class}) samples: {self.n_rare} "
              f"({self.n_rare/len(self.indices)*100:.2f}%)")
        print(f"  Common samples: {self.n_common} "
              f"({self.n_common/len(self.indices)*100:.2f}%)")

    def _create_imbalanced_dataset(self):
        """Create imbalanced dataset with specified rare ratio."""
        # Get all labels
        if hasattr(self.mnist, 'targets'):
            all_labels = self.mnist.targets.numpy()
        else:
            all_labels = np.array([self.mnist[i][1] for i in range(len(self.mnist))])

        # Find rare and common class indices
        rare_mask = (all_labels == self.rare_class)
        rare_indices = np.where(rare_mask)[0]
        common_indices = np.where(~rare_mask)[0]

        print(f"  Natural frequency of class {self.rare_class}: "
              f"{len(rare_indices)}/{len(all_labels)} = "
              f"{len(rare_indices)/len(all_labels)*100:.2f}%")

        # Determine target counts
        if self.train:
            # Create imbalanced training set
            n_total = 50000
            n_rare = int(n_total * self.rare_ratio)
            n_common = n_total - n_rare
        else:
            # Use all samples for test set
            n_rare = len(rare_indices)
            n_common = len(common_indices)

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
        self.labels = all_labels[self.indices]
        self.is_rare = (self.labels == self.rare_class)

        self.n_rare = int(self.is_rare.sum())
        self.n_common = len(self.indices) - self.n_rare

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor of shape (1, 28, 28)
            label: Digit label (0-9)
            is_rare: Boolean indicating if sample is rare class
        """
        mnist_idx = self.indices[idx]
        image, label = self.mnist[mnist_idx]
        is_rare = self.is_rare[idx]

        return image, label, torch.tensor(is_rare, dtype=torch.float32)


def get_mnist_dataloaders(
    root='~/data',
    rare_class=9,
    rare_ratio=0.02,
    batch_size=256,
    num_workers=4,
    download=True
):
    """
    Create MNIST dataloaders with class imbalance.

    Args:
        root: Root directory for MNIST dataset
        rare_class: Which digit is rare
        rare_ratio: Target ratio for rare class
        batch_size: Batch size
        num_workers: Number of data loading workers
        download: Whether to download dataset

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = MNISTImbalanced(
        root=root,
        train=True,
        rare_class=rare_class,
        rare_ratio=rare_ratio,
        download=download
    )

    test_dataset = MNISTImbalanced(
        root=root,
        train=False,
        rare_class=rare_class,
        rare_ratio=0.5,  # Use all samples in test
        download=False
    )

    # Split train into train/val
    n_train = int(0.9 * len(train_dataset))
    n_val = len(train_dataset) - n_train
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [n_train, n_val]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_subset,
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
    print("=" * 80)
    print("Testing MNIST Imbalanced Dataset")
    print("=" * 80)

    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        root='~/data',
        rare_class=9,
        rare_ratio=0.02,
        batch_size=32,
        download=True
    )

    # Check first batch
    print("\nChecking sample batches...")
    for loader_name, loader in [('Train', train_loader), ('Val', val_loader), ('Test', test_loader)]:
        images, labels, is_rare = next(iter(loader))
        n_rare = is_rare.sum().item()

        print(f"\n{loader_name} Loader:")
        print(f"  Batch shape: {images.shape}")
        print(f"  Image range: [{images.min():.2f}, {images.max():.2f}]")
        print(f"  Labels: {labels[:10].tolist()}...")
        print(f"  Rare samples in batch: {n_rare}/{len(is_rare)} ({n_rare/len(is_rare)*100:.1f}%)")

    print("\n" + "=" * 80)
    print("Dataset Test Complete!")
    print("=" * 80)
