"""
Test CelebA dataset loading with rare attributes.
"""
import sys
sys.path.insert(0, '/home/asudjianto/jupyterlab/ga_AE')

from src.datasets.celeba import get_celeba_dataloaders
import torch

# CelebA will be downloaded to this location
DATA_ROOT = '~/data'

print("=" * 80)
print("Testing CelebA Dataset with Rare Attributes")
print("=" * 80)

# Test with different rare attribute combinations
rare_configs = [
    {
        'name': 'Male + Eyeglasses + Bald',
        'attributes': ['Male', 'Eyeglasses', 'Bald'],
        'expected_natural_freq': '~1-2%'
    },
    {
        'name': 'Young + Mustache + Wearing_Hat',
        'attributes': ['Young', 'Mustache', 'Wearing_Hat'],
        'expected_natural_freq': '~1-3%'
    }
]

# Test first configuration
config = rare_configs[0]
print(f"\nTesting Configuration: {config['name']}")
print(f"Expected natural frequency: {config['expected_natural_freq']}")
print("-" * 80)

try:
    train_loader, val_loader, test_loader = get_celeba_dataloaders(
        root=DATA_ROOT,
        rare_attributes=config['attributes'],
        rare_ratio=0.02,  # 2% rare in training
        batch_size=32,
        image_size=64,
        num_workers=2,
        download=True  # Will download if not present
    )

    print("\n" + "=" * 80)
    print("Dataset Loading Successful!")
    print("=" * 80)

    # Check first batch from each split
    print("\nChecking sample batches...")

    for loader_name, loader in [('Train', train_loader), ('Val', val_loader), ('Test', test_loader)]:
        images, attrs, is_rare = next(iter(loader))
        n_rare = is_rare.sum().item()

        print(f"\n{loader_name} Loader:")
        print(f"  Batch shape: {images.shape}")
        print(f"  Image range: [{images.min():.2f}, {images.max():.2f}]")
        print(f"  Attributes shape: {attrs.shape}")
        print(f"  Rare samples in batch: {n_rare}/{len(is_rare)} ({n_rare/len(is_rare)*100:.1f}%)")

        # Check attribute values for rare samples
        if n_rare > 0:
            rare_idx = torch.where(is_rare == 1)[0][0]
            sample_attrs = attrs[rare_idx]
            print(f"  Sample rare attributes: {sample_attrs.sum().item()} active attributes")

    print("\n" + "=" * 80)
    print("Dataset Test Complete!")
    print("=" * 80)

except Exception as e:
    print(f"\nError loading dataset: {e}")
    print("\nIf CelebA download failed, you may need to:")
    print("  1. Download manually from: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
    print(f"  2. Extract to: {DATA_ROOT}/celeba")
    import traceback
    traceback.print_exc()
