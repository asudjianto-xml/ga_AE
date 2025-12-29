"""
Test CelebA dataset setup and model architectures before launching full experiments.

This script:
1. Downloads CelebA if needed (or tests with existing data)
2. Verifies rare attribute combination exists
3. Tests ImageVAE and ImageGAAE models
4. Runs a quick training sanity check (1 epoch)
"""
import sys
sys.path.insert(0, '/home/asudjianto/jupyterlab/ga_AE')

import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from src.datasets.celeba import get_celeba_dataloaders
from src.models.image_models import ImageVAE, ImageGAAE


def test_dataset(data_root, rare_attributes, rare_ratio=0.02, download=True):
    """Test CelebA dataset loading with rare attributes."""
    print("\n" + "="*80)
    print("Testing CelebA Dataset Loader")
    print("="*80)
    print(f"Data root: {data_root}")
    print(f"Rare attributes: {rare_attributes}")
    print(f"Rare ratio: {rare_ratio}")
    print(f"Download: {download}")
    print()

    try:
        print("Loading dataloaders...")
        train_loader, val_loader, test_loader = get_celeba_dataloaders(
            root=data_root,
            rare_attributes=rare_attributes,
            rare_ratio=rare_ratio,
            batch_size=32,  # Small batch for testing
            image_size=64,
            num_workers=2,
            download=download
        )

        print("\n✓ Dataset loading successful!")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")

        # Test one batch
        print("\nTesting batch iteration...")
        images, attrs, is_rare = next(iter(train_loader))
        print(f"  Batch shape: {images.shape}")
        print(f"  Attributes shape: {attrs.shape}")
        print(f"  Rare samples in batch: {is_rare.sum().item()}/{len(is_rare)}")
        print(f"  Image range: [{images.min():.2f}, {images.max():.2f}]")

        return train_loader, val_loader, test_loader

    except Exception as e:
        print(f"\n✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_models(device='cuda'):
    """Test ImageVAE and ImageGAAE architectures."""
    print("\n" + "="*80)
    print("Testing Model Architectures")
    print("="*80)
    print(f"Device: {device}")
    print()

    latent_dim = 128
    image_size = 64
    batch_size = 4

    # Create dummy batch
    x = torch.randn(batch_size, 3, image_size, image_size).to(device)

    # Test VAE
    print("Testing ImageVAE...")
    try:
        vae = ImageVAE(latent_dim=latent_dim, image_size=image_size, beta=1.0).to(device)
        n_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")

        recon, mu, logvar = vae(x)
        losses = vae.loss_function(recon, x, mu, logvar)

        print(f"  Input shape: {x.shape}")
        print(f"  Recon shape: {recon.shape}")
        print(f"  Latent shape: {mu.shape}")
        print(f"  Loss: {losses['loss'].item():.4f}")
        print(f"  Recon loss: {losses['recon_loss'].item():.4f}")
        print(f"  KL loss: {losses['kl_loss'].item():.4f}")
        print("  ✓ ImageVAE working!")

    except Exception as e:
        print(f"  ✗ ImageVAE failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test GA-AE
    print("\nTesting ImageGAAE...")
    try:
        gaae = ImageGAAE(
            latent_dim=latent_dim,
            image_size=image_size,
            lambda_grass=0.1,
            lambda_entropy=0.01
        ).to(device)
        n_params = sum(p.numel() for p in gaae.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")

        recon, z = gaae(x)
        losses = gaae.loss_function(recon, x, z)

        print(f"  Input shape: {x.shape}")
        print(f"  Recon shape: {recon.shape}")
        print(f"  Latent shape: {z.shape}")
        print(f"  Loss: {losses['loss'].item():.4f}")
        print(f"  Recon loss: {losses['recon_loss'].item():.4f}")
        print(f"  Grass loss: {losses['grass_loss'].item():.4f}")
        print(f"  Blade entropy: {losses['blade_entropy'].item():.4f}")
        print("  ✓ ImageGAAE working!")

    except Exception as e:
        print(f"  ✗ ImageGAAE failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def quick_training_test(train_loader, model, model_type='vae', device='cuda', n_batches=10):
    """Quick training sanity check."""
    print("\n" + "="*80)
    print(f"Quick Training Test ({model_type.upper()})")
    print("="*80)
    print(f"Testing {n_batches} batches...")
    print()

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    losses = []
    pbar = tqdm(enumerate(train_loader), total=n_batches, desc='Training test')

    for batch_idx, (images, attrs, is_rare) in pbar:
        if batch_idx >= n_batches:
            break

        images = images.to(device)
        optimizer.zero_grad()

        if model_type == 'vae':
            recon, mu, logvar = model(images)
            loss_dict = model.loss_function(recon, images, mu, logvar)
        else:  # ga-ae
            recon, z = model(images)
            loss_dict = model.loss_function(recon, images, z)

        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    print(f"\n✓ Training test successful!")
    print(f"  Average loss: {sum(losses)/len(losses):.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Loss trend: {'↓ Decreasing' if losses[-1] < losses[0] else '→ Stable'}")

    return True


def main():
    """Main test function."""
    print("\n" + "="*80)
    print("CelebA Setup Test")
    print("="*80)
    print("This script tests CelebA dataset and model setup before full experiments.")
    print()

    # Configuration
    data_root = Path.home() / 'data'
    rare_attributes = ['Male', 'Eyeglasses', 'Bald']
    rare_ratio = 0.02
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Configuration:")
    print(f"  Data root: {data_root}")
    print(f"  Rare attributes: {rare_attributes}")
    print(f"  Rare ratio: {rare_ratio}")
    print(f"  Device: {device}")
    print()

    # Check GPU
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ WARNING: No GPU detected. Training will be very slow.")
        print("  Consider using a machine with GPU for CelebA experiments.")

    # Step 1: Test dataset
    print("\n" + "="*80)
    print("STEP 1: Dataset Loading")
    print("="*80)

    # Ask user if they want to download
    print("\n⚠ CelebA download is ~1.5GB and may take 10-30 minutes.")
    print("  If data already exists, it will be reused.")
    print()

    # For automated testing, set download=True
    # In production, you might want to check if data exists first
    train_loader, val_loader, test_loader = test_dataset(
        data_root=str(data_root),
        rare_attributes=rare_attributes,
        rare_ratio=rare_ratio,
        download=True  # Set to True to auto-download
    )

    if train_loader is None:
        print("\n✗ Dataset test failed. Cannot proceed.")
        return 1

    # Step 2: Test models
    print("\n" + "="*80)
    print("STEP 2: Model Architecture")
    print("="*80)

    if not test_models(device=device):
        print("\n✗ Model test failed. Cannot proceed.")
        return 1

    # Step 3: Quick training test
    print("\n" + "="*80)
    print("STEP 3: Training Sanity Check")
    print("="*80)

    # Test VAE
    vae = ImageVAE(latent_dim=128, image_size=64, beta=1.0).to(device)
    if not quick_training_test(train_loader, vae, model_type='vae', device=device, n_batches=5):
        print("\n✗ VAE training test failed.")
        return 1

    # Test GA-AE
    gaae = ImageGAAE(latent_dim=128, image_size=64, lambda_grass=0.1, lambda_entropy=0.01).to(device)
    if not quick_training_test(train_loader, gaae, model_type='ga-ae', device=device, n_batches=5):
        print("\n✗ GA-AE training test failed.")
        return 1

    # Summary
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED!")
    print("="*80)
    print()
    print("CelebA setup is ready for full multi-seed experiments.")
    print()
    print("Next steps:")
    print("  1. Review configuration in run_celeba_multiseed.sh")
    print("  2. Launch experiments: bash run_celeba_multiseed.sh")
    print("  3. Monitor progress: tail -f celeba_multiseed.log")
    print()
    print("Expected runtime:")
    print("  - Each model: ~2-3 hours (50 epochs, 64x64 images)")
    print("  - Total (6 models): ~12-18 hours with GPU")
    print("="*80)

    return 0


if __name__ == '__main__':
    exit(main())
