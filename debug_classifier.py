"""
Debug script to visualize generated samples and understand classifier behavior.
"""
import sys
sys.path.insert(0, '/home/asudjianto/jupyterlab/ga_AE')

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.datasets.mnist_imbalanced import get_mnist_dataloaders
from src.models.mnist_models import MNISTVAE, MNISTGAAE


def load_model(model_path, model_type='vae', latent_dim=32):
    """Load trained model."""
    checkpoint = torch.load(model_path, map_location='cpu')
    args = checkpoint['args']

    if model_type == 'vae':
        model = MNISTVAE(latent_dim=latent_dim, beta=args.get('beta', 1.0))
    else:
        model = MNISTGAAE(
            latent_dim=latent_dim,
            lambda_grass=args.get('lambda_grass', 0.1),
            lambda_entropy=args.get('lambda_entropy', 0.01)
        )

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model


def visualize_generations(model, model_type, n_samples=100, save_path='debug_samples.png'):
    """Generate and visualize samples."""
    device = next(model.parameters()).device

    with torch.no_grad():
        # Generate samples
        z = torch.randn(n_samples, model.latent_dim, device=device)

        if model_type == 'vae':
            gen_images = model.decode(z)
        else:
            gen_images = model.decoder(z)

        # Move to CPU for visualization
        gen_images = gen_images.cpu().numpy()

    # Plot grid
    n_rows = 10
    n_cols = 10
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    fig.suptitle(f'{model_type.upper()} Generated Samples', fontsize=16)

    for idx in range(n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        img = gen_images[idx, 0]  # First channel
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()


def analyze_classifier(model, test_loader, model_type='vae', n_gen=100):
    """Analyze what's happening with the classifier."""
    device = next(model.parameters()).device
    model.eval()

    # Collect test samples
    print("\nCollecting test samples...")
    test_images = []
    test_labels = []

    with torch.no_grad():
        for images, labels, is_rare in test_loader:
            test_images.append(images)
            test_labels.append(labels)

    test_images = torch.cat(test_images, dim=0).to(device)
    test_labels = torch.cat(test_labels, dim=0).to(device)

    print(f"Test set: {len(test_images)} samples")
    print(f"Test label distribution: {[(i, (test_labels == i).sum().item()) for i in range(10)]}")

    # Generate samples
    print(f"\nGenerating {n_gen} samples...")
    with torch.no_grad():
        z = torch.randn(n_gen, model.latent_dim, device=device)

        if model_type == 'vae':
            gen_images = model.decode(z)
        else:
            gen_images = model.decoder(z)

    # Classify using 1-NN
    print("\nClassifying generated samples...")
    with torch.no_grad():
        gen_flat = gen_images.view(len(gen_images), -1)
        test_flat = test_images.view(len(test_images), -1)

        # Compute distances
        dists = torch.cdist(gen_flat, test_flat, p=2)  # (n_gen, n_test)

        # Find nearest neighbors
        nearest_idx = dists.argmin(dim=1)  # (n_gen,)
        pred_labels = test_labels[nearest_idx]

        # Get minimum distances
        min_dists = dists.min(dim=1).values

    # Analyze results
    pred_labels_cpu = pred_labels.cpu().numpy()
    min_dists_cpu = min_dists.cpu().numpy()

    print(f"\nGenerated samples label distribution:")
    for digit in range(10):
        count = (pred_labels_cpu == digit).sum()
        pct = 100 * count / len(pred_labels_cpu)
        avg_dist = min_dists_cpu[pred_labels_cpu == digit].mean() if count > 0 else 0
        print(f"  Digit {digit}: {count}/{len(pred_labels_cpu)} ({pct:.1f}%) - Avg dist: {avg_dist:.4f}")

    # Check image statistics
    print(f"\nGenerated image statistics:")
    print(f"  Mean pixel value: {gen_images.mean().item():.4f}")
    print(f"  Std pixel value: {gen_images.std().item():.4f}")
    print(f"  Min pixel value: {gen_images.min().item():.4f}")
    print(f"  Max pixel value: {gen_images.max().item():.4f}")

    # Check for mode collapse (all samples similar)
    sample_variance = gen_images.var(dim=0).mean().item()
    print(f"  Variance across samples: {sample_variance:.6f}")

    return pred_labels_cpu, min_dists_cpu


def main():
    """Main debug function."""
    data_root = '~/data'
    rare_class = 9

    # Load test data
    print("Loading test data...")
    _, _, test_loader = get_mnist_dataloaders(
        root=data_root,
        rare_class=rare_class,
        rare_ratio=0.02,
        batch_size=256,
        num_workers=0,
        download=False
    )

    # Debug VAE
    print("\n" + "="*80)
    print("DEBUGGING VAE")
    print("="*80)
    vae_path = Path('results/mnist_experiments/vae/seed_0/best_model.pt')
    if vae_path.exists():
        vae = load_model(vae_path, model_type='vae', latent_dim=32)

        # Visualize
        visualize_generations(vae, 'vae', n_samples=100,
                            save_path='debug_vae_samples.png')

        # Analyze
        analyze_classifier(vae, test_loader, model_type='vae', n_gen=100)
    else:
        print(f"VAE model not found at {vae_path}")

    # Debug GA-AE (first run - bad results)
    print("\n" + "="*80)
    print("DEBUGGING GA-AE (first run - 6.225× lift)")
    print("="*80)
    gaae1_path = Path('results/mnist_experiments/ga-ae/seed_0/best_model.pt')
    if gaae1_path.exists():
        gaae1 = load_model(gaae1_path, model_type='ga-ae', latent_dim=32)

        # Visualize
        visualize_generations(gaae1, 'ga-ae-run1', n_samples=100,
                            save_path='debug_gaae1_samples.png')

        # Analyze
        analyze_classifier(gaae1, test_loader, model_type='ga-ae', n_gen=100)
    else:
        print(f"GA-AE (run 1) model not found at {gaae1_path}")

    # Debug GA-AE (second run - good results)
    print("\n" + "="*80)
    print("DEBUGGING GA-AE (second run - 1.375× lift)")
    print("="*80)
    gaae2_path = Path('results/mnist_experiments_full/ga-ae/seed_0/best_model.pt')
    if gaae2_path.exists():
        gaae2 = load_model(gaae2_path, model_type='ga-ae', latent_dim=32)

        # Visualize
        visualize_generations(gaae2, 'ga-ae-run2', n_samples=100,
                            save_path='debug_gaae2_samples.png')

        # Analyze
        analyze_classifier(gaae2, test_loader, model_type='ga-ae', n_gen=100)
    else:
        print(f"GA-AE (run 2) model not found at {gaae2_path}")


if __name__ == '__main__':
    main()
