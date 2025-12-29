"""
Visualize generated samples from trained MNIST models.
"""
import sys
sys.path.insert(0, '/home/asudjianto/jupyterlab/ga_AE')

import torch
import torchvision
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.models.mnist_models import MNISTVAE, MNISTGAAE


def generate_and_save_samples(model, model_type, output_path, n_samples=100, device='cuda'):
    """Generate samples and save as grid."""
    model.eval()

    with torch.no_grad():
        # Sample from prior N(0, I)
        z = torch.randn(n_samples, model.latent_dim, device=device)

        # Decode
        if model_type == 'vae':
            gen_images = model.decode(z)
        else:  # ga-ae
            gen_images = model.decoder(z)

        # Denormalize from [-1, 1] to [0, 1]
        gen_images = (gen_images + 1) / 2
        gen_images = gen_images.clamp(0, 1)

        # Create grid
        grid = torchvision.utils.make_grid(gen_images, nrow=10, padding=2)

        # Save grid
        torchvision.utils.save_image(grid, output_path)
        print(f"Saved {n_samples} samples to: {output_path}")

        return gen_images, grid


def plot_samples_with_matplotlib(gen_images, title, save_path):
    """Plot samples using matplotlib for better visualization."""
    n_samples = min(64, len(gen_images))
    gen_images = gen_images[:n_samples]

    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    fig.suptitle(title, fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < n_samples:
            img = gen_images[i].cpu().squeeze().numpy()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved matplotlib plot to: {save_path}")
    plt.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    results_dir = Path('results/mnist_experiments')

    # Load VAE
    print("\n" + "="*80)
    print("Loading VAE model...")
    print("="*80)
    vae_checkpoint = torch.load(results_dir / 'vae/seed_0/best_model.pt', map_location=device)
    vae = MNISTVAE(latent_dim=32, beta=1.0).to(device)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    print(f"Loaded VAE from epoch {vae_checkpoint['epoch']}")

    # Generate VAE samples
    print("\nGenerating VAE samples...")
    vae_samples, vae_grid = generate_and_save_samples(
        vae, 'vae',
        results_dir / 'vae/seed_0/generated_samples.png',
        n_samples=100,
        device=device
    )
    plot_samples_with_matplotlib(
        vae_samples,
        'VAE (β=1.0) - Generated Samples',
        results_dir / 'vae/seed_0/generated_samples_detailed.png'
    )

    # Load Standard AE
    print("\n" + "="*80)
    print("Loading Standard AE model...")
    print("="*80)
    ae_checkpoint = torch.load(results_dir / 'ga-ae/seed_0/best_model.pt', map_location=device)
    ae = MNISTGAAE(latent_dim=32).to(device)
    ae.load_state_dict(ae_checkpoint['model_state_dict'])
    print(f"Loaded Standard AE from epoch {ae_checkpoint['epoch']}")

    # Generate Standard AE samples
    print("\nGenerating Standard AE samples...")
    ae_samples, ae_grid = generate_and_save_samples(
        ae, 'ga-ae',
        results_dir / 'ga-ae/seed_0/generated_samples.png',
        n_samples=100,
        device=device
    )
    plot_samples_with_matplotlib(
        ae_samples,
        'Standard AE - Generated Samples',
        results_dir / 'ga-ae/seed_0/generated_samples_detailed.png'
    )

    # Create side-by-side comparison
    print("\n" + "="*80)
    print("Creating side-by-side comparison...")
    print("="*80)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # VAE samples
    vae_grid_np = vae_grid.cpu().numpy().transpose(1, 2, 0)
    axes[0].imshow(vae_grid_np, cmap='gray')
    axes[0].set_title('VAE (β=1.0) - 100 Samples\nRare Lift: 50.0×', fontsize=14)
    axes[0].axis('off')

    # Standard AE samples
    ae_grid_np = ae_grid.cpu().numpy().transpose(1, 2, 0)
    axes[1].imshow(ae_grid_np, cmap='gray')
    axes[1].set_title('Standard AE - 100 Samples\nRare Lift: 6.22×', fontsize=14)
    axes[1].axis('off')

    plt.tight_layout()
    comparison_path = results_dir / 'comparison_samples.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to: {comparison_path}")
    plt.close()

    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)
    print("\nFiles generated:")
    print(f"  VAE samples: {results_dir / 'vae/seed_0/generated_samples.png'}")
    print(f"  VAE detailed: {results_dir / 'vae/seed_0/generated_samples_detailed.png'}")
    print(f"  AE samples: {results_dir / 'ga-ae/seed_0/generated_samples.png'}")
    print(f"  AE detailed: {results_dir / 'ga-ae/seed_0/generated_samples_detailed.png'}")
    print(f"  Comparison: {comparison_path}")


if __name__ == '__main__':
    main()
