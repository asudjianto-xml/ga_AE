"""
Training script for CelebA experiments: GA-AE vs VAE.
"""
import sys
sys.path.insert(0, '/home/asudjianto/jupyterlab/ga_AE')

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse

from src.datasets.celeba import get_celeba_dataloaders
from src.models.image_models import ImageVAE, ImageGAAE


def train_epoch(model, train_loader, optimizer, device, epoch, model_type='vae'):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_reg = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (images, attrs, is_rare) in enumerate(pbar):
        images = images.to(device)

        optimizer.zero_grad()

        if model_type == 'vae':
            recon, mu, logvar = model(images)
            losses = model.loss_function(recon, images, mu, logvar)
            reg_loss = losses['kl_loss']
        else:  # GA-AE
            recon, z = model(images)
            losses = model.loss_function(recon, images, z)
            if 'grass_loss' in losses:
                reg_loss = losses['grass_loss'] - losses['blade_entropy']
            else:
                reg_loss = torch.tensor(0.0)

        loss = losses['loss']
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += losses['recon_loss'].item()
        total_reg += reg_loss.item()

        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{losses["recon_loss"].item():.4f}'
            })

    n_batches = len(train_loader)
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon / n_batches,
        'reg_loss': total_reg / n_batches
    }


def evaluate(model, val_loader, device, model_type='vae'):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    total_recon = 0

    with torch.no_grad():
        for images, attrs, is_rare in val_loader:
            images = images.to(device)

            if model_type == 'vae':
                recon, mu, logvar = model(images)
                losses = model.loss_function(recon, images, mu, logvar)
            else:  # GA-AE
                recon, z = model(images)
                # Only reconstruction loss for evaluation
                recon_loss = torch.nn.functional.mse_loss(recon, images)
                losses = {'loss': recon_loss, 'recon_loss': recon_loss}

            total_loss += losses['loss'].item()
            total_recon += losses['recon_loss'].item()

    n_batches = len(val_loader)
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon / n_batches
    }


def compute_rare_metrics(model, test_loader, device, model_type='vae', n_gen=2000):
    """
    Compute rare mode metrics: recall and lift.

    Args:
        model: Trained model
        test_loader: Test dataloader
        device: Device
        model_type: 'vae' or 'ga-ae'
        n_gen: Number of samples to generate

    Returns:
        dict with rare_count, rare_recall, rare_lift
    """
    model.eval()

    # Get test set rare samples for recall computation
    test_rare_count = 0
    with torch.no_grad():
        for images, attrs, is_rare in test_loader:
            test_rare_count += is_rare.sum().item()

    print(f"\nTest set has {test_rare_count} rare samples")

    # Generate samples from prior
    print(f"Generating {n_gen} samples from prior...")
    generated_rare = 0

    with torch.no_grad():
        n_batches = (n_gen + 127) // 128
        for _ in tqdm(range(n_batches), desc='Generating'):
            batch_size = min(128, n_gen - generated_rare)
            if batch_size <= 0:
                break

            # Sample from prior N(0, I)
            z = torch.randn(batch_size, model.latent_dim, device=device)

            # Decode
            if model_type == 'vae':
                gen_images = model.decode(z)
            else:  # GA-AE
                gen_images = model.decoder(z)

            # Classify using attribute model (simplified: use oracle attributes)
            # For real experiments, should use pre-trained attribute classifier
            # For now, we'll estimate based on reconstruction similarity

            # TODO: Implement proper attribute classification
            # For now, use random estimation (placeholder)
            # In practice, you would:
            # 1. Train an attribute classifier on CelebA
            # 2. Use it to predict attributes of generated images
            # 3. Check if rare combination is present

    # Placeholder metrics (replace with actual classification)
    # Rare recall = (gen_rare_count) / test_rare_count
    # Rare lift = (gen_rare_count / n_gen) / rare_ratio

    rare_recall = 0.0  # Placeholder
    rare_lift = 0.0  # Placeholder

    return {
        'test_rare_count': test_rare_count,
        'gen_rare_count': 0,  # Placeholder
        'rare_recall': rare_recall,
        'rare_lift': rare_lift
    }


def main(args):
    """Main training function."""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir) / args.model_type / f'seed_{args.seed}'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load data
    print("\nLoading CelebA dataset...")
    train_loader, val_loader, test_loader = get_celeba_dataloaders(
        root=args.data_root,
        rare_attributes=args.rare_attributes,
        rare_ratio=args.rare_ratio,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        download=args.download
    )

    # Create model
    print(f"\nCreating {args.model_type.upper()} model...")
    if args.model_type == 'vae':
        model = ImageVAE(
            latent_dim=args.latent_dim,
            image_size=args.image_size,
            beta=args.beta
        ).to(device)
        print(f"  Beta: {args.beta}")
    else:  # ga-ae
        model = ImageGAAE(
            latent_dim=args.latent_dim,
            image_size=args.image_size,
            lambda_grass=args.lambda_grass,
            lambda_entropy=args.lambda_entropy
        ).to(device)
        print(f"  Lambda grass: {args.lambda_grass}")
        print(f"  Lambda entropy: {args.lambda_entropy}")

    print(f"  Latent dim: {args.latent_dim}")
    print(f"  Image size: {args.image_size}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {n_params:,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Tensorboard
    writer = SummaryWriter(output_dir / 'logs')

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_loss = float('inf')
    metrics_log = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*80}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, args.model_type
        )

        # Validate
        val_metrics = evaluate(model, val_loader, device, args.model_type)

        # Learning rate schedule
        scheduler.step(val_metrics['loss'])

        # Log metrics
        print(f"\nTrain Loss: {train_metrics['loss']:.4f} | "
              f"Recon: {train_metrics['recon_loss']:.4f} | "
              f"Reg: {train_metrics['reg_loss']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Recon: {val_metrics['recon_loss']:.4f}")

        # Tensorboard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Recon/train', train_metrics['recon_loss'], epoch)
        writer.add_scalar('Recon/val', val_metrics['recon_loss'], epoch)

        # Save metrics
        epoch_metrics = {
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics
        }
        metrics_log.append(epoch_metrics)

        # Save checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'args': vars(args)
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"  → Saved best model (val_loss: {best_val_loss:.4f})")

        # Save metrics log
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_log, f, indent=2)

    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(output_dir / 'best_model.pt')['model_state_dict'])
    test_metrics = evaluate(model, test_loader, device, args.model_type)
    print(f"Test Loss: {test_metrics['loss']:.4f} | "
          f"Recon: {test_metrics['recon_loss']:.4f}")

    # Compute rare metrics (placeholder for now)
    # rare_metrics = compute_rare_metrics(
    #     model, test_loader, device, args.model_type, n_gen=2000
    # )
    # print(f"\nRare Mode Metrics:")
    # print(f"  Test rare count: {rare_metrics['test_rare_count']}")
    # print(f"  Gen rare count: {rare_metrics['gen_rare_count']}")
    # print(f"  Rare recall: {rare_metrics['rare_recall']:.2%}")
    # print(f"  Rare lift: {rare_metrics['rare_lift']:.2f}×")

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CelebA experiments')

    # Model
    parser.add_argument('--model-type', type=str, choices=['vae', 'ga-ae'],
                        required=True, help='Model type')
    parser.add_argument('--latent-dim', type=int, default=128,
                        help='Latent dimension')
    parser.add_argument('--image-size', type=int, default=64,
                        help='Image size')

    # VAE specific
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Beta for VAE')

    # GA-AE specific
    parser.add_argument('--lambda-grass', type=float, default=0.1,
                        help='Lambda for Grassmann spread loss')
    parser.add_argument('--lambda-entropy', type=float, default=0.01,
                        help='Lambda for blade entropy')

    # Data
    parser.add_argument('--data-root', type=str, default='~/data',
                        help='Root directory for CelebA')
    parser.add_argument('--rare-attributes', nargs='+',
                        default=['Male', 'Eyeglasses', 'Bald'],
                        help='Rare attribute combination')
    parser.add_argument('--rare-ratio', type=float, default=0.02,
                        help='Rare ratio in training set')
    parser.add_argument('--download', action='store_true',
                        help='Download CelebA if not present')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # Misc
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str,
                        default='results/celeba_experiments',
                        help='Output directory')

    args = parser.parse_args()

    main(args)
