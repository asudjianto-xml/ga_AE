"""
E7: GA-Native Prior via Tangent Chamfer Loss

Tests whether geometric projection/rejection can replace or augment KL divergence:
1. VAE (KL only) - baseline
2. VAE (KL + Chamfer) - hybrid approach
3. VAE (Chamfer only) - pure geometric prior (no density matching)

Key question: Can we replace density alignment (KL) with geometric alignment (Chamfer)?
"""
import torch
import numpy as np
from pathlib import Path

from src.datasets import get_dataset
from src.models import VAE, VAE_TangentChamfer
from src.utils.training import Trainer
from src.utils.training_chamfer import ChamferTrainer


def experiment_e7_ga_prior(seed=0, device='cuda'):
    """Run E7 GA-native prior experiment"""
    print("=" * 80)
    print("EXPERIMENT E7: GA-Native Prior (Tangent Chamfer)")
    print("=" * 80)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use MoG 2D with rare tail mode (same as E2/E2c for comparison)
    dataset = get_dataset('mog2d', n_train=10000, n_val=2000, n_test=2000,
                          seed=seed, n_components=8, tail_weight=0.02)

    # Base config
    base_config = {
        'epochs': 200,
        'batch_size': 256,
        'lr': 3e-4,
        'device': device,
        'diagnostic_every': 20,
        'checkpoint_every': 200,
        'k_values': [1, 2],
        'eps_values': [1e-6],
        'bank_update_every': 1  # Update reference bank every batch
    }

    # Variants to test
    variants = {
        'vae_kl_only': {
            'name': 'VAE (KL only)',
            'use_chamfer': False,
            'use_kl': True,
            'beta': 1.0,
            'lambda_chamfer': 0.0
        },
        'vae_kl_chamfer': {
            'name': 'VAE (KL + Chamfer)',
            'use_chamfer': True,
            'use_kl': True,
            'beta': 1.0,
            'lambda_chamfer': 0.1
        },
        'vae_chamfer_only': {
            'name': 'VAE (Chamfer only) - Pure GA Prior',
            'use_chamfer': True,
            'use_kl': False,
            'beta': 0.0,
            'lambda_chamfer': 0.3  # Higher weight since it's the only regularizer
        }
    }

    # Train each variant
    for variant_key, variant_info in variants.items():
        print(f"\n{'=' * 80}")
        print(f"Training {variant_info['name']}...")
        print(f"{'=' * 80}")

        # Create model
        if variant_info['use_chamfer']:
            model = VAE_TangentChamfer(
                input_dim=2,
                latent_dim=2,
                hidden_dims=[64, 32],
                beta=variant_info['beta'],
                use_kl=variant_info['use_kl'],
                lambda_chamfer=variant_info['lambda_chamfer'],
                chamfer_k=1,  # CRITICAL: k must be < latent_dim for meaningful rejection
                ref_bank_size=4096,
                lambda_collapse=0.0,  # Optional: could add collapse barrier
                eps=1e-6
            )
        else:
            # Standard VAE (no Chamfer)
            model = VAE(
                input_dim=2,
                latent_dim=2,
                hidden_dims=[64, 32],
                beta=variant_info['beta']
            )

        model = model.to(device)

        output_dir = f'results/e7_ga_prior/{variant_key}/seed_{seed}'

        config = base_config.copy()
        config['model_type'] = variant_key
        config['beta'] = variant_info['beta']

        # Use appropriate trainer
        if variant_info['use_chamfer']:
            trainer = ChamferTrainer(
                model, dataset['train'], dataset['val'], dataset['test'],
                config, output_dir, metadata=dataset['metadata']
            )
        else:
            trainer = Trainer(
                model, dataset['train'], dataset['val'], dataset['test'],
                config, output_dir, metadata=dataset['metadata']
            )

        trainer.train()
        print(f"\n{variant_info['name']} complete! Saved to {output_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='E7: GA-Native Prior')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    experiment_e7_ga_prior(args.seed, args.device)

    print("\n" + "=" * 80)
    print("E7 GA-NATIVE PRIOR EXPERIMENT COMPLETE!")
    print("=" * 80)
    print("\nKey questions answered:")
    print("1. Can Chamfer replace KL? (compare vae_chamfer_only vs vae_kl_only)")
    print("2. Does Chamfer + KL improve? (compare vae_kl_chamfer vs baselines)")
    print("3. Impact on rare mode coverage and off-manifold stability")
    print("\nNext steps:")
    print("1. Run: python compare_e7.py")
    print("2. Check rare mode recall and generation quality")
    print("=" * 80)
