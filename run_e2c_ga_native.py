"""
E2c: GA-Native Regularizers (Grassmann, Blade Entropy, Blade Matching)

Tests TRUE geometric algebra / exterior algebra regularizers:
- Option A: Grassmann + Blade Entropy
- Option B: Grassmann + Blade Matching
- Option C: All three
"""
import torch
import numpy as np
from pathlib import Path

from src.datasets import get_dataset
from src.models import GA_AE_Grassmann
from src.utils.training import Trainer


def experiment_e2c_ga_native(seed=0, device='cuda'):
    """Run E2c GA-native experiment"""
    print("=" * 80)
    print("EXPERIMENT E2c: GA-Native Regularizers (Grassmann/Blade)")
    print("=" * 80)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Same dataset as E2: MoG 2D with rare tail mode
    dataset = get_dataset('mog2d', n_train=10000, n_val=2000, n_test=2000,
                          seed=seed, n_components=8, tail_weight=0.02)

    # Base config
    base_config = {
        'epochs': 200,
        'batch_size': 256,
        'lr': 3e-4,
        'device': device,
        'diagnostic_every': 20,  # Less frequent for speed
        'checkpoint_every': 200,  # Only final checkpoint
        'k_values': [1, 2],  # 2D data
        'eps_values': [1e-6]
    }

    # GA-native variants
    variants = {
        'option_a_grass_entropy': {
            'name': 'Option A: Grassmann + Blade Entropy',
            'params': {
                'add_grassmann': True,
                'lambda_grassmann': 0.1,
                'add_blade_entropy': True,
                'lambda_blade_entropy': 0.1,
                'add_blade_matching': False
            }
        },
        'option_b_grass_matching': {
            'name': 'Option B: Grassmann + Blade Matching',
            'params': {
                'add_grassmann': True,
                'lambda_grassmann': 0.1,
                'add_blade_entropy': False,
                'add_blade_matching': True,
                'lambda_blade_matching': 0.1,
                'blade_mmd_sigma': 1.0
            }
        },
        'option_c_all_three': {
            'name': 'Option C: All Three (Grass+Entropy+Matching)',
            'params': {
                'add_grassmann': True,
                'lambda_grassmann': 0.1,
                'add_blade_entropy': True,
                'lambda_blade_entropy': 0.1,
                'add_blade_matching': True,
                'lambda_blade_matching': 0.1,
                'blade_mmd_sigma': 1.0
            }
        }
    }

    # Train each variant
    for variant_key, variant_info in variants.items():
        print(f"\n{'=' * 80}")
        print(f"Training {variant_info['name']}...")
        print(f"{'=' * 80}")

        # Create model
        model = GA_AE_Grassmann(
            input_dim=2,
            latent_dim=2,
            hidden_dims=[64, 32],  # Smaller for 2D
            # Original geometry terms (keep for consistency)
            lambda_k_volume=0.1,
            lambda_edc=0.1,
            k_values=[1, 2],
            volume_floor_tau=-10.0,
            eps=1e-6,
            # GA-native terms (variant-specific)
            **variant_info['params']
        )
        model = model.to(device)

        output_dir = f'results/e2c_ga_native/{variant_key}/seed_{seed}'

        config = base_config.copy()
        config['model_type'] = variant_key

        trainer = Trainer(
            model, dataset['train'], dataset['val'], dataset['test'],
            config, output_dir, metadata=dataset['metadata']
        )

        trainer.train()
        print(f"\n{variant_info['name']} complete! Saved to {output_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='E2c: GA-Native Regularizers')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    experiment_e2c_ga_native(args.seed, args.device)

    print("\n" + "=" * 80)
    print("E2c GA-NATIVE EXPERIMENT COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run: python compare_e2c.py")
    print("2. Check rare mode recall vs CAE target (22.7%)")
    print("=" * 80)
