"""
E2b: GA-AE Ablation Study for Coverage Terms

Tests which coverage terms fix the rare mode recall failure:
- Variant 2: +ED (Energy Distance)
- Variant 6: +ED+Repulsion
- Variant 9: +All (ED+Repulsion+Gap+DecVolFloor)
"""
import torch
import numpy as np
from pathlib import Path

from src.datasets import get_dataset
from src.models import GA_AE_Ablation
from src.utils.training import Trainer


def experiment_e2b_ablation(seed=0, device='cuda'):
    """Run E2b ablation experiment"""
    print("=" * 80)
    print("EXPERIMENT E2b: GA-AE Coverage Ablation")
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
        'k_values': [1, 2],  # Reduce for speed (2D data)
        'eps_values': [1e-6]
    }

    # Ablation variants
    variants = {
        'variant2_ed': {
            'name': 'GA-AE + ED',
            'params': {
                'add_ed': True,
                'lambda_ed': 0.05,
                'add_repulsion': False,
                'add_gap': False,
                'add_dec_vol': False
            }
        },
        'variant6_ed_repel': {
            'name': 'GA-AE + ED + Repulsion',
            'params': {
                'add_ed': True,
                'lambda_ed': 0.05,
                'add_repulsion': True,
                'lambda_repel': 0.01,
                'repulsion_sigma': 1.0,
                'add_gap': False,
                'add_dec_vol': False
            }
        },
        'variant9_all': {
            'name': 'GA-AE + All',
            'params': {
                'add_ed': True,
                'lambda_ed': 0.05,
                'add_repulsion': True,
                'lambda_repel': 0.01,
                'repulsion_sigma': 1.0,
                'add_gap': True,
                'lambda_gap': 0.5,
                'add_dec_vol': True,
                'lambda_dec_vol': 0.1,
                'dec_vol_tau': -5.0
            }
        }
    }

    # Train each variant
    for variant_key, variant_info in variants.items():
        print(f"\n{'=' * 80}")
        print(f"Training {variant_info['name']}...")
        print(f"{'=' * 80}")

        # Create model
        model = GA_AE_Ablation(
            input_dim=2,
            latent_dim=2,
            hidden_dims=[64, 32],  # Smaller for 2D
            # Original geometry terms
            lambda_k_volume=0.1,
            lambda_edc=0.1,
            k_values=[1, 2],
            volume_floor_tau=-10.0,
            eps=1e-6,
            # Ablation terms
            **variant_info['params']
        )
        model = model.to(device)

        output_dir = f'results/e2b_ablation/{variant_key}/seed_{seed}'

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

    parser = argparse.ArgumentParser(description='E2b: GA-AE Coverage Ablation')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    experiment_e2b_ablation(args.seed, args.device)

    print("\n" + "=" * 80)
    print("E2b ABLATION COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run: python compare_e2b.py")
    print("2. Check rare mode recall for each variant")
    print("=" * 80)
