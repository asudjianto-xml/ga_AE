"""
Main experiment runner for Geometric Autoencoder paper.

Run all experiments (E1-E6) or specific ones.
"""
import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
import argparse
from pathlib import Path
import json

from src.datasets import get_dataset
from src.models import (
    DeterministicAE, VAE,
    GeometryRegularizedAE, GeometryRegularizedVAE,
    ContractiveAE, SpectralNormAE, SobolevAE
)
from src.utils.training import Trainer
from src.utils.plotting import create_all_plots


def get_model(model_type: str, input_dim: int, latent_dim: int, config: dict):
    """Factory function to create models"""
    hidden_dims = config.get('hidden_dims', [128, 64])
    activation = config.get('activation', 'relu')

    if model_type == 'ae':
        return DeterministicAE(input_dim, latent_dim, hidden_dims, activation)

    elif model_type == 'vae':
        return VAE(
            input_dim, latent_dim, hidden_dims, activation,
            beta=config.get('beta', 1.0)
        )

    elif model_type == 'ga_ae':
        return GeometryRegularizedAE(
            input_dim, latent_dim, hidden_dims, activation,
            lambda_k_volume=config.get('lambda_k_volume', 0.1),
            lambda_edc=config.get('lambda_edc', 0.1),
            k_values=config.get('k_values', [1, 2, 4]),
            volume_floor_tau=config.get('volume_floor_tau', -10.0)
        )

    elif model_type == 'ga_vae':
        return GeometryRegularizedVAE(
            input_dim, latent_dim, hidden_dims, activation,
            beta=config.get('beta', 1.0),
            lambda_k_volume=config.get('lambda_k_volume', 0.1),
            lambda_edc=config.get('lambda_edc', 0.1),
            k_values=config.get('k_values', [1, 2, 4]),
            volume_floor_tau=config.get('volume_floor_tau', -10.0),
            use_mmd_posterior=config.get('use_mmd_posterior', False),
            lambda_mmd=config.get('lambda_mmd', 0.0)
        )

    elif model_type == 'cae':
        return ContractiveAE(
            input_dim, latent_dim, hidden_dims, activation,
            lambda_cae=config.get('lambda_cae', 0.1)
        )

    elif model_type == 'spectral':
        return SpectralNormAE(
            input_dim, latent_dim, hidden_dims, activation
        )

    elif model_type == 'sobolev':
        return SobolevAE(
            input_dim, latent_dim, hidden_dims, activation,
            lambda_sobolev=config.get('lambda_sobolev', 0.1),
            mode=config.get('sobolev_mode', 'decoder_lipschitz')
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def experiment_e1_ae_trap(seed: int = 0, device: str = 'cuda'):
    """
    E1: The AE Trap - reconstruction good, generation bad
    """
    print("=" * 80)
    print("EXPERIMENT E1: The AE Trap")
    print("=" * 80)

    # Use 20D MoG dataset
    dataset = get_dataset('mog20d', n_train=10000, n_val=2000, n_test=2000, seed=seed)

    config = {
        'model_type': 'ae',
        'epochs': 200,
        'batch_size': 256,
        'lr': 3e-4,
        'device': device,
        'diagnostic_every': 5,
        'checkpoint_every': 50,
        'k_values': [1, 2, 4, 8],
        'eps_values': [1e-6]
    }

    model = get_model('ae', input_dim=20, latent_dim=8, config=config)
    model = model.to(device)

    output_dir = f'results/e1_ae_trap/seed_{seed}'

    trainer = Trainer(
        model, dataset['train'], dataset['val'], dataset['test'],
        config, output_dir, metadata=dataset['metadata']
    )

    trainer.train()
    create_all_plots(output_dir, experiment='e1')

    print(f"E1 completed. Results in {output_dir}")


def experiment_e2_tail_stress(seed: int = 0, device: str = 'cuda'):
    """
    E2: Tail stress test - rare mode recall
    """
    print("=" * 80)
    print("EXPERIMENT E2: Tail Stress Test")
    print("=" * 80)

    dataset = get_dataset('mog20d', n_train=10000, n_val=2000, n_test=2000,
                         seed=seed, tail_weight=0.02)

    models_to_test = [
        ('ae', {}),
        ('ga_ae', {'lambda_k_volume': 0.1, 'lambda_edc': 0.1}),
        ('cae', {'lambda_cae': 0.1})
    ]

    for model_type, model_config in models_to_test:
        print(f"\nTraining {model_type}...")

        config = {
            'model_type': model_type,
            'epochs': 200,
            'batch_size': 256,
            'lr': 3e-4,
            'device': device,
            'diagnostic_every': 5,
            'checkpoint_every': 50,
            **model_config
        }

        model = get_model(model_type, input_dim=20, latent_dim=8, config=config)
        model = model.to(device)

        output_dir = f'results/e2_tail_stress/{model_type}/seed_{seed}'

        trainer = Trainer(
            model, dataset['train'], dataset['val'], dataset['test'],
            config, output_dir, metadata=dataset['metadata']
        )

        trainer.train()

    # Create comparative plots
    create_all_plots('results/e2_tail_stress', experiment='e2')
    print("E2 completed.")


def experiment_e3_vae_collapse(seed: int = 0, device: str = 'cuda'):
    """
    E3: VAE posterior collapse sweep
    """
    print("=" * 80)
    print("EXPERIMENT E3: VAE Posterior Collapse")
    print("=" * 80)

    dataset = get_dataset('mog20d', n_train=10000, n_val=2000, n_test=2000, seed=seed)

    beta_values = [0.1, 1.0, 4.0]

    for beta in beta_values:
        for use_anneal in [False, True]:
            model_name = f'vae_beta{beta}_anneal{use_anneal}'
            print(f"\nTraining {model_name}...")

            config = {
                'model_type': 'vae',
                'beta': beta,
                'use_kl_annealing': use_anneal,
                'kl_anneal_epochs': 50,
                'epochs': 200,
                'batch_size': 256,
                'lr': 3e-4,
                'device': device,
                'diagnostic_every': 5,
                'checkpoint_every': 50
            }

            model = get_model('vae', input_dim=20, latent_dim=8, config=config)
            model = model.to(device)

            output_dir = f'results/e3_vae_collapse/{model_name}/seed_{seed}'

            trainer = Trainer(
                model, dataset['train'], dataset['val'], dataset['test'],
                config, output_dir, metadata=dataset['metadata']
            )

            trainer.train()

    create_all_plots('results/e3_vae_collapse', experiment='e3')
    print("E3 completed.")


def experiment_e4_vae_tradeoff(seed: int = 0, device: str = 'cuda'):
    """
    E4: VAE trade-off - KL vs aggregated posterior matching
    """
    print("=" * 80)
    print("EXPERIMENT E4: VAE Trade-off")
    print("=" * 80)

    dataset = get_dataset('mog20d', n_train=10000, n_val=2000, n_test=2000, seed=seed)

    models_to_test = [
        ('vae', {'beta': 1.0}),
        ('vae', {'beta': 1.0, 'use_kl_annealing': True}),
        ('ga_vae', {'beta': 1.0, 'use_mmd_posterior': True, 'lambda_mmd': 10.0,
                    'lambda_k_volume': 0.1, 'lambda_edc': 0.1}),
    ]

    for i, (model_type, model_config) in enumerate(models_to_test):
        model_name = f'{model_type}_config{i}'
        print(f"\nTraining {model_name}...")

        config = {
            'model_type': model_type,
            'epochs': 200,
            'batch_size': 256,
            'lr': 3e-4,
            'device': device,
            'diagnostic_every': 5,
            'checkpoint_every': 50,
            **model_config
        }

        model = get_model(model_type, input_dim=20, latent_dim=8, config=config)
        model = model.to(device)

        output_dir = f'results/e4_vae_tradeoff/{model_name}/seed_{seed}'

        trainer = Trainer(
            model, dataset['train'], dataset['val'], dataset['test'],
            config, output_dir, metadata=dataset['metadata']
        )

        trainer.train()

    create_all_plots('results/e4_vae_tradeoff', experiment='e4')
    print("E4 completed.")


def experiment_e5_baselines(seed: int = 0, device: str = 'cuda'):
    """
    E5: Baselines comparison
    """
    print("=" * 80)
    print("EXPERIMENT E5: Baselines Comparison")
    print("=" * 80)

    # Use swissroll dataset
    dataset = get_dataset('swissroll', n_train=10000, n_val=2000, n_test=2000,
                         seed=seed, embed_dim=50)

    models_to_test = [
        ('ae', {}),
        ('spectral', {}),
        ('sobolev', {'lambda_sobolev': 0.1}),
        ('ga_ae', {'lambda_k_volume': 0.1, 'lambda_edc': 0.1})
    ]

    for model_type, model_config in models_to_test:
        print(f"\nTraining {model_type}...")

        config = {
            'model_type': model_type,
            'epochs': 200,
            'batch_size': 256,
            'lr': 3e-4,
            'device': device,
            'diagnostic_every': 5,
            'checkpoint_every': 50,
            **model_config
        }

        model = get_model(model_type, input_dim=50, latent_dim=16, config=config)
        model = model.to(device)

        output_dir = f'results/e5_baselines/{model_type}/seed_{seed}'

        trainer = Trainer(
            model, dataset['train'], dataset['val'], dataset['test'],
            config, output_dir, metadata=dataset['metadata']
        )

        trainer.train()

    create_all_plots('results/e5_baselines', experiment='e5')
    print("E5 completed.")


def experiment_e6_teacher(seed: int = 0, device: str = 'cuda'):
    """
    E6: Controlled teacher generator
    """
    print("=" * 80)
    print("EXPERIMENT E6: Teacher Generator")
    print("=" * 80)

    for curvature in ['smooth', 'sharp']:
        print(f"\nTeacher curvature: {curvature}")

        dataset = get_dataset(f'teacher_{curvature}', n_train=10000, n_val=2000, n_test=2000,
                             seed=seed, latent_dim=8, output_dim=20)

        models_to_test = [
            ('ae', {}),
            ('ga_ae', {'lambda_k_volume': 0.1, 'lambda_edc': 0.1})
        ]

        for model_type, model_config in models_to_test:
            print(f"  Training {model_type}...")

            config = {
                'model_type': model_type,
                'epochs': 200,
                'batch_size': 256,
                'lr': 3e-4,
                'device': device,
                'diagnostic_every': 5,
                'checkpoint_every': 50,
                **model_config
            }

            model = get_model(model_type, input_dim=20, latent_dim=8, config=config)
            model = model.to(device)

            output_dir = f'results/e6_teacher/{curvature}_{model_type}/seed_{seed}'

            trainer = Trainer(
                model, dataset['train'], dataset['val'], dataset['test'],
                config, output_dir, metadata=dataset['metadata']
            )

            trainer.train()

    create_all_plots('results/e6_teacher', experiment='e6')
    print("E6 completed.")


def main():
    parser = argparse.ArgumentParser(description='Run GA-AE experiments')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6'],
                       help='Which experiment to run')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Run experiments
    if args.experiment == 'all':
        experiment_e1_ae_trap(args.seed, args.device)
        experiment_e2_tail_stress(args.seed, args.device)
        experiment_e3_vae_collapse(args.seed, args.device)
        experiment_e4_vae_tradeoff(args.seed, args.device)
        experiment_e5_baselines(args.seed, args.device)
        experiment_e6_teacher(args.seed, args.device)
    elif args.experiment == 'e1':
        experiment_e1_ae_trap(args.seed, args.device)
    elif args.experiment == 'e2':
        experiment_e2_tail_stress(args.seed, args.device)
    elif args.experiment == 'e3':
        experiment_e3_vae_collapse(args.seed, args.device)
    elif args.experiment == 'e4':
        experiment_e4_vae_tradeoff(args.seed, args.device)
    elif args.experiment == 'e5':
        experiment_e5_baselines(args.seed, args.device)
    elif args.experiment == 'e6':
        experiment_e6_teacher(args.seed, args.device)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)


if __name__ == '__main__':
    main()
