"""
Quick demo of E1: The AE Trap experiment
Runs with fewer epochs for demonstration
"""
import sys
sys.path.insert(0, 'src')

import torch
import numpy as np

from src.datasets import get_dataset
from src.models import DeterministicAE
from src.utils.training import Trainer
from src.utils.plotting import create_all_plots

# Set seeds
torch.manual_seed(0)
np.random.seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

print("=" * 80)
print("EXPERIMENT E1 DEMO: The AE Trap (50 epochs)")
print("=" * 80)

# Generate 20D MoG dataset
print("\nGenerating dataset...")
dataset = get_dataset('mog20d', n_train=5000, n_val=1000, n_test=1000, seed=0)
print(f"Dataset shapes: train={dataset['train'].shape}, val={dataset['val'].shape}")

# Create standard AE
print("\nCreating model...")
model = DeterministicAE(input_dim=20, latent_dim=8, hidden_dims=[128, 64])
model = model.to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training configuration
config = {
    'model_type': 'ae',
    'epochs': 50,  # Reduced from 200 for demo
    'batch_size': 256,
    'lr': 3e-4,
    'device': device,
    'diagnostic_every': 5,
    'checkpoint_every': 25,
    'k_values': [1, 2, 4, 8],
    'eps_values': [1e-6]
}

output_dir = 'results/e1_ae_trap_demo/seed_0'

print(f"\nTraining for {config['epochs']} epochs...")
print(f"Output: {output_dir}")

trainer = Trainer(
    model, dataset['train'], dataset['val'], dataset['test'],
    config, output_dir, metadata=dataset['metadata']
)

trainer.train()

print("\nGenerating plots...")
create_all_plots(output_dir, experiment='e1')

print("\n" + "=" * 80)
print("E1 DEMO COMPLETED!")
print("=" * 80)
print(f"\nResults saved to: {output_dir}")
print(f"Plots saved to: {output_dir}/plots")
print("\nKey findings to look for:")
print("  1. Reconstruction loss decreases over training")
print("  2. Generation metrics (MMD, Energy Distance) remain poor")
print("  3. k-volumes collapse (especially for larger k)")
print("  4. Generative gap increases (on-manifold vs off-manifold)")
print("  5. Decoder instability under radius stress")
