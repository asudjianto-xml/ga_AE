"""Quick test script to verify the implementation works"""
import sys
sys.path.insert(0, 'src')

import torch
import numpy as np

from src.datasets import get_dataset
from src.models import DeterministicAE
from src.utils.training import Trainer

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Check GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Create small dataset
print("\nGenerating dataset...")
dataset = get_dataset('mog20d', n_train=1000, n_val=200, n_test=200, seed=42)
print(f"Dataset shapes: train={dataset['train'].shape}, val={dataset['val'].shape}, test={dataset['test'].shape}")

# Create model
print("\nCreating model...")
model = DeterministicAE(input_dim=20, latent_dim=8, hidden_dims=[64, 32])
model = model.to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
print("\nTesting forward pass...")
x_test = torch.from_numpy(dataset['train'][:10]).float().to(device)
x_recon, z = model(x_test)
print(f"Input shape: {x_test.shape}")
print(f"Latent shape: {z.shape}")
print(f"Reconstruction shape: {x_recon.shape}")

# Test training for a few epochs
print("\nTraining for 5 epochs...")
config = {
    'model_type': 'ae',
    'epochs': 5,
    'batch_size': 128,
    'lr': 3e-4,
    'device': device,
    'diagnostic_every': 2,
    'checkpoint_every': 5,
    'k_values': [1, 2, 4],
    'eps_values': [1e-6]
}

trainer = Trainer(
    model, dataset['train'], dataset['val'], dataset['test'],
    config, 'results/test_quick', metadata=dataset['metadata']
)

trainer.train()

print("\n" + "="*80)
print("QUICK TEST PASSED!")
print("="*80)
print("\nYou can now run full experiments with:")
print("  python run_experiments.py --experiment e1")
print("  python run_experiments.py --experiment all")
