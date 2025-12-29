"""Run only GA-AE for E5 (other models already complete)"""
import torch
import numpy as np
from src.datasets import get_dataset
from src.models import GeometryRegularizedAE
from src.utils.training import Trainer

torch.manual_seed(0)
np.random.seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 80)
print("E5: Training GA-AE (4/4)")
print("=" * 80)

# SwissRoll dataset (same as other E5 models)
dataset = get_dataset('swissroll', n_train=10000, n_val=2000, n_test=2000,
                     seed=0, embed_dim=50)

config = {
    'model_type': 'ga_ae',
    'epochs': 200,
    'batch_size': 256,
    'lr': 3e-4,
    'device': device,
    'diagnostic_every': 5,
    'checkpoint_every': 50,
    'k_values': [1, 2, 4, 8],
    'eps_values': [1e-6]
}

model = GeometryRegularizedAE(
    input_dim=50, latent_dim=16, hidden_dims=[128, 64],
    lambda_k_volume=0.1, lambda_edc=0.1, k_values=[1, 2, 4]
)
model = model.to(device)

output_dir = 'results/e5_baselines/ga_ae/seed_0'

trainer = Trainer(
    model, dataset['train'], dataset['val'], dataset['test'],
    config, output_dir, metadata=dataset['metadata']
)

trainer.train()
print(f"\nGA-AE training complete! Saved to {output_dir}")
