"""Training utilities"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Optional, List
import json
from pathlib import Path
from tqdm import tqdm

from ..diagnostics.geometric_metrics import (
    compute_full_diagnostics,
    compute_generative_gap_index,
    compute_decoder_stability
)
from ..diagnostics.jacobian_utils import (
    compute_data_pca_directions,
    sample_orthonormal_vectors
)
from .metrics import compute_generation_metrics


class Trainer:
    """
    Unified trainer for all autoencoder variants.
    """

    def __init__(
        self,
        model: nn.Module,
        train_data: np.ndarray,
        val_data: np.ndarray,
        test_data: np.ndarray,
        config: dict,
        output_dir: str,
        metadata: dict = None
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata or {}

        # Create dataloaders
        self.train_loader = self._create_dataloader(train_data, config['batch_size'], shuffle=True)
        self.val_loader = self._create_dataloader(val_data, config['batch_size'], shuffle=False)
        self.test_loader = self._create_dataloader(test_data, config['batch_size'], shuffle=False)

        # Store data for diagnostics
        self.train_data_torch = torch.from_numpy(train_data).float().to(config['device'])
        self.val_data_torch = torch.from_numpy(val_data).float().to(config['device'])
        self.test_data_torch = torch.from_numpy(test_data).float().to(config['device'])

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

        # KL annealing for VAE
        self.kl_weight = 0.0 if config.get('use_kl_annealing', False) else config.get('beta', 1.0)
        self.kl_anneal_epochs = config.get('kl_anneal_epochs', 50)

        # Compute PCA directions for diagnostics
        self.pca_directions = compute_data_pca_directions(
            self.train_data_torch[:5000],  # Use subset for efficiency
            k_max=8
        )

        # Logging
        self.metrics_log = []
        self.model_type = config.get('model_type', 'ae')

    def _create_dataloader(self, data: np.ndarray, batch_size: int, shuffle: bool):
        dataset = TensorDataset(torch.from_numpy(data).float())
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _update_kl_weight(self, epoch: int):
        """Update KL weight for annealing"""
        if self.config.get('use_kl_annealing', False):
            self.kl_weight = min(1.0, epoch / self.kl_anneal_epochs) * self.config.get('beta', 1.0)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {}

        for batch in self.train_loader:
            x = batch[0].to(self.config['device'])

            self.optimizer.zero_grad()

            # Forward pass (model-specific)
            if 'vae' in self.model_type.lower():
                x_recon, mu, logvar, z = self.model(x)
                loss_dict = self.model.loss_function(x, x_recon, mu, logvar, self.kl_weight)
            else:
                x_recon, z = self.model(x)
                loss_dict = self.model.loss_function(x, x_recon, z)

            loss = loss_dict['loss']
            loss.backward()
            self.optimizer.step()

            # Accumulate losses
            for k, v in loss_dict.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                epoch_losses[k].append(v if isinstance(v, float) else v.item())

        # Average losses
        epoch_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        return epoch_losses

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, data_torch: torch.Tensor) -> Dict[str, float]:
        """Evaluate on a dataset"""
        self.model.eval()
        total_recon_loss = 0.0
        n_batches = 0

        for batch in data_loader:
            x = batch[0].to(self.config['device'])

            if 'vae' in self.model_type.lower():
                x_recon, mu, logvar, z = self.model(x)
            else:
                x_recon, z = self.model(x)

            recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
            total_recon_loss += recon_loss.item()
            n_batches += 1

        metrics = {
            'recon_loss': total_recon_loss / n_batches
        }

        return metrics

    @torch.no_grad()
    def compute_diagnostics(self, epoch: int) -> Dict[str, any]:
        """Compute geometric diagnostics"""
        self.model.eval()
        diagnostics = {}

        # Sample subset for efficiency
        n_samples = min(1000, len(self.val_data_torch))
        indices = torch.randperm(len(self.val_data_torch))[:n_samples]
        x_batch = self.val_data_torch[indices]

        # Get encoder and decoder
        if 'vae' in self.model_type.lower():
            # For VAE, use mean encoder for diagnostics
            class MeanEncoder(nn.Module):
                def __init__(self, encoder):
                    super().__init__()
                    self.encoder = encoder

                def forward(self, x):
                    h = self.encoder.shared(x)
                    return self.encoder.fc_mu(h)

            encoder = MeanEncoder(self.model.encoder)
            decoder = self.model.decoder
        else:
            encoder = self.model.encoder
            decoder = self.model.decoder

        # Compute full diagnostics on validation data
        diag = compute_full_diagnostics(
            encoder,
            decoder,
            x_batch,
            k_values=self.config.get('k_values', [1, 2, 4, 8]),
            eps_values=self.config.get('eps_values', [1e-6]),
            pca_directions=self.pca_directions,
            compute_edc=True
        )
        diagnostics.update(diag)

        # Compute generative gap index
        # Get latent dimension robustly
        if 'vae' in self.model_type.lower():
            latent_dim = self.model.latent_dim
        elif hasattr(self.model.encoder, 'network'):
            latent_dim = self.model.encoder.network[-1].out_features
        elif hasattr(self.model, 'encoder') and isinstance(self.model.encoder, nn.Sequential):
            # For models like SpectralNormAE where encoder is a Sequential directly
            latent_dim = list(self.model.encoder.children())[-1].out_features
        else:
            # Fallback: encode a sample to get latent dim
            with torch.no_grad():
                z_sample = encoder(x_batch[:1])
                latent_dim = z_sample.shape[1]

        z_prior = torch.randn(n_samples, latent_dim, device=self.config['device'])

        V_k_list = []
        for k in [2, 4, 8]:
            if k <= x_batch.shape[1]:
                V_k = self.pca_directions[:, :k]
                V_k_list.append(V_k)

        if V_k_list:
            gap_metrics = compute_generative_gap_index(
                encoder, decoder, x_batch, z_prior, V_k_list, eps=1e-6
            )
            diagnostics.update(gap_metrics)

        # Decoder stability under radius stress
        for radius in [0.5, 1.0, 2.0, 4.0]:
            z_stress = z_prior * radius
            U_k = sample_orthonormal_vectors(
                z_prior.shape[1], 4, 'random', device=self.config['device']
            )
            stability = compute_decoder_stability(decoder, z_stress, U_k, eps=1e-6)

            for k, v in stability.items():
                diagnostics[f'decoder_r{radius}_{k}'] = v

        return diagnostics

    @torch.no_grad()
    def compute_generation_metrics(self, epoch: int) -> Dict[str, float]:
        """Compute generation quality metrics"""
        self.model.eval()

        # Generate samples
        n_samples = min(5000, len(self.test_data_torch))

        if 'vae' in self.model_type.lower():
            generated = self.model.sample(n_samples, device=self.config['device'])
        else:
            # Sample from standard normal for deterministic AE
            # Get latent dimension robustly
            if hasattr(self.model.encoder, 'network'):
                latent_dim = self.model.encoder.network[-1].out_features
            elif isinstance(self.model.encoder, nn.Sequential):
                latent_dim = list(self.model.encoder.children())[-1].out_features
            else:
                # Fallback: encode a sample
                with torch.no_grad():
                    z_sample = self.model.encoder(self.test_data_torch[:1])
                    latent_dim = z_sample.shape[1]

            z = torch.randn(n_samples, latent_dim, device=self.config['device'])
            generated = self.model.decode(z)

        generated_np = generated.cpu().numpy()
        real_np = self.test_data_torch[:n_samples].cpu().numpy()

        # Get labels if available
        real_labels = self.metadata.get('labels')
        if real_labels is not None:
            real_labels = real_labels[:n_samples]

        n_modes = self.metadata.get('n_components', 8)

        metrics = compute_generation_metrics(
            real_np, generated_np,
            real_labels=real_labels,
            n_modes=n_modes
        )

        return metrics

    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config['epochs']} epochs...")
        print(f"Output directory: {self.output_dir}")

        for epoch in range(self.config['epochs']):
            # Update KL weight
            self._update_kl_weight(epoch)

            # Train
            train_metrics = self.train_epoch(epoch)
            train_metrics['epoch'] = epoch
            train_metrics['kl_weight'] = self.kl_weight

            # Validate
            val_metrics = self.evaluate(self.val_loader, self.val_data_torch)
            train_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})

            # Compute diagnostics periodically
            if epoch % self.config.get('diagnostic_every', 5) == 0:
                diag_metrics = self.compute_diagnostics(epoch)
                train_metrics.update({f'diag_{k}': v for k, v in diag_metrics.items()})

                # Compute generation metrics
                gen_metrics = self.compute_generation_metrics(epoch)
                train_metrics.update({f'gen_{k}': v for k, v in gen_metrics.items()})

            # Log
            self.metrics_log.append(train_metrics)

            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.config['epochs']}: "
                      f"Train Loss = {train_metrics['loss']:.4f}, "
                      f"Val Recon = {train_metrics['val_recon_loss']:.4f}")

            # Save checkpoint periodically
            if epoch % self.config.get('checkpoint_every', 50) == 0 and epoch > 0:
                self.save_checkpoint(epoch)

        # Final save
        self.save_checkpoint(self.config['epochs'])
        self.save_metrics()

        print(f"Training complete! Results saved to {self.output_dir}")

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / f'model_epoch_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, checkpoint_path)

    def save_metrics(self):
        """Save metrics log"""
        def convert_to_serializable(obj):
            """Convert numpy types to Python native types"""
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            return obj

        metrics_path = self.output_dir / 'metrics.jsonl'
        with open(metrics_path, 'w') as f:
            for entry in self.metrics_log:
                serializable_entry = convert_to_serializable(entry)
                f.write(json.dumps(serializable_entry) + '\n')

        # Save config
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
