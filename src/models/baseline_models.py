"""Baseline autoencoder models for comparison"""
import torch
import torch.nn as nn
from typing import List
from .base_models import DeterministicAE, Encoder, Decoder
from ..diagnostics.jacobian_utils import compute_jvp_encoder


class ContractiveAE(DeterministicAE):
    """
    Contractive Autoencoder (Rifai et al., 2011).

    Adds Frobenius norm penalty on encoder Jacobian.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [128, 64],
        activation: str = 'relu',
        lambda_cae: float = 0.1
    ):
        super().__init__(input_dim, latent_dim, hidden_dims, activation)
        self.lambda_cae = lambda_cae

    def compute_jacobian_penalty(self, x):
        """Compute Frobenius norm of encoder Jacobian using random projections"""
        device = x.device
        batch_size, input_dim = x.shape

        # Use random projections to estimate ||J_E||_F^2
        # E[||J v||^2] for v ~ N(0, I) estimates Tr(J^T J) = ||J||_F^2
        n_samples = 5  # Number of random directions

        penalty = 0.0
        for _ in range(n_samples):
            v = torch.randn(batch_size, input_dim, device=device)
            v = v / v.norm(dim=1, keepdim=True)  # Normalize

            # Compute J_E(x) @ v
            jvp_result = compute_jvp_encoder(self.encoder, x, v.unsqueeze(-1))
            jvp_result = jvp_result.squeeze(-1)  # (batch_size, latent_dim)

            # ||J v||^2
            penalty += (jvp_result ** 2).sum(dim=1).mean()

        penalty /= n_samples
        return penalty

    def loss_function(self, x, x_recon, z):
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')

        # Jacobian penalty
        jac_penalty = self.compute_jacobian_penalty(x)

        # Total loss
        loss = recon_loss + self.lambda_cae * jac_penalty

        return {
            'loss': loss,
            'recon_loss': recon_loss.item(),
            'jac_penalty': jac_penalty.item()
        }


def spectral_norm_init(module, name='weight', n_power_iterations=1):
    """Apply spectral normalization to a module"""
    return nn.utils.spectral_norm(module, name=name, n_power_iterations=n_power_iterations)


class SpectralNormAE(nn.Module):
    """
    Autoencoder with spectral normalization on all linear layers.

    Constrains Lipschitz constant of the network.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [128, 64],
        activation: str = 'relu',
        apply_to_encoder: bool = True,
        apply_to_decoder: bool = True
    ):
        super().__init__()

        # Build encoder with optional spectral norm
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'elu':
            act_fn = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            linear = nn.Linear(prev_dim, h_dim)
            if apply_to_encoder:
                linear = spectral_norm_init(linear)
            encoder_layers.extend([linear, act_fn()])
            prev_dim = h_dim

        linear = nn.Linear(prev_dim, latent_dim)
        if apply_to_encoder:
            linear = spectral_norm_init(linear)
        encoder_layers.append(linear)

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        decoder_hidden = hidden_dims[::-1]
        for h_dim in decoder_hidden:
            linear = nn.Linear(prev_dim, h_dim)
            if apply_to_decoder:
                linear = spectral_norm_init(linear)
            decoder_layers.extend([linear, act_fn()])
            prev_dim = h_dim

        linear = nn.Linear(prev_dim, input_dim)
        if apply_to_decoder:
            linear = spectral_norm_init(linear)
        decoder_layers.append(linear)

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def loss_function(self, x, x_recon, z):
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
        return {
            'loss': recon_loss,
            'recon_loss': recon_loss.item()
        }


class SobolevAE(DeterministicAE):
    """
    Autoencoder with Sobolev/Jacobian self-consistency regularization.

    Penalizes variance of J_DE(x) across nearby points or magnitude of decoder Jacobian.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [128, 64],
        activation: str = 'relu',
        lambda_sobolev: float = 0.1,
        mode: str = 'decoder_lipschitz'  # 'decoder_lipschitz' or 'consistency'
    ):
        super().__init__(input_dim, latent_dim, hidden_dims, activation)
        self.lambda_sobolev = lambda_sobolev
        self.mode = mode

    def compute_sobolev_penalty(self, x, z):
        """Compute Sobolev-type regularization"""
        device = x.device
        batch_size, latent_dim = z.shape

        if self.mode == 'decoder_lipschitz':
            # Penalize ||J_D(z)||_F^2 using random projections
            n_samples = 5
            penalty = 0.0

            for _ in range(n_samples):
                u = torch.randn(batch_size, latent_dim, device=device)
                u = u / u.norm(dim=1, keepdim=True)

                # Enable gradients on z
                z_copy = z.detach().requires_grad_(True)

                # Compute decoder output
                x_out = self.decoder(z_copy)

                # Compute gradient
                grad_outputs = u
                grads = torch.autograd.grad(
                    outputs=x_out,
                    inputs=z_copy,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True
                )[0]

                penalty += (grads ** 2).sum(dim=1).mean()

            penalty /= n_samples

        elif self.mode == 'consistency':
            # Penalize variance of reconstruction Jacobian
            # This is simplified: just penalize squared norm
            x.requires_grad_(True)
            x_recon = self.forward(x)[0]

            # Random direction for JVP
            v = torch.randn_like(x)
            v = v / v.norm(dim=1, keepdim=True)

            grads = torch.autograd.grad(
                outputs=x_recon,
                inputs=x,
                grad_outputs=v,
                create_graph=True,
                retain_graph=True
            )[0]

            penalty = (grads ** 2).sum(dim=1).mean()

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return penalty

    def loss_function(self, x, x_recon, z):
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')

        # Sobolev penalty
        sobolev_penalty = self.compute_sobolev_penalty(x, z)

        # Total loss
        loss = recon_loss + self.lambda_sobolev * sobolev_penalty

        return {
            'loss': loss,
            'recon_loss': recon_loss.item(),
            'sobolev_penalty': sobolev_penalty.item()
        }
