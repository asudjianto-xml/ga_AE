"""Base autoencoder models"""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class Encoder(nn.Module):
    """MLP Encoder"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [128, 64],
        activation: str = 'relu'
    ):
        super().__init__()

        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'elu':
            act_fn = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                act_fn()
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, latent_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Decoder(nn.Module):
    """MLP Decoder"""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [64, 128],
        activation: str = 'relu'
    ):
        super().__init__()

        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'elu':
            act_fn = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        prev_dim = latent_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                act_fn()
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, z):
        return self.network(z)


class VAEEncoder(nn.Module):
    """VAE Encoder with mean and log-variance outputs"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [128, 64],
        activation: str = 'relu'
    ):
        super().__init__()

        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'elu':
            act_fn = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                act_fn()
            ])
            prev_dim = h_dim

        self.shared = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x):
        h = self.shared(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def get_mean_map(self):
        """Return the mean map (mu(x)) as a function"""
        def mean_map(x):
            h = self.shared(x)
            return self.fc_mu(h)
        return mean_map


class DeterministicAE(nn.Module):
    """Standard deterministic autoencoder"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [128, 64],
        activation: str = 'relu'
    ):
        super().__init__()

        self.encoder = Encoder(input_dim, latent_dim, hidden_dims, activation)
        # Reverse hidden dims for decoder
        decoder_hidden = hidden_dims[::-1]
        self.decoder = Decoder(latent_dim, input_dim, decoder_hidden, activation)

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


class VAE(nn.Module):
    """Variational Autoencoder"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [128, 64],
        activation: str = 'relu',
        beta: float = 1.0
    ):
        super().__init__()

        self.encoder = VAEEncoder(input_dim, latent_dim, hidden_dims, activation)
        decoder_hidden = hidden_dims[::-1]
        self.decoder = Decoder(latent_dim, input_dim, decoder_hidden, activation)

        self.beta = beta
        self.latent_dim = latent_dim

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

    def loss_function(self, x, x_recon, mu, logvar, kl_weight=None):
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum') / x.size(0)

        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        # Total loss with beta weighting
        beta = kl_weight if kl_weight is not None else self.beta
        loss = recon_loss + beta * kl_div

        return {
            'loss': loss,
            'recon_loss': recon_loss.item(),
            'kl_div': kl_div.item(),
            'beta': beta
        }

    def sample(self, n_samples: int, device: str = 'cpu'):
        """Sample from prior"""
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decode(z)
