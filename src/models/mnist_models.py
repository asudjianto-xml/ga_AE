"""
MLP-based autoencoder models for MNIST.
Simpler than CNN models, faster training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """MLP encoder for MNIST."""
    def __init__(self, latent_dim=32, input_dim=784):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 1, 28, 28) or (batch, 784)
        Returns:
            z: (batch, latent_dim)
        """
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        return self.encoder(x)


class MLPDecoder(nn.Module):
    """MLP decoder for MNIST."""
    def __init__(self, latent_dim=32, output_dim=784):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

            nn.Linear(512, output_dim),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z):
        """
        Args:
            z: (batch, latent_dim)
        Returns:
            x: (batch, 1, 28, 28)
        """
        x = self.decoder(z)
        return x.view(x.size(0), 1, 28, 28)


class MNISTVAE(nn.Module):
    """Variational Autoencoder for MNIST."""
    def __init__(self, latent_dim=32, beta=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder outputs mu and logvar
        self.encoder_base = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder = MLPDecoder(latent_dim)

    def encode(self, x):
        """Encode to latent distribution parameters."""
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        h = self.encoder_base(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode from latent."""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss_function(self, recon, x, mu, logvar):
        """VAE loss = reconstruction + beta * KL divergence."""
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, x, reduction='mean')

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # Average over batch

        # Total loss
        loss = recon_loss + self.beta * kl_loss

        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


class MNISTGAAE(nn.Module):
    """
    Geometric Autoencoder for MNIST with Grassmann spread and blade entropy.
    Simplified version with faster geometric computations.
    """
    def __init__(
        self,
        latent_dim=32,
        lambda_grass=0.1,
        lambda_entropy=0.01,
        k_values=(2, 4, 8)
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.lambda_grass = lambda_grass
        self.lambda_entropy = lambda_entropy
        self.k_values = k_values

        # Encoder and decoder
        self.encoder = MLPEncoder(latent_dim)
        self.decoder = MLPDecoder(latent_dim)

    def forward(self, x):
        """Forward pass."""
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def grassmann_spread_loss(self, z, k=2, n_pairs=16):
        """
        Grassmann spread loss: repel tangent k-blades.
        Simplified for MLP decoder.
        """
        batch_size = z.size(0)
        device = z.device

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # Sample pairs
        indices = torch.randperm(batch_size, device=device)[:min(n_pairs * 2, batch_size)]
        if len(indices) < 2:
            return torch.tensor(0.0, device=device)

        n_actual_pairs = len(indices) // 2
        z_i = z[indices[:n_actual_pairs]].requires_grad_(True)
        z_j = z[indices[n_actual_pairs:2*n_actual_pairs]].requires_grad_(True)

        # Sample random directions in latent space
        W_k = torch.randn(n_actual_pairs, self.latent_dim, k, device=device)
        W_k, _ = torch.linalg.qr(W_k)

        # Compute decoder Jacobian via batched autograd
        # For each direction, compute J_D(z) @ w
        U_i_list = []
        U_j_list = []

        for dir_idx in range(k):
            w = W_k[:, :, dir_idx]  # (n_pairs, latent_dim)

            # Decode
            x_i = self.decoder(z_i).view(n_actual_pairs, -1)  # (n_pairs, 784)
            x_j = self.decoder(z_j).view(n_actual_pairs, -1)

            # Compute J_D(z_i) @ w via autograd
            # Use torch.autograd.grad with vector-Jacobian product
            grad_outputs_i = torch.ones_like(x_i)
            jvp_i = torch.autograd.grad(
                outputs=x_i,
                inputs=z_i,
                grad_outputs=grad_outputs_i,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]  # (n_pairs, latent_dim)

            # Project onto direction w
            jvp_i_w = (jvp_i * w).sum(dim=1, keepdim=True)  # (n_pairs, 1)

            # Actually, we need J_D(z_i) @ w which is a vector in data space
            # Let me compute this properly using forward-mode AD

            # Simplified: use finite differences
            eps = 1e-4
            x_i_plus = self.decoder(z_i + eps * w).view(n_actual_pairs, -1)
            jvp_i_w = (x_i_plus - x_i) / eps  # (n_pairs, 784)

            x_j_plus = self.decoder(z_j + eps * w).view(n_actual_pairs, -1)
            jvp_j_w = (x_j_plus - x_j) / eps

            U_i_list.append(jvp_i_w)
            U_j_list.append(jvp_j_w)

        # Stack into frames: (n_pairs, 784, k)
        U_i = torch.stack(U_i_list, dim=2)
        U_j = torch.stack(U_j_list, dim=2)

        # Orthonormalize
        U_i, _ = torch.linalg.qr(U_i)
        U_j, _ = torch.linalg.qr(U_j)

        # Compute Grassmann similarity
        M = torch.bmm(U_i.transpose(1, 2), U_j)  # (n_pairs, k, k)
        MM = torch.bmm(M, M.transpose(1, 2))  # (n_pairs, k, k)

        # Stabilization
        eps = 1e-6
        MM = MM + eps * torch.eye(k, device=device).unsqueeze(0)

        # Log determinant
        log_det = torch.logdet(MM + 1e-8)
        sim = torch.exp(0.5 * log_det.clamp(min=-10, max=10))

        # Loss: minimize similarity (repulsion)
        loss = sim.mean()

        return loss

    def blade_entropy_loss(self, z, k_values=None, n_samples=32):
        """
        Blade entropy loss: maximize entropy across k-blade scales.
        Simplified for faster computation.
        """
        if k_values is None:
            k_values = self.k_values

        device = z.device
        batch_size = z.size(0)

        # Sample subset
        if batch_size > n_samples:
            indices = torch.randperm(batch_size, device=device)[:n_samples]
            z_sample = z[indices]
        else:
            z_sample = z

        # Compute s_k for each k (simplified: use norm-based approximation)
        s_k_list = []
        for k in k_values:
            # Sample random k-directions
            W_k = torch.randn(len(z_sample), self.latent_dim, k, device=device)
            W_k, _ = torch.linalg.qr(W_k)

            # Compute approximate k-volume via finite differences
            vols = []
            for i in range(len(z_sample)):
                z_i = z_sample[i:i+1]
                W_i = W_k[i]

                # Decode at z and z + small perturbations
                x_base = self.decoder(z_i).view(-1)
                jvps = []
                for j in range(k):
                    eps = 1e-4
                    w = W_i[:, j].unsqueeze(0)  # (1, latent_dim)
                    x_pert = self.decoder(z_i + eps * w).view(-1)
                    jvp = (x_pert - x_base) / eps
                    jvps.append(jvp)

                # Stack and compute Gram matrix
                J = torch.stack(jvps, dim=1)  # (784, k)
                G = torch.mm(J.T, J)  # (k, k)

                # k-volume = sqrt(det(G))
                vol = torch.sqrt(torch.det(G + 1e-6 * torch.eye(k, device=device)).clamp(min=1e-10))
                vols.append(vol)

            s_k = torch.stack(vols).mean()
            s_k_list.append(s_k)

        # Convert to distribution
        s_k_tensor = torch.stack(s_k_list)
        delta = 1e-8
        p_k = (s_k_tensor + delta) / (s_k_tensor.sum() + delta * len(k_values))

        # Compute entropy
        entropy = -(p_k * torch.log(p_k + 1e-10)).sum()

        return entropy

    def loss_function(self, recon, x, z):
        """GA-AE loss = reconstruction + grassmann spread - blade entropy."""
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction='mean')

        # Geometric regularization
        if self.training and self.lambda_grass > 0:
            grass_loss = self.grassmann_spread_loss(z, k=2, n_pairs=4)
        else:
            grass_loss = torch.tensor(0.0, device=z.device)

        if self.training and self.lambda_entropy > 0:
            entropy = self.blade_entropy_loss(z, (2, 4), n_samples=16)
        else:
            entropy = torch.tensor(0.0, device=z.device)

        # Total loss
        loss = recon_loss + self.lambda_grass * grass_loss - self.lambda_entropy * entropy

        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'grass_loss': grass_loss,
            'blade_entropy': entropy
        }


if __name__ == '__main__':
    # Test models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Testing MNIST VAE...")
    vae = MNISTVAE(latent_dim=32, beta=1.0).to(device)
    x = torch.randn(4, 1, 28, 28).to(device)
    recon, mu, logvar = vae(x)
    losses = vae.loss_function(recon, x, mu, logvar)
    print(f"  Input shape: {x.shape}")
    print(f"  Recon shape: {recon.shape}")
    print(f"  Loss: {losses['loss'].item():.4f}")

    print("\nTesting MNIST GA-AE...")
    gaae = MNISTGAAE(latent_dim=32).to(device)
    recon, z = gaae(x)
    losses = gaae.loss_function(recon, x, z)
    print(f"  Input shape: {x.shape}")
    print(f"  Recon shape: {recon.shape}")
    print(f"  Latent shape: {z.shape}")
    print(f"  Loss: {losses['loss'].item():.4f}")
