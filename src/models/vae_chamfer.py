"""
VAE with GA-Native Prior via Tangent Chamfer Loss

Replaces or augments KL divergence with geometric projection/rejection constraints.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, List
from .base_models import VAE
from .tangent_chamfer import (
    TangentReferenceBank,
    TangentChamferLoss,
    BladeCollapseBarrier,
    build_reference_tangent_from_posterior,
    decoder_jvp_columns
)
from ..diagnostics.jacobian_utils import sample_orthonormal_vectors


class VAE_TangentChamfer(VAE):
    """
    VAE with Tangent Chamfer Loss: GA-Native Prior via Projection/Rejection.

    Instead of (or in addition to) KL divergence, uses:
      L_chamfer = E[|| Rej_Q(U_prior) ||^2]
    where Q comes from nearest posterior sample's tangent blade.

    Options:
      - use_kl=True, lambda_chamfer > 0: KL + Chamfer (hybrid)
      - use_kl=False, lambda_chamfer > 0: Pure Chamfer (E7 experiment)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [128, 64],
        activation: str = 'relu',
        beta: float = 1.0,
        use_kl: bool = True,
        # Tangent Chamfer params
        lambda_chamfer: float = 0.1,
        chamfer_k: int = 4,
        ref_bank_size: int = 4096,
        # Optional collapse barrier
        lambda_collapse: float = 0.0,
        collapse_tau: float = 0.0,
        collapse_k_list: tuple = (2, 4),
        eps: float = 1e-6
    ):
        super().__init__(input_dim, latent_dim, hidden_dims, activation, beta)

        self.use_kl = use_kl
        self.lambda_chamfer = lambda_chamfer
        self.chamfer_k = chamfer_k
        self.lambda_collapse = lambda_collapse
        self.eps = eps

        # Reference bank for Grassmannian nearest neighbor
        self.ref_bank = TangentReferenceBank(max_items=ref_bank_size, device='cpu')

        # Chamfer loss module
        self.tangent_chamfer = TangentChamferLoss(
            ref_bank=self.ref_bank,
            k=chamfer_k,
            eps=eps
        )

        # Optional collapse barrier
        if lambda_collapse > 0:
            self.collapse_barrier = BladeCollapseBarrier(
                k_list=collapse_k_list,
                tau=collapse_tau,
                eps=eps
            )
        else:
            self.collapse_barrier = None

        # Store V_k for JVP computations (sample once, reuse)
        self.register_buffer('V_k', None)

    def _get_or_sample_V_k(self, device):
        """Get or sample tangent directions V_k in latent space."""
        if self.V_k is None or self.V_k.shape != (self.latent_dim, self.chamfer_k):
            V_k = sample_orthonormal_vectors(
                self.latent_dim, self.chamfer_k, 'random', device=device
            )
            self.V_k = V_k
        return self.V_k.to(device)

    def update_reference_bank(self, x: torch.Tensor):
        """
        Update the reference bank with current batch's posterior tangent blades.

        Called during training to populate the bank with (z_post, Q_ref) pairs.
        """
        with torch.no_grad():
            # Encode to posterior
            h = self.encoder.shared(x)
            mu = self.encoder.fc_mu(h)

            # Build reference tangent basis at mu using decoder
            V_k = self._get_or_sample_V_k(x.device)
            Q_ref = build_reference_tangent_from_posterior(
                self.decoder, mu, V_k
            )  # [B, output_dim, k]

            # Add to bank
            self.ref_bank.add(mu, Q_ref)

    def compute_chamfer_loss(self, x: torch.Tensor, z_prior: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute tangent chamfer loss for prior samples.

        If z_prior not provided, samples from N(0, I).
        """
        batch_size = x.size(0)
        device = x.device

        if z_prior is None:
            z_prior = torch.randn(batch_size, self.latent_dim, device=device)

        V_k = self._get_or_sample_V_k(device)

        # Compute Chamfer loss (queries nearest neighbor from bank)
        chamfer_loss = self.tangent_chamfer(self.decoder, z_prior, V_k)

        return chamfer_loss

    def compute_collapse_barrier(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute collapse barrier on encoder mean map.

        Computes k-volumes for mean encoder and penalizes low volumes.
        """
        if self.collapse_barrier is None:
            return torch.tensor(0.0, device=x.device)

        device = x.device

        # Build mean encoder wrapper
        class MeanEncoder(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder

            def forward(self, x):
                h = self.encoder.shared(x)
                return self.encoder.fc_mu(h)

        mean_encoder = MeanEncoder(self.encoder)

        # Compute JVPs for different k
        jacobian_cols_by_k = {}
        for k in self.collapse_barrier.k_list:
            if k > min(x.shape[1], self.latent_dim):
                continue

            V_k = sample_orthonormal_vectors(x.shape[1], k, 'random', device=device)
            U_k = decoder_jvp_columns(mean_encoder, x, V_k)  # [B, latent_dim, k]
            jacobian_cols_by_k[k] = U_k

        return self.collapse_barrier(jacobian_cols_by_k)

    def loss_function(self, x, x_recon, mu, logvar, kl_weight=None):
        """
        Compute loss with optional KL and Chamfer terms.

        Args:
            x: input data [B, n]
            x_recon: reconstruction [B, n]
            mu: encoder mean [B, d]
            logvar: encoder log variance [B, d]
            kl_weight: optional KL weight (for annealing)

        Returns:
            dict with loss components
        """
        # Base reconstruction loss
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum') / x.size(0)

        # KL divergence (optional)
        if self.use_kl:
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            beta = kl_weight if kl_weight is not None else self.beta
        else:
            kl_div = torch.tensor(0.0, device=x.device)
            beta = 0.0

        # Tangent Chamfer loss (geometric prior)
        if self.lambda_chamfer > 0:
            chamfer_loss = self.compute_chamfer_loss(x)
        else:
            chamfer_loss = torch.tensor(0.0, device=x.device)

        # Collapse barrier (optional)
        if self.lambda_collapse > 0:
            collapse_loss = self.compute_collapse_barrier(x)
        else:
            collapse_loss = torch.tensor(0.0, device=x.device)

        # Total loss
        loss = recon_loss + \
               beta * kl_div + \
               self.lambda_chamfer * chamfer_loss + \
               self.lambda_collapse * collapse_loss

        return {
            'loss': loss,
            'recon_loss': recon_loss.item(),
            'kl_div': kl_div.item(),
            'chamfer_loss': chamfer_loss.item() if self.lambda_chamfer > 0 else 0.0,
            'collapse_loss': collapse_loss.item() if self.lambda_collapse > 0 else 0.0,
            'beta': beta
        }
