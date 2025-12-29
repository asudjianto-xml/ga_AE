"""
Tangent Chamfer Loss: GA-Native Prior via Projection/Rejection

Implements the "moving target" blade field approach using Grassmannian nearest neighbors.
Instead of matching to a single global blade, each prior sample finds the nearest
posterior sample and matches to its local tangent blade geometry.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp
from typing import Optional, Dict


def orthonormalize_cols(U: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Orthonormalize the columns of U (shape: [n, k]) using QR.
    Returns Q with shape [n, k].
    """
    Q, R = torch.linalg.qr(U, mode="reduced")
    return Q


def batched_orthonormalize(U: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    U: [B, n, k] -> Q: [B, n, k]
    """
    Qs = []
    for b in range(U.shape[0]):
        Qs.append(orthonormalize_cols(U[b], eps=eps))
    return torch.stack(Qs, dim=0)


def decoder_jvp_columns(decoder_fn, z: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute U = J_D(z) V where:
      z: [B, d]
      V: [d, k] (columns are direction vectors in latent space)
    Returns:
      U: [B, n, k]
    """
    B, d = z.shape
    d2, k = V.shape
    assert d2 == d, f"Dimension mismatch: z has dim {d}, V has dim {d2}"

    # Compute k JVPs; each gives [B, n]
    cols = []
    for j in range(k):
        vj = V[:, j].expand(B, d)  # [B, d]

        # jvp wants a function of z only
        def f(zz):
            return decoder_fn(zz)

        _, jvp_out = jvp(f, (z,), (vj,))
        cols.append(jvp_out)  # [B, n]

    U = torch.stack(cols, dim=-1)  # [B, n, k]
    return U


class TangentReferenceBank:
    """
    Stores posterior codes and corresponding reference tangent bases Q_ref in data space.
    Used to implement Grassmannian nearest-neighbor blade selection.

    This is the "moving target" fix: instead of using a single global blade,
    each prior sample queries the nearest posterior sample's tangent geometry.
    """
    def __init__(self, max_items: int = 8192, device: str = "cpu"):
        self.max_items = max_items
        self.device = device
        self.z_codes = None      # [M, d]
        self.Q_refs = None       # [M, n, k]
        self.ptr = 0
        self.full = False

    @torch.no_grad()
    def add(self, z_post: torch.Tensor, Q_ref: torch.Tensor):
        """
        z_post: [B, d] - posterior latent codes
        Q_ref:  [B, n, k] - orthonormal tangent bases in data space
        """
        z_post = z_post.detach().to(self.device)
        Q_ref = Q_ref.detach().to(self.device)

        B = z_post.shape[0]
        if self.z_codes is None:
            d = z_post.shape[1]
            n, k = Q_ref.shape[1], Q_ref.shape[2]
            self.z_codes = torch.zeros((self.max_items, d), device=self.device)
            self.Q_refs = torch.zeros((self.max_items, n, k), device=self.device)

        # Circular buffer
        end = self.ptr + B
        if end <= self.max_items:
            self.z_codes[self.ptr:end] = z_post
            self.Q_refs[self.ptr:end] = Q_ref
        else:
            first = self.max_items - self.ptr
            self.z_codes[self.ptr:] = z_post[:first]
            self.Q_refs[self.ptr:] = Q_ref[:first]
            remain = B - first
            self.z_codes[:remain] = z_post[first:]
            self.Q_refs[:remain] = Q_ref[first:]
            self.full = True

        self.ptr = (self.ptr + B) % self.max_items
        if self.ptr == 0:
            self.full = True

    @torch.no_grad()
    def query_nearest(self, z_prior: torch.Tensor) -> torch.Tensor:
        """
        z_prior: [B, d]
        Returns Q_ref_nn: [B, n, k] corresponding to nearest z_post in the bank.

        This implements the Grassmannian nearest-neighbor selection:
        For each z ~ p(z), find z* = argmin ||z - z_i|| in posterior codes,
        and return the tangent blade Q_ref(z*).
        """
        assert self.z_codes is not None, "Reference bank is empty. Call add() first."
        z_prior = z_prior.detach().to(self.device)

        M = self.max_items if self.full else self.ptr
        if M == 0:
            # Bank is empty, return zeros (will be handled by calling code)
            B = z_prior.shape[0]
            n, k = self.Q_refs.shape[1], self.Q_refs.shape[2]
            return torch.zeros((B, n, k), device=z_prior.device)

        Z = self.z_codes[:M]  # [M, d]
        # Compute nearest by squared Euclidean distance
        dists = torch.cdist(z_prior, Z, p=2.0)  # [B, M]
        nn_idx = torch.argmin(dists, dim=1)     # [B]
        Q_nn = self.Q_refs[nn_idx]              # [B, n, k]
        return Q_nn


@torch.no_grad()
def build_reference_tangent_from_posterior(decoder_fn, z_post: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute Q_ref(z_post) = orth( J_D(z_post) V ).

    This builds the "valid tangent blade" at posterior samples by computing
    the decoder's local tangent directions and orthonormalizing them.

    z_post: [B, d]
    V:      [d, k]
    returns Q_ref: [B, n, k]
    """
    U_post = decoder_jvp_columns(decoder_fn, z_post, V)  # [B, n, k]
    Q_ref = batched_orthonormalize(U_post)               # [B, n, k]
    return Q_ref


class RejectionLoss(nn.Module):
    """
    Projection/Rejection loss in geometric algebra framework.

    L_rej = E_{z~p(z)} || (I - Q_ref Q_ref^T) U(z) ||_F^2

    where:
      U(z) = J_D(z) V_k  (k tangent directions in data space)
      Q_ref spans the locally valid tangent blade (from nearest posterior sample)

    This forces decoder tangents at prior samples to lie within the blade
    defined by nearby posterior samples, preventing invalid extrapolation.
    """
    def __init__(self, k: int, eps: float = 1e-8):
        super().__init__()
        self.k = k
        self.eps = eps

    def forward(self, U: torch.Tensor, Q_ref: torch.Tensor) -> torch.Tensor:
        """
        U:     [B, n, k] - decoder tangent directions at prior samples
        Q_ref: [B, n, k] - orthonormal reference blades from nearest posterior

        Returns scalar loss (mean rejection norm squared).
        """
        # Compute projection: Q Q^T U
        # QtU = Q^T @ U: [B, k, k]
        QtU = torch.bmm(Q_ref.transpose(1, 2), U)  # [B, k, n] @ [B, n, k] -> [B, k, k]

        # Proj = Q @ Q^T @ U: [B, n, k]
        Proj = torch.bmm(Q_ref, QtU)  # [B, n, k] @ [B, k, k] -> [B, n, k]

        # Rejection = U - Proj
        Rej = U - Proj

        # Return mean squared Frobenius norm
        return (Rej.pow(2).sum(dim=(1, 2))).mean()


class TangentChamferLoss(nn.Module):
    """
    Tangent-space Chamfer distance: match decoder tangents to nearest posterior tangents.

    For each z_prior ~ p(z):
      1. Find nearest z_post in reference bank -> Q_ref
      2. Compute U_prior = J_D(z_prior) V
      3. Penalize rejection of U_prior from Q_ref

    This implements the "moving target" blade field on the Grassmannian bundle.
    """
    def __init__(self, ref_bank: TangentReferenceBank, k: int, eps: float = 1e-8):
        super().__init__()
        self.ref_bank = ref_bank
        self.k = k
        self.eps = eps
        self.rejection = RejectionLoss(k=k, eps=eps)

    def forward(self, decoder_fn, z_prior: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        decoder_fn: callable, decoder(z) -> x
        z_prior: [B, d] - samples from prior p(z)
        V: [d, k] - tangent directions in latent space

        Returns scalar rejection loss.
        """
        # Query reference tangent bases (nearest neighbor in latent)
        Q_ref = self.ref_bank.query_nearest(z_prior).to(z_prior.device)  # [B, n, k]

        # Compute U_prior = J_D(z_prior) V
        U_prior = decoder_jvp_columns(decoder_fn, z_prior, V)  # [B, n, k]

        # Optional: orthonormalize U_prior for stability
        # (not required for rejection, but helps with numerical scale)
        # U_prior = batched_orthonormalize(U_prior)

        return self.rejection(U_prior, Q_ref)


def k_volume_logdet_from_jvp(jvp_cols: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute k-volume from JVP columns via Gram determinant.

    jvp_cols: [B, m, k] (m is output dim of the map whose Jacobian is used)

    Computes 0.5 * logdet( U^T U + eps I ) for each batch item.
    Returns [B].
    """
    B, m, k = jvp_cols.shape
    # Gram: [B, k, k]
    G = torch.bmm(jvp_cols.transpose(1, 2), jvp_cols)  # [B, k, k]
    G = G + eps * torch.eye(k, device=jvp_cols.device).unsqueeze(0)

    # logdet via Cholesky (more stable than direct logdet)
    try:
        L = torch.linalg.cholesky(G)
        logdet = 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1) + 1e-8), dim=-1)
    except:
        # Fallback to direct logdet if Cholesky fails
        logdet = torch.logdet(G + 1e-8)

    return 0.5 * logdet


class BladeCollapseBarrier(nn.Module):
    """
    Hard anti-collapse constraint: make posterior collapse illegal rather than unlikely.

    L_collapse = sum_k ReLU(tau - log|∧^k J_μ(x)|)

    This penalizes k-volumes that fall below threshold tau.
    """
    def __init__(self, k_list=(2, 4, 8), tau=0.0, eps=1e-8):
        super().__init__()
        self.k_list = k_list
        self.tau = tau
        self.eps = eps

    def forward(self, jacobian_cols_by_k: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        jacobian_cols_by_k: dict k -> U_k where U_k = J_map(x) V_k has shape [B, m, k]

        Penalize max(0, tau - logvol_k).
        """
        losses = []
        for k, U in jacobian_cols_by_k.items():
            if k not in self.k_list:
                continue
            logvol = k_volume_logdet_from_jvp(U, eps=self.eps)  # [B]
            losses.append(F.relu(self.tau - logvol).mean())

        if len(losses) == 0:
            return torch.tensor(0.0, device=next(iter(jacobian_cols_by_k.values())).device)

        return sum(losses) / len(losses)
