"""Jacobian-vector product utilities using torch.func"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np


def compute_jvp_encoder(
    encoder: nn.Module,
    x: torch.Tensor,
    v: torch.Tensor
) -> torch.Tensor:
    """
    Compute J_E(x) @ v using JVP.

    Args:
        encoder: Encoder network
        x: Input tensor (batch_size, input_dim) or (input_dim,)
        v: Direction vectors (batch_size, input_dim, k) or (input_dim, k) or (input_dim,)

    Returns:
        J_E(x) @ v: (batch_size, latent_dim, k) or matching input shape
    """
    from torch.func import jvp

    # Handle different input shapes
    single_sample = x.dim() == 1
    single_direction = v.dim() <= 2

    if single_sample:
        x = x.unsqueeze(0)

    batch_size = x.shape[0]

    if single_direction:
        if v.dim() == 1:
            v = v.unsqueeze(0).unsqueeze(-1)  # (1, input_dim, 1)
        elif v.dim() == 2:
            v = v.unsqueeze(0)  # (1, input_dim, k)
        v = v.expand(batch_size, -1, -1)

    # v is now (batch_size, input_dim, k)
    k = v.shape[2]
    results = []

    for i in range(k):
        v_i = v[:, :, i]  # (batch_size, input_dim)
        _, jvp_result = jvp(encoder, (x,), (v_i,))
        results.append(jvp_result)

    output = torch.stack(results, dim=-1)  # (batch_size, latent_dim, k)

    if single_sample:
        output = output.squeeze(0)

    return output


def compute_jvp_decoder(
    decoder: nn.Module,
    z: torch.Tensor,
    u: torch.Tensor
) -> torch.Tensor:
    """
    Compute J_D(z) @ u using JVP.

    Args:
        decoder: Decoder network
        z: Latent tensor (batch_size, latent_dim) or (latent_dim,)
        u: Direction vectors (batch_size, latent_dim, k) or (latent_dim, k) or (latent_dim,)

    Returns:
        J_D(z) @ u: (batch_size, output_dim, k) or matching input shape
    """
    from torch.func import jvp

    single_sample = z.dim() == 1
    single_direction = u.dim() <= 2

    if single_sample:
        z = z.unsqueeze(0)

    batch_size = z.shape[0]

    if single_direction:
        if u.dim() == 1:
            u = u.unsqueeze(0).unsqueeze(-1)
        elif u.dim() == 2:
            u = u.unsqueeze(0)
        u = u.expand(batch_size, -1, -1)

    k = u.shape[2]
    results = []

    for i in range(k):
        u_i = u[:, :, i]
        _, jvp_result = jvp(decoder, (z,), (u_i,))
        results.append(jvp_result)

    output = torch.stack(results, dim=-1)

    if single_sample:
        output = output.squeeze(0)

    return output


def compute_jvp_composed(
    encoder: nn.Module,
    decoder: nn.Module,
    x: torch.Tensor,
    v: torch.Tensor
) -> torch.Tensor:
    """
    Compute J_D(E(x)) @ J_E(x) @ v using composed JVP.

    Args:
        encoder: Encoder network
        decoder: Decoder network
        x: Input tensor (batch_size, input_dim)
        v: Direction vectors (batch_size, input_dim, k)

    Returns:
        J_DE(x) @ v: (batch_size, output_dim, k)
    """
    # First compute J_E(x) @ v
    u = compute_jvp_encoder(encoder, x, v)  # (batch_size, latent_dim, k)

    # Then compute J_D(E(x)) @ u
    z = encoder(x)
    output = compute_jvp_decoder(decoder, z, u)  # (batch_size, output_dim, k)

    return output


def sample_orthonormal_vectors(
    dim: int,
    k: int,
    method: str = 'random',
    data_cov: Optional[torch.Tensor] = None,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample k orthonormal vectors in R^dim.

    Args:
        dim: Dimension of the space
        k: Number of vectors
        method: 'random' or 'pca'
        data_cov: Data covariance matrix (dim, dim) for PCA method
        device: Device

    Returns:
        V_k: (dim, k) orthonormal matrix
    """
    if method == 'random':
        # Sample random Gaussian and QR decompose
        A = torch.randn(dim, k, device=device)
        V_k, _ = torch.linalg.qr(A)
        return V_k

    elif method == 'pca':
        if data_cov is None:
            raise ValueError("data_cov required for PCA method")

        # Compute top-k eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(data_cov)
        # Sort in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        V_k = eigenvectors[:, idx[:k]]
        return V_k

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_data_pca_directions(
    X: torch.Tensor,
    k_max: int = 8
) -> torch.Tensor:
    """
    Compute PCA directions from data.

    Args:
        X: Data tensor (n_samples, dim)
        k_max: Maximum number of principal components

    Returns:
        V: (dim, k_max) top principal directions
    """
    X_centered = X - X.mean(dim=0, keepdim=True)
    cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    idx = torch.argsort(eigenvalues, descending=True)
    V = eigenvectors[:, idx[:k_max]]
    return V
