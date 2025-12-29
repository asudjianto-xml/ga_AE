"""Geometric metrics based on Jacobians"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from .jacobian_utils import (
    compute_jvp_encoder,
    compute_jvp_decoder,
    compute_jvp_composed,
    sample_orthonormal_vectors
)


def compute_log_volume(
    jacobian_gram: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute log volume from J @ J^T using stabilized logdet.

    Args:
        jacobian_gram: J @ J^T matrix (batch_size, d, d) or (d, d)
        eps: Regularization for numerical stability

    Returns:
        log_vol: 0.5 * logdet(J @ J^T + eps * I)
    """
    single_sample = jacobian_gram.dim() == 2

    if single_sample:
        jacobian_gram = jacobian_gram.unsqueeze(0)

    batch_size, d, _ = jacobian_gram.shape
    device = jacobian_gram.device

    # Add regularization
    eye = torch.eye(d, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    regularized = jacobian_gram + eps * eye

    # Compute logdet via Cholesky (more stable)
    try:
        L = torch.linalg.cholesky(regularized)
        log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)
    except RuntimeError:
        # Fallback to eigenvalue decomposition
        eigenvalues = torch.linalg.eigvalsh(regularized)
        log_det = torch.sum(torch.log(eigenvalues.clamp(min=1e-10)), dim=-1)

    log_vol = 0.5 * log_det

    if single_sample:
        log_vol = log_vol.squeeze(0)

    return log_vol


def compute_k_volume(
    encoder: nn.Module,
    x: torch.Tensor,
    V_k: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute k-volume: log vol of J_E(x) @ V_k.

    Args:
        encoder: Encoder network
        x: Input tensor (batch_size, input_dim)
        V_k: Orthonormal directions (input_dim, k)
        eps: Regularization

    Returns:
        log_vol_k: (batch_size,) log k-volumes
    """
    # Compute A_k = J_E(x) @ V_k via JVP
    A_k = compute_jvp_encoder(encoder, x, V_k)  # (batch_size, latent_dim, k)

    # Compute A_k^T @ A_k (Gram matrix)
    gram = torch.bmm(A_k.transpose(1, 2), A_k)  # (batch_size, k, k)

    # Compute log volume
    log_vol_k = compute_log_volume(gram, eps)  # (batch_size,)

    return log_vol_k


def compute_k_volume_decoder(
    decoder: nn.Module,
    z: torch.Tensor,
    U_k: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute k-volume for decoder: log vol of J_D(z) @ U_k.

    Args:
        decoder: Decoder network
        z: Latent tensor (batch_size, latent_dim)
        U_k: Orthonormal directions (latent_dim, k)
        eps: Regularization

    Returns:
        log_vol_k: (batch_size,) log k-volumes
    """
    # Compute B_k = J_D(z) @ U_k via JVP
    B_k = compute_jvp_decoder(decoder, z, U_k)  # (batch_size, output_dim, k)

    # Compute B_k^T @ B_k
    gram = torch.bmm(B_k.transpose(1, 2), B_k)  # (batch_size, k, k)

    # Compute log volume
    log_vol_k = compute_log_volume(gram, eps)

    return log_vol_k


def compute_edc_k(
    encoder: nn.Module,
    decoder: nn.Module,
    x: torch.Tensor,
    V_k: torch.Tensor
) -> torch.Tensor:
    """
    Compute encoder-decoder consistency: ||J_DE(x) @ V_k - V_k||_F^2.

    Args:
        encoder: Encoder network
        decoder: Decoder network
        x: Input tensor (batch_size, input_dim)
        V_k: Orthonormal directions (input_dim, k)

    Returns:
        edc: (batch_size,) consistency errors
    """
    # Compute J_DE(x) @ V_k
    composed_v = compute_jvp_composed(encoder, decoder, x, V_k)  # (batch_size, input_dim, k)

    # Compare to V_k
    V_k_expanded = V_k.unsqueeze(0).expand(x.shape[0], -1, -1)  # (batch_size, input_dim, k)
    diff = composed_v - V_k_expanded

    # Frobenius norm squared
    edc = torch.sum(diff ** 2, dim=(1, 2))  # (batch_size,)

    return edc


def compute_decoder_stability(
    decoder: nn.Module,
    z_samples: torch.Tensor,
    U_k: torch.Tensor,
    eps: float = 1e-6
) -> Dict[str, torch.Tensor]:
    """
    Compute decoder stability metrics for given latent samples.

    Args:
        decoder: Decoder network
        z_samples: Latent samples (n_samples, latent_dim)
        U_k: Orthonormal directions (latent_dim, k)
        eps: Regularization

    Returns:
        Dictionary with stability metrics
    """
    log_vol_k = compute_k_volume_decoder(decoder, z_samples, U_k, eps)

    return {
        'log_vol_k_mean': log_vol_k.mean().item(),
        'log_vol_k_std': log_vol_k.std().item(),
        'log_vol_k_p10': torch.quantile(log_vol_k, 0.1).item(),
        'log_vol_k_p50': torch.quantile(log_vol_k, 0.5).item(),
        'log_vol_k_p90': torch.quantile(log_vol_k, 0.9).item()
    }


def compute_generative_gap_index(
    encoder: nn.Module,
    decoder: nn.Module,
    x_data: torch.Tensor,
    z_prior: torch.Tensor,
    V_k_list: List[torch.Tensor],
    eps: float = 1e-6
) -> Dict[str, float]:
    """
    Compute the generative gap index: difference between on-manifold and off-manifold metrics.

    Args:
        encoder: Encoder network
        decoder: Decoder network
        x_data: Real data samples (n_samples, input_dim)
        z_prior: Prior samples (n_samples, latent_dim)
        V_k_list: List of orthonormal direction sets [(input_dim, k1), (input_dim, k2), ...]
        eps: Regularization

    Returns:
        Dictionary with gap metrics
    """
    with torch.no_grad():
        # On-manifold: encode real data
        z_post = encoder(x_data)

        # Compute decoder volumes for both regimes
        gap_metrics = {}

        for i, V_k in enumerate(V_k_list):
            k = V_k.shape[1]

            # Sample U_k in latent space (random)
            U_k = sample_orthonormal_vectors(
                z_post.shape[1], k,
                method='random',
                device=z_post.device
            )

            # On-manifold decoder volume
            log_vol_post = compute_k_volume_decoder(decoder, z_post, U_k, eps).mean()

            # Off-manifold decoder volume
            log_vol_prior = compute_k_volume_decoder(decoder, z_prior, U_k, eps).mean()

            # Gap
            gap = (log_vol_prior - log_vol_post).item()
            gap_metrics[f'gap_kvol_k{k}'] = gap

        # Encoder-decoder consistency (on-manifold)
        edc_values = []
        for V_k in V_k_list:
            edc = compute_edc_k(encoder, decoder, x_data, V_k).mean()
            edc_values.append(edc.item())

        gap_metrics['edc_mean'] = np.mean(edc_values)

        # Overall gap score (average across k)
        gap_scores = [v for k, v in gap_metrics.items() if k.startswith('gap_kvol')]
        gap_metrics['gap_overall'] = np.mean(gap_scores) if gap_scores else 0.0

    return gap_metrics


def compute_full_diagnostics(
    encoder: nn.Module,
    decoder: nn.Module,
    x_batch: torch.Tensor,
    k_values: List[int] = [1, 2, 4, 8],
    eps_values: List[float] = [1e-8, 1e-6, 1e-4],
    pca_directions: Optional[torch.Tensor] = None,
    compute_edc: bool = True
) -> Dict[str, any]:
    """
    Compute full suite of geometric diagnostics for a batch.

    Args:
        encoder: Encoder network
        decoder: Decoder network
        x_batch: Input batch (batch_size, input_dim)
        k_values: List of k values for k-volume
        eps_values: List of epsilon values for regularization
        pca_directions: Optional PCA directions (input_dim, k_max)
        compute_edc: Whether to compute EDC metrics

    Returns:
        Dictionary with all diagnostic metrics
    """
    diagnostics = {}
    device = x_batch.device
    batch_size, input_dim = x_batch.shape

    with torch.no_grad():
        z_batch = encoder(x_batch)
        latent_dim = z_batch.shape[1]

        # For each eps and k, compute metrics
        for eps in eps_values:
            for k in k_values:
                if k > min(input_dim, latent_dim):
                    continue

                # Random directions
                V_k_random = sample_orthonormal_vectors(input_dim, k, 'random', device=device)
                log_vol_random = compute_k_volume(encoder, x_batch, V_k_random, eps)

                key_prefix = f'eps{eps:.0e}_k{k}_random'
                diagnostics[f'{key_prefix}_mean'] = log_vol_random.mean().item()
                diagnostics[f'{key_prefix}_std'] = log_vol_random.std().item()
                diagnostics[f'{key_prefix}_p10'] = torch.quantile(log_vol_random, 0.1).item()
                diagnostics[f'{key_prefix}_p50'] = torch.quantile(log_vol_random, 0.5).item()
                diagnostics[f'{key_prefix}_p90'] = torch.quantile(log_vol_random, 0.9).item()

                # PCA directions if provided
                if pca_directions is not None and k <= pca_directions.shape[1]:
                    V_k_pca = pca_directions[:, :k]
                    log_vol_pca = compute_k_volume(encoder, x_batch, V_k_pca, eps)

                    key_prefix = f'eps{eps:.0e}_k{k}_pca'
                    diagnostics[f'{key_prefix}_mean'] = log_vol_pca.mean().item()
                    diagnostics[f'{key_prefix}_std'] = log_vol_pca.std().item()
                    diagnostics[f'{key_prefix}_p10'] = torch.quantile(log_vol_pca, 0.1).item()
                    diagnostics[f'{key_prefix}_p50'] = torch.quantile(log_vol_pca, 0.5).item()
                    diagnostics[f'{key_prefix}_p90'] = torch.quantile(log_vol_pca, 0.9).item()

                # EDC if requested
                if compute_edc:
                    edc_random = compute_edc_k(encoder, decoder, x_batch, V_k_random)
                    diagnostics[f'edc_k{k}_random_mean'] = edc_random.mean().item()
                    diagnostics[f'edc_k{k}_random_std'] = edc_random.std().item()

                    if pca_directions is not None and k <= pca_directions.shape[1]:
                        V_k_pca = pca_directions[:, :k]
                        edc_pca = compute_edc_k(encoder, decoder, x_batch, V_k_pca)
                        diagnostics[f'edc_k{k}_pca_mean'] = edc_pca.mean().item()
                        diagnostics[f'edc_k{k}_pca_std'] = edc_pca.std().item()

    return diagnostics
