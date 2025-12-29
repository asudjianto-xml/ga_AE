"""Evaluation metrics for generative models"""
import torch
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from scipy.stats import entropy


def compute_mmd(x: torch.Tensor, y: torch.Tensor, kernel: str = 'rbf', sigma: float = 1.0) -> float:
    """
    Compute Maximum Mean Discrepancy between two samples.

    Args:
        x: First sample (n, d)
        y: Second sample (m, d)
        kernel: Kernel type ('rbf')
        sigma: Kernel bandwidth

    Returns:
        MMD value
    """
    x = x.detach().cpu()
    y = y.detach().cpu()

    def rbf_kernel(a, b, sigma):
        a = a.unsqueeze(1)  # (n, 1, d)
        b = b.unsqueeze(0)  # (1, m, d)
        dist = torch.sum((a - b) ** 2, dim=2)
        return torch.exp(-dist / (2 * sigma ** 2))

    k_xx = rbf_kernel(x, x, sigma).mean()
    k_yy = rbf_kernel(y, y, sigma).mean()
    k_xy = rbf_kernel(x, y, sigma).mean()

    mmd = k_xx + k_yy - 2 * k_xy
    return mmd.item()


def compute_energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Energy Distance between two samples.

    ED = 2 E[||X - Y||] - E[||X - X'||] - E[||Y - Y'||]

    Args:
        x: First sample (n, d)
        y: Second sample (m, d)

    Returns:
        Energy distance
    """
    # Pairwise distances
    dist_xy = pairwise_distances(x, y, metric='euclidean')
    dist_xx = pairwise_distances(x, x, metric='euclidean')
    dist_yy = pairwise_distances(y, y, metric='euclidean')

    # Compute expectations
    ed = 2 * dist_xy.mean() - dist_xx.mean() - dist_yy.mean()
    return ed


def compute_knn_precision_recall(
    real: np.ndarray,
    generated: np.ndarray,
    k: int = 5
) -> Tuple[float, float]:
    """
    Compute k-NN precision and recall.

    Precision: fraction of generated samples whose k-NN contains a real sample
    Recall: fraction of real samples whose k-NN contains a generated sample

    Args:
        real: Real samples (n, d)
        generated: Generated samples (m, d)
        k: Number of neighbors

    Returns:
        (precision, recall)
    """
    from sklearn.neighbors import NearestNeighbors

    # Precision: for each generated sample, check if nearest neighbor is real
    nbrs_real = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(real)
    distances_gen, _ = nbrs_real.kneighbors(generated)

    # Also fit on generated
    nbrs_gen = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(generated)
    distances_real, _ = nbrs_gen.kneighbors(real)

    # Precision: fraction of generated with small distance to real
    # Use median distance in real data as threshold
    dist_real_to_real = nbrs_real.kneighbors(real)[0]
    threshold = np.median(dist_real_to_real[:, -1])

    precision = (distances_gen[:, 0] < threshold).mean()

    # Recall: fraction of real covered by generated
    recall = (distances_real[:, 0] < threshold).mean()

    return float(precision), float(recall)


def compute_mode_coverage(
    generated: np.ndarray,
    real_labels: np.ndarray,
    generated_labels: np.ndarray,
    n_modes: int
) -> Dict[str, float]:
    """
    Compute mode coverage and rare mode recall.

    Args:
        generated: Generated samples (m, d)
        real_labels: True mode labels for real data (n,)
        generated_labels: Predicted mode labels for generated data (m,)
        n_modes: Total number of modes

    Returns:
        Dictionary with mode coverage metrics
    """
    # Count samples per mode in real and generated
    real_counts = np.bincount(real_labels, minlength=n_modes)
    gen_counts = np.bincount(generated_labels, minlength=n_modes)

    # Coverage: fraction of modes with at least one generated sample
    coverage = (gen_counts > 0).sum() / n_modes

    # KL divergence between distributions
    real_dist = real_counts / real_counts.sum()
    gen_dist = gen_counts / (gen_counts.sum() + 1e-10)

    # Avoid log(0)
    gen_dist = np.clip(gen_dist, 1e-10, 1.0)
    real_dist = np.clip(real_dist, 1e-10, 1.0)

    kl_div = entropy(real_dist, gen_dist)

    # Rare mode recall (last mode is rare by construction)
    rare_mode_idx = n_modes - 1
    rare_real = real_counts[rare_mode_idx]
    rare_gen = gen_counts[rare_mode_idx]

    rare_recall = rare_gen / (rare_real + 1e-10)

    return {
        'mode_coverage': float(coverage),
        'kl_divergence': float(kl_div),
        'rare_mode_recall': float(rare_recall),
        'rare_mode_count_real': int(rare_real),
        'rare_mode_count_gen': int(rare_gen)
    }


def assign_to_modes(
    samples: np.ndarray,
    centroids: np.ndarray
) -> np.ndarray:
    """
    Assign samples to nearest mode centroid.

    Args:
        samples: (n, d) samples
        centroids: (n_modes, d) mode centers

    Returns:
        labels: (n,) mode assignments
    """
    distances = cdist(samples, centroids, metric='euclidean')
    labels = np.argmin(distances, axis=1)
    return labels


def compute_generation_metrics(
    real_data: np.ndarray,
    generated_data: np.ndarray,
    real_labels: np.ndarray = None,
    n_modes: int = None
) -> Dict[str, float]:
    """
    Compute comprehensive generation quality metrics.

    Args:
        real_data: Real samples (n, d)
        generated_data: Generated samples (m, d)
        real_labels: Optional mode labels for mode coverage
        n_modes: Number of modes

    Returns:
        Dictionary with all metrics
    """
    metrics = {}

    # Convert to appropriate types
    real_torch = torch.from_numpy(real_data).float()
    gen_torch = torch.from_numpy(generated_data).float()

    # MMD
    metrics['mmd'] = compute_mmd(real_torch, gen_torch)

    # Energy Distance
    metrics['energy_distance'] = compute_energy_distance(real_data, generated_data)

    # k-NN precision/recall
    precision, recall = compute_knn_precision_recall(real_data, generated_data)
    metrics['knn_precision'] = precision
    metrics['knn_recall'] = recall
    metrics['knn_f1'] = 2 * precision * recall / (precision + recall + 1e-10)

    # Mode coverage if labels provided
    if real_labels is not None and n_modes is not None:
        from sklearn.cluster import KMeans

        # Fit KMeans on real data to get centroids
        kmeans = KMeans(n_clusters=n_modes, random_state=0, n_init=10)
        kmeans.fit(real_data)
        centroids = kmeans.cluster_centers_

        # Assign generated samples
        gen_labels = assign_to_modes(generated_data, centroids)

        mode_metrics = compute_mode_coverage(
            generated_data, real_labels, gen_labels, n_modes
        )
        metrics.update(mode_metrics)

    return metrics
