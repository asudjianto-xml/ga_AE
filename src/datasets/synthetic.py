"""Synthetic dataset generators for GA experiments"""
import numpy as np
import torch
from torch import nn
from typing import Tuple, Dict, Any
from sklearn.datasets import make_swiss_roll


def generate_mixture_of_gaussians(
    n_samples: int,
    dim: int = 2,
    n_components: int = 8,
    tail_weight: float = 0.02,
    seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate mixture of Gaussians with different correlation structures.

    Args:
        n_samples: Total number of samples
        dim: Dimension of each sample
        n_components: Number of mixture components
        tail_weight: Weight for the rare tail mode
        seed: Random seed

    Returns:
        samples: (n_samples, dim) array
        labels: (n_samples,) component assignments
    """
    rng = np.random.RandomState(seed)

    # Create mixture weights - one tail mode, rest equal
    weights = np.ones(n_components)
    weights[-1] = tail_weight * (n_components - 1)  # Rare mode
    weights = weights / weights.sum()

    # Assign samples to components
    labels = rng.choice(n_components, size=n_samples, p=weights)
    samples = np.zeros((n_samples, dim))

    # Generate components with different correlation structures
    for k in range(n_components):
        mask = labels == k
        n_k = mask.sum()
        if n_k == 0:
            continue

        # Create random mean
        mean = rng.randn(dim) * 3

        # Create covariance with different structures
        if k == 0:
            # Isotropic
            cov = np.eye(dim)
        elif k < n_components - 1:
            # Random rotation with varying condition numbers
            A = rng.randn(dim, dim)
            U, _, Vt = np.linalg.svd(A)
            # Eigenvalues with controlled condition number
            eigs = np.exp(np.linspace(0, -k, dim))  # Increasing ill-conditioning
            cov = U @ np.diag(eigs) @ U.T
        else:
            # Tail mode: highly anisotropic
            eigs = np.exp(np.linspace(0, -5, dim))
            A = rng.randn(dim, dim)
            U, _, _ = np.linalg.svd(A)
            cov = U @ np.diag(eigs) @ U.T

        # Sample from this component
        samples[mask] = rng.multivariate_normal(mean, cov, size=n_k)

    return samples.astype(np.float32), labels


def generate_swissroll_embedded(
    n_samples: int,
    embed_dim: int = 50,
    noise_scale: float = 0.1,
    seed: int = 0
) -> np.ndarray:
    """
    Generate Swiss roll in 3D and embed to higher dimension with anisotropic noise.

    Args:
        n_samples: Number of samples
        embed_dim: Target embedding dimension
        noise_scale: Scale of additive noise
        seed: Random seed

    Returns:
        samples: (n_samples, embed_dim) array
    """
    rng = np.random.RandomState(seed)

    # Generate 3D Swiss roll
    X_3d, _ = make_swiss_roll(n_samples, noise=0.05, random_state=seed)

    # Create random embedding matrix (3 -> embed_dim)
    embed_matrix = rng.randn(3, embed_dim)
    X_embedded = X_3d @ embed_matrix

    # Add anisotropic noise
    noise_scales = np.exp(np.linspace(0, -3, embed_dim))  # Decreasing noise
    noise = rng.randn(n_samples, embed_dim) * noise_scales * noise_scale

    samples = X_embedded + noise
    return samples.astype(np.float32)


class TeacherNetwork(nn.Module):
    """Teacher generator network for controlled experiments"""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: list,
        curvature: str = 'smooth',
        seed: int = 0
    ):
        super().__init__()
        torch.manual_seed(seed)

        layers = []
        prev_dim = latent_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if curvature == 'smooth':
                layers.append(nn.Tanh())
            else:  # 'sharp'
                layers.append(nn.ReLU())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # Initialize with small weights for smooth, large for sharp
        scale = 0.1 if curvature == 'smooth' else 1.0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=scale)
                nn.init.zeros_(module.bias)

    def forward(self, z):
        return self.network(z)


def generate_teacher_network_data(
    n_samples: int,
    latent_dim: int,
    output_dim: int,
    curvature: str = 'smooth',
    noise_scale: float = 0.1,
    mixture_mode: bool = False,
    rare_teacher_weight: float = 0.02,
    seed: int = 0,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[np.ndarray, np.ndarray, TeacherNetwork]:
    """
    Generate data from teacher network(s).

    Args:
        n_samples: Number of samples
        latent_dim: Latent dimension
        output_dim: Output dimension
        curvature: 'smooth' or 'sharp' for teacher network
        noise_scale: Additive noise scale
        mixture_mode: If True, use two teachers (rare + common)
        rare_teacher_weight: Weight for rare teacher in mixture mode
        seed: Random seed
        device: Device for computation

    Returns:
        samples: (n_samples, output_dim) array
        latents: (n_samples, latent_dim) ground truth latents
        teacher: The primary teacher network
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # Create primary teacher
    teacher = TeacherNetwork(
        latent_dim, output_dim,
        hidden_dims=[128, 64],
        curvature=curvature,
        seed=seed
    ).to(device)
    teacher.eval()

    if not mixture_mode:
        # Simple single teacher
        z = torch.randn(n_samples, latent_dim, device=device)
        with torch.no_grad():
            x = teacher(z)
        noise = torch.randn_like(x) * noise_scale
        samples = (x + noise).cpu().numpy()
        latents = z.cpu().numpy()
    else:
        # Mixture of two teachers
        teacher_rare = TeacherNetwork(
            latent_dim, output_dim,
            hidden_dims=[128, 64],
            curvature='sharp',  # Rare teacher is always sharp
            seed=seed + 1
        ).to(device)
        teacher_rare.eval()

        # Assign samples
        n_rare = int(n_samples * rare_teacher_weight)
        n_common = n_samples - n_rare

        z_common = torch.randn(n_common, latent_dim, device=device)
        z_rare = torch.randn(n_rare, latent_dim, device=device)

        with torch.no_grad():
            x_common = teacher(z_common)
            x_rare = teacher_rare(z_rare)

        # Combine
        x = torch.cat([x_common, x_rare], dim=0)
        z = torch.cat([z_common, z_rare], dim=0)

        # Add noise
        noise = torch.randn_like(x) * noise_scale
        samples = (x + noise).cpu().numpy()
        latents = z.cpu().numpy()

        # Shuffle
        perm = rng.permutation(n_samples)
        samples = samples[perm]
        latents = latents[perm]

    return samples.astype(np.float32), latents.astype(np.float32), teacher


def get_dataset(
    name: str,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """
    Get dataset by name.

    Args:
        name: Dataset name ('mog2d', 'mog20d', 'swissroll', 'teacher_smooth', 'teacher_sharp', etc.)
        n_train: Number of training samples
        n_val: Number of validation samples
        n_test: Number of test samples
        seed: Random seed
        **kwargs: Additional dataset-specific parameters

    Returns:
        Dictionary with 'train', 'val', 'test' arrays and metadata
    """
    total = n_train + n_val + n_test

    if name == 'mog2d':
        X_all, labels = generate_mixture_of_gaussians(
            total, dim=2, seed=seed, **kwargs
        )
        metadata = {'dim': 2, 'labels': labels}

    elif name == 'mog20d':
        X_all, labels = generate_mixture_of_gaussians(
            total, dim=20, seed=seed, **kwargs
        )
        metadata = {'dim': 20, 'labels': labels}

    elif name == 'swissroll':
        X_all = generate_swissroll_embedded(
            total, seed=seed, **kwargs
        )
        metadata = {'dim': X_all.shape[1]}

    elif name.startswith('teacher'):
        curvature = 'smooth' if 'smooth' in name else 'sharp'
        mixture = 'mixture' in name

        latent_dim = kwargs.get('latent_dim', 8)
        output_dim = kwargs.get('output_dim', 20)

        X_all, Z_all, teacher = generate_teacher_network_data(
            total,
            latent_dim=latent_dim,
            output_dim=output_dim,
            curvature=curvature,
            mixture_mode=mixture,
            seed=seed,
            **kwargs
        )
        metadata = {
            'dim': output_dim,
            'latent_dim': latent_dim,
            'latents': Z_all,
            'teacher': teacher
        }
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Split
    X_train = X_all[:n_train]
    X_val = X_all[n_train:n_train + n_val]
    X_test = X_all[n_train + n_val:]

    return {
        'train': X_train,
        'val': X_val,
        'test': X_test,
        'metadata': metadata
    }
