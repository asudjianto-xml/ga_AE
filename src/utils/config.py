"""Configuration dataclasses for experiments"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml


@dataclass
class DatasetConfig:
    name: str  # 'mog2d', 'mog20d', 'swissroll', 'teacher', 'tabular'
    n_train: int = 10000
    n_val: int = 2000
    n_test: int = 2000
    seed: int = 0
    # Dataset-specific params
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    model_type: str  # 'ae', 'vae', 'ga_ae', 'ga_vae', 'cae', 'spectral_ae', 'sobolev_ae'
    input_dim: int
    latent_dim: int
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    activation: str = 'relu'
    # VAE-specific
    beta: float = 1.0
    use_kl_annealing: bool = False
    kl_anneal_epochs: int = 50
    # Geometry-regularized params
    lambda_k_volume: float = 0.0
    lambda_edc: float = 0.0
    lambda_dist: float = 0.0
    k_values: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    volume_floor_tau: float = -10.0
    # CAE params
    lambda_cae: float = 0.0
    # Sobolev params
    lambda_sobolev: float = 0.0
    # WAE-like params
    use_mmd_posterior: bool = False
    lambda_mmd: float = 0.0


@dataclass
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 256
    lr: float = 3e-4
    optimizer: str = 'adam'
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    checkpoint_every: int = 5
    device: str = 'cuda'


@dataclass
class DiagnosticConfig:
    compute_every: int = 5
    eps_values: List[float] = field(default_factory=lambda: [1e-8, 1e-6, 1e-4])
    k_values: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    use_pca_directions: bool = True
    radius_stress_test: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0, 8.0])
    n_samples_diagnostic: int = 1000


@dataclass
class ExperimentConfig:
    name: str
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    diagnostics: DiagnosticConfig
    output_dir: str = 'results'

    def save(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
