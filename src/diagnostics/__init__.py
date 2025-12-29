"""Geometric diagnostics for autoencoders"""
from .jacobian_utils import (
    compute_jvp_encoder,
    compute_jvp_decoder,
    compute_jvp_composed,
    sample_orthonormal_vectors
)
from .geometric_metrics import (
    compute_log_volume,
    compute_k_volume,
    compute_edc_k,
    compute_decoder_stability,
    compute_generative_gap_index
)

__all__ = [
    'compute_jvp_encoder',
    'compute_jvp_decoder',
    'compute_jvp_composed',
    'sample_orthonormal_vectors',
    'compute_log_volume',
    'compute_k_volume',
    'compute_edc_k',
    'compute_decoder_stability',
    'compute_generative_gap_index'
]
