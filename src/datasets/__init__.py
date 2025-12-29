"""Dataset generators"""
from .synthetic import (
    generate_mixture_of_gaussians,
    generate_swissroll_embedded,
    generate_teacher_network_data,
    get_dataset
)

__all__ = [
    'generate_mixture_of_gaussians',
    'generate_swissroll_embedded',
    'generate_teacher_network_data',
    'get_dataset'
]
