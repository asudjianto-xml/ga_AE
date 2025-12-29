"""Plotting utilities for experiment results"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd

sns.set_style('whitegrid')
sns.set_palette('husl')


def load_metrics(metrics_path: str) -> pd.DataFrame:
    """Load metrics from JSONL file"""
    records = []
    with open(metrics_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def plot_training_curves(
    df: pd.DataFrame,
    output_path: str,
    metrics_to_plot: List[str] = None
):
    """Plot training curves"""
    if metrics_to_plot is None:
        metrics_to_plot = ['loss', 'val_recon_loss']

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 4))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_to_plot):
        if metric in df.columns:
            ax.plot(df['epoch'], df[metric], linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric}')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_geometric_diagnostics(
    df: pd.DataFrame,
    output_path: str,
    k_values: List[int] = [1, 2, 4, 8]
):
    """Plot geometric diagnostic evolution"""
    n_k = len(k_values)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # 1. k-volume evolution (mean)
    ax = axes[0]
    for k in k_values:
        col_name = f'diag_eps1e-06_k{k}_pca_mean'
        if col_name in df.columns:
            ax.plot(df['epoch'], df[col_name], label=f'k={k}', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Log k-volume (mean)')
    ax.set_title('Encoder k-volume Evolution (PCA directions)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. EDC evolution
    ax = axes[1]
    for k in k_values:
        col_name = f'diag_edc_k{k}_pca_mean'
        if col_name in df.columns:
            ax.plot(df['epoch'], df[col_name], label=f'k={k}', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('EDC (mean)')
    ax.set_title('Encoder-Decoder Consistency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Generative gap
    ax = axes[2]
    gap_cols = [c for c in df.columns if c.startswith('diag_gap_kvol')]
    for col in gap_cols:
        ax.plot(df['epoch'], df[col], label=col.replace('diag_gap_kvol_', ''), linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gap (prior - post)')
    ax.set_title('Generative Gap Index')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Overall gap score
    ax = axes[3]
    if 'diag_gap_overall' in df.columns:
        ax.plot(df['epoch'], df['diag_gap_overall'], linewidth=2, color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Overall Gap Score')
    ax.set_title('Overall Generative Gap')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_generation_quality(
    df: pd.DataFrame,
    output_path: str
):
    """Plot generation quality metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    metrics = [
        ('gen_mmd', 'MMD'),
        ('gen_energy_distance', 'Energy Distance'),
        ('gen_knn_f1', 'k-NN F1 Score'),
        ('gen_rare_mode_recall', 'Rare Mode Recall')
    ]

    for ax, (col, title) in zip(axes, metrics):
        if col in df.columns:
            ax.plot(df['epoch'], df[col], linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_recon_vs_generation_gap(
    df: pd.DataFrame,
    output_path: str
):
    """Plot the AE trap: reconstruction vs generation gap"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Reconstruction error decreasing
    ax = axes[0]
    if 'val_recon_loss' in df.columns:
        ax.plot(df['epoch'], df['val_recon_loss'], linewidth=2, color='blue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reconstruction MSE')
        ax.set_title('Reconstruction Performance (On-Manifold)')
        ax.grid(True, alpha=0.3)

    # Right: Generation quality worsening or plateauing
    ax = axes[1]
    if 'gen_energy_distance' in df.columns:
        ax.plot(df['epoch'], df['gen_energy_distance'], linewidth=2, color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Energy Distance')
        ax.set_title('Generation Quality (Off-Manifold)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_decoder_stability(
    df: pd.DataFrame,
    output_path: str,
    radii: List[float] = [0.5, 1.0, 2.0, 4.0]
):
    """Plot decoder stability under radius stress test"""
    fig, ax = plt.subplots(figsize=(8, 6))

    for radius in radii:
        col_name = f'diag_decoder_r{radius}_log_vol_k_mean'
        if col_name in df.columns:
            ax.plot(df['epoch'], df[col_name], label=f'r={radius}', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Decoder Log Volume (mean)')
    ax.set_title('Decoder Stability Under Radius Stress')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_vae_collapse(
    df: pd.DataFrame,
    output_path: str
):
    """Plot VAE posterior collapse indicators"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # 1. KL divergence
    ax = axes[0]
    if 'kl_div' in df.columns:
        ax.plot(df['epoch'], df['kl_div'], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('KL Divergence')
        ax.set_title('KL Divergence Over Training')
        ax.grid(True, alpha=0.3)

    # 2. Mean encoder k-volume
    ax = axes[1]
    k_values = [1, 2, 4, 8]
    for k in k_values:
        col_name = f'diag_eps1e-06_k{k}_pca_mean'
        if col_name in df.columns:
            ax.plot(df['epoch'], df[col_name], label=f'k={k}', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Log k-volume (mean)')
    ax.set_title('Mean Encoder k-volume')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Generation quality
    ax = axes[2]
    if 'gen_energy_distance' in df.columns:
        ax.plot(df['epoch'], df['gen_energy_distance'], linewidth=2, color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Energy Distance')
        ax.set_title('Generation Quality')
        ax.grid(True, alpha=0.3)

    # 4. Beta/KL weight
    ax = axes[3]
    if 'kl_weight' in df.columns:
        ax.plot(df['epoch'], df['kl_weight'], linewidth=2, color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('KL Weight (β)')
        ax.set_title('KL Annealing Schedule')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison_scatter(
    results_dirs: List[str],
    labels: List[str],
    output_path: str
):
    """Plot scatter of gap score vs generation quality for multiple models"""
    fig, ax = plt.subplots(figsize=(8, 6))

    for results_dir, label in zip(results_dirs, labels):
        metrics_path = Path(results_dir) / 'metrics.jsonl'
        if not metrics_path.exists():
            continue

        df = load_metrics(str(metrics_path))

        # Get final values
        if 'diag_gap_overall' in df.columns and 'gen_energy_distance' in df.columns:
            gap = df['diag_gap_overall'].iloc[-1]
            ed = df['gen_energy_distance'].iloc[-1]

            ax.scatter(gap, ed, s=100, label=label, alpha=0.7)

    ax.set_xlabel('Generative Gap Index')
    ax.set_ylabel('Energy Distance (Generation Quality)')
    ax.set_title('Gap Score vs Generation Quality')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_all_plots(output_dir: str, experiment: str):
    """Create all relevant plots for an experiment"""
    output_dir = Path(output_dir)

    # Find all metrics files
    metrics_files = list(output_dir.rglob('metrics.jsonl'))

    if not metrics_files:
        print(f"No metrics files found in {output_dir}")
        return

    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Plot each run
    for metrics_file in metrics_files:
        df = load_metrics(str(metrics_file))
        run_name = metrics_file.parent.name

        print(f"Creating plots for {run_name}...")

        # Basic training curves
        plot_training_curves(
            df, str(plots_dir / f'{run_name}_training.png'),
            metrics_to_plot=['loss', 'val_recon_loss']
        )

        # Geometric diagnostics
        plot_geometric_diagnostics(
            df, str(plots_dir / f'{run_name}_diagnostics.png')
        )

        # Generation quality
        plot_generation_quality(
            df, str(plots_dir / f'{run_name}_generation.png')
        )

        # Experiment-specific plots
        if experiment in ['e1', 'e2']:
            plot_recon_vs_generation_gap(
                df, str(plots_dir / f'{run_name}_ae_trap.png')
            )
            plot_decoder_stability(
                df, str(plots_dir / f'{run_name}_decoder_stability.png')
            )

        if experiment in ['e3', 'e4']:
            plot_vae_collapse(
                df, str(plots_dir / f'{run_name}_vae_collapse.png')
            )

    # Create comparison plots if multiple runs
    if len(metrics_files) > 1 and experiment in ['e2', 'e5']:
        results_dirs = [str(mf.parent) for mf in metrics_files]
        labels = [mf.parent.name for mf in metrics_files]

        plot_comparison_scatter(
            results_dirs, labels,
            str(plots_dir / 'comparison_scatter.png')
        )

    print(f"Plots saved to {plots_dir}")
