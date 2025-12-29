"""
Analyze experimental results and extract key numbers for the paper.

Usage:
    python analyze_results.py results/e1_ae_trap/seed_0
    python analyze_results.py results/e2_tail_stress --compare
"""
import sys
sys.path.insert(0, 'src')

import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def load_metrics(metrics_path):
    """Load metrics from JSONL file"""
    records = []
    with open(metrics_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def analyze_single_run(result_dir):
    """Analyze a single experiment run"""
    result_dir = Path(result_dir)
    metrics_path = result_dir / 'metrics.jsonl'

    if not metrics_path.exists():
        print(f"No metrics file found at {metrics_path}")
        return None

    df = load_metrics(metrics_path)
    config_path = result_dir / 'config.json'

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"\n{'='*80}")
    print(f"Analysis: {result_dir}")
    print(f"{'='*80}\n")

    # Basic info
    print(f"Model type: {config.get('model_type', 'unknown')}")
    print(f"Total epochs: {len(df)}")
    print(f"Latent dim: {config.get('latent_dim', 'N/A')}")

    # Training metrics
    print(f"\n## Training Metrics")
    print(f"{'Metric':<30} {'Initial':>12} {'Final':>12} {'Improvement':>12}")
    print("-" * 70)

    metrics_to_track = [
        ('loss', 'Loss'),
        ('val_recon_loss', 'Val Recon MSE'),
    ]

    if 'kl_div' in df.columns:
        metrics_to_track.append(('kl_div', 'KL Divergence'))

    for col, name in metrics_to_track:
        if col in df.columns:
            initial = df[col].iloc[0]
            final = df[col].iloc[-1]
            improvement = ((initial - final) / initial * 100) if initial != 0 else 0
            print(f"{name:<30} {initial:>12.4f} {final:>12.4f} {improvement:>11.1f}%")

    # Generation metrics (from last diagnostic epoch)
    diag_df = df[df.apply(lambda row: any(k.startswith('gen_') for k in row.keys()), axis=1)]

    if not diag_df.empty:
        print(f"\n## Generation Quality Metrics (Final)")
        print(f"{'Metric':<40} {'Value':>15}")
        print("-" * 60)

        final_diag = diag_df.iloc[-1]

        gen_metrics = [
            ('gen_mmd', 'MMD'),
            ('gen_energy_distance', 'Energy Distance'),
            ('gen_knn_precision', 'k-NN Precision'),
            ('gen_knn_recall', 'k-NN Recall'),
            ('gen_knn_f1', 'k-NN F1'),
            ('gen_mode_coverage', 'Mode Coverage'),
            ('gen_rare_mode_recall', 'Rare Mode Recall'),
        ]

        for col, name in gen_metrics:
            if col in final_diag:
                value = final_diag[col]
                print(f"{name:<40} {value:>15.4f}")

    # Geometric diagnostics
    if not diag_df.empty:
        print(f"\n## Geometric Diagnostics (Final)")
        print(f"{'Metric':<40} {'Value':>15}")
        print("-" * 60)

        final_diag = diag_df.iloc[-1]

        # k-volumes
        for k in [1, 2, 4, 8]:
            col_random = f'diag_eps1e-06_k{k}_random_mean'
            col_pca = f'diag_eps1e-06_k{k}_pca_mean'

            if col_random in final_diag:
                print(f"k={k} volume (random directions) {final_diag[col_random]:>15.4f}")
            if col_pca in final_diag:
                print(f"k={k} volume (PCA directions) {final_diag[col_pca]:>15.4f}")

        # Gap metrics
        if 'diag_gap_overall' in final_diag:
            print(f"\n{'Gap Score (Overall)':<40} {final_diag['diag_gap_overall']:>15.4f}")

        for col in final_diag.keys():
            if col.startswith('diag_gap_kvol'):
                k_str = col.replace('diag_gap_kvol_', '')
                print(f"Gap Score ({k_str}){'':<27} {final_diag[col]:>15.4f}")

        # Decoder stability
        print(f"\n## Decoder Stability (Final)")
        for r in [0.5, 1.0, 2.0, 4.0]:
            col = f'diag_decoder_r{r}_log_vol_k_mean'
            if col in final_diag:
                print(f"Radius={r:<4} log-volume {final_diag[col]:>15.4f}")

    # Evolution analysis
    if not diag_df.empty and len(diag_df) > 2:
        print(f"\n## Temporal Evolution")

        # When does k-volume collapse?
        k4_col = 'diag_eps1e-06_k4_pca_mean'
        if k4_col in diag_df.columns:
            k4_values = diag_df[k4_col].values
            k4_initial = k4_values[0]
            k4_threshold = k4_initial - 3  # 3 units drop

            collapse_epochs = diag_df[diag_df[k4_col] < k4_threshold]['epoch'].values
            if len(collapse_epochs) > 0:
                print(f"k=4 volume collapse detected at epoch {collapse_epochs[0]}")

        # When does gap widen?
        if 'diag_gap_overall' in diag_df.columns:
            gap_values = diag_df['diag_gap_overall'].values
            if gap_values[0] < gap_values[-1]:
                print(f"Generative gap widened from {gap_values[0]:.3f} to {gap_values[-1]:.3f}")

    return {
        'result_dir': str(result_dir),
        'config': config,
        'final_metrics': final_diag.to_dict() if not diag_df.empty else {}
    }


def compare_multiple_runs(base_dir):
    """Compare multiple runs (e.g., different models in E2)"""
    base_dir = Path(base_dir)

    # Find all metrics files
    metrics_files = list(base_dir.rglob('metrics.jsonl'))

    if not metrics_files:
        print(f"No metrics files found in {base_dir}")
        return

    print(f"\n{'='*80}")
    print(f"Comparison: {base_dir}")
    print(f"{'='*80}\n")

    results = []

    for metrics_path in metrics_files:
        df = load_metrics(metrics_path)
        run_name = metrics_path.parent.name
        model_name = metrics_path.parent.parent.name

        # Get final generation metrics
        diag_df = df[df.apply(lambda row: any(k.startswith('gen_') for k in row.keys()), axis=1)]

        if not diag_df.empty:
            final = diag_df.iloc[-1]

            result = {
                'Model': f"{model_name}/{run_name}",
                'Recon MSE': df['val_recon_loss'].iloc[-1],
            }

            if 'gen_energy_distance' in final:
                result['Energy Distance'] = final['gen_energy_distance']
            if 'gen_mmd' in final:
                result['MMD'] = final['gen_mmd']
            if 'gen_rare_mode_recall' in final:
                result['Rare Mode Recall'] = final['gen_rare_mode_recall']
            if 'diag_gap_overall' in final:
                result['Gap Score'] = final['diag_gap_overall']

            results.append(result)

    if results:
        comparison_df = pd.DataFrame(results)
        print("\n## Model Comparison")
        print(comparison_df.to_string(index=False))

        # Statistical summary
        print(f"\n## Statistical Summary")
        print(comparison_df.describe().to_string())


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('result_dir', type=str, help='Path to results directory')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple runs in the directory')

    args = parser.parse_args()

    if args.compare:
        compare_multiple_runs(args.result_dir)
    else:
        analyze_single_run(args.result_dir)


if __name__ == '__main__':
    main()
