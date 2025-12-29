"""
Aggregate multi-seed MNIST experimental results for ICML paper.

Computes mean ± std across seeds for:
- Rare mode metrics (recall, lift, count)
- Sample diversity (variance)
- Reconstruction quality
"""
import sys
sys.path.insert(0, '/home/asudjianto/jupyterlab/ga_AE')

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


def load_seed_results(results_dir, model_type, seeds):
    """Load results from all seeds for a model."""
    all_results = []

    for seed in seeds:
        seed_path = Path(results_dir) / model_type / f'seed_{seed}' / 'final_metrics.json'

        if not seed_path.exists():
            print(f"WARNING: Missing results for {model_type} seed {seed}")
            print(f"  Expected: {seed_path}")
            continue

        with open(seed_path, 'r') as f:
            results = json.load(f)

        all_results.append({
            'seed': seed,
            'test_loss': results['test']['loss'],
            'recon_loss': results['test']['recon_loss'],
            'gen_rare_count': results['rare']['gen_rare_count'],
            'rare_recall': results['rare']['rare_recall'],
            'rare_lift': results['rare']['rare_lift'],
        })

    return all_results


def compute_statistics(results, metric):
    """Compute mean and std for a metric across seeds."""
    values = [r[metric] for r in results]
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'values': values
    }


def print_results_table(vae_results, gaae_results):
    """Print formatted results table."""
    print("\n" + "="*80)
    print("MNIST Multi-Seed Results (Mean ± Std)")
    print("="*80)
    print()

    # Header
    print(f"{'Metric':<30} {'VAE':>20} {'GA-AE':>20}")
    print("-"*80)

    # Metrics to report
    metrics = [
        ('Test Reconstruction Loss', 'recon_loss'),
        ('Generated Rare Count', 'gen_rare_count'),
        ('Rare Recall@2000', 'rare_recall'),
        ('Rare Mode Lift (RML)', 'rare_lift'),
    ]

    for label, key in metrics:
        vae_stat = compute_statistics(vae_results, key)
        gaae_stat = compute_statistics(gaae_results, key)

        if key == 'gen_rare_count':
            # Integer counts
            vae_str = f"{vae_stat['mean']:.0f} ± {vae_stat['std']:.1f}"
            gaae_str = f"{gaae_stat['mean']:.0f} ± {gaae_stat['std']:.1f}"
        elif key == 'rare_recall':
            # Percentages
            vae_str = f"{vae_stat['mean']*100:.1f}% ± {vae_stat['std']*100:.1f}%"
            gaae_str = f"{gaae_stat['mean']*100:.1f}% ± {gaae_stat['std']*100:.1f}%"
        elif key == 'rare_lift':
            # Lift ratios
            vae_str = f"{vae_stat['mean']:.2f}× ± {vae_stat['std']:.2f}×"
            gaae_str = f"{gaae_stat['mean']:.2f}× ± {gaae_stat['std']:.2f}×"
        else:
            # Regular floats
            vae_str = f"{vae_stat['mean']:.4f} ± {vae_stat['std']:.4f}"
            gaae_str = f"{gaae_stat['mean']:.4f} ± {gaae_stat['std']:.4f}"

        print(f"{label:<30} {vae_str:>20} {gaae_str:>20}")

    print("-"*80)
    print(f"Number of seeds: {len(vae_results)} (VAE), {len(gaae_results)} (GA-AE)")
    print("="*80)
    print()


def create_comparison_plot(vae_results, gaae_results, output_path):
    """Create comparison visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Metrics to plot
    plot_configs = [
        ('rare_lift', 'Rare Mode Lift (RML)', 'Target: 1.0×'),
        ('rare_recall', 'Rare Recall@2000', None),
        ('recon_loss', 'Test Reconstruction Loss', None),
    ]

    for ax, (metric, title, hline_label) in zip(axes, plot_configs):
        vae_stat = compute_statistics(vae_results, metric)
        gaae_stat = compute_statistics(gaae_results, metric)

        # Bar plot with error bars
        x = np.arange(2)
        means = [vae_stat['mean'], gaae_stat['mean']]
        stds = [vae_stat['std'], gaae_stat['std']]
        labels = ['VAE', 'GA-AE']
        colors = ['#E74C3C', '#3498DB']

        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.7,
                     capsize=5, edgecolor='black', linewidth=1.5)

        # Add individual seed points
        for i, (stat, label) in enumerate([(vae_stat, 'VAE'), (gaae_stat, 'GA-AE')]):
            x_jitter = x[i] + np.random.normal(0, 0.05, len(stat['values']))
            ax.scatter(x_jitter, stat['values'], color='black', s=30,
                      alpha=0.6, zorder=10)

        # Add target line for rare lift
        if metric == 'rare_lift' and hline_label:
            ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2,
                      alpha=0.7, label=hline_label)
            ax.legend(fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Format y-axis based on metric
        if metric == 'rare_recall':
            ax.set_ylabel('Recall (%)')
            ax.set_ylim(0, max(max(vae_stat['values']), max(gaae_stat['values'])) * 1.2)
            # Convert to percentage
            vals = ax.get_yticks()
            ax.set_yticklabels([f'{v*100:.0f}%' for v in vals])
        elif metric == 'rare_lift':
            ax.set_ylabel('Lift Ratio')
            # Set appropriate y-limit for lift
            max_val = max(max(vae_stat['values']), max(gaae_stat['values']))
            ax.set_ylim(0, max_val * 1.2)
        else:
            ax.set_ylabel('MSE Loss')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot: {output_path}")
    plt.close()


def save_latex_table(vae_results, gaae_results, output_path):
    """Save LaTeX table for paper."""
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{MNIST Multi-Seed Results (Mean $\\pm$ Std)}")
    lines.append("\\label{tab:mnist_multiseed}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Metric & VAE & GA-AE \\\\")
    lines.append("\\midrule")

    # Add metrics
    metrics = [
        ('Test Recon Loss', 'recon_loss', '.4f'),
        ('Gen Rare Count', 'gen_rare_count', '.1f'),
        ('Rare Recall@2000 (\\%)', 'rare_recall', '.1f', 100),  # multiply by 100
        ('Rare Mode Lift', 'rare_lift', '.2f'),
    ]

    for label, key, fmt, *scale in metrics:
        vae_stat = compute_statistics(vae_results, key)
        gaae_stat = compute_statistics(gaae_results, key)

        multiplier = scale[0] if scale else 1

        vae_mean = vae_stat['mean'] * multiplier
        vae_std = vae_stat['std'] * multiplier
        gaae_mean = gaae_stat['mean'] * multiplier
        gaae_std = gaae_stat['std'] * multiplier

        line = f"{label} & ${vae_mean:{fmt}} \\pm {vae_std:{fmt}}$ & "
        line += f"${gaae_mean:{fmt}} \\pm {gaae_std:{fmt}}$ \\\\"
        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved LaTeX table: {output_path}")


def main():
    """Main aggregation function."""
    results_dir = Path('results/mnist_multiseed')
    seeds = [0, 1, 2, 3, 4]

    print("\n" + "="*80)
    print("Aggregating MNIST Multi-Seed Results")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Seeds: {seeds}")
    print()

    # Load results
    print("Loading VAE results...")
    vae_results = load_seed_results(results_dir, 'vae', seeds)
    print(f"  Loaded {len(vae_results)} seeds")

    print("Loading GA-AE results...")
    gaae_results = load_seed_results(results_dir, 'ga-ae', seeds)
    print(f"  Loaded {len(gaae_results)} seeds")

    if not vae_results or not gaae_results:
        print("\nERROR: Missing results. Please run experiments first:")
        print("  bash run_mnist_multiseed.sh")
        return 1

    # Print results table
    print_results_table(vae_results, gaae_results)

    # Create comparison plot
    output_plot = results_dir / 'comparison_multiseed.png'
    create_comparison_plot(vae_results, gaae_results, output_plot)

    # Save LaTeX table
    output_latex = results_dir / 'results_table.tex'
    save_latex_table(vae_results, gaae_results, output_latex)

    # Save aggregated JSON
    aggregated = {
        'vae': {
            'n_seeds': len(vae_results),
            'rare_lift': compute_statistics(vae_results, 'rare_lift'),
            'rare_recall': compute_statistics(vae_results, 'rare_recall'),
            'gen_rare_count': compute_statistics(vae_results, 'gen_rare_count'),
            'recon_loss': compute_statistics(vae_results, 'recon_loss'),
        },
        'ga-ae': {
            'n_seeds': len(gaae_results),
            'rare_lift': compute_statistics(gaae_results, 'rare_lift'),
            'rare_recall': compute_statistics(gaae_results, 'rare_recall'),
            'gen_rare_count': compute_statistics(gaae_results, 'gen_rare_count'),
            'recon_loss': compute_statistics(gaae_results, 'recon_loss'),
        }
    }

    output_json = results_dir / 'aggregated_results.json'
    with open(output_json, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"Saved aggregated results: {output_json}")

    print("\n" + "="*80)
    print("Aggregation Complete!")
    print("="*80)
    print("\nFiles generated:")
    print(f"  - {output_plot}")
    print(f"  - {output_latex}")
    print(f"  - {output_json}")
    print("\nUse these results to update the paper!")
    print("="*80)

    return 0


if __name__ == '__main__':
    exit(main())
