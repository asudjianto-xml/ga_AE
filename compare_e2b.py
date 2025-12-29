"""Compare E2b ablation results"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Variants
variants = {
    'variant2_ed': {'label': 'GA-AE + ED', 'color': '#1f77b4'},
    'variant6_ed_repel': {'label': 'GA-AE + ED+Repul', 'color': '#ff7f0e'},
    'variant9_all': {'label': 'GA-AE + All', 'color': '#2ca02c'}
}

# Also load original E2 results for comparison
e2_models = {
    'ae': {'label': 'AE (Baseline)', 'color': 'gray', 'linestyle': '--'},
    'cae': {'label': 'CAE (Winner)', 'color': 'purple', 'linestyle': '--'},
    'ga_ae': {'label': 'GA-AE (Original)', 'color': 'red', 'linestyle': '--'}
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Collect data
all_data = {}

# Load E2b ablation results
for variant_key, info in variants.items():
    data = {'epochs': [], 'recon': [], 'energy_dist': [], 'rare_recall': [],
            'ed_loss': [], 'repulsion': [], 'gap': [], 'dec_vol': []}

    metrics_file = f'results/e2b_ablation/{variant_key}/seed_0/metrics.jsonl'
    try:
        with open(metrics_file, 'r') as f:
            for line in f:
                record = json.loads(line)
                data['epochs'].append(record['epoch'])
                data['recon'].append(record.get('recon_loss', 0))
                data['ed_loss'].append(record.get('ed_loss', 0))
                data['repulsion'].append(record.get('repulsion_loss', 0))
                data['gap'].append(record.get('gap_loss', 0))
                data['dec_vol'].append(record.get('dec_vol_loss', 0))

                if 'gen_energy_distance' in record:
                    data['energy_dist'].append(record['gen_energy_distance'])
                    data['rare_recall'].append(record.get('gen_rare_mode_recall', 0))
                else:
                    data['energy_dist'].append(np.nan)
                    data['rare_recall'].append(np.nan)

        all_data[variant_key] = data
        print(f"Loaded {variant_key}: {len(data['epochs'])} epochs")
    except Exception as e:
        print(f"Error loading {variant_key}: {e}")

# Load E2 baseline results for comparison
for model_key, info in e2_models.items():
    data = {'epochs': [], 'recon': [], 'energy_dist': [], 'rare_recall': []}

    metrics_file = f'results/e2_tail_stress/{model_key}/seed_0/metrics.jsonl'
    try:
        with open(metrics_file, 'r') as f:
            for line in f:
                record = json.loads(line)
                data['epochs'].append(record['epoch'])
                data['recon'].append(record.get('recon_loss', 0))

                if 'gen_energy_distance' in record:
                    data['energy_dist'].append(record['gen_energy_distance'])
                    data['rare_recall'].append(record.get('gen_rare_mode_recall', 0))
                else:
                    data['energy_dist'].append(np.nan)
                    data['rare_recall'].append(np.nan)

        all_data[model_key] = data
    except Exception as e:
        print(f"Warning: Could not load {model_key}: {e}")

# Plot 1: Reconstruction Loss
for variant_key, data in all_data.items():
    if variant_key in variants:
        axes[0, 0].plot(data['epochs'], data['recon'],
                       label=variants[variant_key]['label'],
                       linewidth=2, color=variants[variant_key]['color'], alpha=0.9)
    elif variant_key in e2_models:
        axes[0, 0].plot(data['epochs'], data['recon'],
                       label=e2_models[variant_key]['label'],
                       linewidth=1.5, color=e2_models[variant_key]['color'],
                       linestyle=e2_models[variant_key]['linestyle'], alpha=0.6)

axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Reconstruction MSE')
axes[0, 0].set_title('Reconstruction Loss')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# Plot 2: Energy Distance (generation quality)
for variant_key, data in all_data.items():
    ed = [x for x in data['energy_dist'] if not np.isnan(x)]
    ep = [data['epochs'][i] for i, x in enumerate(data['energy_dist']) if not np.isnan(x)]

    if ed and variant_key in variants:
        axes[0, 1].plot(ep, ed, label=variants[variant_key]['label'],
                       linewidth=2, color=variants[variant_key]['color'],
                       marker='o', markersize=5, alpha=0.9)
    elif ed and variant_key in e2_models:
        axes[0, 1].plot(ep, ed, label=e2_models[variant_key]['label'],
                       linewidth=1.5, color=e2_models[variant_key]['color'],
                       linestyle=e2_models[variant_key]['linestyle'],
                       marker='s', markersize=4, alpha=0.6)

axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Energy Distance')
axes[0, 1].set_title('Generation Quality (Lower = Better)')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Rare Mode Recall *** KEY METRIC ***
for variant_key, data in all_data.items():
    rr = [x * 100 for x in data['rare_recall'] if not np.isnan(x)]
    ep = [data['epochs'][i] for i, x in enumerate(data['rare_recall']) if not np.isnan(x)]

    if rr and variant_key in variants:
        axes[0, 2].plot(ep, rr, label=variants[variant_key]['label'],
                       linewidth=2, color=variants[variant_key]['color'],
                       marker='o', markersize=5, alpha=0.9)
    elif rr and variant_key in e2_models:
        axes[0, 2].plot(ep, rr, label=e2_models[variant_key]['label'],
                       linewidth=1.5, color=e2_models[variant_key]['color'],
                       linestyle=e2_models[variant_key]['linestyle'],
                       marker='s', markersize=4, alpha=0.6)

axes[0, 2].axhline(y=22.7, color='purple', linestyle=':', linewidth=2, label='CAE Target (22.7%)')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Rare Mode Recall (%)')
axes[0, 2].set_title('Rare Mode Coverage (Higher = Better)')
axes[0, 2].legend(fontsize=8)
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: ED Loss Term (for variants with ED)
for variant_key, data in all_data.items():
    if variant_key in variants and variant_key in all_data:
        ed_vals = [x for x in data['ed_loss'] if x > 0]
        ep_vals = [data['epochs'][i] for i, x in enumerate(data['ed_loss']) if x > 0]
        if ed_vals:
            axes[1, 0].plot(ep_vals, ed_vals, label=variants[variant_key]['label'],
                           linewidth=2, color=variants[variant_key]['color'], alpha=0.9)

axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('ED Loss Term')
axes[1, 0].set_title('Energy Distance Loss Component')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Repulsion Loss Term
for variant_key, data in all_data.items():
    if variant_key in variants and variant_key in all_data:
        rep_vals = [x for x in data['repulsion'] if x > 0]
        ep_vals = [data['epochs'][i] for i, x in enumerate(data['repulsion']) if x > 0]
        if rep_vals:
            axes[1, 1].plot(ep_vals, rep_vals, label=variants[variant_key]['label'],
                           linewidth=2, color=variants[variant_key]['color'], alpha=0.9)

axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Repulsion Loss')
axes[1, 1].set_title('Repulsion Loss Component')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Final Comparison Bar Chart
model_labels = []
final_recon = []
final_ed = []
final_rare_recall = []
final_colors = []

# Add E2b variants
for variant_key in ['variant2_ed', 'variant6_ed_repel', 'variant9_all']:
    if variant_key in all_data:
        data = all_data[variant_key]
        ed = [x for x in data['energy_dist'] if not np.isnan(x)]
        rr = [x * 100 for x in data['rare_recall'] if not np.isnan(x)]

        if ed and rr:
            model_labels.append(variants[variant_key]['label'].replace('GA-AE + ', ''))
            final_recon.append(data['recon'][-1])
            final_ed.append(ed[-1])
            final_rare_recall.append(rr[-1])
            final_colors.append(variants[variant_key]['color'])

# Add E2 baselines for comparison
for model_key in ['ae', 'cae', 'ga_ae']:
    if model_key in all_data:
        data = all_data[model_key]
        ed = [x for x in data['energy_dist'] if not np.isnan(x)]
        rr = [x * 100 for x in data['rare_recall'] if not np.isnan(x)]

        if ed and rr:
            model_labels.append(e2_models[model_key]['label'])
            final_recon.append(data['recon'][-1])
            final_ed.append(ed[-1])
            final_rare_recall.append(rr[-1])
            final_colors.append(e2_models[model_key]['color'])

x = np.arange(len(model_labels))
width = 0.25

axes[1, 2].bar(x - width, final_rare_recall, width, label='Rare Recall (%)', alpha=0.8, color='lightgreen')
axes[1, 2].bar(x, final_ed, width, label='Energy Dist', alpha=0.8, color='lightcoral')
axes[1, 2].bar(x + width, final_recon, width, label='Recon MSE', alpha=0.8, color='lightblue')

axes[1, 2].axhline(y=22.7, color='purple', linestyle='--', linewidth=2, label='CAE Target')
axes[1, 2].set_xlabel('Model')
axes[1, 2].set_ylabel('Metric Value')
axes[1, 2].set_title('Final Performance Comparison')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(model_labels, rotation=30, ha='right', fontsize=8)
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/e2b_ablation/e2b_comparison.png', dpi=300, bbox_inches='tight')
print("\nSaved to results/e2b_ablation/e2b_comparison.png")

# Print summary
print("\n" + "=" * 80)
print("E2b ABLATION RESULTS SUMMARY")
print("=" * 80)

for variant_key in ['variant2_ed', 'variant6_ed_repel', 'variant9_all']:
    if variant_key in all_data:
        data = all_data[variant_key]
        ed = [x for x in data['energy_dist'] if not np.isnan(x)]
        rr = [x * 100 for x in data['rare_recall'] if not np.isnan(x)]

        if ed and rr:
            print(f"\n{variants[variant_key]['label']}:")
            print(f"  Final Recon:       {data['recon'][-1]:.4f}")
            print(f"  Final Energy Dist: {ed[-1]:.4f}")
            print(f"  Final Rare Recall: {rr[-1]:.2f}%")

            if rr[-1] > 15:
                print(f"  ✓ SUCCESS: Rare recall > 15%!")
            else:
                print(f"  ✗ FAILED: Rare recall < 15%")

print("\n" + "=" * 80)
print("E2 BASELINE COMPARISON")
print("=" * 80)

for model_key in ['ae', 'cae', 'ga_ae']:
    if model_key in all_data:
        data = all_data[model_key]
        ed = [x for x in data['energy_dist'] if not np.isnan(x)]
        rr = [x * 100 for x in data['rare_recall'] if not np.isnan(x)]

        if ed and rr:
            print(f"\n{e2_models[model_key]['label']}:")
            print(f"  Final Rare Recall: {rr[-1]:.2f}%")
