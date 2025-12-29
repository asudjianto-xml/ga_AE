"""Compare E2c GA-native results"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Variants
variants = {
    'option_a_grass_entropy': {'label': 'Option A:\nGrass+Entropy', 'color': '#1f77b4'},
    'option_b_grass_matching': {'label': 'Option B:\nGrass+Matching', 'color': '#ff7f0e'},
    'option_c_all_three': {'label': 'Option C:\nAll Three', 'color': '#2ca02c'}
}

# Also load E2 baselines for comparison
e2_models = {
    'ae': {'label': 'AE', 'color': 'gray', 'linestyle': '--'},
    'cae': {'label': 'CAE (Winner)', 'color': 'purple', 'linestyle': '--'},
    'ga_ae': {'label': 'GA-AE (Original)', 'color': 'red', 'linestyle': '--'}
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Collect data
all_data = {}

# Load E2c GA-native results
for variant_key, info in variants.items():
    data = {'epochs': [], 'recon': [], 'energy_dist': [], 'rare_recall': [],
            'grassmann': [], 'blade_entropy': [], 'blade_matching': []}

    metrics_file = f'results/e2c_ga_native/{variant_key}/seed_0/metrics.jsonl'
    try:
        with open(metrics_file, 'r') as f:
            for line in f:
                record = json.loads(line)
                data['epochs'].append(record['epoch'])
                data['recon'].append(record.get('recon_loss', 0))
                data['grassmann'].append(record.get('grassmann_loss', 0))
                data['blade_entropy'].append(record.get('blade_entropy_loss', 0))
                data['blade_matching'].append(record.get('blade_matching_loss', 0))

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

axes[0, 2].axhline(y=22.7, color='purple', linestyle=':', linewidth=2, label='CAE Target')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Rare Mode Recall (%)')
axes[0, 2].set_title('★ Rare Mode Coverage (TARGET: >22.7%)')
axes[0, 2].legend(fontsize=7)
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Grassmann Loss
for variant_key, data in all_data.items():
    if variant_key in variants and variant_key in all_data:
        grass_vals = [x for x in data['grassmann'] if x > 0]
        ep_vals = [data['epochs'][i] for i, x in enumerate(data['grassmann']) if x > 0]
        if grass_vals:
            axes[1, 0].plot(ep_vals, grass_vals, label=variants[variant_key]['label'],
                           linewidth=2, color=variants[variant_key]['color'], alpha=0.9)

axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Grassmann Spread Loss')
axes[1, 0].set_title('Grassmann k-Blade Repulsion')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Blade Entropy Loss
for variant_key, data in all_data.items():
    if variant_key in variants and variant_key in all_data:
        ent_vals = [x for x in data['blade_entropy'] if x != 0]
        ep_vals = [data['epochs'][i] for i, x in enumerate(data['blade_entropy']) if x != 0]
        if ent_vals:
            axes[1, 1].plot(ep_vals, ent_vals, label=variants[variant_key]['label'],
                           linewidth=2, color=variants[variant_key]['color'], alpha=0.9)

axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Blade Entropy Loss')
axes[1, 1].set_title('Multi-Grade Entropy (Lower = More Diverse)')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Final Comparison Bar Chart
model_labels = []
final_recon = []
final_ed = []
final_rare_recall = []
final_colors = []

# Add E2c variants
for variant_key in ['option_a_grass_entropy', 'option_b_grass_matching', 'option_c_all_three']:
    if variant_key in all_data:
        data = all_data[variant_key]
        ed = [x for x in data['energy_dist'] if not np.isnan(x)]
        rr = [x * 100 for x in data['rare_recall'] if not np.isnan(x)]

        if ed and rr:
            label = variants[variant_key]['label'].replace('\n', ' ')
            model_labels.append(label[:15])  # Truncate for readability
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
axes[1, 2].bar(x + width, [r*100 for r in final_recon], width, label='Recon×100', alpha=0.8, color='lightblue')

axes[1, 2].axhline(y=22.7, color='purple', linestyle='--', linewidth=2, label='CAE Target')
axes[1, 2].set_xlabel('Model')
axes[1, 2].set_ylabel('Metric Value')
axes[1, 2].set_title('Final Performance Comparison')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(model_labels, rotation=40, ha='right', fontsize=7)
axes[1, 2].legend(fontsize=7)
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/e2c_ga_native/e2c_comparison.png', dpi=300, bbox_inches='tight')
print("\nSaved to results/e2c_ga_native/e2c_comparison.png")

# Print summary
print("\n" + "=" * 80)
print("E2c GA-NATIVE RESULTS SUMMARY")
print("=" * 80)

for variant_key in ['option_a_grass_entropy', 'option_b_grass_matching', 'option_c_all_three']:
    if variant_key in all_data:
        data = all_data[variant_key]
        ed = [x for x in data['energy_dist'] if not np.isnan(x)]
        rr = [x * 100 for x in data['rare_recall'] if not np.isnan(x)]

        if ed and rr:
            print(f"\n{variants[variant_key]['label'].replace(chr(10), ' ')}:")
            print(f"  Final Recon:       {data['recon'][-1]:.4f}")
            print(f"  Final Energy Dist: {ed[-1]:.4f}")
            print(f"  Final Rare Recall: {rr[-1]:.2f}%")

            if rr[-1] > 22.7:
                print(f"  ✓✓ BEATS CAE (22.7%)!")
            elif rr[-1] > 15:
                print(f"  ✓ SUCCESS: Rare recall > 15%")
            elif rr[-1] > 0:
                print(f"  ~ PARTIAL: Some coverage")
            else:
                print(f"  ✗ FAILED: No rare mode coverage")

print("\n" + "=" * 80)
print("COMPARISON TO E2 BASELINES")
print("=" * 80)
print("  AE:         0.00% rare recall")
print("  CAE:       22.73% rare recall ← TARGET")
print("  GA-AE:      0.00% rare recall")
print("=" * 80)
