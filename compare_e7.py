"""Compare E7 GA-Native Prior results"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Variants
variants = {
    'vae_kl_only': {'label': 'VAE\n(KL only)', 'color': '#1f77b4'},
    'vae_kl_chamfer': {'label': 'VAE\n(KL+Chamfer)', 'color': '#ff7f0e'},
    'vae_chamfer_only': {'label': 'VAE\n(Chamfer only)', 'color': '#2ca02c'}
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Collect data
all_data = {}

# Load E7 results
for variant_key, info in variants.items():
    data = {'epochs': [], 'recon': [], 'kl': [], 'chamfer': [],
            'energy_dist': [], 'rare_recall': []}

    metrics_file = f'results/e7_ga_prior/{variant_key}/seed_0/metrics.jsonl'
    try:
        with open(metrics_file, 'r') as f:
            for line in f:
                record = json.loads(line)
                data['epochs'].append(record['epoch'])
                data['recon'].append(record.get('recon_loss', 0))
                data['kl'].append(record.get('kl_div', 0))
                data['chamfer'].append(record.get('chamfer_loss', 0))

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

# Plot 1: Reconstruction Loss
for variant_key, data in all_data.items():
    axes[0, 0].plot(data['epochs'], data['recon'],
                   label=variants[variant_key]['label'],
                   linewidth=2, color=variants[variant_key]['color'], alpha=0.9)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Reconstruction MSE')
axes[0, 0].set_title('Reconstruction Loss')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# Plot 2: KL Divergence
for variant_key, data in all_data.items():
    if max(data['kl']) > 0:  # Only plot if KL is used
        axes[0, 1].plot(data['epochs'], data['kl'],
                       label=variants[variant_key]['label'],
                       linewidth=2, color=variants[variant_key]['color'], alpha=0.9)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('KL Divergence')
axes[0, 1].set_title('KL Divergence (Density Prior)')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Rare Mode Recall *** KEY METRIC ***
for variant_key, data in all_data.items():
    rr = [x * 100 for x in data['rare_recall'] if not np.isnan(x)]
    ep = [data['epochs'][i] for i, x in enumerate(data['rare_recall']) if not np.isnan(x)]

    if rr:
        axes[0, 2].plot(ep, rr, label=variants[variant_key]['label'],
                       linewidth=2, color=variants[variant_key]['color'],
                       marker='o', markersize=5, alpha=0.9)

axes[0, 2].axhline(y=22.7, color='purple', linestyle=':', linewidth=2, label='CAE Target')
axes[0, 2].axhline(y=40.91, color='green', linestyle='--', linewidth=2, label='E2c Best (Grass+Ent)')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Rare Mode Recall (%)')
axes[0, 2].set_title('★ Rare Mode Coverage')
axes[0, 2].legend(fontsize=7)
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Chamfer Loss
for variant_key, data in all_data.items():
    if max(data['chamfer']) > 0:
        axes[1, 0].plot(data['epochs'], data['chamfer'],
                       label=variants[variant_key]['label'],
                       linewidth=2, color=variants[variant_key]['color'], alpha=0.9)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Chamfer Loss')
axes[1, 0].set_title('Tangent Chamfer Loss (Geometric Prior)')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Energy Distance
for variant_key, data in all_data.items():
    ed = [x for x in data['energy_dist'] if not np.isnan(x)]
    ep = [data['epochs'][i] for i, x in enumerate(data['energy_dist']) if not np.isnan(x)]

    if ed:
        axes[1, 1].plot(ep, ed, label=variants[variant_key]['label'],
                       linewidth=2, color=variants[variant_key]['color'],
                       marker='o', markersize=5, alpha=0.9)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Energy Distance')
axes[1, 1].set_title('Generation Quality (Lower = Better)')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Final Comparison Bar Chart
model_labels = []
final_kl = []
final_chamfer = []
final_rare_recall = []
final_colors = []

for variant_key in ['vae_kl_only', 'vae_kl_chamfer', 'vae_chamfer_only']:
    if variant_key in all_data:
        data = all_data[variant_key]
        rr = [x * 100 for x in data['rare_recall'] if not np.isnan(x)]

        if rr:
            label = variants[variant_key]['label'].replace('\n', ' ')
            model_labels.append(label)
            final_kl.append(data['kl'][-1])
            final_chamfer.append(data['chamfer'][-1])
            final_rare_recall.append(rr[-1])
            final_colors.append(variants[variant_key]['color'])

x = np.arange(len(model_labels))
width = 0.25

axes[1, 2].bar(x - width, final_rare_recall, width, label='Rare Recall (%)', alpha=0.8, color='lightgreen')
axes[1, 2].bar(x, [k*10 for k in final_kl], width, label='KL×10', alpha=0.8, color='lightcoral')
axes[1, 2].bar(x + width, [c*100 for c in final_chamfer], width, label='Chamfer×100', alpha=0.8, color='lightyellow')

axes[1, 2].axhline(y=22.7, color='purple', linestyle='--', linewidth=2, label='CAE Target')
axes[1, 2].axhline(y=40.91, color='green', linestyle='--', linewidth=2, label='E2c Best')
axes[1, 2].set_xlabel('Model')
axes[1, 2].set_ylabel('Metric Value')
axes[1, 2].set_title('Final Performance Comparison')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(model_labels, rotation=20, ha='right', fontsize=8)
axes[1, 2].legend(fontsize=7)
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/e7_ga_prior/e7_comparison.png', dpi=300, bbox_inches='tight')
print("\nSaved to results/e7_ga_prior/e7_comparison.png")

# Print summary
print("\n" + "=" * 80)
print("E7 GA-NATIVE PRIOR RESULTS SUMMARY")
print("=" * 80)

for variant_key in ['vae_kl_only', 'vae_kl_chamfer', 'vae_chamfer_only']:
    if variant_key in all_data:
        data = all_data[variant_key]
        ed = [x for x in data['energy_dist'] if not np.isnan(x)]
        rr = [x * 100 for x in data['rare_recall'] if not np.isnan(x)]

        if ed and rr:
            print(f"\n{variants[variant_key]['label'].replace(chr(10), ' ')}:")
            print(f"  Final KL:          {data['kl'][-1]:.4f}")
            print(f"  Final Chamfer:     {data['chamfer'][-1]:.4f}")
            print(f"  Final Recon:       {data['recon'][-1]:.4f}")
            print(f"  Final Energy Dist: {ed[-1]:.4f}")
            print(f"  Final Rare Recall: {rr[-1]:.2f}%")

            if rr[-1] > 40.91:
                print(f"  ✓✓ BEATS E2c BEST (40.91%)!")
            elif rr[-1] > 22.7:
                print(f"  ✓✓ BEATS CAE (22.7%)")
            elif rr[-1] > 15:
                print(f"  ✓ SUCCESS (>15%)")
            else:
                print(f"  ~ Partial or failed")

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print("1. Can Chamfer replace KL?")
print("   Compare 'VAE (Chamfer only)' vs 'VAE (KL only)'")
print("\n2. Does hybrid approach help?")
print("   Compare 'VAE (KL+Chamfer)' vs individual terms")
print("\n3. Rare mode coverage:")
print("   Target: CAE 22.7%, E2c Best 40.91%")
print("=" * 80)
