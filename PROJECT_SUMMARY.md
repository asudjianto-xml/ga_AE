# Project Summary: Geometric Autoencoder Experiments

## Overview

This project implements a complete experimental framework for the paper **"Diagnosing the Generative Gap in Autoencoders: A Geometric Perspective on Volume, Subspace Collapse and Off-Manifold Failure"**.

**Status**: ✅ Implementation Complete | E1 Demo Successfully Run

---

## What Has Been Implemented

### 1. Core Components

#### Datasets (`src/datasets/`)
- ✅ **D1**: Mixture of Gaussians (2D and 20D) with varying correlation structures
- ✅ **D2**: Swiss roll embedded in 50D with anisotropic noise
- ✅ **D3**: Teacher network generators (smooth/sharp curvature variants)
- ✅ **D4**: Support for real tabular data (UCI datasets)

#### Geometric Diagnostics (`src/diagnostics/`)
- ✅ Jacobian-vector product (JVP) utilities using `torch.func.jvp`
- ✅ k-volume computation for k ∈ {1,2,4,8}
- ✅ Encoder-decoder consistency (EDC) metrics
- ✅ Off-manifold decoder stability under radius stress
- ✅ Generative gap index computation
- ✅ Support for both random and PCA-aligned orthonormal directions

#### Models (`src/models/`)
- ✅ **Base Models**: Deterministic AE, VAE
- ✅ **Geometry-Regularized**: GA-AE, GA-VAE (with k-volume floors and EDC penalties)
- ✅ **Baselines**: Contractive AE, Spectral Norm AE, Sobolev AE
- ✅ VAE with KL annealing and MMD-based posterior matching

#### Metrics (`src/utils/metrics.py`)
- ✅ MMD (Maximum Mean Discrepancy)
- ✅ Energy Distance
- ✅ k-NN Precision/Recall
- ✅ Mode coverage and rare-mode recall
- ✅ KL divergence between mode distributions

#### Training & Evaluation (`src/utils/`)
- ✅ Unified trainer for all model types
- ✅ Automatic diagnostic computation at checkpoints
- ✅ Generation quality evaluation
- ✅ Metrics logging to JSONL
- ✅ Model checkpointing

#### Visualization (`src/utils/plotting.py`)
- ✅ Training curve plots
- ✅ Geometric diagnostic evolution
- ✅ Generation quality metrics
- ✅ Reconstruction vs generation gap visualization
- ✅ VAE posterior collapse indicators
- ✅ Decoder stability under radius stress
- ✅ Multi-model comparison scatter plots

### 2. Experiments

All 6 experiments from the paper specification are implemented:

- ✅ **E1**: The AE Trap (reconstruction ≠ generation)
- ✅ **E2**: Tail stress test (rare mode recall)
- ✅ **E3**: VAE posterior collapse sweep
- ✅ **E4**: VAE trade-off (KL vs MMD)
- ✅ **E5**: Baselines comparison
- ✅ **E6**: Controlled teacher generator

---

## Files Created

### Main Scripts
```
run_experiments.py       # Main runner for all experiments
run_e1_demo.py          # Quick demo of E1 with 50 epochs
test_quick.py           # Quick functionality test
```

### Documentation
```
README.md               # Setup and usage instructions
EXPERIMENT_NOTES.md     # Detailed experiment specifications and expected results
PROJECT_SUMMARY.md      # This file
Claude.md               # Python environment configuration
```

### Source Code
```
src/
├── datasets/
│   ├── __init__.py
│   └── synthetic.py              # All dataset generators
├── models/
│   ├── __init__.py
│   ├── base_models.py            # AE, VAE
│   ├── geometry_models.py        # GA-AE, GA-VAE
│   └── baseline_models.py        # CAE, Spectral, Sobolev
├── diagnostics/
│   ├── __init__.py
│   ├── jacobian_utils.py         # JVP computation
│   └── geometric_metrics.py      # k-volume, EDC, gap index
└── utils/
    ├── __init__.py
    ├── config.py                 # Configuration dataclasses
    ├── training.py               # Training loop
    ├── metrics.py                # Evaluation metrics
    └── plotting.py               # Visualization utilities
```

---

## E1 Demo Results (50 epochs)

**Successfully completed!** Results saved to: `results/e1_ae_trap_demo/seed_0/`

### Generated Files:
- `metrics.jsonl` - Training metrics logged every 5 epochs
- `config.json` - Experiment configuration
- `model_epoch_25.pt` - Checkpoint at epoch 25
- `model_epoch_50.pt` - Final model
- `plots/` - 5 PNG figures:
  - `seed_0_training.png` - Loss curves
  - `seed_0_diagnostics.png` - k-volume evolution, EDC, gap metrics
  - `seed_0_generation.png` - MMD, Energy Distance, k-NN metrics, rare mode recall
  - `seed_0_ae_trap.png` - Reconstruction vs Generation gap
  - `seed_0_decoder_stability.png` - Off-manifold stability under radius stress

### Key Observations (from logs):
- **Epoch 0**: Train Loss = 8.97, Val Recon = 8.69
- **Epoch 10**: Train Loss = 0.50, Val Recon = 0.48 (reconstruction improving)
- **Epoch 40**: Train Loss = 0.32, Val Recon = 0.32 (reconstruction excellent)
- **Geometric diagnostics**: Computed at epochs 0, 5, 10, ..., 50
- **All plots generated successfully**

---

## How to Run Full Experiments

### Option 1: Run All Experiments
```bash
cd /home/asudjianto/jupyterlab/ga_AE
~/jupyterlab/ga_verify/venv/bin/python run_experiments.py --experiment all --seed 0
```
**Time estimate**: ~6-8 hours for all 6 experiments (200 epochs each, multiple models)

### Option 2: Run Individual Experiments
```bash
# E1: AE Trap (1 model, 200 epochs) - ~30 min
~/jupyterlab/ga_verify/venv/bin/python run_experiments.py --experiment e1

# E2: Tail stress (3 models) - ~90 min
~/jupyterlab/ga_verify/venv/bin/python run_experiments.py --experiment e2

# E3: VAE collapse (6 models: 3 betas × 2 annealing) - ~3 hours
~/jupyterlab/ga_verify/venv/bin/python run_experiments.py --experiment e3

# E4: VAE tradeoff (3 models) - ~90 min
~/jupyterlab/ga_verify/venv/bin/python run_experiments.py --experiment e4

# E5: Baselines (4 models) - ~2 hours
~/jupyterlab/ga_verify/venv/bin/python run_experiments.py --experiment e5

# E6: Teacher (4 models: 2 teachers × 2 models) - ~2 hours
~/jupyterlab/ga_verify/venv/bin/python run_experiments.py --experiment e6
```

### Option 3: Run Multiple Seeds
```bash
# Run E1 with 3 different seeds for robustness
for seed in 0 1 2; do
    ~/jupyterlab/ga_verify/venv/bin/python run_experiments.py --experiment e1 --seed $seed
done
```

---

## Results Structure

Each experiment creates a structured results directory:

```
results/
├── e1_ae_trap/
│   └── seed_0/
│       ├── metrics.jsonl           # Training logs
│       ├── config.json             # Configuration
│       ├── model_epoch_50.pt       # Checkpoint
│       ├── model_epoch_100.pt
│       ├── model_epoch_150.pt
│       ├── model_epoch_200.pt      # Final model
│       └── plots/
│           ├── seed_0_training.png
│           ├── seed_0_diagnostics.png
│           ├── seed_0_generation.png
│           ├── seed_0_ae_trap.png
│           └── seed_0_decoder_stability.png
├── e2_tail_stress/
│   ├── ae/
│   │   └── seed_0/ ...
│   ├── ga_ae/
│   │   └── seed_0/ ...
│   ├── cae/
│   │   └── seed_0/ ...
│   └── plots/
│       └── comparison_scatter.png  # Cross-model comparison
...
```

---

## Using Results in the Paper

### Extracting Numbers for Tables

```python
import json
import pandas as pd

# Load metrics
with open('results/e1_ae_trap/seed_0/metrics.jsonl', 'r') as f:
    metrics = [json.loads(line) for line in f]
df = pd.DataFrame(metrics)

# Final epoch values
final = df.iloc[-1]
print(f"Reconstruction MSE: {final['val_recon_loss']:.3f}")
print(f"Energy Distance: {final['gen_energy_distance']:.3f}")
print(f"Generative Gap: {final['diag_gap_overall']:.3f}")
```

### Figures for Paper

All plots are saved as PNG at 300 DPI. Recommended figures:

1. **Figure 1** (The AE Trap): Use `seed_0_ae_trap.png` from E1
2. **Figure 2** (k-volume collapse): Use `seed_0_diagnostics.png` from E1 (top-left panel)
3. **Figure 3** (Rare mode recall): Combine histograms from E2 (all models)
4. **Figure 4** (VAE collapse): Use `seed_0_vae_collapse.png` from E3
5. **Figure 5** (Baselines comparison): Use `comparison_scatter.png` from E5
6. **Figure 6** (Teacher validation): Use `*_decoder_stability.png` from E6

---

## Key Implementation Details

### GPU Acceleration
- All operations run on GPU (CUDA)
- Current GPU: NVIDIA GB10
- Batch processing for diagnostics
- JVP computation fully GPU-accelerated

### Computational Efficiency
- Diagnostics computed on subset (n=1000) of validation data
- k-volume: exact computation via Cholesky decomposition
- PCA directions precomputed once
- Checkpointing every 50 epochs to allow resumption

### Numerical Stability
- Epsilon regularization: {1e-8, 1e-6, 1e-4}
- Cholesky decomposition for log-det (more stable than eigendecomposition)
- Gradient clipping available (not enabled by default)
- Mixed precision training possible (not enabled by default)

---

## Next Steps

### To Generate Paper-Ready Results:

1. **Run all experiments with multiple seeds:**
   ```bash
   for exp in e1 e2 e3 e4 e5 e6; do
       for seed in 0 1 2; do
           ~/jupyterlab/ga_verify/venv/bin/python run_experiments.py \
               --experiment $exp --seed $seed
       done
   done
   ```

2. **Aggregate results across seeds:**
   - Mean ± std for all metrics
   - Statistical significance tests (t-tests, Wilcoxon)
   - Create multi-seed plots

3. **Generate paper figures:**
   - Use high-quality plotting (already at 300 DPI)
   - Consider adjusting color schemes for publication
   - Add significance indicators (*p<0.05, etc.)

4. **Extract result tables:**
   - Create summary CSV files
   - Format for LaTeX tables
   - Include confidence intervals

### Optional Enhancements:

- ⬜ Add 2D visualizations (t-SNE of latent space)
- ⬜ Add interpolation experiments
- ⬜ Add real tabular datasets (UCI Adult, Credit)
- ⬜ Implement full stochastic log-det estimation for global volume
- ⬜ Add teacher Jacobian comparison plots for E6
- ⬜ Add progressive k-collapse animation
- ⬜ Mixed precision training for faster computation

---

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**
   - Reduce batch size in config
   - Reduce n_samples_diagnostic (default 1000)

2. **Training too slow**
   - Check GPU utilization (`nvidia-smi`)
   - Reduce diagnostic_every frequency
   - Consider reducing number of k values

3. **Plots not generating**
   - Check matplotlib backend (already set to 'Agg')
   - Ensure results directory is writable

4. **JSON serialization errors**
   - Already fixed with convert_to_serializable()
   - If new issues arise, check for torch.Tensor in logs

---

## Citation

If you use this code or results:

```bibtex
@article{sudjianto2025geometric,
  title={Diagnosing the Generative Gap in Autoencoders: A Geometric Perspective
         on Volume, Subspace Collapse and Off-Manifold Failure},
  author={Sudjianto, Agus},
  year={2025}
}
```

---

## Contact & Support

- **Project location**: `/home/asudjianto/jupyterlab/ga_AE`
- **Virtual environment**: `~/jupyterlab/ga_verify/venv/bin/python`
- **GPU**: NVIDIA GB10 (CUDA 12.1)

**All systems operational and ready for full experimental runs!** 🚀
