# Experiment Execution Status

**Start Time**: 2025-12-28 21:03 UTC
**Command**: `python run_experiments.py --experiment all --seed 0`
**Estimated Duration**: 6-8 hours
**Expected Completion**: ~03:00-05:00 UTC (next morning)

---

## Experiment Queue

All 6 experiments will run sequentially:

### E1: The AE Trap ⏳ STARTING
- **Dataset**: 20D MoG (10K train, 2K val, 2K test)
- **Model**: Standard AE
- **Training**: 200 epochs
- **Est. Time**: ~30 minutes
- **Status**: Initializing...

### E2: Tail Stress Test ⏳ QUEUED
- **Dataset**: 20D MoG with 2% rare mode
- **Models**: AE, GA-AE, CAE (3 models)
- **Training**: 200 epochs each
- **Est. Time**: ~90 minutes
- **Status**: Waiting for E1

### E3: VAE Posterior Collapse ⏳ QUEUED
- **Dataset**: 20D MoG
- **Models**: VAE with β ∈ {0.1, 1.0, 4.0} × {no anneal, anneal} (6 models)
- **Training**: 200 epochs each
- **Est. Time**: ~3 hours
- **Status**: Waiting for E2

### E4: VAE Trade-off ⏳ QUEUED
- **Dataset**: 20D MoG
- **Models**: VAE (standard), VAE (anneal), GA-VAE (MMD) (3 models)
- **Training**: 200 epochs each
- **Est. Time**: ~90 minutes
- **Status**: Waiting for E3

### E5: Baselines Comparison ⏳ QUEUED
- **Dataset**: Swiss roll 50D
- **Models**: AE, Spectral AE, Sobolev AE, GA-AE (4 models)
- **Training**: 200 epochs each
- **Est. Time**: ~2 hours
- **Status**: Waiting for E4

### E6: Teacher Generator ⏳ QUEUED
- **Dataset**: Teacher networks (smooth + sharp)
- **Models**: 2 teachers × 2 models (4 models)
- **Training**: 200 epochs each
- **Est. Time**: ~2 hours
- **Status**: Waiting for E5

---

## Total Progress

**Models to Train**: 1 + 3 + 6 + 3 + 4 + 4 = **21 models**
**Total Epochs**: 21 × 200 = **4,200 epochs**

---

## Monitoring

### Real-time Log
```bash
tail -f full_experiments.log
```

### Progress Check
```bash
bash monitor_progress.sh
```

### GPU Utilization
```bash
nvidia-smi
```

### Disk Space
```bash
du -sh results/
```

---

## Expected Outputs

For each experiment, the following will be generated:

```
results/<experiment_name>/<model_name>/seed_0/
├── metrics.jsonl           # Training logs (diagnostic every 5 epochs)
├── config.json             # Configuration
├── model_epoch_50.pt       # Checkpoints
├── model_epoch_100.pt
├── model_epoch_150.pt
├── model_epoch_200.pt      # Final model
└── plots/
    ├── *_training.png
    ├── *_diagnostics.png
    ├── *_generation.png
    ├── *_ae_trap.png (E1, E2)
    ├── *_decoder_stability.png (E1, E2, E6)
    └── *_vae_collapse.png (E3, E4)
```

Plus comparison plots in experiment-level `plots/` directories.

---

## Resource Usage

### GPU
- **Device**: NVIDIA GB10
- **Utilization**: Expected 90-100%
- **Memory**: ~2-4 GB VRAM per model

### Disk
- **Current**: ~2 MB (from E1 demo)
- **Expected Final**: ~3-5 GB
  - Metrics: ~50 KB per model
  - Checkpoints: ~1 MB per checkpoint × 4 per model × 21 models ≈ 84 MB
  - Plots: ~1 MB per model × 21 ≈ 21 MB
  - Data: ~100 MB

### CPU
- **Cores**: Multi-threaded (PyTorch DataLoader)
- **Expected**: 400-600% utilization (4-6 cores)

---

## What Happens Next

### During Execution (Automatic)
1. ✅ Datasets generated for each experiment
2. ✅ Models initialized
3. ✅ Training for 200 epochs per model
4. ✅ Diagnostics computed every 5 epochs
5. ✅ Checkpoints saved every 50 epochs
6. ✅ Plots generated after each model completes
7. ✅ Comparison plots created per experiment

### After Completion (Manual)
1. Run analysis: `python analyze_results.py results/<exp> --compare`
2. Extract key numbers for paper tables
3. Select best figures for paper
4. Optional: Run with more seeds for robustness

---

## Troubleshooting

### If Experiments Crash

1. **Check last log**:
   ```bash
   tail -100 full_experiments.log
   ```

2. **Find incomplete experiments**:
   ```bash
   find results -name "metrics.jsonl" -exec wc -l {} \;
   ```

3. **Resume from specific experiment**:
   ```bash
   python run_experiments.py --experiment e3 --seed 0
   ```

### If Running Slow

1. **Check GPU utilization**:
   ```bash
   watch -n 1 nvidia-smi
   ```
   - Should see ~90-100% GPU utilization
   - Memory usage ~2-4 GB

2. **Check diagnostic overhead**:
   - If diagnostics taking too long, can reduce `diagnostic_every` in config
   - Or reduce `n_samples_diagnostic` (default 1000)

### If Disk Full

1. **Remove demo results**:
   ```bash
   rm -rf results/e1_ae_trap_demo results/test_quick
   ```

2. **Remove intermediate checkpoints**:
   ```bash
   find results -name "model_epoch_50.pt" -delete
   find results -name "model_epoch_100.pt" -delete
   find results -name "model_epoch_150.pt" -delete
   # Keep only final checkpoints
   ```

---

## Progress Milestones

Expected timeline (started 21:03):

| Time | Milestone | Cumulative Progress |
|------|-----------|---------------------|
| 21:03 | Start E1 | 0% |
| 21:30 | Complete E1 | 5% (1/21 models) |
| 21:35 | Start E2 | 5% |
| 23:00 | Complete E2 | 19% (4/21 models) |
| 23:05 | Start E3 | 19% |
| 02:00 | Complete E3 | 48% (10/21 models) |
| 02:05 | Start E4 | 48% |
| 03:30 | Complete E4 | 62% (13/21 models) |
| 03:35 | Start E5 | 62% |
| 05:30 | Complete E5 | 81% (17/21 models) |
| 05:35 | Start E6 | 81% |
| 07:30 | Complete E6 | 100% (21/21 models) |

**Note**: Times are approximate and may vary based on GPU load and diagnostic computation time.

---

## Success Criteria

Experiments are successful if:

1. ✅ All 21 models train to 200 epochs
2. ✅ All metrics files contain ~40 entries (diagnostic every 5 epochs)
3. ✅ All final checkpoints exist (`model_epoch_200.pt`)
4. ✅ All plots generated without errors
5. ✅ No NaN values in key metrics
6. ✅ Expected patterns observed:
   - E1: Recon ↓, Gen flat/bad
   - E2: GA-AE rare mode recall > AE
   - E3: k-volume predicts collapse
   - E4: MMD+geom > KL+geom
   - E5: GA-AE > Spectral
   - E6: GA-AE closer to teacher

---

## Next Steps After Completion

### Immediate Analysis
```bash
# Check all experiments completed
bash monitor_progress.sh

# Analyze each experiment
for exp in e1_ae_trap e2_tail_stress e3_vae_collapse e4_vae_tradeoff e5_baselines e6_teacher; do
    python analyze_results.py results/$exp --compare
done
```

### Extract Paper Results
```python
# Create results aggregation script
# Extract key numbers from all experiments
# Generate summary tables
# Compile figures for paper
```

### Optional: Multi-Seed Runs
```bash
# Run with 5 seeds for robustness
for seed in 0 1 2 3 4; do
    python run_experiments.py --experiment all --seed $seed
done

# Then compute mean ± std across seeds
```

---

**Status**: 🟢 Running (Check `monitor_progress.sh` for live updates)
