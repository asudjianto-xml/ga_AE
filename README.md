# Geometric Autoencoder Experiments

Implementation of experiments for "Diagnosing the Generative Gap in Autoencoders: A Geometric Perspective on Volume, Subspace Collapse and Off-Manifold Failure"

## Project Structure

```
ga_AE/
├── src/
│   ├── datasets/          # Dataset generators (MoG, Swiss roll, Teacher networks)
│   ├── models/            # Model architectures (AE, VAE, GA-AE, GA-VAE, baselines)
│   ├── diagnostics/       # Geometric diagnostics (JVP, k-volume, EDC)
│   ├── experiments/       # Individual experiment scripts
│   └── utils/             # Training, metrics, plotting utilities
├── results/               # Experiment results (created during runs)
├── configs/               # Configuration files
├── run_experiments.py     # Main experiment runner
├── test_quick.py          # Quick functionality test
└── README.md
```

## Setup

### Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA support
- NumPy, SciPy, scikit-learn
- Matplotlib, Seaborn
- tqdm

### Installation

```bash
# Using the configured virtual environment
source ~/jupyterlab/ga_verify/venv/bin/activate

# Install additional dependencies if needed
pip install matplotlib seaborn scikit-learn tqdm
```

## Quick Test

Run a quick test to verify everything works:

```bash
cd /home/asudjianto/jupyterlab/ga_AE
~/jupyterlab/ga_verify/venv/bin/python test_quick.py
```

This will train a small model for 5 epochs and verify all components are working.

## Running Experiments

### Run All Experiments

```bash
~/jupyterlab/ga_verify/venv/bin/python run_experiments.py --experiment all --seed 0
```

### Run Individual Experiments

```bash
# E1: AE Trap (reconstruction good, generation bad)
~/jupyterlab/ga_verify/venv/bin/python run_experiments.py --experiment e1

# E2: Tail stress test (rare mode recall)
~/jupyterlab/ga_verify/venv/bin/python run_experiments.py --experiment e2

# E3: VAE posterior collapse sweep
~/jupyterlab/ga_verify/venv/bin/python run_experiments.py --experiment e3

# E4: VAE trade-off (KL vs MMD)
~/jupyterlab/ga_verify/venv/bin/python run_experiments.py --experiment e4

# E5: Baselines comparison
~/jupyterlab/ga_verify/venv/bin/python run_experiments.py --experiment e5

# E6: Controlled teacher generator
~/jupyterlab/ga_verify/venv/bin/python run_experiments.py --experiment e6
```

### Options

- `--experiment`: Which experiment to run (e1-e6 or all)
- `--seed`: Random seed (default: 0)
- `--device`: Device to use (cuda or cpu, default: cuda)

## Experiments

### E1: The AE Trap

**Goal**: Demonstrate that low reconstruction error does not imply good generation quality.

**Dataset**: 20D Mixture of Gaussians

**Models**: Standard AE

**Key Metrics**:
- Reconstruction MSE vs Energy Distance
- k-volume collapse indicators
- Off-manifold decoder stability

**Expected Result**: Reconstruction error decreases, but generation quality plateaus or worsens. Geometric diagnostics predict this gap.

### E2: Tail Stress Test

**Goal**: Show that AEs miss rare modes while GA-AE captures them better.

**Dataset**: 20D MoG with 2% rare mode

**Models**: AE, GA-AE, CAE

**Key Metrics**:
- Rare mode recall
- Mode coverage
- k-volume preservation

**Expected Result**: GA-AE achieves better rare mode recall than baselines.

### E3: VAE Posterior Collapse Sweep

**Goal**: Show that KL divergence alone doesn't predict collapse; k-volume does.

**Dataset**: 20D MoG

**Models**: VAE with β ∈ {0.1, 1.0, 4.0} and KL annealing variants

**Key Metrics**:
- KL divergence
- Mean encoder k-volume
- Generation quality

**Expected Result**: k-volume metrics predict generation failure earlier than KL.

### E4: VAE Trade-off

**Goal**: Compare KL-based vs MMD-based posterior matching with geometry regularization.

**Dataset**: 20D MoG

**Models**: Standard VAE, VAE + annealing, GA-VAE with MMD

**Key Metrics**:
- Latent space match quality
- Generation metrics
- Geometric gap scores

**Expected Result**: MMD-based approach better accommodates geometry preservation.

### E5: Baselines Comparison

**Goal**: Show that existing Jacobian-based methods don't fully address selective k-collapse.

**Dataset**: Swiss roll embedded in 50D

**Models**: AE, Spectral Norm AE, Sobolev AE, GA-AE

**Key Metrics**:
- k-volume for different k
- EDC metrics
- Generation quality

**Expected Result**: Spectral norm helps stability but doesn't restore k-subspace structure like GA-AE.

### E6: Controlled Teacher Generator

**Goal**: Validate diagnostics against ground-truth generator with known Jacobian.

**Dataset**: Teacher network synthetic (smooth vs sharp curvature)

**Models**: AE, GA-AE

**Key Metrics**:
- Decoder Jacobian vs teacher Jacobian
- Radius stress test
- Generation error

**Expected Result**: GA diagnostics correctly predict mismatch under off-manifold sampling.

## Results

Results are saved in `results/<experiment_name>/`:

- `metrics.jsonl`: Training metrics logged at each checkpoint
- `config.json`: Experiment configuration
- `model_epoch_X.pt`: Model checkpoints
- `plots/`: Generated figures

## Diagnostic Metrics

### Geometric Diagnostics

- **Log k-volume**: `log_vol_{E,k}(x)` - measures preservation of k-dimensional subspaces
- **EDC (Encoder-Decoder Consistency)**: `||J_DE(x)V_k - V_k||_F^2` - round-trip distortion
- **Generative Gap Index**: Difference between on-manifold and off-manifold metrics
- **Decoder Stability**: Off-manifold volume under radius stress test

### Generation Metrics

- **MMD**: Maximum Mean Discrepancy between real and generated
- **Energy Distance**: Statistical distance measure
- **k-NN Precision/Recall**: Coverage metrics
- **Mode Coverage**: Fraction of modes with generated samples
- **Rare Mode Recall**: Capture rate of rare modes

## Implementation Details

### Jacobian Computation

Uses `torch.func.jvp` for efficient Jacobian-vector products. For k orthonormal directions V_k:
- Compute `J_E(x) @ V_k` via k JVP calls
- Form Gram matrix `A^T A` where `A = J_E(x) V_k`
- Compute log-det with Cholesky decomposition

### Regularization

**GA-AE Loss**:
```
L = L_recon + λ_k * sum_k max(0, τ_k - log_vol_k) + λ_edc * EDC_k
```

**GA-VAE Loss**:
```
L = L_recon + β * KL + λ_k * vol_penalty + λ_edc * EDC_k
```
Or with MMD:
```
L = L_recon + λ_mmd * MMD(q_agg, p) + λ_k * vol_penalty + λ_edc * EDC_k
```

### Computational Efficiency

- Diagnostics computed on subset of validation data (n=1000)
- k-volume: exact for small k (≤8), stochastic estimation for global volume
- PCA directions precomputed once on training data
- GPU acceleration for all operations

## Citation

If you use this code, please cite:

```bibtex
@article{sudjianto2025geometric,
  title={Diagnosing the Generative Gap in Autoencoders: A Geometric Perspective},
  author={Sudjianto, Agus},
  year={2025}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on the repository.
