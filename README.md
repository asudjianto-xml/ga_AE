# Grassmannian Autoencoder (GA-AE)

Implementation of "Escaping the Autoencoder Trap: Grassmannian Tangent-Space Regularization for Tail Coverage"

**Key Innovation**: Geometric regularization using Grassmann manifolds and exterior algebra to improve rare mode coverage in autoencoders, avoiding the mode collapse and tail mass misallocation observed in VAEs.

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

### E1-E7: 2D Gaussian Mixture Experiments

**Goal**: Validate geometric diagnostics and tail coverage on controlled synthetic data.

**Dataset**: 2D Mixture of Gaussians with 2% rare tail mode

**Models**: Standard AE, Contractive AE, Spectral Norm AE, VAE (β=0.1, 1.0, 4.0), GA-AE

**Key Results**:
- Standard AE: 0 rare samples captured (0% recall)
- Contractive AE: 10/44 rare samples (23% recall)
- **GA-AE: 18/44 rare samples (41% recall)**
- VAE: 243-249/2000 samples classified as rare (6× overproduction, tail mass misallocation)

### E8: MNIST with Class Imbalance

**Goal**: Validate on real image data at scale.

**Dataset**: MNIST with digit 9 reduced to 2% training frequency (1,000/50,000 samples)

**Models**: VAE (β=1.0), GA-AE

**Training**:
```bash
# VAE
python train_mnist.py --model-type vae --beta 1.0 --epochs 50 --seed 0

# GA-AE
python train_mnist.py --model-type ga-ae --lambda-grass 0.1 --lambda-entropy 0.01 --epochs 50 --seed 0
```

**Key Results**:
- **VAE**: Severe mode collapse
  - Generated: 2000/2000 samples → digit 9 (100%)
  - Rare Mode Lift: 50× (massive overproduction)
  - Sample variance: 0.0005 (all nearly identical)
- **GA-AE**: Near-perfect calibration
  - Generated: 55/2000 samples → digit 9 (2.75%)
  - Rare Mode Lift: 1.375× (excellent calibration)
  - Sample variance: 0.240 (480× better diversity)

**Finding**: Geometric regularization prevents mode collapse and achieves calibrated rare mode coverage on real images.

### E9: CelebA with Rare Attributes (In Progress)

**Goal**: Validate on high-resolution face images with rare attribute combinations.

**Dataset**: CelebA 64×64 with rare attribute combination (e.g., Male + Eyeglasses + Bald ≈ 1-2% natural frequency)

**Models**: ImageVAE, ImageGAAE (convolutional architectures)

**Training**:
```bash
# VAE
python train_celeba.py --model-type vae --beta 1.0 --image-size 64 --latent-dim 128 --epochs 50

# GA-AE
python train_celeba.py --model-type ga-ae --lambda-grass 0.1 --lambda-entropy 0.01 --image-size 64 --latent-dim 128 --epochs 50
```

**Status**: Dataset loader and CNN architectures implemented, ready for training.

## Results

Results are saved in `results/<experiment_name>/`:

- `metrics.jsonl`: Training metrics logged at each checkpoint
- `config.json`: Experiment configuration
- `model_epoch_X.pt`: Model checkpoints
- `plots/`: Generated figures

## Diagnostic Metrics

### Rare Mode Evaluation

- **Rare Mode Rate (RMR)**: Fraction of generated samples in rare class
  - For MNIST: RMR = (# generated digit 9) / 2000
  - Target: 2% (matching training frequency)
- **Rare Mode Lift (RML)**: RMR / target_ratio
  - RML = 1.0× indicates perfect calibration
  - RML >> 1 indicates overproduction (e.g., VAE mode collapse)
  - RML << 1 indicates underproduction
- **Rare Recall@N**: Fraction of test rare samples covered
  - Recall = (# gen rare) / (# test rare)
  - Measures coverage rather than calibration

### Sample Quality

- **Sample Variance**: var(generated images)
  - Measures diversity across generated samples
  - Low variance (< 0.001) indicates mode collapse
  - High variance (> 0.1) indicates healthy diversity
- **Reconstruction Loss**: MSE on test set
  - Validates on-manifold performance
- **Energy Distance**: Statistical distance between real and generated distributions
  - Used for 2D Gaussian experiments

## Methodology

### Core Idea

The autoencoder trap: good reconstruction ≠ good generation. Standard autoencoders optimize reconstruction loss on-manifold (conditioned on real data) but provide no guarantees off-manifold (sampling from prior). This leads to mode collapse and failure to capture rare modes.

**Our Solution**: Explicitly regularize the geometry of the decoder's tangent space using:
1. **Grassmann Spread Loss**: Repels decoder tangent k-blades on the Grassmann manifold
2. **Blade Entropy Loss**: Maximizes diversity across multi-grade volumes (k=2, 4, 8)

### Loss Functions

**GA-AE Loss**:
```
L = L_recon + λ_grass * L_grass - λ_entropy * H_blade
```

Where:
- `L_recon`: MSE reconstruction loss
- `L_grass`: Grassmann spread loss = E[sim_Grass(blade_k(D, z_i), blade_k(D, z_j))]
  - Penalizes similarity between decoder tangent k-blades at different latent points
  - Computed via: sim = sqrt(det(U_i^T U_j U_j^T U_i)) where U_i, U_j are orthonormalized decoder Jacobian frames
- `H_blade`: Blade entropy = -Σ p_k log(p_k)
  - Encourages balanced expansion across k-dimensional scales
  - p_k = (s_k + δ) / Σ(s_k' + δ) where s_k = E[exp(log_vol_k)]
- Typical hyperparameters: λ_grass = 0.1, λ_entropy = 0.01

**VAE Loss** (baseline):
```
L = L_recon + β * KL(q(z|x) || p(z))
```

Where:
- `KL`: KL divergence between posterior and prior N(0, I)
- β: KL weighting (typically 1.0)

### Jacobian Computation

Decoder Jacobian-vector products computed via:
1. Sample k random orthonormal directions W_k in latent space (via QR decomposition of Gaussian samples)
2. Compute J_D(z) @ W_k using finite differences: (D(z + ε*w) - D(z)) / ε
3. Orthonormalize resulting frame: U = qf(J_D(z) @ W_k)
4. Compute Grassmann similarity or k-volumes via Gram determinants

### Computational Efficiency

- Geometric losses computed on subsampled pairs/batches during training
- Grassmann spread: typically 4-16 pairs per batch
- Blade entropy: 16-32 samples for expectation
- Finite difference approximation (ε=1e-4) for MLP decoders
- Total overhead: ~10-20% vs standard autoencoder training

## Citation

If you use this code, please cite:

```bibtex
@article{sudjianto2025geometric,
  title={Escaping the Autoencoder Trap: Grassmannian Tangent-Space Regularization for Tail Coverage},
  author={Sudjianto, Agus},
  year={2025}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on the repository.
