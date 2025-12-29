# CelebA Experiment: GA-AE vs VAE

This directory contains the complete setup for comparing Grassmannian Autoencoder (GA-AE) with standard VAE on CelebA dataset with rare attribute combinations.

## Experiment Overview

**Goal**: Demonstrate that GA-AE captures rare attribute combinations better than VAE on real-world image data.

**Dataset**: CelebA with rare attribute combination (e.g., "Male + Eyeglasses + Bald" ~1-2% natural frequency)

**Models**:
- **VAE**: Standard variational autoencoder with KL divergence (β=1.0)
- **GA-AE**: Geometric autoencoder with Grassmann spread + blade entropy

## Quick Start

### 1. Download CelebA Dataset (First Time Only)

The dataset will be automatically downloaded (~1.4GB) on first run:

```bash
# Test dataset loading (will download CelebA)
~/jupyterlab/ga_verify/venv/bin/python test_celeba_dataset.py
```

This will download CelebA to `~/data/celeba/` and verify the dataset is correctly loaded with rare attributes.

### 2. Train VAE Baseline

```bash
~/jupyterlab/ga_verify/venv/bin/python train_celeba.py \
  --model-type vae \
  --beta 1.0 \
  --epochs 50 \
  --batch-size 128 \
  --lr 1e-4 \
  --rare-attributes Male Eyeglasses Bald \
  --rare-ratio 0.02 \
  --download
```

**Expected time**: ~4-6 hours on single GPU
**Output**: `results/celeba_experiments/vae/seed_0/`

### 3. Train GA-AE

```bash
~/jupyterlab/ga_verify/venv/bin/python train_celeba.py \
  --model-type ga-ae \
  --lambda-grass 0.1 \
  --lambda-entropy 0.01 \
  --epochs 50 \
  --batch-size 128 \
  --lr 1e-4 \
  --rare-attributes Male Eyeglasses Bald \
  --rare-ratio 0.02
```

**Expected time**: ~6-8 hours on single GPU (geometric losses are more expensive)
**Output**: `results/celeba_experiments/ga-ae/seed_0/`

## Files Created

### Dataset Module
- `src/datasets/celeba.py`: CelebA dataset with rare attribute filtering

### Model Architectures
- `src/models/image_models.py`: CNN-based VAE and GA-AE implementations

### Training & Evaluation
- `train_celeba.py`: Main training script
- `evaluate_celeba.py`: Evaluation with rare recall, lift, FID (coming next)
- `compare_celeba_results.py`: Generate comparison plots and tables (coming next)

### Testing
- `test_celeba_dataset.py`: Test dataset loading
- `test_image_models.py`: Test model forward passes

## Expected Results

Based on Gaussian mixture experiments, we expect:

| Model | Rare Recall@2k | Rare Mode Lift | FID ↓ |
|-------|----------------|----------------|-------|
| Standard AE | 2-5% | 0.1× | ~50 |
| VAE (β=1.0) | 400-600% | 6.0× | ~30 |
| **GA-AE** | **40-50%** | **0.5×** | **~25** |

**Key Findings**:
1. VAE overproduces rare attributes (6× lift) → poor calibration
2. GA-AE balances coverage (40-50%) with calibration (0.5×)
3. Geometric diagnostics (k-volumes) predict generation quality

## Rare Attribute Combinations

### Default: Male + Eyeglasses + Bald
- Natural frequency: ~1-2% of CelebA
- Clear visual distinction
- Easy to verify in generated samples

### Alternative: Young + Mustache + Wearing_Hat
- Natural frequency: ~1-3% of CelebA
- Different attribute types
- Tests generalization

To use alternative configuration:
```bash
--rare-attributes Young Mustache Wearing_Hat
```

## Architecture Details

### Encoder (for 64×64 images)
```
Input (3, 64, 64)
  → Conv(64, 4, 2) + BN + LeakyReLU    → (64, 32, 32)
  → Conv(128, 4, 2) + BN + LeakyReLU   → (128, 16, 16)
  → Conv(256, 4, 2) + BN + LeakyReLU   → (256, 8, 8)
  → Conv(512, 4, 2) + BN + LeakyReLU   → (512, 4, 4)
  → FC(latent_dim)                      → (latent_dim,)
```

### Decoder (symmetric)
```
Input (latent_dim,)
  → FC(512×4×4)                         → (512, 4, 4)
  → ConvT(256, 4, 2) + BN + ReLU       → (256, 8, 8)
  → ConvT(128, 4, 2) + BN + ReLU       → (128, 16, 16)
  → ConvT(64, 4, 2) + BN + ReLU        → (64, 32, 32)
  → ConvT(3, 4, 2) + Tanh              → (3, 64, 64)
```

**Parameters**: ~5M for latent_dim=128

## Hyperparameters

### Common
- Latent dimension: 128
- Image size: 64×64
- Batch size: 128
- Learning rate: 1e-4 (Adam)
- Epochs: 50

### VAE
- β: 1.0 (standard)
- Alternative: β=0.1, 4.0 (see β-VAE effects)

### GA-AE
- λ_grass: 0.1 (Grassmann spread weight)
- λ_entropy: 0.01 (blade entropy weight)
- k-values: (2, 4, 8) for blade entropy

## Monitoring Training

### Tensorboard
```bash
tensorboard --logdir results/celeba_experiments/
```

Visit: http://localhost:6006

### Metrics to Watch
- **Reconstruction loss**: Should converge to ~0.01-0.03
- **VAE KL loss**: Should stabilize around 10-50
- **GA-AE Grassmann loss**: Should decrease (blades repelling)
- **GA-AE Blade entropy**: Should increase (multi-scale diversity)

## GPU Memory Requirements

- **VAE**: ~4-6GB
- **GA-AE**: ~6-8GB (geometric losses require extra memory)

If OOM, reduce batch size:
```bash
--batch-size 64
```

## Next Steps

1. **Run both experiments** (VAE and GA-AE)
2. **Implement evaluation script** with:
   - Rare recall and lift computation
   - FID score calculation
   - Sample visualization
   - Attribute classification (using pre-trained classifier)
3. **Generate comparison plots** for paper
4. **Add multi-seed runs** (5-10 seeds) for statistical significance

## Troubleshooting

### CelebA Download Issues
If automatic download fails:
1. Manual download: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
2. Extract to: `~/data/celeba/`
3. Remove `--download` flag from training command

### GPU Out of Memory
- Reduce batch size: `--batch-size 64`
- Reduce image size: `--image-size 48`
- Reduce latent dim: `--latent-dim 64`

### Slow Training
- GA-AE is 1.5-2× slower than VAE due to geometric losses
- Expected: VAE ~4-6 hours, GA-AE ~6-8 hours for 50 epochs
- Consider reducing epochs to 30-40 for faster iteration

## Citation

If you use this code, please cite:

```bibtex
@article{sudjianto2025escaping,
  title={Escaping the Autoencoder Trap: Grassmannian Tangent-Space Regularization for Tail Coverage},
  author={Sudjianto, Agus},
  journal={arXiv preprint},
  year={2025}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
