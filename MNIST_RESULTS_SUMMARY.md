# MNIST Experiments: Initial Results

## Experiment Setup

**Dataset**: MNIST with class imbalance
- Rare class: Digit 9
- Rare ratio: 2% in training set
- Total training: 50,000 samples (1,000 rare)
- Total test: 10,000 samples (1,009 rare)

**Models Trained**:
1. VAE (β=1.0) - ✅ Complete
2. Standard AE - 🔄 Training (~40% done)

---

## VAE Results (β=1.0)

**Training**:
- Epochs: 50
- Final train loss: 0.0157
- Final recon loss: 0.0157
- Final KL loss: Very small (converged)
- Best val loss: 0.2746

**Generation Metrics**:
- Test rare count: 1009 (10.09% of test set)
- Gen rare count: 2000 (from 2000 generated samples)
- **Rare recall@2000**: 198.22% (all generated classified as rare!)
- **Rare lift**: 50.00× (severe overproduction)

**Analysis**:
⚠️ The 1-NN classifier appears to be broken - it's classifying ALL generated samples as rare class 9. This needs fixing.

**Expected behavior** (based on Gaussian experiments):
- VAE should show 6× lift (overproduction)
- Not 50× (which suggests classifier issue)

---

## Standard AE Results

🔄 **Currently training** (Epoch ~12/30)

Early metrics:
- Train loss: ~0.0225
- Recon loss: ~0.0225
- Training progressing smoothly on GPU

---

## Next Steps

### 1. Fix Rare Class Classifier ⚠️ **URGENT**
The 1-NN classifier is classifying everything as rare. Possible issues:
- Distance computation error
- Test set loading issue
- Label matching problem

### 2. Complete Standard AE Training
- ETA: ~10 minutes
- Will provide baseline for comparison

### 3. Implement Proper Evaluation
- Use pre-trained MNIST classifier (more reliable than 1-NN)
- Compute FID scores
- Generate sample visualizations

### 4. Re-enable Geometric Losses
- Fix BatchNorm issues in GA-AE
- Train full GA-AE with geometric regularization
- Compare: Standard AE vs VAE vs GA-AE

---

## Expected Final Results (Based on Gaussian Experiments)

| Model | Rare Recall@2k | Rare Mode Lift | Interpretation |
|-------|----------------|----------------|----------------|
| Standard AE | 2-5% | 0.10× | Ignores rare class |
| VAE (β=1.0) | 400-600% | 6.0× | Overproduces rare class |
| GA-AE | 40-50% | 0.50× | Balanced coverage |

---

## Files Generated

**Models**:
- `results/mnist_experiments/vae/seed_0/best_model.pt`
- `results/mnist_experiments/vae/seed_0/final_metrics.json`
- `results/mnist_experiments/ga-ae/seed_0/` (in progress)

**Logs**:
- `results/mnist_experiments/training.log` (VAE complete log)
- `results/mnist_experiments/ga-ae_training.log` (GA-AE progress)

**Tensorboard**:
- `results/mnist_experiments/vae/seed_0/logs/`
- `results/mnist_experiments/ga-ae/seed_0/logs/`

---

## Current Issues to Resolve

1. ⚠️ **1-NN classifier bug** - Classifying all samples as rare
2. 🔄 **Geometric losses disabled** in GA-AE due to BatchNorm issue
3. 📊 **Need visualization** - Generate sample images to visually verify

---

## Time Invested

- **Setup**: 30 min (dataset + models + training script)
- **VAE training**: ~1 hour (50 epochs)
- **GA-AE training**: ~10 min (in progress)
- **Total**: ~1.5 hours for initial MNIST validation

**vs CelebA**: Would have taken 10-14 hours + download issues

**Conclusion**: MNIST was the right choice for quick validation! 🎯
