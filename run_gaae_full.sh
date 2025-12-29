#!/bin/bash

# Train full GA-AE with geometric losses enabled
PYTHON=~/jupyterlab/ga_verify/venv/bin/python

echo "================================================================================"
echo "Training GA-AE with Geometric Regularization (Grassmann + Blade Entropy)"
echo "================================================================================"
echo ""

$PYTHON train_mnist.py \
  --model-type ga-ae \
  --lambda-grass 0.1 \
  --lambda-entropy 0.01 \
  --epochs 50 \
  --batch-size 256 \
  --lr 1e-3 \
  --rare-class 9 \
  --rare-ratio 0.02 \
  --output-dir results/mnist_experiments_full \
  --seed 0

if [ $? -eq 0 ]; then
  echo ""
  echo "✓ GA-AE training complete!"
  echo ""
  echo "Results saved to: results/mnist_experiments_full/ga-ae/seed_0/"
else
  echo ""
  echo "✗ GA-AE training failed"
  exit 1
fi
