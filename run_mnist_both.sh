#!/bin/bash

# Run both MNIST experiments: VAE and GA-AE
PYTHON=~/jupyterlab/ga_verify/venv/bin/python

mkdir -p results/mnist_experiments

echo "================================================================================"
echo "Starting MNIST Experiments: VAE and GA-AE"
echo "================================================================================"
echo ""

# Train VAE first
echo "Step 1/2: Training VAE (β=1.0)..."
echo "--------------------------------------------------------------------------------"
$PYTHON train_mnist.py \
  --model-type vae \
  --beta 1.0 \
  --epochs 50 \
  --batch-size 256 \
  --lr 1e-3 \
  --rare-class 9 \
  --rare-ratio 0.02 \
  --download

if [ $? -eq 0 ]; then
  echo ""
  echo "✓ VAE training complete!"
  echo ""

  # Train GA-AE
  echo "Step 2/2: Training GA-AE..."
  echo "--------------------------------------------------------------------------------"
  $PYTHON train_mnist.py \
    --model-type ga-ae \
    --lambda-grass 0.1 \
    --lambda-entropy 0.01 \
    --epochs 50 \
    --batch-size 256 \
    --lr 1e-3 \
    --rare-class 9 \
    --rare-ratio 0.02

  if [ $? -eq 0 ]; then
    echo ""
    echo "✓ GA-AE training complete!"
    echo ""
    echo "================================================================================"
    echo "Both experiments complete!"
    echo "================================================================================"
    echo ""
    echo "Results:"
    echo "  VAE:   results/mnist_experiments/vae/seed_0/final_metrics.json"
    echo "  GA-AE: results/mnist_experiments/ga-ae/seed_0/final_metrics.json"
    echo ""
    echo "View tensorboard:"
    echo "  tensorboard --logdir results/mnist_experiments/"
  else
    echo ""
    echo "✗ GA-AE training failed"
    exit 1
  fi
else
  echo ""
  echo "✗ VAE training failed"
  exit 1
fi
