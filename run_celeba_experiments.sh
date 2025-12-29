#!/bin/bash

# Run CelebA experiments: VAE vs GA-AE
# This script trains both models sequentially or in parallel

PYTHON=~/jupyterlab/ga_verify/venv/bin/python
RARE_ATTRS="Male Eyeglasses Bald"
EPOCHS=50
BATCH_SIZE=128

echo "================================================================================"
echo "CelebA Experiments: GA-AE vs VAE"
echo "================================================================================"
echo "Rare attributes: $RARE_ATTRS"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "================================================================================"

# Check if GPU is available
$PYTHON -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

# Ask user which experiment to run
echo ""
echo "Which experiment would you like to run?"
echo "  1) VAE only"
echo "  2) GA-AE only"
echo "  3) Both (sequential)"
echo "  4) Test dataset loading first"
read -p "Enter choice [1-4]: " choice

case $choice in
  1)
    echo ""
    echo "================================================================================"
    echo "Training VAE (β=1.0)..."
    echo "================================================================================"
    $PYTHON train_celeba.py \
      --model-type vae \
      --beta 1.0 \
      --epochs $EPOCHS \
      --batch-size $BATCH_SIZE \
      --lr 1e-4 \
      --rare-attributes $RARE_ATTRS \
      --rare-ratio 0.02 \
      --download
    ;;

  2)
    echo ""
    echo "================================================================================"
    echo "Training GA-AE..."
    echo "================================================================================"
    $PYTHON train_celeba.py \
      --model-type ga-ae \
      --lambda-grass 0.1 \
      --lambda-entropy 0.01 \
      --epochs $EPOCHS \
      --batch-size $BATCH_SIZE \
      --lr 1e-4 \
      --rare-attributes $RARE_ATTRS \
      --rare-ratio 0.02 \
      --download
    ;;

  3)
    echo ""
    echo "================================================================================"
    echo "Training Both Models (VAE first, then GA-AE)..."
    echo "================================================================================"

    echo ""
    echo "Step 1/2: Training VAE..."
    $PYTHON train_celeba.py \
      --model-type vae \
      --beta 1.0 \
      --epochs $EPOCHS \
      --batch-size $BATCH_SIZE \
      --lr 1e-4 \
      --rare-attributes $RARE_ATTRS \
      --rare-ratio 0.02 \
      --download

    if [ $? -eq 0 ]; then
      echo ""
      echo "VAE training complete!"
      echo ""
      echo "Step 2/2: Training GA-AE..."
      $PYTHON train_celeba.py \
        --model-type ga-ae \
        --lambda-grass 0.1 \
        --lambda-entropy 0.01 \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr 1e-4 \
        --rare-attributes $RARE_ATTRS \
        --rare-ratio 0.02
    else
      echo "VAE training failed. Skipping GA-AE."
      exit 1
    fi
    ;;

  4)
    echo ""
    echo "================================================================================"
    echo "Testing CelebA Dataset Loading..."
    echo "================================================================================"
    $PYTHON test_celeba_dataset.py
    echo ""
    echo "Dataset test complete! You can now train models with options 1-3."
    ;;

  *)
    echo "Invalid choice. Exiting."
    exit 1
    ;;
esac

echo ""
echo "================================================================================"
echo "Experiment(s) Complete!"
echo "================================================================================"
echo "Results saved to: results/celeba_experiments/"
echo ""
echo "To view tensorboard logs:"
echo "  tensorboard --logdir results/celeba_experiments/"
echo ""
echo "Next steps:"
echo "  1. Run evaluation script (coming next)"
echo "  2. Generate comparison plots"
echo "  3. Add multi-seed runs for statistical significance"
echo "================================================================================"
