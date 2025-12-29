#!/bin/bash
#
# Run CelebA experiments with multiple seeds
# Uses rare attribute combination: Male + Eyeglasses + Bald (~1-2% natural frequency)
#

PYTHON="/home/asudjianto/jupyterlab/ga_verify/venv/bin/python"
SCRIPT="train_celeba.py"
OUTPUT_DIR="results/celeba_multiseed"
EPOCHS=50
IMAGE_SIZE=64
LATENT_DIM=128
RARE_RATIO=0.02

# Rare attribute combination
RARE_ATTRS="Male Eyeglasses Bald"

# Seeds to run (fewer for CelebA due to longer training time)
SEEDS=(0 1 2)

echo "========================================================================"
echo "CelebA Multi-Seed Experiments for ICML"
echo "========================================================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "Seeds: ${SEEDS[@]}"
echo "Epochs: ${EPOCHS}"
echo "Image size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "Latent dim: ${LATENT_DIM}"
echo "Rare attributes: ${RARE_ATTRS}"
echo "Rare ratio: ${RARE_RATIO}% in training"
echo "========================================================================"
echo ""

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Check if GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. GPU may not be available."
    echo "CelebA training will be very slow without GPU."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Function to run a single experiment
run_experiment() {
    local model_type=$1
    local seed=$2
    local output_path="${OUTPUT_DIR}/${model_type}/seed_${seed}"

    echo ""
    echo "--------------------------------------------------------------------"
    echo "Running ${model_type} with seed ${seed}"
    echo "Output: ${output_path}"
    echo "--------------------------------------------------------------------"

    if [ "${model_type}" == "vae" ]; then
        ${PYTHON} ${SCRIPT} \
            --model-type vae \
            --beta 1.0 \
            --latent-dim ${LATENT_DIM} \
            --image-size ${IMAGE_SIZE} \
            --rare-attributes ${RARE_ATTRS} \
            --rare-ratio ${RARE_RATIO} \
            --epochs ${EPOCHS} \
            --batch-size 128 \
            --lr 1e-4 \
            --num-workers 4 \
            --seed ${seed} \
            --output-dir ${output_path} \
            --data-root ~/data
    else
        ${PYTHON} ${SCRIPT} \
            --model-type ga-ae \
            --lambda-grass 0.1 \
            --lambda-entropy 0.01 \
            --latent-dim ${LATENT_DIM} \
            --image-size ${IMAGE_SIZE} \
            --rare-attributes ${RARE_ATTRS} \
            --rare-ratio ${RARE_RATIO} \
            --epochs ${EPOCHS} \
            --batch-size 128 \
            --lr 1e-4 \
            --num-workers 4 \
            --seed ${seed} \
            --output-dir ${output_path} \
            --data-root ~/data
    fi

    local exit_code=$?
    if [ ${exit_code} -eq 0 ]; then
        echo "✓ ${model_type} seed ${seed} completed successfully"
    else
        echo "✗ ${model_type} seed ${seed} failed with exit code ${exit_code}"
    fi

    return ${exit_code}
}

# Track start time
start_time=$(date +%s)

# Run all experiments
echo ""
echo "========================================================================"
echo "Phase 1: Training VAE (Baseline)"
echo "========================================================================"

for seed in "${SEEDS[@]}"; do
    run_experiment "vae" ${seed}
done

echo ""
echo "========================================================================"
echo "Phase 2: Training GA-AE (Our Method)"
echo "========================================================================"

for seed in "${SEEDS[@]}"; do
    run_experiment "ga-ae" ${seed}
done

# Calculate elapsed time
end_time=$(date +%s)
elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))

echo ""
echo "========================================================================"
echo "All CelebA Experiments Complete!"
echo "========================================================================"
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Run aggregation script: python aggregate_celeba_results.py"
echo "  2. Generate sample visualizations"
echo "  3. Update paper with CelebA results"
echo "========================================================================"
