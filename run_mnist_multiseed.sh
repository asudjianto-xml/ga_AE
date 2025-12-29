#!/bin/bash
#
# Run MNIST experiments with multiple seeds for statistical validation
# This script trains both GA-AE and VAE models with 5 different random seeds
#

PYTHON="/home/asudjianto/jupyterlab/ga_verify/venv/bin/python"
SCRIPT="train_mnist.py"
OUTPUT_DIR="results/mnist_multiseed"
EPOCHS=50
RARE_CLASS=9
RARE_RATIO=0.02
LATENT_DIM=32

# Seeds to run
SEEDS=(0 1 2 3 4)

echo "========================================================================"
echo "MNIST Multi-Seed Experiments for ICML"
echo "========================================================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "Seeds: ${SEEDS[@]}"
echo "Epochs: ${EPOCHS}"
echo "Rare class: ${RARE_CLASS} (${RARE_RATIO}% training frequency)"
echo "========================================================================"
echo ""

# Create output directory
mkdir -p ${OUTPUT_DIR}

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
            --rare-class ${RARE_CLASS} \
            --rare-ratio ${RARE_RATIO} \
            --epochs ${EPOCHS} \
            --batch-size 256 \
            --lr 1e-3 \
            --seed ${seed} \
            --output-dir ${output_path} \
            --data-root ~/data
    else
        ${PYTHON} ${SCRIPT} \
            --model-type ga-ae \
            --lambda-grass 0.1 \
            --lambda-entropy 0.01 \
            --latent-dim ${LATENT_DIM} \
            --rare-class ${RARE_CLASS} \
            --rare-ratio ${RARE_RATIO} \
            --epochs ${EPOCHS} \
            --batch-size 256 \
            --lr 1e-3 \
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
echo "All Experiments Complete!"
echo "========================================================================"
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Run aggregation script: python aggregate_mnist_results.py"
echo "  2. Generate comparison plots"
echo "  3. Update paper with multi-seed results (mean ± std)"
echo "========================================================================"
