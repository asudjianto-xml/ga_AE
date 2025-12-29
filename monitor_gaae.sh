#!/bin/bash

# Monitor GA-AE training progress
OUTPUT_FILE="/tmp/claude/-home-asudjianto-jupyterlab/tasks/be4414e.output"
LOG_FILE="results/mnist_experiments_full/ga-ae_training.log"

echo "================================================================================"
echo "Monitoring GA-AE Training Progress"
echo "================================================================================"
echo ""

# Copy output to log file
cp "$OUTPUT_FILE" "$LOG_FILE" 2>/dev/null

# Show last 30 lines
echo "Latest progress:"
echo "--------------------------------------------------------------------------------"
tail -30 "$OUTPUT_FILE" | grep -E "Epoch|loss=|Train Loss|Val Loss|Saved|complete"
echo ""
echo "================================================================================"
echo "Full log: $LOG_FILE"
echo "To monitor live: tail -f $LOG_FILE"
echo "================================================================================"
