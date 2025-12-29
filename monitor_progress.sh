#!/bin/bash
# Monitor experiment progress

echo "=========================================="
echo "Experiment Progress Monitor"
echo "=========================================="
echo ""

# Check running processes
echo "Running Python processes:"
ps aux | grep "run_experiments.py" | grep -v grep
echo ""

# Check latest log
if [ -f full_experiments.log ]; then
    echo "Last 20 lines of experiment log:"
    echo "------------------------------------------"
    tail -20 full_experiments.log
    echo ""
fi

# Check completed experiments
echo "Completed experiments:"
echo "------------------------------------------"
for exp in e1_ae_trap e2_tail_stress e3_vae_collapse e4_vae_tradeoff e5_baselines e6_teacher; do
    if [ -d "results/$exp" ]; then
        count=$(find results/$exp -name "model_epoch_200.pt" 2>/dev/null | wc -l)
        echo "  $exp: $count models completed (200 epochs)"
    fi
done
echo ""

# Check current epoch in latest metrics file
echo "Current training status:"
echo "------------------------------------------"
latest_metrics=$(find results -name "metrics.jsonl" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
if [ -n "$latest_metrics" ]; then
    echo "Latest metrics file: $latest_metrics"
    last_epoch=$(tail -1 "$latest_metrics" | python3 -c "import sys, json; print(json.loads(sys.stdin.read())['epoch'])" 2>/dev/null || echo "N/A")
    echo "Last recorded epoch: $last_epoch / 200"
fi
echo ""

# Disk usage
echo "Disk usage:"
du -sh results/
echo ""

echo "=========================================="
echo "To monitor in real-time, run:"
echo "  tail -f full_experiments.log"
echo "=========================================="
