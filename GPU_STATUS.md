# GPU Configuration Status

## Hardware

- **GPU**: NVIDIA GB10
- **Memory**: 128.5 GB
- **CUDA Capability**: 12.1
- **Temperature**: ~53°C (normal operating range)

## Current Status

✓ **All experiments configured to use GPU automatically**

### Active Training

```
Process: train_mnist.py (GA-AE seed 1)
Status: Running
GPU Utilization: ~37%
Progress: Epoch 10/50
Speed: ~17-20 iterations/second
```

### GPU Configuration in Code

All training scripts automatically detect and use CUDA:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Files with GPU support:**
- ✓ `train_mnist.py` - MNIST experiments
- ✓ `train_celeba.py` - CelebA experiments
- ✓ `src/models/mnist_models.py` - Models move to device
- ✓ `src/models/image_models.py` - Models move to device

## Verification Commands

### Check GPU Usage
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Single check
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,temperature.gpu --format=csv
```

### Check Training Process
```bash
# See active Python processes
ps aux | grep python | grep train

# Monitor training logs
tail -f mnist_multiseed.log
tail -f celeba_multiseed.log
```

## Expected GPU Utilization

| Experiment | Batch Size | GPU Memory | Utilization | Speed |
|------------|------------|------------|-------------|-------|
| MNIST VAE  | 256 | ~2-3 GB | 30-40% | 140-170 it/s |
| MNIST GA-AE | 256 | ~2-3 GB | 30-40% | 17-20 it/s |
| CelebA VAE | 128 | ~8-12 GB | 50-70% | 5-10 it/s |
| CelebA GA-AE | 128 | ~8-12 GB | 50-70% | 3-5 it/s |

**Note**: GA-AE is slower due to geometric loss computations (Jacobian-vector products, QR decomposition, determinants).

## PyTorch CUDA Compatibility

Current setup shows a warning:
```
Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
```

**Impact**: Minor warning only. Training works correctly despite CUDA capability being 12.1 vs max 12.0. No performance degradation observed.

**If issues arise**: Update PyTorch to latest version with:
```bash
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Multi-Seed Experiment Timeline

### MNIST (Currently Running)
- 10 models × 50 epochs × ~2 min/model = **~1.5-2 hours total**
- Running in background: `nohup bash run_mnist_multiseed.sh`
- Log file: `mnist_multiseed.log`

### CelebA (Pending Data Download)
- 6 models × 50 epochs × ~2-3 hours/model = **~12-18 hours total**
- Requires: CelebA dataset download (~1.5GB)
- Will run: `nohup bash run_celeba_multiseed.sh`

## Monitoring GPU Health

### Temperature
- Current: ~53°C
- Safe range: < 85°C
- Throttling: Occurs at ~90°C

### Memory
- MNIST: < 5% of 128GB (very light load)
- CelebA: ~10-15% expected
- Available headroom: Excellent

### Power
```bash
# Check power draw
nvidia-smi --query-gpu=power.draw,power.limit --format=csv
```

## Troubleshooting

### If GPU not detected:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
```

### If out of memory:
- Reduce batch size in training scripts
- MNIST: batch_size=256 → 128
- CelebA: batch_size=128 → 64

### Force CPU (not recommended):
```bash
CUDA_VISIBLE_DEVICES="" python train_mnist.py ...
```

---

**Last updated**: December 29, 2025
**Status**: ✓ All systems operational
