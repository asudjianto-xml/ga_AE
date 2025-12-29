# CelebA Dataset Download Guide

## Issue: Google Drive Rate Limiting

The official CelebA dataset is hosted on Google Drive and is frequently rate-limited due to high traffic. You may encounter this error:

```
Too many users have viewed or downloaded this file recently. Please try accessing the file again later.
```

## Solutions

### Option 1: Manual Download (Recommended)

1. **Download from Browser:**
   - Visit: https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM
   - Download the file manually to `~/data/celeba/`

2. **Alternative Sources:**
   - Kaggle: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
   - Official Website: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

3. **Expected Structure:**
   ```
   ~/data/celeba/
   ├── img_align_celeba/
   │   ├── 000001.jpg
   │   ├── 000002.jpg
   │   └── ... (202,599 images)
   ├── list_attr_celeba.txt
   ├── list_bbox_celeba.txt
   ├── list_eval_partition.txt
   └── list_landmarks_align_celeba.txt
   ```

### Option 2: Use Kaggle API

```bash
# Install kaggle
pip install kaggle

# Setup API credentials (from kaggle.com/account)
mkdir -p ~/.kaggle
# Place your kaggle.json there

# Download dataset
kaggle datasets download -d jessicali9530/celeba-dataset
unzip celeba-dataset.zip -d ~/data/celeba/
```

### Option 3: Wait and Retry

Google Drive rate limits typically reset after 24 hours. You can:
```bash
# Retry test script tomorrow
python test_celeba_setup.py
```

### Option 4: Use Alternative Dataset

For validation of high-resolution image generation, consider:
- **FFHQ** (Flickr-Faces-HQ): 70k high-quality face images
- **LSUN Bedrooms**: Indoor scene images
- **ImageNet subset**: Natural images with class labels

## Testing After Download

Once data is downloaded, verify setup:

```bash
# Test dataset loading
python test_celeba_setup.py

# Should show:
# ✓ Dataset loading successful!
# ✓ ImageVAE working!
# ✓ ImageGAAE working!
```

## Running Experiments

After successful setup:

```bash
# Launch multi-seed experiments
bash run_celeba_multiseed.sh

# Monitor progress
tail -f celeba_multiseed.log
```

## Current Status

- ✓ CelebA dataset loader implemented
- ✓ CNN models (ImageVAE, ImageGAAE) ready
- ✓ Training scripts prepared
- ⚠ Dataset download rate-limited

**Recommendation**: Focus on MNIST multi-seed experiments first (currently running). Return to CelebA once:
1. MNIST results are complete and analyzed
2. Dataset download is successful
3. GPU time is available (~12-18 hours for full experiments)

## Contact

For CelebA dataset issues, refer to:
- CelebA Official: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- PyTorch Issues: https://github.com/pytorch/vision/issues

---

Last updated: December 29, 2025
