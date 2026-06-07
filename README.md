# DeblurGAN — Image Deblurring with Generative Adversarial Networks

**Author:** Vaidehi  
**Repo:** [github.com/vaidehi-builds/DeblurGAN-DL](https://github.com/vaidehi-builds/DeblurGAN-DL)  
**Status:** Baseline training complete — analysis and loss modification in progress

---

## Project Overview

This project implements DeblurGAN, a conditional GAN-based model for removing motion blur from images. The goal is to train a generator network that takes a blurry image as input and outputs a sharp, deblurred version.

The project is structured as:
1. Train the baseline DeblurGAN model
2. Evaluate on test images (PSNR / SSIM)
3. Identify failure modes
4. Propose and implement a loss function modification
5. Compare baseline vs modified model

---

## Environment

| Component | Version |
|-----------|---------|
| OS | Windows 11 |
| Python | 3.10 |
| PyTorch | 2.5.1 |
| CUDA | 12.1 |
| GPU | NVIDIA RTX 4060 |

---

## Dataset

- **Source:** `blurred_sharp.zip` — 1151 blurry/sharp image pairs
- **Content:** Dashcam-style car footage with motion blur
- **Preprocessing:** Images combined into side-by-side pairs using `combine_flat.py`
- **Training path:** `D:\DeblurGAN_DL\blurred_sharp\combined\train`
- **Original blurry images:** `D:\DeblurGAN_DL\blurred_sharp\blurred`
- **Original sharp images:** `D:\DeblurGAN_DL\blurred_sharp\sharp`

---

## Training

### Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 300 |
| Batch size | 1 |
| Generator | ResNet-9 blocks |
| Discriminator | PatchGAN |
| GAN type | WGAN-GP |
| Content loss | Perceptual (VGG19) |
| Image size | 256 × 256 |
| GPU | RTX 4060 |
| Training time | ~7–8 hours |

### Training Command

```bash
python train.py --dataroot D:\DeblurGAN_DL\blurred_sharp\combined\train \
                --name experiment_name \
                --model content_gan \
                --which_model_netG resnet_9blocks \
                --which_direction AtoB \
                --lambda_A 100 \
                --dataset_mode aligned \
                --norm instance \
                --pool_size 0 \
                --no_dropout \
                --niter 150 \
                --niter_decay 150 \
                --gpu_ids 0
```

### Loss Curves

| Loss | Behavior |
|------|----------|
| G_L1 (Perceptual) | Dropped ~55% from epoch 1 to 300 |
| G_GAN | Steadily increased — generator learning to fool discriminator |
| D_real+fake | Collapsed to ~0.000 by epoch 2 |

Training curves saved as `training_curves.png`.

---

## Inference

### Command

```bash
python test.py --dataroot D:\DeblurGAN_DL\blurred_sharp\blurred \
               --name experiment_name \
               --model test \
               --dataset_mode single \
               --which_epoch latest \
               --how_many 50 \
               --results_dir D:\DeblurGAN_DL\results \
               --display_id 0 \
               --nThreads 0
```

### Results

Output images saved to:
```
D:\DeblurGAN_DL\results\experiment_name\test_latest\images\
```

Each image has a pair:
- `N_real_A.png` — original blurry input
- `N_fake_B.png` — model's deblurred output

---

## Evaluation

### Quantitative Results (Baseline)

| Metric | Score |
|--------|-------|
| PSNR | 10.51 dB |
| SSIM | 0.2608 |

Evaluation script: `eval.py`

```bash
python eval.py
```

### Qualitative Results

The baseline model outputs **near-gray images** across all checkpoints (epoch 50 through 300). No meaningful deblurring is visible.

---

## Failure Analysis

### Root Cause: Discriminator Collapse

The discriminator loss (`D_real+fake`) dropped to **~0.000 by epoch 2** and remained there for the remainder of training.

**What this means:**
- The discriminator became unable to distinguish real from fake images extremely early
- With no useful adversarial feedback, the generator had only the perceptual loss to guide it
- The generator converged to outputting near-constant gray images — a safe minimum that satisfies a collapsed discriminator

**Evidence:**
- All `fake_B` outputs are gray regardless of checkpoint (epoch 50, 100, 150, 300)
- PSNR of 10.51 dB is well below the expected range of 28–35 dB for a working model
- SSIM of 0.26 confirms near-zero structural similarity to ground truth

### Why WGAN-GP Failed Here

WGAN-GP is sensitive to the balance between generator and discriminator training. With the current setup, the discriminator collapsed before it could provide stable gradients, leaving the generator without meaningful adversarial supervision.

---

## Proposed Next Steps

Three candidate loss modifications to address discriminator collapse:

**Option 1 — Edge-Aware Loss**
Add a Sobel-based edge detection term to the content loss. Forces the generator to preserve sharp edges without relying on the discriminator signal.

**Option 2 — FFT Loss**
Add a frequency-domain loss term that penalizes differences in high-frequency components (edges and textures). Directly targets the sharpness deficit.

**Option 3 — Multi-Scale Loss**
Compute perceptual loss at multiple image scales, encouraging the generator to learn both global structure and fine detail.

*Decision pending discussion with supervisor.*

---

## Repository Structure

```
DeblurGAN/
├── train.py                  # Training entry point
├── test.py                   # Inference entry point
├── eval.py                   # PSNR/SSIM evaluation script
├── combine_flat.py           # Dataset preprocessing
├── training_curves.png       # Loss curves over 300 epochs
├── models/
│   ├── conditional_gan_model.py
│   ├── test_model.py
│   ├── networks.py
│   └── losses.py
├── data/
├── util/
└── checkpoints/
    └── experiment_name/      # Saved model weights (every 5 epochs)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `models/losses.py` | Loss function definitions — modification target |
| `models/conditional_gan_model.py` | Main training loop |
| `models/networks.py` | Generator and Discriminator architectures |
| `eval.py` | Computes PSNR/SSIM against ground truth |
| `training_curves.png` | Visual training analysis |

---

## Notes

- Windows multiprocessing fix: `--nThreads 0` required for inference on Windows
- Visdom display disabled with `--display_id 0` (no display server running)
- All 1151 image pairs used for training — no held-out validation split during training
- Checkpoints saved every 5 epochs from epoch 5 to 300
