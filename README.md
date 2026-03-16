# Diffusion Neuron

Conditional next-frame video prediction using live human brain cells (CL1, Cortical Labs) as a biological bottleneck. Frames are encoded into electrical stimulation, sent to real neurons over UDP, and the spike responses are decoded into a predicted next frame.

---

## Architecture

```
frame_t + class
      |
      v
 StimEncoder     CNN + REINFORCE policy (3 rounds: full / spatial / color)
      |
      |  freqs + amps (64 channels, UDP)
      v
  CL1 Cells      Real neurons
      |
      |  spike counts (64 channels x 3 rounds = 177 values)
      v
 SpikeDecoder    Spatial spike grid -> Conv2d -> 3x bilinear upsample
      |
      v
 predicted frame_t+1  (64x64 RGB)
```

### Encoder

CNN that outputs stimulation parameters (frequency + amplitude) per active channel. Non-differentiable through the biological bottleneck, so the encoder is trained with REINFORCE using SSIM as the reward.

Three rounds per frame, each with a different preprocessed input:

| Round | Mode | Input | Channels |
|-------|------|-------|----------|
| 1 | `full` | Raw RGB | 42 |
| 2 | `spatial` | Sobel edge map | 21 |
| 3 | `color` | YUV chroma | 21 |

### Decoder

177 spike values are reshaped into a `(3, 8, 8)` spatial grid, processed through Conv2d layers, then upsampled 8->16->32->64 via bilinear interpolation. GroupNorm throughout to avoid grey collapse at batch size 1. No class conditioning; output is driven purely by spike patterns.

### Feedback

After each step, feedback is packed into the next stim message on dedicated channels:
- SSIM >= 0.6: synchronous burst on 8 positive channels
- SSIM <= 0.4: chaotic noise on 8 negative channels
- Otherwise: minimum safe stimulus

### Channel Layout

| Channels | Role |
|----------|------|
| `[0, 4, 7, 56, 63]` | Dead (hardware) |
| Active[0:42] | Encoder (full / spatial / color) |
| Active[42:50] | Positive feedback |
| Active[50:58] | Negative feedback |

---

## Files

### Entry points

| File | Description |
|------|-------------|
| `train.py` | Main training loop. Encoder -> stim -> spikes -> decoder -> REINFORCE + MSE update. Checkpoints every 500 steps. |
| `infer.py` | Inference. Supports single frame, multi-frame, custom image, and full train/test set sweeps. Outputs side-by-side GIF (input / predicted) to `inference_out/`. |
| `download_ucf101.py` | Downloads UCF101.rar (~6.5 GB) and extracts the configured classes to `data/train/`. |
| `cl1_neural_interface.py` | Runs on the CL1 device. Receives stim packets, runs `create_stim_plan()`, collects spikes, replies. |

### Models

| File | Description |
|------|-------------|
| `models/encoder.py` | `StimEncoder` - CNN backbone with three policy heads. Outputs (mu, sigma) per channel for frequency and amplitude. |
| `models/decoder.py` | `SpikeDecoder` - Spatial spike grid -> Conv2d -> bilinear upsample to 64x64. No class conditioning. |

### Protocol layer

| File | Description |
|------|-------------|
| `cl1/protocol.py` | UDP packet pack/unpack. STIM = 520 bytes, SPIKE = 264 bytes. Channels with amp=0 skip validation. |
| `cl1/interface.py` | `CL1Interface` - async UDP socket pair. Handles artifact rejection window and spike receive with timeout. |
| `cl1/channel_map.py` | `build_stim_arrays()` - maps encoder and feedback params into full 64-channel arrays. |

### Data

| File | Description |
|------|-------------|
| `data/ucf101_subset.py` | UCF101 loader. Returns `(frame_t, frame_t+1, class_idx, is_new_clip)`. 80/20 train/test split. Clips are shuffled but intra-clip frame order is preserved so inter-clip pauses fire correctly. |

### Utilities

| File | Description |
|------|-------------|
| `config.py` | All constants. |
| `feedback.py` | `compute_feedback()` - positive/negative stim arrays from SSIM value. |
| `utils/ssim.py` | Differentiable SSIM used as the REINFORCE reward. |
| `utils/session.py` | `SessionManager` - enforces the CL1 rest cycle (2.5hr train / 1hr rest). |

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+, PyTorch 2.0+. CUDA optional.

### Dataset

Run the download script to fetch and extract UCF101 automatically:

```bash
python download_ucf101.py
```

This downloads UCF101.rar (~6.5 GB), extracts the classes listed in `UCF101_CLASSES`, and deletes the RAR afterwards. To keep the RAR or extract different classes:

```bash
python download_ucf101.py --keep-rar --classes IceDancing Biking Basketball
```

On Windows, UnRAR must be on your PATH. Download from https://www.rarlab.com/rar_add.htm.

Active classes are set in `config.py` via `UCF101_CLASSES`.

---

## Running

### Hardware

On the CL1 device:
```bash
python cl1_neural_interface.py
```

On the training machine:
```bash
python train.py --cl1-host cl1-2544-122.corticalcloud
```

Training resumes from `checkpoints/latest.pt` automatically if it exists.

### Inference

Output goes to `inference_out/` by default. Override with `--out`.

```bash
# From dataset
python infer.py --class IceDancing --frames 30 --cl1-host cl1-2544-122.corticalcloud

# From a custom image
python infer.py --class Biking --image my_frame.png --cl1-host cl1-2544-122.corticalcloud

# Test set
python infer.py --test-set --cl1-host cl1-2544-122.corticalcloud

# Training set (capped)
python infer.py --train-set --class IceDancing --max-samples 30 --cl1-host cl1-2544-122.corticalcloud
```

---

## Training metrics

| Metric | Meaning |
|--------|---------|
| `ssim` | Structural similarity between predicted and actual next frame. Higher is better. |
| `mse` | Pixel-level mean squared error. Lower is better. |
| `enc_loss` | REINFORCE loss (-log_prob x advantage). Negative means the encoder took actions that beat the baseline. |
| `baseline` | Exponential moving average of SSIM (decay 0.99). Tracks rolling average reward. |

---

## Why this problem is hard

In standard RL, the policy selects from a discrete or low-dimensional action space and the reward signal is relatively unambiguous. Here the "action" is a 64x64 RGB image (roughly 12,000 continuous values) and the reward (SSIM) is a weak pixel-level similarity score that happily rewards predicting a blurry mean frame.

Several compounding problems:

**The bottleneck is non-differentiable.** Gradients cannot flow through the biological cells. The encoder is trained with REINFORCE, but it is not trying to elicit a fixed response. The cells are biological and their outputs change over time as they adapt to stimulation. The encoder's role is closer to a catalyst: continuously adapting its inputs to the cells' current state, providing structured stimulation that gives the cells useful signal to respond to. The cells and encoder co-adapt, but with no direct gradient signal the encoder can only guide this through reward-shaped trial and error, which is slow and high-variance.

**The spike signal is low-dimensional and noisy.** 177 spike counts (59 channels x 3 rounds) must carry enough information to reconstruct a 64x64 frame. The cells are biological; the same stimulation does not produce identical spikes each time. This noise looks identical to the decoder as meaningful variation, making it hard to learn a stable mapping.

**The decoder can cheat.** MSE loss rewards predicting the mean of the training distribution. A decoder with enough capacity can achieve reasonable SSIM purely by memorizing per-class average frames, completely ignoring the spike input. The noise ablation (`ablation_noise.py`) quantifies exactly how much SSIM is available for free this way. Real CL1 training must beat that ceiling to prove the spikes are contributing.

**The reward is delayed and indirect.** Feedback to the cells (positive/negative stim channels) is based on SSIM from the previous step. The cells are not being told what to do; they are being shaped over many steps toward responses that correlate with better predictions, which is a much weaker signal than supervised learning.

The result is that progress is slow, noisy, and difficult to interpret. Small SSIM improvements may reflect the decoder overfitting rather than the cells learning. The ablation workflow exists to separate these two effects.

---

## Ablations and tuning

### Random noise ablation

`ablation_noise.py` trains the decoder with random noise in place of real CL1 spikes. This establishes a baseline: the SSIM a decoder can achieve purely by learning mean-frame statistics, with no biological signal.

```bash
python ablation_noise.py
```

Use `--ablation` in `infer.py` to run inference with the ablation checkpoint:

```bash
python infer.py --class IceDancing --frames 30 --ablation
```

**Interpreting results:** if real CL1 training SSIM does not consistently exceed the noise ablation SSIM, the decoder is ignoring the spikes and learning from dataset statistics alone.

### Decoder capacity

The decoder capacity (`LR_DECODER`, channel widths in `models/decoder.py`) controls how much the decoder can learn independently of the spike signal.

**The core tradeoff:**
- Too large: the decoder has enough capacity to memorize mean frames per class, making spike content irrelevant. It will match or exceed the noise ablation SSIM without needing real spikes.
- Too small: the decoder cannot express the spike-to-frame mapping even when spikes carry real information. Outputs will be blurry regardless of spike quality.

**Tuning guide:**
1. Run `ablation_noise.py` to get a noise baseline SSIM
2. Train with real CL1 spikes and compare SSIM curves
3. If the curves are similar, reduce decoder capacity (`LR_DECODER`, channel counts) and retrain
4. If real training SSIM drops significantly below the noise baseline, the decoder is too smallincrease capacity
5. The target is the smallest decoder where real CL1 training measurably outperforms random noise

**Current settings** (tuned against noise ablation):
- `LR_DECODER = 1e-5` (10x lower than encoder)
- Channel progression: 3->32->64->32->16->8->3

### Class conditioning

The decoder does not receive the class label. This prevents it from learning per-class mean frames, which would let it produce plausible-looking outputs with no spike information. If outputs collapse to grey or become unrecognisable, re-adding class conditioning (`class_emb` + `class_proj` in `models/decoder.py`) gives the decoder a prior to work from while the encoder learns.
