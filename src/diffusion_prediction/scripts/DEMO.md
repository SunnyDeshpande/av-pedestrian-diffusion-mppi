# Diffusion Prediction Demos

Standalone visualization scripts that demonstrate the diffusion pedestrian trajectory prediction models. No ROS, sensors, or data required.

## Prerequisites

```bash
conda activate cs588  # or adapt
cd ~/Adapt/cs_588_g10/src/diffusion_prediction
```

Trained weights must exist at:
- Single-agent: `models/diffusion/av2_pretrain_v1/ema_best.pt`
- Joint multi-agent: `models/diffusion/av2_joint_v1/ema_best.pt`

---

## 1. Animated Demo (`demo_live.py`)

Simulates pedestrians walking along trajectories and shows the model predicting their next 5 seconds in real-time. Outputs an MP4 or GIF video.

### Single-agent (4 scenarios: straight, turn, stop, swerve)

```bash
python scripts/demo_live.py \
    --model single \
    --weights ../../models/diffusion/av2_pretrain_v1/ema_best.pt \
    --output ../../figures/demo_live_single.mp4
```

### Joint multi-agent (2 scenarios: crossing, parallel walking)

```bash
python scripts/demo_live.py \
    --model joint \
    --weights ../../models/diffusion/av2_joint_v1/ema_best.pt \
    --output ../../figures/demo_live_joint.mp4
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | `single` or `joint` |
| `--weights` | required | Path to `.pt` checkpoint |
| `--output` | `figures/demo_live.mp4` | Output path (`.mp4` or `.gif`) |
| `--live` | off | Show in a live window instead of saving (needs X11) |
| `--device` | `cuda:0` | `cuda:0` or `cpu` |
| `--K` | `20` | Number of trajectory samples per pedestrian |
| `--fps` | `4` | Animation framerate (matches 4 Hz sensor rate) |

### Live window (requires X11 forwarding)

```bash
python scripts/demo_live.py \
    --model single \
    --weights ../../models/diffusion/av2_pretrain_v1/ema_best.pt \
    --live
```

---

## 2. Static Grid Demo (`visualize.py`)

Generates a grid of subplots, each showing one synthetic pedestrian scenario with observed history and the model's multi-modal prediction fan.

### Single-agent (9 scenarios)

```bash
python scripts/visualize.py \
    --model single \
    --weights ../../models/diffusion/av2_pretrain_v1/ema_best.pt \
    --demo \
    --output ../../figures/demo_single.png
```

### Joint multi-agent (4 scenes)

```bash
python scripts/visualize.py \
    --model joint \
    --weights ../../models/diffusion/av2_joint_v1/ema_best.pt \
    --demo \
    --output ../../figures/demo_joint.png
```

### With real validation data (requires `.npz` shards)

```bash
# Single-agent on val data
python scripts/visualize.py \
    --model single \
    --weights ../../models/diffusion/av2_pretrain_v1/ema_best.pt \
    --data /path/to/val_shards/ \
    --num-samples 16 --seed 42 \
    --output ../../figures/eval_single.png

# Joint multi-agent on val data
python scripts/visualize.py \
    --model joint \
    --weights ../../models/diffusion/av2_joint_v1/ema_best.pt \
    --data /path/to/joint_val_shards/ \
    --num-samples 9 --seed 42 \
    --output ../../figures/eval_joint.png
```

This mode also generates a `_hist.png` file with the minFDE error distribution.

---

## Reading the plots

| Element | Meaning |
|---------|---------|
| Blue solid line + dots | Observed pedestrian history (past 5s) |
| Blue square | Current position |
| Green dashed line | Ground truth future (where the person actually went) |
| Orange thin lines | K=20 predicted trajectory samples (multi-modal spread) |
| Orange thick line | Best mode (sample closest to the mean prediction) |

The spread of the orange fan shows the model's uncertainty. Wider fan = more uncertain about where the pedestrian will go.
