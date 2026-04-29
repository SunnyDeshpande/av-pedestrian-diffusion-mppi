# Adapt -- Action Items & Next Steps

**Last updated:** 2026-04-28

---

## Current Status Summary

| Component | Status |
|---|---|
| MPPI controller (torch backend) | Complete, verified |
| AutoShield perception pipeline (LiDAR + RGBD + fusion) | Complete (from AutoShield) |
| Single-agent diffusion model (AV2 pretrain) | Complete -- 200 epochs, minFDE = 0.693 m |
| Joint multi-agent diffusion model (AV2 pretrain) | Complete -- 200 epochs, minFDE = 0.529 m |
| 3-mode prediction integration | Complete -- `single-default`, `single-diffusion`, `multi-diffusion` |
| MPPI prediction tensor consumption | Complete -- `prediction_source=predicted` wires tensor to MPPI |
| Unified launch file | Complete -- `adapt_prediction_launch.py` |
| GEM e4 rosbag data collection | **Not started** |
| Diffusion model finetuning on GEM data | **Not started** |
| On-vehicle closed-loop testing | **Not started** |
| Gazebo sim closed-loop testing | **Not started** |

---

## 1. Collect GEM e4 Rosbag Data (HIGH PRIORITY)

The diffusion models are pretrained on Argoverse 2 (urban crosswalk scenarios). Distribution shift to the UIUC highbay is the biggest deployment risk. Finetuning on self-collected data is the primary mitigation.

### What to record

**Target: 4-6 hours of rosbag across 2 recording days (different lighting).**

| Scenario | Events | Time | Description |
|---|---|---|---|
| Crossing | ~150 | ~2 hr | Confederate pedestrians cross perpendicular at 5/10/15/20 m standoff, varying speeds |
| Parallel-walk | ~80 | ~1.5 hr | Ped walks alongside vehicle, same or opposite direction, varying lateral offset |
| Stationary | ~50 | ~30 min | Ped stands in or near the ego path |
| Group | ~20 | ~30 min | 2-4 confederates walk together |

### Topics to record

```bash
ros2 bag record -o data/gem_bags/<date>_<scenario>_<n> \
    /fusion_pedestrian_position \
    /lidar_pedestrian_position \
    /rgbd_pedestrian_position \
    /vehicle_rpt \
    /pacmod/vehicle_speed_rpt \
    /tf /tf_static
```

### Recording setup

```bash
# Terminal 1: Sensor bringup
ros2 launch basic_launch sensor_init.launch.py vehicle_name:=e4

# Terminal 2: Perception (LiDAR + YOLO)
ros2 run adapt_full lidar_processing &
ros2 run yolo_person_detector rgbd_pedestrain_detector

# Terminal 3: Fusion
ros2 launch adapt_full lidar_camera_fusion_launch.py

# Terminal 4: Record
ros2 bag record -o data/gem_bags/<name> \
    /fusion_pedestrian_position /vehicle_rpt /tf /tf_static
```

Vehicle speed during recording: 1-4 m/s. Drive manually; PACMod engagement is NOT necessary.

### Diversity checklist

- [ ] Two recording days, different lighting conditions
- [ ] Multiple confederate clothing colors
- [ ] Vary intersection/standoff geometry every 3-5 events
- [ ] Include both slow-stroll and brisk-walk speeds
- [ ] Include hesitation/stop-and-go events at curbs

---

## 2. Finetune Diffusion Models on GEM Data

### Auto-labeling (no manual annotation)

The recorded `/fusion_pedestrian_position` IS the label. Replay bags through the tracker offline to extract (history, future) windows.

```bash
# Extract training windows from bags
python -m diffusion_prediction.scripts.extract_gem_windows \
    --bags data/gem_bags/ \
    --output data/gem_processed/
```

Expected yield: ~5,000-10,000 windows from 4-6 hours of bag.

Reserve ~20% of bags (from a different recording day) as a UIUC validation holdout.

### Finetune recipe

**Single-agent model:**
```bash
conda activate adapt
python -m diffusion_prediction.finetune \
    --pretrained models/diffusion/av2_pretrain_v1/ema_best.pt \
    --data data/gem_processed \
    --epochs 20 --lr 2e-5 --batch-size 32 \
    --device cuda:0 --run-name gem_single_v1
```

**Joint multi-agent model:**
```bash
python -m diffusion_prediction.finetune \
    --pretrained models/diffusion/av2_joint_v1/ema_best.pt \
    --data data/gem_processed \
    --epochs 20 --lr 2e-5 --batch-size 32 \
    --device cuda:0 --run-name gem_joint_v1
```

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| LR | 2e-5 constant |
| Epochs | 20 |
| Batch size | 32 |
| Start from | AV2 pretrained EMA weights |
| Optional | Freeze `ego_in` / `t_embed` if < 5,000 windows |
| Walltime | ~1 hour on RTX 3060 Laptop |

### Success criteria

- UIUC holdout minFDE-20 within 1.5x of AV2 val minFDE-20
- If FAIL: extend recording to 12-15 hours (~600 crossing events) and re-finetune

---

## 3. Tuning the Diffusion Model

### If predictions are too noisy / jittery

1. **Increase `min_history_count`** (default 5) -- requires more observed frames before predicting
2. **Reduce K** at runtime (20 -> 10) -- fewer samples, less mode diversity but more stable
3. **Tune sticky-mode threshold** -- the 1.5x cost ratio in `infer_node.py` controls when the best-mode selection switches. Raise to 2.0x for more stability, lower to 1.2x for faster mode switching
4. **Reduce DDIM steps** (10 -> 5) -- faster but slightly lower quality

### If latency exceeds budget (> 50 ms per tick)

Apply in order:
1. DDIM steps: 10 -> 5
2. d_model: 128 -> 96 (requires retraining)
3. K (runtime only): 20 -> 10
4. Last resort: distill to a 1-step student

### If distribution shift is too large

- Extend GEM recording to 12-15 hours
- Add speed augmentation during AV2 pretraining: randomly rescale pedestrian speed by 0.5-1.5x
- Consider domain-randomization augmentations (translation jitter, rotation noise)

---

## 4. Gazebo Simulation Testing

### Host-side setup (one-time)

```bash
mkdir -p ~/gem_simulation_ws/src
ln -s ~/UIUC/AVSE/POLARIS_GEM_Simulator ~/gem_simulation_ws/src/POLARIS_GEM_Simulator
cd ~/UIUC/AVSE/POLARIS_GEM_Simulator
bash setup/build_docker_image.sh
```

### Run closed-loop sim

**Container Terminal A -- sim:**
```bash
cd ~/UIUC/AVSE/POLARIS_GEM_Simulator && bash run_docker_container.sh
# Inside container:
cd ~/host/gem_simulation_ws && catkin_make && source devel/setup.bash
roslaunch gem_launch gem_init.launch world_name:=track1.world vehicle_name:=e4
```

**Container Terminal B -- MPPI bridge:**
```bash
bash run_docker_container.sh
# Inside container:
cd ~/host/gem_simulation_ws && source devel/setup.bash
roslaunch gem_mppi_sim mppi_sim.launch vehicle_name:=e4
```

---

## 5. On-Vehicle Deployment Testing

### Shadow mode (A/B comparison)

Run the diffusion predictor alongside the original predictor on remapped topics:

```bash
# Original predictor on canonical topics
ros2 run yolo_person_detector pedestrian_behaviour_predictor

# Diffusion predictor on remapped topics
ros2 run diffusion_prediction infer_node \
    --ros-args -r __ns:=/diffusion \
    -p weights:=models/diffusion/gem_joint_v1/ema_best.pt
```

Record both and compare ADE @ 5s on matched track IDs offline.

### Full deployment

```bash
ros2 launch adapt_full adapt_prediction_launch.py \
    prediction_mode:=multi-diffusion \
    controller:=mppi \
    weights:=models/diffusion/gem_joint_v1/ema_best.pt \
    enable_fusion:=true \
    enable_safety:=true
```

### Validation checklist

- [ ] Latency < 30 ms per inference tick (check node log warnings)
- [ ] `/pedestrian_predictions_tensor` publishes at ~10 Hz
- [ ] MPPI log shows `obs=M` matching the number of detected pedestrians
- [ ] No near-miss events (ego-ped distance < 1.5 m) that are worse than baseline
- [ ] TTC values are stable (no frame-to-frame jitter > 1 s)
- [ ] Vehicle swerves/slows appropriately for crossing pedestrians
- [ ] `ESS/K` stays in [0.05, 0.5] range

---

## 6. MPPI Tuning Guide

Do this in order. Only move on when the previous step passes.

### Step 0 -- Standalone test

```bash
python3 src/vehicle_drivers/mppi_controller/test/test_phase1.py
```

Must pass with mean |lateral error| < 0.05 m.

### Step 1 -- Tracking (no obstacles)

Target: |lateral error| < 0.5 m, speed converges in 3-5 s, ESS/K in [0.05, 0.5].

| Symptom | Knob | Direction |
|---|---|---|
| Lateral error bouncing | `mppi/w_pos` | Up (15 -> 30) |
| Heading lag on curves | `mppi/lookahead_m` | Down (8 -> 5) |
| Jerky steering | `mppi/sigma_steer` | Down (0.15 -> 0.08) |
| Sluggish speed response | `pid/kp` | Up (0.6 -> 0.9) |
| ESS/K < 0.05 | `mppi/lambda_` | Up (0.1 -> 0.5) |
| ESS/K saturating at 1.0 | `mppi/sigma_steer`, `mppi/sigma_accel` | Up |

### Step 2 -- Obstacle response

```bash
ros2 topic pub --once /fusion_pedestrian_position \
    std_msgs/msg/Int32MultiArray "{data: [10, 0]}"
```

| Symptom | Knob |
|---|---|
| Plows through pedestrian | `mppi/w_obs` up (150 -> 300); `mppi/clearance` up (3.0 -> 4.0) |
| Freezes far from obstacle | `mppi/w_obs` down; check clearance isn't > lane half-width |
| Swerves violently | `mppi/sigma_steer` down; `mppi/w_curv` up |

### Step 3 -- Real pedestrian detections

Watch for:
- Intermittent detections causing obstacle flicker -- consider buffering last N snapshots
- Noisy bearing from YOLO at distance -- tighten `matching_threshold` in `sensor_fusion_params.yaml`
- Integer quantization (~0.5-1 m noise) sits inside MPPI's 3 m clearance, so it's fine

---

## 7. Repo Hygiene

- [ ] Ensure `.gitignore` includes: `data/`, `models/`, `*.pt`, `*.ckpt`, `logs/`
- [ ] Verify with `git status` that no weights, datasets, or logs appear as untracked
- [ ] Keep `adapt_requirements.text` and `cs588_requirements.txt` up to date

---

## 8. Known Risks

| Risk | Severity | Mitigation |
|---|---|---|
| AV2 -> UIUC distribution shift | High | GEM finetune; extend recording if minFDE > 1.5x |
| Inference latency > 50 ms at M > 1 | Medium | Drop DDIM steps -> d_model -> K in order |
| Mode-flip jitter in predictions | Medium | Sticky-mode selection with 1.5x threshold |
| DDIM frame-to-frame stochasticity | Low | Seed initial noise per (track_id, tick_index) |
| Vehicle GPU differs from dev GPU | Medium | Re-benchmark latency on vehicle at deployment |
| GNSS heading incorrect on launch | Known | Relaunch or restart machine (per CLAUDE.md) |
