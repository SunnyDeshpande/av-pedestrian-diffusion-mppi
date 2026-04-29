# Adapt -- Project Description

**UIUC CS 588 (Autonomous Vehicle Systems) -- Group 10**
**Author:** Sunny Deshpande (sunnynd2)
**Vehicle:** Polaris GEM e4
**Framework:** ROS 2 (colcon), Python 3.12, PyTorch

---

## 1. What Is Adapt

Adapt replaces AutoShield's Stanley lateral controller on the UIUC Polaris GEM e4 with a **Model Predictive Path Integral (MPPI)** motion planner that reasons jointly about steering and acceleration, and natively accepts obstacle costs from a **diffusion-based pedestrian trajectory predictor**.

The system offers 3 selectable prediction modes (constant-velocity baseline, single-agent diffusion, joint multi-agent diffusion) wired to 2 controllers (MPPI or Stanley), giving a configurable pipeline from perception through control.

---

## 2. Full Pipeline

```
                    +-------------------+     +-----------------------+
                    |   Ouster OS1-128  |     |  OAK-D LR (RGB+D)    |
                    |  /ouster/points   |     | /oak/rgb/image_raw    |
                    |  PointCloud2      |     | /oak/stereo/image_raw |
                    +--------+----------+     +----------+------------+
                             |                           |
                             v                           v
                  +--------------------+     +-------------------------+
                  | lidar_processing   |     | rgbd_pedestrian_detector|
                  | DBSCAN clustering  |     | YOLOv11 + depth proj   |
                  | + human geometry   |     | COCO class 0 = person  |
                  +--------+-----------+     +----------+--------------+
                           |                            |
               /lidar_pedestrian_position   /rgbd_pedestrian_position
               Int32MultiArray              Int32MultiArray
               [dist_m, bear_deg, ...]      [dist_m, bear_deg, ...]
                           |                            |
                           +------------+---------------+
                                        |
                                        v
                             +---------------------+
                             | lidar_camera_fusion  |
                             | LiDAR 80% dist /    |
                             | cam 20% dist;       |
                             | cam 70% bearing /   |
                             | LiDAR 30% bearing   |
                             +----------+----------+
                                        |
                         /fusion_pedestrian_position
                         Int32MultiArray [d1,b1,d2,b2,...]
                         (multi-ped, polar, ego-frame, ~10 Hz)
                                        |
               +------------------------+------------------------+
               |                        |                        |
               v                        v                        v
    +--------------------+   +---------------------+   +-------------------+
    | PREDICTOR          |   | high_level_          |   | MPPI (raw mode)  |
    | (one of 3 modes)   |   | decision_node        |   | direct obstacle  |
    |                    |   | reads /ped_motion,    |   | /fusion_ped_pos  |
    | see Section 5      |   | /ped_ttc, /fusion_   |   | (legacy path)    |
    +--------+-----------+   | ped_pos              |   +-------------------+
             |               +----------+-----------+
             |                          ^
             |                reads /pedestrian_motion
             |                reads /pedestrian_ttc
             |
             +---> /pedestrian_motion (Twist)
             +---> /pedestrian_ttc (Float64)
             +---> /person_prediction (Marker)
             +---> /pedestrian_predictions_tensor (Float32MultiArray, M x 20 x 2)
                                        |
               +------------------------+
               |
    +----------v-----------+       +-------------------------+
    | CONTROLLER           |       | CONTROLLER              |
    | (if mppi)            |       | (if stanley)            |
    |                      |       |                         |
    | adapt_mppi_node      |       | stanley_controller_node |
    | prediction_source=   |       | Stanley cross-track +   |
    |   predicted: reads   |       | PID speed control       |
    |   /ped_pred_tensor   |       | No obstacle avoidance   |
    | K=600, H=30, 10 Hz   |       | (relies on high_level)  |
    +----------+-----------+       +----------+--------------+
               |                              |
               v                              v
    +---------------------------------------------+
    | PACMod2 Interface                            |
    | /pacmod/steering_cmd  (PositionWithSpeed)    |
    | /pacmod/accel_cmd     (SystemCmdFloat)       |
    | /pacmod/brake_cmd     (SystemCmdFloat)       |
    | /pacmod/shift_cmd     (SystemCmdInt)          |
    | /pacmod/global_cmd    (GlobalCmd)             |
    +----------------------+-----------------------+
                           |
                           v
                  +------------------+
                  | GEM e4 Vehicle   |
                  | Steering + Drive |
                  +------------------+
```

---

## 3. Repository Structure

```
cs_588_g10/
+-- docs/                           # This folder
+-- data/                           # Rosbags, processed datasets (gitignored)
+-- models/                         # Trained weights (gitignored)
+-- adapt_requirements.text         # Conda env recipe (py3.12, GPU torch)
+-- cs588_requirements.txt          # CPU-only dev env (py3.11)
+-- run_tmux.sh                     # Tmux multi-pane launcher
+-- src/
    +-- adapt_full/                 # Perception + fusion + safety + launch files
    |   +-- adapt_full/
    |   |   +-- adapt_lidar_processing.py       # LiDAR DBSCAN pedestrian detector
    |   |   +-- adapt_lidar_camera_fusion.py    # Sensor fusion node
    |   |   +-- adapt_high_level_command.py      # Safety state machine
    |   |   +-- adapt_safety_controller.py       # Emergency brake
    |   |   +-- adapt_stanley_controller.py      # Stanley controller (fallback)
    |   +-- launch/
    |   |   +-- adapt_prediction_launch.py       # ** Unified 3-mode launch **
    |   |   +-- adapt_mppi_launch.py             # MPPI-only launch
    |   |   +-- adapt_full_launch.py             # Full stack launch
    |   |   +-- stanley_controller_launch.py
    |   +-- config/                              # YAML param files
    |   +-- waypoints/                           # Track CSV files
    |
    +-- diffusion_prediction/       # ** Diffusion trajectory prediction **
    |   +-- diffusion_prediction/
    |   |   +-- model.py             # TrajectoryDenoiser (single-agent, 0.47M params)
    |   |   +-- model_joint.py       # JointTrajectoryDenoiser (multi-agent, 0.93M params)
    |   |   +-- ddpm.py              # Cosine schedule, DDIM sampling (single + joint)
    |   |   +-- infer_node.py        # ROS 2 inference node (3 prediction modes)
    |   |   +-- tracker.py           # Greedy Euclidean tracker with EMA smoothing
    |   |   +-- utils.py             # Message building, TTC, frame transforms
    |   |   +-- dataset.py           # AV2 single-agent dataloader
    |   |   +-- dataset_joint.py     # AV2 joint multi-agent dataloader
    |   |   +-- train.py             # Single-agent pretraining
    |   |   +-- train_joint.py       # Joint multi-agent pretraining
    |   |   +-- finetune.py          # GEM rosbag finetuning
    |   |   +-- eval.py              # Offline evaluation (ADE/FDE/miss-rate)
    |   +-- scripts/
    |       +-- preprocess_av2.py         # AV2 -> Parquet (single-agent)
    |       +-- preprocess_av2_joint.py   # AV2 -> Parquet (multi-agent scenes)
    |       +-- extract_gem_windows.py    # Rosbag -> training windows
    |       +-- bench_latency.py          # GPU latency benchmark
    |
    +-- yolo_person_detector/       # YOLO + original predictor
    |   +-- yolo_person_detector/
    |       +-- rgbd_pedestrain_detector.py      # YOLOv11 RGBD detection
    |       +-- pedestrian_behaviour_predictor.py # Const-velocity predictor (extended to multi-ped)
    |       +-- detect_node.py
    |
    +-- vehicle_drivers/
    |   +-- mppi_controller/        # ** MPPI controller **
    |   |   +-- mppi_controller/
    |   |   |   +-- mppi.py                  # Torch MPPI optimizer (K=600 rollouts)
    |   |   |   +-- adapt_mppi_node.py       # ROS 2 node (AutoShield topic contract)
    |   |   |   +-- adapt_mppi_generic_node.py  # Generic test harness
    |   |   |   +-- reference_path.py        # Waypoint path with nearest-point
    |   |   +-- rviz/adapt_main.rviz
    |   +-- gem_gnss_control/       # Pure pursuit controller (fallback)
    |   +-- gem_visualization/      # URDF, RViz config
    |
    +-- basic_launch/               # Sensor bringup launch files
    +-- hardware_drivers/           # Ouster, cameras, Septentrio, PACMod2
    +-- utilities/                  # CAN bus, radar setup scripts
```

---

## 4. Perception Pipeline

### LiDAR Pedestrian Detection (`adapt_lidar_processing`)

- Input: `/ouster/points` (PointCloud2)
- Method: DBSCAN clustering + human geometry filtering (height, width)
- Output: `/lidar_pedestrian_position` (Int32MultiArray, polar `[dist_m, bearing_deg, ...]`)

### RGBD Pedestrian Detection (`rgbd_pedestrian_detector`)

- Input: `/oak/rgb/image_raw`, `/oak/stereo/image_raw`
- Method: YOLOv11 (COCO class 0 = person) + depth projection
- Output: `/rgbd_pedestrian_position` (Int32MultiArray, polar)

### Sensor Fusion (`lidar_camera_fusion`)

- Inputs: Both LiDAR and RGBD pedestrian positions
- Sync: ApproximateTimeSync (100 ms slop)
- Fusion weights:
  - **Distance:** 80% LiDAR, 20% camera
  - **Bearing:** 30% LiDAR, 70% camera
- Matching: 2 m Euclidean gate
- Output: `/fusion_pedestrian_position` (Int32MultiArray, polar, ~10 Hz)

### Message Format

The fusion topic encodes multiple pedestrians as a flat integer array:
```
[dist1_m, bearing1_deg, dist2_m, bearing2_deg, ...]
```
Distances are in integer meters, bearings in integer degrees.

Polar-to-Cartesian conversion (ego frame, x-forward, y-left):
```python
theta = np.deg2rad(bearing_deg)
x = dist * np.sin(theta)    # forward
y = -dist * np.cos(theta)   # left
```

---

## 5. Prediction Modes

### Mode 1: `single-default` -- Constant-Velocity Predictor

**Package:** `yolo_person_detector`
**Node:** `pedestrian_behaviour_predictor`

The original AutoShield predictor, extended to handle multiple pedestrians.

- **Tracker:** Greedy 2 m Euclidean association, EMA smoothing (alpha=0.6), median spike filter, 7-point moving average, 15-point velocity estimation
- **Prediction:** Linear constant-velocity extrapolation, 5 s horizon, 20 points (dt=0.25 s), velocity clamped at 3 m/s, path clipped at 15 m radius
- **No GPU required**
- **Latency:** < 1 ms

### Mode 2: `single-diffusion` -- Per-Pedestrian Diffusion Model

**Package:** `diffusion_prediction`
**Node:** `infer_node` (with `prediction_mode=single`)
**Model:** `TrajectoryDenoiser` (0.47M params)

Each pedestrian is predicted independently using a MID-style Transformer denoiser.

- **Architecture:** 4-layer Transformer encoder, 4 attention heads, d_model=128, cross-attention decoder
- **Diffusion:** DDPM with 100 train steps (cosine schedule), DDIM-10 at inference
- **Samples:** K=20 per pedestrian, best-mode selection via closest-to-mean + sticky temporal
- **Input:** History (T_hist=20, 4) = [x, y, vx, vy] + ego velocity (2,)
- **Output:** Future (T_fut=20, 2) = [x, y] at dt=0.25 s
- **Pretrained weights:** `models/diffusion/av2_pretrain_v1/ema_best.pt` (minFDE = 0.693 m)
- **Latency:** ~9 ms on RTX 3060

### Mode 3: `multi-diffusion` -- Joint Multi-Agent Diffusion Model

**Package:** `diffusion_prediction`
**Node:** `infer_node` (with `prediction_mode=joint`)
**Model:** `JointTrajectoryDenoiser` (0.93M params)

All pedestrians in the scene are predicted jointly with cross-agent attention, so each pedestrian's prediction is informed by the trajectories of all others.

- **Architecture:** Per-agent encoder (4 layers, shared weights) -> cross-agent attention (2 layers) -> per-agent decoder
- **Max agents:** 16 (padded with agent_mask)
- **Diffusion:** Same as single-agent (DDPM-100 train, DDIM-10 inference)
- **Samples:** K=20 joint scene samples
- **Pretrained weights:** `models/diffusion/av2_joint_v1/ema_best.pt` (minFDE = 0.529 m)
- **Latency:** ~11 ms on RTX 3060

### Best-Mode Selection

All diffusion modes use sticky closest-to-mean selection to pick one trajectory per pedestrian from K=20 samples:

1. Compute mean path across all K samples
2. Find sample closest to the mean (by summed squared distance)
3. Sticky temporal: only switch mode when the current mode's cost exceeds 1.5x the candidate for 3+ consecutive ticks

This prevents TTC jitter from frame-to-frame sample reordering.

---

## 6. Controllers

### MPPI Controller (`adapt_mppi_node`)

Sampling-based model predictive control. Samples K=600 candidate control sequences per tick, evaluates them against a cost function, and computes a softmax-weighted average.

**Vehicle model:** Kinematic bicycle, state = (x, y, yaw, v), input = (accel, steer).

**Cost function (per rollout step):**

| Term | Weight | Formula |
|---|---|---|
| Goal position | `w_pos=15.0` | L2 distance to lookahead point on reference path |
| Velocity tracking | `w_vel=5.0` | \|v - v_ref\| |
| Stability | `w_curv=2.0` | \|steer\| * v (penalizes high-steer-at-high-speed) |
| Pedestrian Gaussian | `w_obs=150.0` | Velocity-aware 2D Gaussian repulsion per pedestrian |
| Hard clearance | `w_obs_hard=250.0` | Step penalty inside clearance radius (3 m) |
| Soft falloff | `w_obs_soft=40.0` | Exponential decay repulsion |

**Obstacle handling:**

When `prediction_source=raw` (default): subscribes to `/fusion_pedestrian_position`, treats each detection as a static `(x,y)` obstacle in world frame.

When `prediction_source=predicted`: subscribes to `/pedestrian_predictions_tensor`, decodes `(M, 20, 2)` trajectories, computes velocities from position differences, and builds `(M, 5)` obstacles `[x, y, vx, vy, conf]` that feed into the velocity-aware Gaussian cost.

**Confidence-growth model:** The obstacle cost uses a temporal confidence-growth model where the uncertainty ellipse around each predicted pedestrian position widens as confidence decays over the rollout horizon. This means MPPI is more cautious about pedestrians farther in the future where predictions are less certain.

### Stanley Controller (`adapt_stanley_controller`)

Classic cross-track error controller with PID speed control. No obstacle avoidance -- relies on `high_level_decision_node` for stop/go decisions based on TTC and safety distance.

Used as a simpler fallback when MPPI is not needed.

---

## 7. High-Level Decision Node

**Node:** `adapt_high_level_command`

State machine that reads `/pedestrian_motion` and `/pedestrian_ttc` from whichever predictor is active, plus `/fusion_pedestrian_position` directly. Outputs `/safety_decision` (String) with states:

| State | Action |
|---|---|
| `CRUISE` | Normal driving |
| `STOP_YIELD` | Full brake for pedestrian |
| `SLOW_CAUTION` | Reduce speed |
| `CREEP_PASS` | Edge forward slowly |

Works identically with all 3 prediction modes because they all publish the same topics.

---

## 8. How to Run

### Prerequisites

```bash
# GPU training/inference env (vehicle or dev machine)
conda create -n adapt python=3.12 -y
conda activate adapt
pip install -r adapt_requirements.text

# Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

# Build workspace
cd ~/Adapt/cs_588_g10
source /opt/ros/<distro>/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### Mode A -- Standalone MPPI Test (no ROS)

```bash
python3 src/vehicle_drivers/mppi_controller/test/test_phase1.py
```

Expected: `PHASE 1 TEST PASSED`, mean |lateral error| ~ 0.007 m.

### Mode B -- Rosbag Replay

```bash
# Terminal 1: Replay
ros2 bag play /path/to/bag --clock

# Terminal 2: Run stack with prediction mode
source install/setup.bash
ros2 launch adapt_full adapt_prediction_launch.py \
    prediction_mode:=multi-diffusion \
    controller:=mppi \
    weights:=models/diffusion/av2_joint_v1/ema_best.pt \
    enable_rviz:=true
```

### Mode C -- Live Vehicle

```bash
# Terminal 1: Sensor bringup
ros2 launch basic_launch sensor_init.launch.py vehicle_name:=e4

# Terminal 2: Perception
ros2 run adapt_full lidar_processing &
ros2 run yolo_person_detector rgbd_pedestrain_detector

# Terminal 3: Full stack
ros2 launch adapt_full adapt_prediction_launch.py \
    prediction_mode:=multi-diffusion \
    controller:=mppi \
    weights:=models/diffusion/av2_joint_v1/ema_best.pt \
    enable_fusion:=true \
    enable_safety:=true \
    enable_high_level:=true
```

### Launch File Arguments

```bash
ros2 launch adapt_full adapt_prediction_launch.py \
    prediction_mode:=<single-default|single-diffusion|multi-diffusion> \
    controller:=<mppi|stanley> \
    vehicle_name:=<e2|e4> \
    desired_speed:=2.0 \
    weights:=<path/to/ema_best.pt> \
    device:=<cuda:0|cpu|''> \
    enable_lidar:=true \
    enable_fusion:=false \
    enable_safety:=false \
    enable_high_level:=false \
    enable_rviz:=true
```

### Quick Smoke Test

```bash
# Inject a fake pedestrian 10 m ahead
ros2 topic pub --once /fusion_pedestrian_position \
    std_msgs/msg/Int32MultiArray "{data: [10, 0]}"

# Watch MPPI react
ros2 topic echo /pacmod/accel_cmd
```

---

## 9. Parameter Reference

### MPPI Sampling

| Param | Default | Range | Effect |
|---|---|---|---|
| `mppi/K` | 600 | 200-2000 | Rollout sample count |
| `mppi/H` | 30 | 20-50 | Horizon steps (at dt=0.1 = 3 s) |
| `mppi/dt` | 0.1 | 0.05-0.2 | Integration timestep |
| `mppi/sigma_steer` | 0.15 | 0.05-0.30 | Steering noise std (rad) |
| `mppi/sigma_accel` | 0.5 | 0.2-1.0 | Acceleration noise std (m/s^2) |
| `mppi/lambda_` | 0.1 | 0.01-2.0 | Softmax temperature |
| `mppi/device` | `''` | `cuda:0`, `cpu`, `''` | Torch device; `''` = auto |

### Tracking Cost

| Param | Default | Effect |
|---|---|---|
| `mppi/w_pos` | 15.0 | Goal position error weight |
| `mppi/w_vel` | 5.0 | Velocity tracking weight |
| `mppi/w_curv` | 2.0 | Stability penalty |steer| * v |
| `mppi/lookahead_m` | 8.0 | Look-ahead distance on reference path |

### Obstacle Cost

| Param | Default | Effect |
|---|---|---|
| `mppi/w_obs` | 150.0 | Peak Gaussian repulsion per pedestrian |
| `mppi/w_obs_hard` | 250.0 | Step penalty inside clearance radius |
| `mppi/w_obs_soft` | 40.0 | Exponential decay repulsion |
| `mppi/clearance` | 3.0 | Hard clearance radius (m) |
| `prediction_source` | `raw` | `raw` = static obstacles, `predicted` = tensor trajectories |

### Longitudinal / PID

| Param | Default | Effect |
|---|---|---|
| `desired_speed` | 2.0 | Cruise target m/s (hard-capped at 5.0) |
| `max_acceleration` | 0.5 | Throttle cap (hard-capped at 2.0) |
| `pid/kp` | 0.6 | Speed PID proportional gain |
| `pid/ki` | 0.0 | Speed PID integral gain |
| `pid/kd` | 0.1 | Speed PID derivative gain |
| `wheelbase` | 1.75 | GEM e4 kinematic bicycle wheelbase (m) |
| `offset` | 1.26 | GPS antenna to rear-axle offset (m) |

### Diffusion Predictor

| Param | Default | Effect |
|---|---|---|
| `weights` | `''` | Path to model checkpoint (.pt) |
| `device` | `cuda:0` | Torch device for inference |
| `prediction_mode` | `joint` | `single` or `joint` model |
| `K` | 20 | Number of trajectory samples |
| `ddim_steps` | 10 | DDIM denoising steps |
| `min_history_count` | 5 | Min observed frames before predicting |
| `prediction_time` | 5.0 | Prediction horizon (seconds) |
| `prediction_points` | 20 | Number of prediction timesteps |
| `collision_distance_threshold` | 1.0 | TTC collision distance (m) |
| `latency_warn_ms` | 80.0 | Log warning if cycle exceeds this (ms) |
| `max_agents` | 16 | Max agents for joint mode padding |

### Operational

| Param | Default | Effect |
|---|---|---|
| `rate_hz` | 10.0 | Control loop frequency |
| `require_pacmod_enable` | `true` | Wait for PACMod enable gate |
| `waypoints_csv` | (auto) | Path to waypoint CSV (auto-resolved via ament) |
| `viz/num_samples` | 19 | Top-weighted MPPI rollouts to visualize |
| `viz/frame_id` | `map` | RViz frame for MPPI markers |

---

## 10. Diffusion Model Details

### Architecture

**Single-agent (`TrajectoryDenoiser`, 0.47M params):**
- History encoder: Linear(4 -> 128) + learned positional embedding (20 tokens) + ego velocity injection
- Diffusion step: sinusoidal embedding -> MLP -> AdaLN-style modulation
- Backbone: 4-layer Transformer encoder (4 heads, d_model=128, FFN=256, dropout=0.1)
- Decoder: Cross-attention from 20 learned future queries onto encoder output -> MLP head -> (20, 2) epsilon prediction

**Joint multi-agent (`JointTrajectoryDenoiser`, 0.93M params):**
- Same per-agent encoder (shared weights across all agents in scene)
- Cross-agent attention: 2-layer Transformer that lets agents attend to each other
- Agent embeddings: learned per-slot embeddings for up to 16 agents
- Per-agent decoder: cross-attention + MLP head (same as single-agent)
- Padding: inactive agent slots masked with `agent_mask`, NaN-safe attention masking

### Diffusion Process

**Training:**
- DDPM with 100 timesteps, cosine beta schedule
- Loss: MSE between predicted and true noise (epsilon-prediction)
- AMP (fp16) with GradScaler

**Inference:**
- DDIM with 10 steps: tau = [99, 88, 77, 66, 55, 44, 33, 22, 11, 0]
- Deterministic given initial noise
- K=20 samples per pedestrian (or per scene for joint model)

### Training Recipe

| | AV2 Pretrain | GEM Finetune |
|---|---|---|
| Optimizer | AdamW (wd=1e-4) | AdamW (wd=1e-4) |
| LR | 2e-4 -> 1e-5 cosine | 2e-5 constant |
| Epochs | 200 | 20 |
| Batch | 32 | 32 |
| AMP | fp16 | fp16 |
| EMA decay | 0.999 | 0.999 |
| Walltime | ~12 hr (RTX 3060) | ~1 hr |

### Pretrained Model Performance (AV2 val)

| Model | minFDE-20 @ 5s | Latency (M=8, K=20) |
|---|---|---|
| Single-agent | 0.693 m | ~9 ms |
| Joint multi-agent | 0.529 m | ~11 ms |
| Target | <= 1.0 m | < 30 ms |

---

## 11. ROS 2 Topic Contract

### Inputs (subscribed by MPPI + predictor)

| Topic | Type | Source | Format |
|---|---|---|---|
| `/navsatfix` | `sensor_msgs/NavSatFix` | Septentrio GNSS | GPS lat/lon |
| `/insnavgeod` | `septentrio_gnss_driver/INSNavGeod` | Septentrio INS | Heading |
| `/pacmod/enabled` | `std_msgs/Bool` | PACMod2 | Enable gate |
| `/pacmod/vehicle_speed_rpt` | `pacmod2_msgs/VehicleSpeedRpt` | PACMod2 | Filtered speed m/s |
| `/fusion_pedestrian_position` | `std_msgs/Int32MultiArray` | Sensor fusion | Polar `[dist_m, bear_deg, ...]` |

### Outputs (published by MPPI)

| Topic | Type | Content |
|---|---|---|
| `/pacmod/global_cmd` | `pacmod2_msgs/GlobalCmd` | Enable + clear-override |
| `/pacmod/steering_cmd` | `pacmod2_msgs/PositionWithSpeed` | Steering wheel angle (rad) |
| `/pacmod/accel_cmd` | `pacmod2_msgs/SystemCmdFloat` | Throttle (m/s^2) |
| `/pacmod/brake_cmd` | `pacmod2_msgs/SystemCmdFloat` | Brake (0.0-1.0) |
| `/pacmod/shift_cmd` | `pacmod2_msgs/SystemCmdInt` | Gear (3=Forward) |
| `/pacmod/turn_cmd` | `pacmod2_msgs/SystemCmdInt` | Turn signal |

### Outputs (published by predictor)

| Topic | Type | Content |
|---|---|---|
| `/person_prediction` | `visualization_msgs/Marker` | LINE_STRIP, 20 pts, primary ped trajectory |
| `/pedestrian_motion` | `geometry_msgs/Twist` | Primary ped x,y in ego frame |
| `/pedestrian_ttc` | `std_msgs/Float64` | Time-to-collision (seconds, inf if none) |
| `/pedestrian_predictions_tensor` | `std_msgs/Float32MultiArray` | (M, 20, 2) all peds, best-mode trajectories |

---

## 12. PACMod2 Vehicle Control

### Control Sequence

**Enabling autonomous control:**
```
1. /pacmod/global_cmd   -> enable=True, clear_override=True
2. /pacmod/shift_cmd    -> command=3 (Forward)
3. /pacmod/brake_cmd    -> command=0.0
4. /pacmod/accel_cmd    -> command=0.0
```

**Autonomous driving loop (10 Hz):**
```
1. Read GNSS position + INS heading
2. Read vehicle speed
3. MPPI.update(state, ref_path, obstacles) -> (steer, accel)
4. Front-wheel angle -> steering-wheel angle (polynomial)
5. PID speed control from accel -> throttle
6. Publish all PACMod commands
```

### Steering Calibration

Front-wheel angle (deg) to steering-wheel angle (deg):
```
sw = -0.1084 * |angle|^2 + 21.775 * |angle|
```
Front-wheel angle clamped to +/- 35 deg, steering-wheel clamped to +/- 450 deg.

### Safety Caps

| Limit | Value |
|---|---|
| Max speed | 5.0 m/s |
| Max acceleration | 2.0 m/s^2 |
| Max front wheel angle | +/- 35 deg |
| Steering rate limit | 4.0 rad/s |

---

## 13. Coordinate Frames

- **Ego/base_link:** x = forward, y = left, z = up. All perception outputs and predictions are in this frame.
- **World/ENU:** GPS -> East-North-Up via WGS-84 geodetic conversion. MPPI operates in this frame. Origin: (40.0927422, -88.2359639).
- **Polar (fusion messages):** distance in meters, bearing in degrees. Converted to Cartesian in each consuming node.

---

## 14. Conda Environments

| Env | Python | Torch | Use |
|---|---|---|---|
| `adapt` | 3.12 | 2.5.1+cu121 (GPU) | Training, inference, vehicle runtime |
| `cs588` | 3.11 | 2.11.0+cpu | CPU-only algorithm dev |

**Rule:** NEVER install Python packages via `apt` or into `~/.local`. All deps go through conda/pip only.

```bash
# GPU env
conda create -n adapt python=3.12 -y && conda activate adapt
pip install -r adapt_requirements.text

# CPU dev env
conda create -n cs588 python=3.11 -y && conda activate cs588
pip install -r cs588_requirements.txt
```

ROS 2 (Humble/Jazzy) comes from apt. Source `/opt/ros/<distro>/setup.bash` before any ROS work.

---

## 15. Visualization (RViz)

The `adapt_main.rviz` config shows:

- **Reference path:** Yellow line strip (latched, published once)
- **Chosen trajectory:** Bold path = MPPI weighted-mean rollout
- **Sampled trajectories:** 19 top-weighted MPPI rollouts in pastel rainbow
- **Obstacle markers:** Red translucent cylinders at clearance radius around each pedestrian
- **Prediction markers:** Person prediction LINE_STRIP from the active predictor

Launch with `enable_rviz:=true` (default) or manually:
```bash
rviz2 -d src/vehicle_drivers/mppi_controller/rviz/adapt_main.rviz
```
