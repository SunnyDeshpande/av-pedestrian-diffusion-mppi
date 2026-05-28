# ADAPT: MPPI Motion Planning with Diffusion-Based Pedestrian Prediction

![ROS2](https://img.shields.io/badge/ROS2-Humble%20%7C%20Jazzy-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2BCUDA-orange.svg)
![Platform](https://img.shields.io/badge/Platform-UIUC%20GEM%20e4-orange)
![Course](https://img.shields.io/badge/CS588-Autonomous%20Vehicle%20Systems-blue)
![UIUC](https://img.shields.io/badge/Institution-UIUC-orange)
![Date](https://img.shields.io/badge/Date-Spring%202026-green)

> **A pedestrian-aware autonomy stack for the UIUC Polaris GEM e4 that replaces the AutoShield Stanley lateral controller with a torch-based MPPI motion planner, and feeds it multi-modal pedestrian futures from a diffusion-based trajectory predictor - LiDAR + RGB-D perception, weighted sensor fusion, learned prediction, and sampling-based control on the GPU at 10 Hz.**

---

## Overview

Reactive pedestrian handling on autonomous vehicles tends to rely on proximity thresholds and late braking, which is brittle under uncertainty, occlusion, and partial observability. ADAPT replaces that pattern with two coordinated changes:

1. A **diffusion-based pedestrian trajectory predictor** (MID-style Transformer denoiser, joint multi-agent variant) that emits multi-modal 5-second futures rather than a single point estimate.
2. A **torch MPPI motion planner** that rolls out 600 candidate trajectories per cycle on the GPU, scores them against the predicted pedestrian distributions, and outputs steering + accel/brake commands to the PACMod2 drive-by-wire stack.

The system is a configurable ROS 2 pipeline with three selectable prediction modes (constant-velocity baseline, single-agent diffusion, joint multi-agent diffusion) wired to two controllers (MPPI or fallback Stanley).

```
LiDAR ──┐                                            ┌─► /pedestrian_predictions_tensor
        ├──► Sensor Fusion ──► Diffusion Predictor ──┤    (M, 20, 2) base_link, dt=0.25s
RGB-D ──┘                                            └─► MPPI Planner ──► PACMod2 ──► GEM e4
                                                          K=600, H=40, dt=0.1, 10 Hz
```

---

## Hardware Platform - UIUC GEM e4

| Sensor | Spec |
|---|---|
| **Top LiDAR** | Ouster OS1-128 - 128ch, 360° HFoV, 45° VFoV, 10-20 Hz, ~200m range |
| **Front LiDAR** | Livox HAP - 120°×25° FoV, ~452k pts/s |
| **Front Stereo RGB-D** | OAK-D LR - 1152×720 @ 23 FPS, global shutter |
| **Corner Cameras** | Lucid - 1920×1200 @ 48.3 FPS, PoE |
| **GNSS/INS** | Septentrio AsteRx SBi3 Pro+ with RTK |
| **Drive-by-wire** | PACMod2 via USB-to-CAN (steering, throttle, brake) |
| **Compute** | NVIDIA GPU on-vehicle for MPPI + diffusion inference |

---

## System Architecture

### ROS 2 Topic Interfaces

| Topic | Type | Description |
|---|---|---|
| `/ouster/points` | `PointCloud2` | Raw top-LiDAR returns |
| `/oak/rgb/image_raw`, `/oak/stereo/image_raw` | `Image` | RGB + depth from front stereo |
| `/lidar_pedestrian_position` | `Int32MultiArray [d, θ, …]` | LiDAR pedestrian estimate (polar, ego-frame) |
| `/rgbd_pedestrian_position` | `Int32MultiArray [d, θ, …]` | RGB-D pedestrian estimate |
| `/fusion_pedestrian_position` | `Int32MultiArray [d₁, θ₁, d₂, θ₂, …]` | Fused multi-pedestrian estimate, ~10 Hz |
| `/fusion_pedestrian_paths` | `MarkerArray` | Per-track LINE_STRIPs for RViz + history-buffer source |
| `/pedestrian_predictions_tensor` | `Float32MultiArray (M, 20, 2)` | **Diffusion output** - multi-modal trajectories, base_link Cartesian, dt=0.25 s |
| `/person_prediction` | `Marker` | RViz visualization of predicted futures |
| `/pedestrian_motion`, `/pedestrian_ttc` | `Twist`, `Float64` | Aggregated motion + time-to-collision signals |
| `/pacmod/steering_cmd`, `/pacmod/accel_cmd`, `/pacmod/brake_cmd` | PACMod messages | Vehicle commands from MPPI |

---

## Module Details

### 1. LiDAR Pedestrian Pipeline

Converts raw `PointCloud2` into a stable pedestrian estimate in the ego frame:

1. **Preprocessing** - voxelization, ground filtering, outlier removal
2. **Clustering** - DBSCAN to separate static / dynamic objects
3. **Human filtering** - geometric gates on height + width + aspect
4. **Tracking** - EMA smoothing on cluster centroids

Publishes `(d, θ)` in ego frame to `/lidar_pedestrian_position`. Source: `src/adapt_full/adapt_full/adapt_lidar_processing.py`.

---

### 2. RGB-D Pedestrian Pipeline

1. **Detection** - YOLOv11 (COCO class 0 = person) on OAK-D RGB
2. **Depth extraction** - bounding-box → depth crop → closest-pedestrian range
3. **Pose transform** - pixel + depth → ego-frame `(distance, bearing)`

Source: `src/yolo_person_detector/yolo_person_detector/rgbd_pedestrain_detector.py`.

---

### 3. Sensor Fusion

Combines LiDAR and RGB-D into a single robust estimate with modality-weighted fusion:

| Quantity | LiDAR weight | Camera weight |
|---|---|---|
| Distance | 0.8 | 0.2 |
| Bearing | 0.3 | 0.7 |

- **Time sync** - `ApproximateTimeSync`, slop ≤ 0.1 s
- **Data association** - 2.0 m Euclidean gate; unmatched detections published standalone
- **Smoothing** - EMA on live track XY (`track_smooth_alpha`) + centered moving-average over the published path (`path_smoothing_window`, visualization-only)

Source: `src/adapt_full/adapt_full/adapt_lidar_camera_fusion.py`. Tunable in `src/adapt_full/config/sensor_fusion_params.yaml`.

---

### 4. Diffusion Pedestrian Trajectory Prediction

MID-style Transformer denoiser predicting multi-modal pedestrian futures from a history of fused positions.

| Variant | Parameters | Training data |
|---|---|---|
| **Single-agent** (`TrajectoryDenoiser`) | 0.47 M | Argoverse 2 |
| **Joint multi-agent** (`JointTrajectoryDenoiser`) | 0.93 M | Argoverse 2 with cross-agent attention |

- **Schedule** - DDPM 100-step cosine schedule (training), DDIM 10-step sampling (inference)
- **Output contract** - `Float32MultiArray (M, 20, 2)`, base_link Cartesian (x=forward, y=left), fixed `H=20`, `dt=0.25 s`, 5 s horizon
- **Modes selectable at runtime** - `const_vel` baseline, `single`, `joint`
- **Side outputs** - `/pedestrian_motion` (Twist), `/pedestrian_ttc` (Float64), `/person_prediction` (RViz Marker)

The output contract is frozen on both sides - the MPPI obstacle cost indexes the tensor on the same `(H, dt)` grid. Sources: `src/diffusion_prediction/diffusion_prediction/{model.py, model_joint.py, ddpm.py, infer_node.py}`.

---

### 5. Torch MPPI Motion Planner

Sampling-based receding-horizon planner that natively accepts the predicted pedestrian distribution as a soft cost.

| Setting | Value |
|---|---|
| Rollouts (`K` / `num_samples`) | 600 |
| Horizon (`H`) | 40 steps |
| Step size (`dt`) | 0.1 s (4 s lookahead) |
| Cycle rate | 10 Hz |
| Backend | PyTorch on CUDA (auto-falls-back to CPU) |
| Vehicle model | Kinematic bicycle |

Cost terms: reference-path tracking, longitudinal-speed regulation, control-effort regularization, and a time-aware obstacle cost against `/pedestrian_predictions_tensor`. Outputs PACMod2 `steering_cmd`, `accel_cmd`, `brake_cmd`.

Source: `src/vehicle_drivers/gem_mppi_control/mppi_ros.py` (ROS 2 node), `src/vehicle_drivers/gem_mppi_control/mppi_t.py` (offline matplotlib harness, no ROS).

---

### 6. Fallback - Stanley + Safety State Machine

For modes that do not run MPPI, an AutoShield-style Stanley lateral controller + PID longitudinal controller is available, gated by a high-level safety state machine (`CRUISE` / `SLOW_CAUTION` / `STOP_YIELD`) consuming `/pedestrian_motion` and `/pedestrian_ttc`.

Sources: `src/adapt_full/adapt_full/{adapt_stanley_controller.py, adapt_high_level_command.py, adapt_safety_controller.py}`.

---

## Installation & Launch

### Environment

Strict conda-only environment policy - no system Python, no apt, no `~/.local/`:

```bash
conda create -n adapt python=3.12 -y
conda activate adapt
pip install -r adapt_requirements.text
```

Matches the Jazzy vehicle's Python 3.12. For the Humble dev host (Python 3.10), create a parallel `adapt-py310` env using the same requirements file.

### Build

```bash
source /opt/ros/humble/setup.bash   # or jazzy on the vehicle
colcon build --symlink-install --packages-ignore livox_ros_driver2
source install/setup.bash
export VEHICLE_NAME=e4
```

### Run (4-terminal session)

```bash
# T1 - sensors + RViz
ros2 launch basic_launch sensor_init.launch.py

# T2 - perception + fusion
ros2 launch adapt_full lidar_rgb_tracking_launch.py

# T3 - diffusion predictor (default: joint mode, av2_joint_v2 weights)
ros2 launch diffusion_prediction diffusion.launch.py

# T4 - MPPI controller
python3 src/vehicle_drivers/gem_mppi_control/mppi_ros.py
```

### Standalone MPPI without ROS

For numerical sanity-checking or algorithm dev:

```bash
python3 src/vehicle_drivers/gem_mppi_control/mppi_t.py
```

### Smoke test (any run mode)

Inject a fake pedestrian and watch the predictor + MPPI react:

```bash
ros2 topic pub --once /fusion_pedestrian_position \
    std_msgs/msg/Int32MultiArray "{data: [10, 0]}"
```

> **Safety note**: All on-vehicle operation requires a safety driver. Physical interlocks (emergency button + brake-pedal button) sever the PACMod connection. The safety driver retains final stopping authority.

---

## Project Structure

```
cs_588_g10/
├── docs/                              # PROJECT.md (full design spec), diffusion summary
├── src/
│   ├── adapt_full/                    # Perception + fusion + safety state machine + launch
│   │   ├── adapt_full/
│   │   │   ├── adapt_lidar_processing.py        # DBSCAN clustering, EMA tracking
│   │   │   ├── adapt_lidar_camera_fusion.py     # Weighted sensor fusion
│   │   │   ├── adapt_high_level_command.py      # CRUISE / SLOW_CAUTION / STOP_YIELD
│   │   │   ├── adapt_safety_controller.py
│   │   │   └── adapt_stanley_controller.py
│   │   ├── config/                              # YAML param files
│   │   └── waypoints/                           # Track CSVs
│   ├── diffusion_prediction/          # Diffusion trajectory predictor
│   │   ├── diffusion_prediction/
│   │   │   ├── model.py                         # Single-agent denoiser (0.47M)
│   │   │   ├── model_joint.py                   # Joint multi-agent denoiser (0.93M)
│   │   │   ├── ddpm.py                          # Cosine schedule, DDIM sampling
│   │   │   ├── infer_node.py                    # ROS 2 inference node (3 modes)
│   │   │   ├── train.py, train_joint.py         # Argoverse 2 pretraining
│   │   │   └── finetune.py                      # GEM rosbag fine-tuning
│   │   └── models/diffusion/                    # Trained weights
│   ├── yolo_person_detector/          # YOLOv11 RGB-D detector
│   ├── vehicle_drivers/
│   │   ├── gem_mppi_control/                    # MPPI motion planner
│   │   │   ├── mppi_ros.py                      # ROS 2 node, 10 Hz on GPU
│   │   │   └── mppi_t.py                        # Offline matplotlib harness
│   │   ├── gem_gnss_control/                    # Pure pursuit fallback
│   │   └── gem_visualization/                   # URDF, RViz config
│   ├── basic_launch/                            # Sensor bringup
│   └── hardware_drivers/                        # Ouster, OAK-D, Septentrio, PACMod2
└── models/diffusion/                            # Pretrained AV2 weights
```

---

## Key Parameters

| Parameter | Value |
|---|---|
| MPPI rollouts `K` | 600 |
| MPPI horizon `H` | 40 steps |
| MPPI step `dt` | 0.1 s |
| MPPI cycle rate | 10 Hz |
| Diffusion horizon `H_ped` | 20 steps |
| Diffusion step `dt_ped` | 0.25 s |
| Diffusion prediction window | 5 s |
| DDIM inference steps | 10 |
| Fusion association threshold | 2.0 m |
| Fusion time sync slop | 0.1 s |
| Fusion distance weight (LiDAR / cam) | 0.8 / 0.2 |
| Fusion bearing weight (LiDAR / cam) | 0.3 / 0.7 |

---

## Limitations

- **Heuristic safety thresholds** - TTC critical range and fusion gates are tuned empirically; richer learned safety models could reduce false slowdowns.
- **Partial observability** - occlusions and sparse depth returns can destabilize fused estimates; EMA smoothing mitigates but does not eliminate this.
- **Intent ambiguity** - a pedestrian near the path is not always a crossing intent; the diffusion predictor's multi-modal output helps but is bounded by training distribution.
- **Actuation boundary** - PACMod2 emergency interlocks sever autonomy but do not apply brakes. Hard safety guarantees require a safety driver.
- **Compute coupling** - MPPI + diffusion both rely on the on-vehicle GPU; CPU fallback is supported but loses real-time guarantees.

---

## Acknowledgments

UIUC CS 588 (Autonomous Vehicle Systems) course staff and the UIUC Polaris GEM platform maintainers for vehicle infrastructure, ROS drivers, and safety procedures. This project is built on top of the [`UIUC-Robotics/gem_ws`](https://github.com/UIUC-Robotics/gem_ws) base vehicle / sensor-driver workspace, which provides the PACMod2 interface, sensor bringup, and ROS 2 driver layer. The diffusion predictor draws on the MID (Motion Indeterminacy Diffusion) line of work for trajectory generation. The pipeline extends the AutoShield safety-controller architecture with sampling-based MPC and learned prediction.

---

## Authors

**Sunny Deshpande** - MEng Autonomy & Robotics, UIUC
[sunnynd2@illinois.edu](mailto:sunnynd2@illinois.edu) · [sunnydeshpande.com](https://sunnydeshpande.com)

Het Patel · Aditya Potnis · Keisuke Ogawa · Francisco Affonso

---

*Built on ROS 2 (Humble / Jazzy), deployed on the UIUC Polaris GEM e4 platform.*
