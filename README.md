# Adapt — MPPI Motion Planner + Diffusion Pedestrian Prediction

**CS 588 (Autonomous Vehicle Systems) — Group 10, UIUC**

Adapt replaces the Stanley lateral controller on the Polaris GEM e4 with a sampling-based **MPPI** motion planner and adds a **diffusion-based pedestrian trajectory predictor** that generates multi-modal 5-second futures. The system offers 3 selectable prediction modes wired to either controller.

See `docs/PROJECT.md` for full architecture details and `docs/TODO.md` for outstanding action items.

---

## Quick Start

### 1. Environment Setup

```bash
# Create conda env (GPU)
conda create -n adapt python=3.12 -y
conda activate adapt
pip install -r adapt_requirements.text

# Source ROS 2
source /opt/ros/<humble|jazzy>/setup.bash

# Build
cd ~/Adapt/cs_588_g10
colcon build --symlink-install
source install/setup.bash
```

Verify GPU:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### 2. Run the Full Stack

The unified launch file supports 3 prediction modes and 2 controllers:

```bash
ros2 launch adapt_full adapt_prediction_launch.py \
    prediction_mode:=<mode> \
    controller:=<controller> \
    [weights:=<path>]
```

| `prediction_mode` | Description | GPU | `weights` needed |
|---|---|---|---|
| `single-default` | Constant-velocity extrapolation (original predictor) | No | No |
| `single-diffusion` | Per-pedestrian diffusion model (0.47M params) | Yes | Yes |
| `multi-diffusion` | Joint multi-agent diffusion with cross-attention (0.93M params) | Yes | Yes |

| `controller` | Description |
|---|---|
| `mppi` | MPPI sampling-based MPC (K=600 rollouts, obstacle avoidance) |
| `stanley` | Stanley cross-track + PID (no obstacle avoidance) |

---

## Running Modes

### A. Standalone MPPI Test (no ROS, no sensors)

```bash
python3 src/vehicle_drivers/mppi_controller/test/test_phase1.py
```

### B. Rosbag Replay

```bash
# Terminal 1 — replay
ros2 bag play /path/to/bag --clock

# Terminal 2 — stack
source install/setup.bash
ros2 launch adapt_full adapt_prediction_launch.py \
    prediction_mode:=multi-diffusion \
    controller:=mppi \
    weights:=models/diffusion/av2_joint_v1/ema_best.pt
```

### C. Live Vehicle

```bash
# Terminal 1 — sensors
ros2 launch basic_launch sensor_init.launch.py vehicle_name:=e4

# Terminal 2 — perception (LiDAR + YOLO detectors)
ros2 run adapt_full lidar_processing &
ros2 run yolo_person_detector rgbd_pedestrain_detector

# Terminal 3 — prediction + control
ros2 launch adapt_full adapt_prediction_launch.py \
    prediction_mode:=multi-diffusion \
    controller:=mppi \
    weights:=models/diffusion/av2_joint_v1/ema_best.pt \
    enable_fusion:=true \
    enable_safety:=true \
    enable_high_level:=true
```

### D. Smoke Test

Inject a fake pedestrian 10 m ahead and verify MPPI reacts:

```bash
ros2 topic pub --once /fusion_pedestrian_position \
    std_msgs/msg/Int32MultiArray "{data: [10, 0]}"

# Check: MPPI log should show obs=1, throttle should drop
```

---

## How It Works

```
  Ouster LiDAR          OAK-D Camera
       |                      |
       v                      v
  lidar_processing    rgbd_pedestrian_detector (YOLOv11)
       |                      |
       +----------+-----------+
                  |
                  v
       lidar_camera_fusion
       (80/20 dist, 30/70 bearing)
                  |
    /fusion_pedestrian_position
    Int32MultiArray [dist, bearing, ...]
                  |
          +-------+-------+
          |               |
          v               v
     PREDICTOR         MPPI (raw mode,
     (1 of 3 modes)    legacy fallback)
          |
          +---> /pedestrian_predictions_tensor (M x 20 x 2)
          +---> /pedestrian_motion, /pedestrian_ttc
          |
          v
     CONTROLLER (mppi or stanley)
          |
          v
     PACMod2 --> GEM e4 actuators
```

**MPPI** samples K=600 candidate control trajectories each tick, evaluates them against a cost function (goal tracking + velocity + stability + pedestrian avoidance), and applies a softmax-weighted average. When `prediction_source=predicted`, it consumes the diffusion model's predicted trajectories with velocity-aware Gaussian repulsion costs.

**Diffusion models** use a MID-style Transformer denoiser trained on Argoverse 2 with DDPM (100 steps, cosine schedule) and infer via DDIM in 10 steps. The joint model adds cross-agent attention so pedestrians' predictions are informed by each other's motion.

---

## Launch Arguments

| Argument | Default | Description |
|---|---|---|
| `prediction_mode` | `single-default` | `single-default`, `single-diffusion`, `multi-diffusion` |
| `controller` | `mppi` | `mppi` or `stanley` |
| `vehicle_name` | `e4` | Vehicle identifier |
| `desired_speed` | `2.0` | Cruise speed m/s (max 5.0) |
| `weights` | `''` | Path to diffusion model `.pt` file |
| `device` | `''` | Torch device (`cuda:0`, `cpu`, or `''` for auto) |
| `enable_lidar` | `true` | Launch LiDAR processing node |
| `enable_fusion` | `false` | Launch sensor fusion node |
| `enable_safety` | `false` | Launch safety controller |
| `enable_high_level` | `false` | Launch high-level decision node |
| `enable_rviz` | `true` | Launch RViz2 with MPPI visualization |

MPPI parameters are overridable at launch via `--ros-args -p`:

```bash
ros2 run mppi_controller adapt_mppi_node --ros-args \
    -p mppi/K:=800 -p mppi/w_obs:=200.0 -p mppi/clearance:=4.0
```

---

## Trained Models

| Model | Weights | minFDE @ 5s | Latency |
|---|---|---|---|
| Single-agent | `models/diffusion/av2_pretrain_v1/ema_best.pt` | 0.693 m | ~9 ms |
| Joint multi-agent | `models/diffusion/av2_joint_v1/ema_best.pt` | 0.529 m | ~11 ms |

Both pretrained on Argoverse 2 (~150k pedestrian sequences). See `docs/TODO.md` for GEM-specific finetuning instructions.

---

## Legacy Commands

### Sensors

```bash
ros2 launch basic_launch sensor_init.launch.py vehicle_name:=e4
```

### GNSS Visualization

```bash
ros2 launch basic_launch visualization.launch.py
```

### Joystick Control

```bash
ros2 launch basic_launch dbw_joystick.launch.py
```

### Pure Pursuit (fallback, no MPPI)

**Note:** Verify GNSS heading is correct on launch. Relaunch GNSS or restart the machine if needed.

```bash
# Terminal 1
ros2 launch pacmod2 pacmod2.launch.xml

# Terminal 2
ros2 run gem_gnss_control pure_pursuit
```

### Corner Cameras

Not compatible with backup PC (post e4 incident).

---

## Project Structure

```
cs_588_g10/
├── docs/
│   ├── PROJECT.md              # Full architecture & parameter reference
│   └── TODO.md                 # Action items & tuning guide
├── LICENSE                      # MIT license
├── requirements.txt            # Core runtime dependencies
├── adapt_requirements.text     # Full GPU conda env recipe
├── cs588_requirements.txt      # CPU-only dev env recipe
├── src/
│   ├── adapt_full/             # Perception, fusion, safety, launch files
│   ├── diffusion_prediction/   # Diffusion trajectory prediction models
│   ├── yolo_person_detector/   # YOLOv11 detection + original predictor
│   ├── vehicle_drivers/
│   │   ├── mppi_controller/    # MPPI controller (torch backend)
│   │   ├── gem_gnss_control/   # Pure pursuit (fallback)
│   │   └── gem_visualization/  # URDF, RViz
│   ├── basic_launch/           # Sensor bringup
│   ├── hardware_drivers/       # LiDAR, camera, GNSS, PACMod2 drivers
│   └── utilities/              # CAN bus, radar scripts
├── models/                     # Trained weights (gitignored)
└── data/                       # Rosbags & datasets (gitignored)
```
