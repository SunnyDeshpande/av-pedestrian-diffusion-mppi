# Adapt — MPPI Motion Planner + Diffusion Pedestrian Prediction

**CS 588 (Autonomous Vehicle Systems) — Group 10, UIUC**

Replaces the Stanley lateral controller on the Polaris GEM e4 with a sampling-based **MPPI** motion planner and adds a **diffusion-based pedestrian trajectory predictor** that produces multi-modal 5-second futures.

See `docs/PROJECT.md` for full architecture details and `docs/TODO.md` for outstanding action items.

---

## Build

```bash
source /opt/ros/humble/setup.bash
conda activate adapt_perception
cd ~/CS588/group10/cs_588_g10
colcon build --symlink-install --packages-ignore livox_ros_driver2
source install/setup.bash
export VEHICLE_NAME=e4
```

---

## Run

Each block is one terminal. Source ROS + `install/setup.bash` first (and `export VEHICLE_NAME=e4`).

### 1. Sensor bringup + RViz

Brings up Ouster, OAK-D, Lucid corner cameras, GNSS, TF, robot model, and RViz with all displays (lidar, OAK pointcloud, fusion paths, person prediction).

```bash
ros2 launch basic_launch sensor_init.launch.py
```

### 2. LiDAR + RGB pedestrian tracker (perception + fusion)

Runs `lidar_processing`, `rgbd_pedestrian_detector` (YOLOv11), and `lidar_camera_fusion`. Publishes `/fusion_pedestrian_position`, `/fusion_pedestrian_path`, `/fusion_pedestrian_paths`.

```bash
ros2 launch adapt_full lidar_rgb_tracking_launch.py
```

Optional overrides:
```bash
ros2 launch adapt_full lidar_rgb_tracking_launch.py matching_threshold:=2.0
```

### 3. Diffusion pedestrian predictor

Subscribes to `/fusion_pedestrian_paths` for history, publishes `/person_prediction` (Marker), `/pedestrian_predictions_tensor`, `/pedestrian_motion`, `/pedestrian_ttc`.

```bash
ros2 launch diffusion_prediction diffusion.launch.py
```

Default weights: `src/diffusion_prediction/models/diffusion/av2_joint_v2/ema_best.pt` in `joint` mode. Override:
```bash
ros2 launch diffusion_prediction diffusion.launch.py \
  weights:=$PWD/src/diffusion_prediction/models/diffusion/eth_ucy_ft_joint/ema_best.pt \
  prediction_mode:=joint device:=cuda:0
```

### 4. MPPI controller

Standalone Torch-based MPPI node (not a ROS launch — direct Python script). Subscribes to PACMod / GNSS / pedestrian topics; publishes vehicle commands.

```bash
python3 src/vehicle_drivers/gem_mppi_control/mppi_ros.py
```

Offline simulator (no ROS, plots in matplotlib):
```bash
python3 src/vehicle_drivers/gem_mppi_control/mppi_t.py
```

### 5. GNSS visualization

Full GNSS bringup — septentrio driver + `gem_gnss_image` overlay + RViz:
```bash
ros2 launch basic_launch gnss.launch.py
```

Fake fixed-position publisher for bench testing (no real GNSS hardware needed):
```bash
./run_gnss_pub.sh
# env overrides: LAT, LON, ALT, RATE_HZ, TOPIC
```

---

## Typical session

```
T1: ros2 launch basic_launch sensor_init.launch.py
T2: ros2 launch adapt_full lidar_rgb_tracking_launch.py
T3: ros2 launch diffusion_prediction diffusion.launch.py
T4: python3 src/vehicle_drivers/gem_mppi_control/mppi_ros.py
```

RViz (started by T1) shows: robot model, both LiDARs, OAK pointcloud, radar, corner camera image, range image, **FUSION_PEDESTRIAN_PATHS** (per-track lines from fusion), **PERSON_PREDICTION** (diffusion output).

---

## Smoke test

Inject a fake pedestrian 10 m straight ahead and watch the predictor / MPPI react:
```bash
ros2 topic pub --once /fusion_pedestrian_position \
    std_msgs/msg/Int32MultiArray "{data: [10, 0]}"
```

---

## Architecture

```
  Ouster LiDAR          OAK-D Camera
       |                      |
       v                      v
  lidar_processing    rgbd_pedestrian_detector (YOLOv11)
       |                      |
       +----------+-----------+
                  v
       lidar_camera_fusion
       (80/20 dist, 30/70 bearing)
                  |
    /fusion_pedestrian_position    /fusion_pedestrian_paths (multi-track LINE_STRIPs)
                  |                          |
                  v                          v
                  +------> diffusion_predictor
                                     |
                                     +---> /pedestrian_predictions_tensor (M × 20 × 2)
                                     +---> /person_prediction (RViz Marker)
                                     +---> /pedestrian_motion, /pedestrian_ttc
                                     |
                                     v
                              MPPI controller
                                     |
                                     v
                              PACMod2 → GEM e4 actuators
```

**Fusion path smoothing.** EMA on live track XY (`track_smooth_alpha`) + centered moving-average over the published path (`path_smoothing_window`, visualization-only). Tune in `src/adapt_full/config/sensor_fusion_params.yaml`.

**Diffusion.** MID-style Transformer denoiser; DDPM 100-step cosine schedule trained, DDIM 10-step inference. Joint variant adds cross-agent attention.

---

## Trained weights

| Mode | File |
|---|---|
| `joint` | `src/diffusion_prediction/models/diffusion/av2_joint_v2/ema_best.pt` (current default) |
| `joint` | `src/diffusion_prediction/models/diffusion/eth_ucy_ft_joint/ema_best.pt` |
| `joint` | `models/diffusion/av2_joint_v1/ema_best.pt` |
| `single` | `models/diffusion/av2_pretrain_v1/ema_best.pt` |

---

## Project structure

```
cs_588_g10/
├── docs/                       PROJECT.md, TODO.md, mid-report
├── src/
│   ├── adapt_full/             lidar_processing, lidar_camera_fusion, lidar_rgb_tracking_launch
│   ├── diffusion_prediction/   inference node, model, training, weights
│   ├── yolo_person_detector/   YOLOv11 RGBD detector
│   ├── vehicle_drivers/
│   │   ├── gem_mppi_control/   mppi_ros.py (standalone)
│   │   ├── gem_gnss_control/   pure_pursuit fallback
│   │   ├── gem_gnss_image/     RViz overlay for GNSS
│   │   └── gem_visualization/  URDF + RViz config
│   ├── basic_launch/           sensor_init, gnss, tf2, rviz_display
│   └── hardware_drivers/       LiDAR / camera / GNSS / PACMod2 drivers
├── models/                     trained weights (gitignored)
└── data/                       rosbags (gitignored)
```
