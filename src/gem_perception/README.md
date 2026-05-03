# gem_perception

Text-promptable 2D detection / segmentation (YOLO-World or LangSAM) fused with
LiDAR to produce a goal `PoseStamped` in `map` and `base_link` frames. Built
for the Polaris GEM platform but reusable on any camera + LiDAR + GNSS rig.

Two packages live in this repo:

| Path | ROS distro | Workspace tool |
|---|---|---|
| `./` (this dir) | ROS1 noetic — Gazebo simulator | catkin |
| `./ros2/` | ROS2 humble — real GEM e4 | colcon (ament_python) |

The two share the same Python core (`geometry.py`, `pipeline.py`, detectors,
`ros_common.py`); only the framework wrappers differ. Edit either side and
keep the byte-for-byte mirror in sync.

## ROS1 (sim) — quickstart

Inside the noetic docker container:

```bash
# one-time deps
pip3 install --user torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118
pip3 install --user ultralytics opencv-python scikit-learn \
  groundingdino-py segment-anything 'git+https://github.com/openai/CLIP.git'

# clone + build
cd ~/host/Downloads/temp/gem_simulation_ws/src
git clone git@github.com:keeesuke/gem_perception.git
cd ~/host/Downloads/temp/gem_simulation_ws
catkin_make --only-pkg-with-deps gem_perception
source devel/setup.bash

# one-time model download (persistent on host)
python3 src/gem_perception/scripts/download_models.py

# run
roslaunch gem_perception perception_yolo.launch default_prompt:="red sign"
# or LangSAM
roslaunch gem_perception perception_sam.launch default_prompt:="red sign"

# change the prompt at runtime
rostopic pub -1 /perception/prompt std_msgs/String "red cone"
```

## ROS2 (real GEM e4, humble) — quickstart

Run on the **real-car computer** (the e4 onboard with humble installed). Full
deployment guide: [`docs/real_car_deploy.md`](docs/real_car_deploy.md). Pre-flight
checklist (commands to gather camera/LiDAR/TF info before launching):
[`docs/real_car_checklist.md`](docs/real_car_checklist.md).

```bash
# one-time deps (humble + Python ML)
sudo apt install -y \
  ros-humble-cv-bridge ros-humble-image-geometry \
  ros-humble-tf2-ros ros-humble-tf2-geometry-msgs \
  ros-humble-message-filters ros-humble-sensor-msgs-py \
  ros-humble-rviz2 python3-pip
pip3 install --user torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118   # x86 + dGPU; Jetson uses NVIDIA's wheel
pip3 install --user ultralytics opencv-python scikit-learn \
  'git+https://github.com/openai/CLIP.git' lang-sam

# clone + build inside your humble workspace (e.g. ~/gem_ws)
cd ~/gem_ws/src
git clone git@github.com:keeesuke/gem_perception.git
ln -s gem_perception/ros2 gem_perception_ros2     # or copy the dir
cd ~/gem_ws
colcon build --symlink-install --packages-select gem_perception_ros2
source install/setup.bash

# one-time model download (persistent at ~/gem_perception_models/)
ros2 run gem_perception_ros2 download_models

# launch (real e4 preset)
ros2 launch gem_perception_ros2 perception_real_e4.launch.py \
  default_prompt:="red sign"

# if your URDF doesn't have an *_optical_link child for the camera:
ros2 launch gem_perception_ros2 perception_real_e4.launch.py \
  default_prompt:="red sign" \
  publish_optical_tf:=true \
  parent_camera_frame:=front_single_camera_link \
  optical_frame_name:=front_single_camera_optical_link

# switch detector
ros2 launch gem_perception_ros2 perception_real_e4.launch.py \
  detector:=sam default_prompt:="pedestrian"

# change the prompt at runtime
ros2 topic pub -1 /perception/prompt std_msgs/String "data: 'red cone'"
```

A bare-bones `perception_yolo.launch.py` / `perception_sam.launch.py` are
also provided; they consume the default `config/perception.yaml` (sim-style
topic names) — the real-car preset above loads `config/perception_real_e4.yaml`
instead.

## Topics

| Topic | Type | Frame | Purpose |
|---|---|---|---|
| `/perception/goal_pose` | geometry_msgs/PoseStamped | **map** | navigation goal in world |
| `/perception/goal_pose_base_link` | geometry_msgs/PoseStamped | **base_link** | goal relative to the car |
| `/perception/goal_is_estimated` | std_msgs/Bool | — | true when LiDAR was too far → 15 m ray estimate |
| `/perception/image_annotated` | sensor_msgs/Image | camera | 2D bbox / mask overlay |
| `/perception/lidar_projected_image` | sensor_msgs/Image | camera | LiDAR points projected onto image (calibration check) |
| `/perception/object_cluster` | sensor_msgs/PointCloud2 | base_link | filtered + clustered LiDAR points of the winning object |
| `/perception/object_bbox_3d` | visualization_msgs/MarkerArray | base_link | 3D AABB + goal sphere |
| `/motion_image_with_goal` | sensor_msgs/Image | — | GNSS BEV with goal marker |
| `/perception/prompt` (sub) | std_msgs/String | — | live prompt change |

## Pipeline (per synced camera + LiDAR frame)

1. **2D detection** — YOLO-World or LangSAM, top-1 by confidence.
2. **Project all LiDAR points into the image** using K (and **D**, if the
   camera publishes a non-zero distortion vector), keep those whose pixel
   falls inside the mask.
3. **Z-axis filter** in `base_link` (default `0.15 < z < 5.0` m) — drops
   ground and sky returns.
4. **Statistical outlier removal**.
5. **DBSCAN** in `base_link`; pick the cluster whose centroid projects
   closest to the mask centroid (image-pixel distance, distortion-aware).
6. Cluster found → centroid is the goal. None → 15 m along the camera ray
   through the mask centroid (`goal_is_estimated=True`); the same code path
   transparently switches back to a measured goal as the car closes in.
7. **Last goal held** for 2 s after detection loss before going silent.

Distortion correction is automatic: when `CameraInfo.D` has any non-zero
element, projection / pixel-to-ray / cluster scoring use OpenCV's
`projectPoints` and `undistortPoints` instead of the cheap pinhole math.
Sim cameras (D = zeros) get the cheap path; real Lucid / OAK cameras
(populated `plumb_bob` D) get accurate edge-of-image projection.

## Models

Weights are resolved in this order, both at download and at runtime:

1. `$GEM_PERCEPTION_MODELS` env var (explicit override).
2. `~/host/gem_perception_models/` if it exists (docker-on-host pattern).
3. `~/gem_perception_models/` (real-car / non-docker default).

```bash
python3 scripts/download_models.py            # ROS1 (sim)
ros2 run gem_perception_ros2 download_models  # ROS2 (real car)
```

- `yolov8s-worldv2.pt` (~26 MB) — YOLO-World small (open-vocabulary)
- `groundingdino_swint_ogc.pth` + `sam_vit_b_01ec64.pth` — Grounding-DINO + SAM1, the Python 3.8 fallback used in the noetic container
- LangSAM (SAM2) downloads on first use via the HuggingFace cache (Python ≥ 3.10, used on the real car)

## Frames assumed

| Frame | Default name (sim) | Default name (real e4) | Purpose |
|---|---|---|---|
| Camera optical | `front_single_camera_optical_link` | configurable; add via static TF if URDF lacks it | image / depth optical (Z-forward) |
| LiDAR | `ouster` | `os_lidar` | point cloud |
| Vehicle body | `base_link` | `base_link` | clustering, marker output |
| World | `map` | `map` (must be broadcast; helper `map_tf_broadcaster` provided) | nav goal output |

All four are ROS params, so they can be remapped without rebuilds. The
real-car config in `ros2/config/perception_real_e4.yaml` carries the e4
defaults and a few `[TBD]` lines for site-specific values.

## Real-car deployment

1. Run `docs/real_car_checklist.md` on the e4 to capture: camera choice,
   frame names, encoding/resolution, whether `map → base_link` is already
   published, distortion vector status, and GPU/torch sanity.
2. Edit `ros2/config/perception_real_e4.yaml` with the values you got.
3. Build + launch per `docs/real_car_deploy.md`.

## License

MIT — see `LICENSE`.
