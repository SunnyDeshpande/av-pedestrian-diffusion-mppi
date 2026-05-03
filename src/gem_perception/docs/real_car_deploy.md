# Deploying gem_perception on the real GEM e4 (ROS2 humble)

This guide assumes you have answered every question in
`docs/real_car_checklist.md` and pasted the output back so the perception
config can be finalised. Run the steps **on the real-car computer**.

## 1. One-time install

```bash
# 1a. ROS2 humble + system deps (skip whatever's already installed)
sudo apt update
sudo apt install -y \
  ros-humble-cv-bridge ros-humble-image-geometry \
  ros-humble-tf2-ros ros-humble-tf2-geometry-msgs \
  ros-humble-message-filters ros-humble-sensor-msgs-py \
  ros-humble-rviz2 \
  python3-pip

# 1b. Python ML stack (use the wheel that matches your platform)
# x86 + dGPU (RTX-class):
pip3 install --user torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
# Jetson Orin (JetPack 5.1+): use NVIDIA's wheel index — see
#   https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html

pip3 install --user ultralytics opencv-python scikit-learn
pip3 install --user 'git+https://github.com/openai/CLIP.git'
# Choose ONE SAM backend:
pip3 install --user lang-sam            # Python ≥ 3.10 (recommended on humble)
# OR (only if lang-sam isn't available)
pip3 install --user groundingdino-py segment-anything

# 1c. Pre-download model weights to ~/gem_perception_models
mkdir -p ~/gem_perception_models
python3 path/to/gem_perception/ros2/gem_perception_ros2/download_models.py
```

## 2. Build the perception package

```bash
cd ~/gem_ws/src         # your humble workspace src dir
ln -s /path/to/cs_588_g10/src/gem_perception/ros2 gem_perception_ros2
# (or copy: cp -r /path/to/cs_588_g10/src/gem_perception/ros2 gem_perception_ros2)

cd ~/gem_ws
colcon build --symlink-install --packages-select gem_perception_ros2
source install/setup.bash
```

## 3. Edit `config/perception_real_e4.yaml`

Open `gem_perception_ros2/config/perception_real_e4.yaml` and update only
the few `[TBD]` lines based on the checklist results:

- `image_topic` / `camera_info_topic` — chosen front camera (`/oak/...` or
  `/lucid/camera_fl/...`)
- `camera_frame` — the optical frame name. **If your URDF doesn't have an
  optical-link child for the chosen camera, leave the default and pass the
  optical-TF launch args (see step 4 below).**
- `lidar_frame` — usually `os_lidar` (matches what `/ouster/points.header.frame_id`
  reports)
- `ref_lat` / `ref_lon` — anchor for the local map origin (any fixed point
  near where you operate)

## 4. Launch

```bash
# 4a. Make sure sensor drivers + URDF (TF) are already running, e.g.
ros2 launch basic_launch sensor_init.launch.py

# 4b. Then perception
# (URDF already publishes <camera>_optical_link → no extra args needed)
ros2 launch gem_perception_ros2 perception_real_e4.launch.py \
  default_prompt:="red sign"

# (URDF lacks the optical link → publish a static TF for it)
ros2 launch gem_perception_ros2 perception_real_e4.launch.py \
  default_prompt:="red sign" \
  publish_optical_tf:=true \
  parent_camera_frame:=front_single_camera_link \
  optical_frame_name:=front_single_camera_optical_link

# Switch detector
ros2 launch gem_perception_ros2 perception_real_e4.launch.py \
  detector:=sam default_prompt:="pedestrian"

# Headless (no rviz)
ros2 launch gem_perception_ros2 perception_real_e4.launch.py \
  default_prompt:="red sign" run_rviz:=false
```

## 5. Change the prompt at runtime

```bash
ros2 topic pub -1 /perception/prompt std_msgs/String "data: 'orange cone'"
```

## 6. End-to-end smoke test (in this order)

```bash
# 6a. TF
ros2 run tf2_tools view_frames -o /tmp/check
# expect: map → base_link, base_link → <camera_optical>, base_link → os_lidar

# 6b. Topics published
ros2 topic list | grep perception
# expect /perception/{image_annotated,lidar_projected_image,object_cluster,
#                     object_bbox_3d,goal_pose,goal_pose_base_link,
#                     goal_is_estimated} all present

# 6c. Image overlay (rviz Image displays)
# - /perception/lidar_projected_image  → LiDAR points should align with edges
#   in the image. If they look offset, calibration is off.
# - /perception/image_annotated        → the prompted object should be boxed
#   in green when LiDAR confirms it, yellow when estimated.

# 6d. Goal sanity
ros2 topic echo /perception/goal_pose_base_link --once
ros2 topic echo /perception/goal_pose --once         # only when map TF is up
```

## 7. Troubleshooting cheat sheet

| Symptom | Cause | Fix |
|---|---|---|
| `TF lookup failed` | wrong frame name in config | edit `config/perception_real_e4.yaml`; verify with `tf2_echo` |
| `/perception/goal_pose` silent, base_link goal works | no `map → base_link` TF | run an EKF, or use the included `map_tf_broadcaster` against Septentrio topics |
| Cloud projection looks rotated 90° in rviz | camera_frame is body-conv, not optical | enable `publish_optical_tf:=true` or fix URDF |
| `Expected all tensors to be on the same device` | YOLO-World cuda/cpu mismatch | already handled in `yolo_detector.set_prompt`; only happens if a stale build is loaded |
| `cuda: False` from torch | wrong wheel for the GPU | reinstall the right torch wheel (Jetson vs x86+dGPU) |
| Camera rate OK, sync callback never fires | timestamps drift > 0.2 s | increase `slop` in the launch (currently hard-coded; can be made a param) |
| LiDAR too sparse on objects > 30 m | normal for OS1-128 | rely on the estimated-goal branch (yellow in rviz) until closer |
| Image colors look wrong (BGR vs RGB) | Lucid encodes rgb8; cv_bridge converts to bgr8 (auto), so usually fine | if not, set `desired_encoding` explicitly in cv_bridge call |

## 8. After-deploy hygiene

- `git pull` on this repo before each test session — sim updates flow into
  the ROS2 core via the `src/gem_perception/*.py ↔ ros2/gem_perception_ros2/*.py`
  byte-for-byte mirror.
- Save a rosbag of the first successful run for regression testing later:
  `ros2 bag record /perception/{goal_pose,goal_is_estimated,image_annotated} /tf /tf_static /ouster/points /oak/rgb/{image_raw,camera_info}`.
