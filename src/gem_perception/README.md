# gem_perception_ros2

ROS2 **humble** parallel of `gem_perception` (ROS1 noetic). Same nodes, same
topics, same behavior - designed to run on the real GEM e4 onboard computer.
Drop this directory into your humble workspace and build with
`colcon build --symlink-install`.

For full deployment instructions see the top-level
[`docs/real_car_deploy.md`](../docs/real_car_deploy.md) and the pre-flight
[`docs/real_car_checklist.md`](../docs/real_car_checklist.md).

## Build

```bash
# inside a humble workspace, e.g. ~/gem_ws
cd ~/gem_ws/src
ln -s /path/to/gem_perception/ros2 gem_perception_ros2     # or: cp -r /path/to/gem_perception/ros2 ./gem_perception_ros2
cd ~/gem_ws
colcon build --symlink-install --packages-select gem_perception_ros2
source install/setup.bash
```

## Model weights

Resolved in order: `$GEM_PERCEPTION_MODELS` → `~/host/gem_perception_models/` →
`~/gem_perception_models/`. Run once to download:

```bash
ros2 run gem_perception_ros2 download_models
```

## Run - real e4 (recommended preset)

```bash
# URDF already publishes the camera optical-link
ros2 launch gem_perception_ros2 perception_real_e4.launch.py \
  default_prompt:="red sign"

# URDF lacks an optical-link → publish a static TF for it
ros2 launch gem_perception_ros2 perception_real_e4.launch.py \
  default_prompt:="red sign" \
  publish_optical_tf:=true \
  parent_camera_frame:=front_single_camera_link \
  optical_frame_name:=front_single_camera_optical_link

# Switch to LangSAM
ros2 launch gem_perception_ros2 perception_real_e4.launch.py \
  detector:=sam default_prompt:="pedestrian"

# Headless (no rviz)
ros2 launch gem_perception_ros2 perception_real_e4.launch.py \
  default_prompt:="red sign" run_rviz:=false
```

The real-car preset loads `config/perception_real_e4.yaml`; edit that for any
site-specific values (frame names, GPS reference lat/lon, model path, etc.).

## Run - sim-style defaults

```bash
ros2 launch gem_perception_ros2 perception_yolo.launch.py default_prompt:="red sign"
ros2 launch gem_perception_ros2 perception_sam.launch.py  default_prompt:="red sign"
```

Loads `config/perception.yaml` with sim topic names (`/oak/rgb/image_raw`,
`/ouster/points`, `front_single_camera_optical_link`, etc.).

## Runtime prompt change

```bash
ros2 topic pub -1 /perception/prompt std_msgs/String "data: 'red cone'"
```
