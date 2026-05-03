# gem_perception

Text-promptable 2D detection / segmentation (YOLO-World or LangSAM) fused with
LiDAR to produce a goal `PoseStamped` in `map` and `base_link` frames. Built
for the Polaris GEM platform but reusable on any camera + LiDAR + GNSS rig.

Two packages live in this repo:

| Path | ROS distro | Workspace tool |
|---|---|---|
| `./` (this dir) | ROS1 noetic | catkin |
| `./ros2/` | ROS2 jazzy | colcon (ament_python) |

The two share the same Python core (`geometry.py`, `pipeline.py`, detector
wrappers); only the framework wrappers differ.

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

## ROS2 (real car, jazzy) — quickstart

```bash
cd ~/jazzy_ws/src
git clone git@github.com:keeesuke/gem_perception.git
ln -s gem_perception/ros2 gem_perception_ros2  # or: cp -r gem_perception/ros2 ./gem_perception_ros2
cd ~/jazzy_ws
colcon build --symlink-install
source install/setup.bash

ros2 run gem_perception_ros2 download_models   # one-time
ros2 launch gem_perception_ros2 perception_yolo.launch.py default_prompt:="red sign"
```

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

1. 2D detection → top-1 by confidence.
2. Project all LiDAR points into the image, keep those whose pixel falls inside the mask.
3. Z-axis filter in `base_link` (default `0.15 < z < 5.0` m).
4. Statistical outlier removal.
5. DBSCAN; pick cluster whose centroid projects closest to the mask centroid.
6. Cluster found → centroid is the goal. None → 15 m along camera ray (estimated).
7. Last goal held for 2 s after detection loss.

## Models

Weights are downloaded once into `~/host/gem_perception_models/` (host-mounted)
so container restarts don't re-download. Re-fetch with:

```bash
python3 scripts/download_models.py            # ROS1
ros2 run gem_perception_ros2 download_models  # ROS2
```

- `yolov8s-worldv2.pt` (~26 MB) — YOLO-World small (open-vocabulary)
- `groundingdino_swint_ogc.pth` + `sam_vit_b_01ec64.pth` — Grounding-DINO + SAM1 (Python 3.8 fallback used in the noetic container)
- LangSAM (SAM2) downloads on first use via the HuggingFace cache (Python ≥ 3.10, used on the real car)

## Frames assumed

| Frame | Default name | Purpose |
|---|---|---|
| Camera optical | `front_single_camera_optical_link` | image / depth optical (Z-forward) |
| LiDAR | `ouster` | point cloud |
| Vehicle body | `base_link` | clustering, marker output |
| World | `map` | nav goal output (must be broadcast; helper `map_tf_broadcaster` provided) |

All four are `~params` on the nodes, so they can be remapped without rebuilds.

## License

MIT — see `LICENSE`.
