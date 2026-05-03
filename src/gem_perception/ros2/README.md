# gem_perception_ros2

ROS2 (jazzy-compatible) parallel of `gem_perception` (ROS1 noetic). Same nodes,
same topics, same behavior. Copy this directory into a jazzy workspace on the
real car and build with `colcon build --symlink-install`.

## Build

```bash
# inside a jazzy workspace
cp -r /path/to/gem_perception_ros2 ~/jazzy_ws/src/
cd ~/jazzy_ws
colcon build --symlink-install
source install/setup.bash
```

## Run

```bash
ros2 launch gem_perception_ros2 perception_yolo.launch.py default_prompt:="red sign"
# or
ros2 launch gem_perception_ros2 perception_sam.launch.py default_prompt:="red sign"
```

## Runtime prompt change

```bash
ros2 topic pub -1 /perception/prompt std_msgs/String "data: 'red cone'"
```

## Model weights

Run once to prefill caches at `~/gem_perception_models/`:

```bash
ros2 run gem_perception_ros2 download_models
```
