# gem_perception — Comprehensive Overview

Text-promptable 2D perception (YOLO-World or LangSAM) fused with LiDAR to
produce a navigation goal as a `geometry_msgs/PoseStamped` in both the world
(`map`) frame and the vehicle (`base_link`) frame. Designed for the Polaris
GEM platform; reusable on any rig with a calibrated camera + LiDAR + GNSS.

The same pipeline ships in two flavors:

- **ROS1 noetic**, for the Gazebo simulator (`./` — repo root).
- **ROS2 jazzy**, for the real car (`./ros2/`).

The two share a single Python core; only the framework wrappers differ.

---

## 1. What it does, end to end

You give the system a free-text prompt — `"red cone"`, `"red sign"`,
`"pedestrian"`, etc. — over a ROS topic. On every synced camera + LiDAR
frame it:

1. Runs a 2D detector (YOLO-World) or text-conditioned segmenter (LangSAM /
   GroundingDINO + SAM) and keeps the **single highest-confidence** match.
2. Projects the LiDAR cloud into the camera image with the calibrated
   intrinsics + extrinsics, keeping only the points whose pixel falls inside
   the detection mask. This is a 3D frustum-clip driven by the 2D detector.
3. Filters that subset: z-axis ground/sky cut in `base_link`, then
   statistical outlier removal.
4. Clusters with DBSCAN. Among the surviving clusters it picks the one whose
   centroid projects closest to the 2D mask centroid — this guarantees the
   3D answer matches the visual target even when the frustum contains
   background objects.
5. Publishes the cluster centroid as a goal in both `base_link` and `map`
   frames, plus a 3D bounding-box marker, the cluster's filtered point
   cloud, and an annotated camera image.
6. **If the LiDAR has no points on the object** (too far / occluded), the
   node emits an *estimated* goal 15 m along the camera ray through the
   mask centroid. As the car closes in and LiDAR starts seeing the object,
   the estimate is replaced by the real cluster — same topic, same
   `goal_pose`, just with `goal_is_estimated` flipped from `True` to `False`.
7. After detection is lost, the last goal is held for 2 s before going
   silent (so a single skipped frame doesn't yank the goal away).

This entire flow runs at the camera + LiDAR sync rate, GPU-accelerated.

---

## 2. Repository layout

```
gem_perception/                       ← repo root (= ROS1 catkin package)
├── README.md                         quickstart for both distros
├── LICENSE                           MIT
├── package.xml, CMakeLists.txt, setup.py
├── src/gem_perception/               framework-agnostic Python core
│   ├── geometry.py                   projection / clustering / TF helpers
│   ├── pipeline.py                   detection → fusion → goal
│   ├── yolo_detector.py              YOLO-World wrapper (ultralytics)
│   ├── sam_detector.py               LangSAM (py≥3.10) + GroundingDINO/SAM1 fallback (py3.8)
│   └── ros_common.py                 ROS-agnostic helpers (image overlays, GoalHold, K parsing)
├── scripts/                          ROS1 node executables
│   ├── yolo_perception_node.py
│   ├── sam_perception_node.py
│   ├── map_tf_broadcaster.py         publishes map→base_link from /septentrio_gnss/insnavgeod
│   ├── bev_overlay_node.py           draws the goal onto /motion_image (GNSS BEV)
│   └── download_models.py            one-shot weight fetch
├── launch/perception_yolo.launch
├── launch/perception_sam.launch
├── config/perception.yaml            params for all nodes
├── rviz/perception.rviz              visualisation layout
└── ros2/                             ROS2 ament_python package
    ├── CATKIN_IGNORE                 so catkin in a sim ws skips this folder
    ├── package.xml, setup.py
    ├── gem_perception_ros2/          rclpy node modules + duplicated core
    ├── launch/perception_*.launch.py
    ├── config/perception.yaml
    └── rviz/perception.rviz
```

The Python core (`geometry.py`, `pipeline.py`, detectors, `ros_common.py`)
is duplicated under `ros2/` so each side is a self-contained installable
package. They are intentionally byte-identical — keep them in sync when
editing.

---

## 3. The pipeline in detail

```
camera image ──┐
camera_info ───┤   ApproximateTime
LiDAR cloud ───┘   (slop ~0.2 s)
                 │
                 ▼
         ┌────────────────────┐
         │ 2D detect / segment│   YOLO-World OR LangSAM
         │   top-1 only       │
         └─────────┬──────────┘
                   │ bbox + mask + score
   ┌───────────────▼─────────────────┐
   │ Project LiDAR → camera image    │
   │   p_cam = T_cam_lidar · p_lidar │
   │   (u,v) = K · p_cam / p_cam.z   │
   │   keep mask[v,u] == True, z > 0 │
   └───────────────┬─────────────────┘
                   │ N points in base_link
   ┌───────────────▼──────────────────┐
   │ z-axis filter (no RANSAC)         │
   │   z_min < z_base < z_max          │
   │   (default 0.15 → 5.0 m)          │
   ├───────────────────────────────────┤
   │ Statistical outlier removal       │
   │   k=8, std_mul=2.0                │
   ├───────────────────────────────────┤
   │ DBSCAN  eps=0.4 m, min=3          │
   └───────────────┬──────────────────┘
                   │ ≥1 candidate cluster
        ┌──────────▼──────────────┐
        │ Pick cluster whose      │
        │ centroid projects       │
        │ closest to mask center  │
        └──────────┬──────────────┘
                   │
       ┌───────────▼───────────────┐
       │ goal_base = cluster mean  │   measured goal
       │ is_estimated = False      │
       └───────────┬───────────────┘
                   │
         OR (no cluster)
                   │
       ┌───────────▼───────────────┐
       │ goal_base = ray_cam·15 m  │   estimated goal
       │ is_estimated = True       │
       └───────────┬───────────────┘
                   │
       ┌───────────▼───────────────┐
       │ GoalHold (2 s)            │   smooth dropouts
       │ TF lookup map ← base_link │
       │ Publish all topics        │
       └───────────────────────────┘
```

### 3.1 Frustum clip (image → 3D)

All math is in `geometry.transform_points` + `project_to_image`:

```
T_cam_lidar = TF.lookup("front_single_camera_optical_link", "ouster")
P_cam       = T_cam_lidar · P_lidar           # (N,4) × (4,N) → (N,3)
(u,v)       = K · P_cam / Z                   # only Z>0 retained
inside      = mask[v_round, u_round] == True
P_keep      = P_lidar[inside]
```

This is dense (vectorised numpy) and works on the full ouster cloud at 10 Hz.

### 3.2 Why DBSCAN + closest-to-ray, not just closest-by-distance

The frustum clip can include LiDAR returns from objects *behind* the target
(another cone further down the line, a wall, etc.). Picking by 3D distance
to the camera would prefer near objects that aren't actually the target.
Picking by **2D image-plane distance** to the mask centroid is the correct
metric: it's what made the detector emit the bbox in the first place.

### 3.3 Estimated goal when LiDAR is too far

Past ~50 m the OS1-128 has only a few returns per object — many frames
will yield zero in-mask points. Rather than drop the goal entirely, the
node ray-casts the mask centroid forward 15 m and publishes that as a
provisional target with `goal_is_estimated=True`. As the car closes the
distance, real LiDAR points start landing on the object and the goal
snaps to the measured cluster automatically.

### 3.4 GoalHold — last-known persistence

`ros_common.GoalHold` keeps the most recent goal for 2 s after detection
stops. This bridges single-frame dropouts (occlusion, motion blur) without
the goal vanishing, and falls quiet only when the target is truly gone.

---

## 4. Detector backends

### 4.1 YOLO-World (`yolo_detector.py`)

Open-vocabulary YOLOv8. We use `yolov8s-worldv2.pt` (~26 MB).

- Prompt is set via `model.set_classes([...])`, which encodes the prompt
  with CLIP into class-conditional weights baked into the head.
- Inference is just `model.predict()`. Top-1 by `boxes.conf`.
- Mask is the rectangle of the top-1 bbox (YOLO doesn't produce masks).

#### Known Ultralytics quirk we patched around

After the first `predict(device="cuda")`, the YOLO and its cached CLIP are
on GPU. The next `set_classes(...)` calls `clip.tokenize(text)` which
returns CPU tensors — but CLIP is on GPU → `Expected all tensors on the
same device`. Workaround in `set_prompt`: temporarily move both the YOLO
and the cached CLIP to CPU, run `set_classes`, move back to the target
device. Implemented at `yolo_detector.py:set_prompt`.

### 4.2 LangSAM / SAM fallback (`sam_detector.py`)

A version-aware wrapper that picks one of two backends based on the
runtime Python:

- **Python ≥ 3.10** (real car / jazzy): `lang-sam` (Grounding-DINO + SAM 2).
- **Python < 3.10** (noetic docker, py 3.8): `groundingdino-py` +
  `segment-anything` (SAM 1, ViT-B). Same `set_prompt` / `infer` API.

Both produce real binary segmentation masks (not just bboxes), so the
frustum clip is much tighter and there's less chance of pulling in
background LiDAR.

### 4.3 Why both?

- YOLO-World is fast (~10 ms / frame on a 3090 Ti at 1280²) and gives a
  rectangle. Good when the target is not occluded by objects of similar
  shape, e.g. a single sign in an open scene.
- LangSAM is slower but produces a tight mask, so the LiDAR clip is much
  more precise for irregular shapes (people, bicycles, traffic cones with
  bases hidden in grass).

You pick which to run via the launch file. Both publish identical topics.

---

## 5. ROS API

### 5.1 Subscriptions

| Topic | Type | Purpose |
|---|---|---|
| `/oak/rgb/image_raw` | sensor_msgs/Image | front camera RGB |
| `/oak/rgb/camera_info` | sensor_msgs/CameraInfo | intrinsics K + D |
| `/ouster/points` | sensor_msgs/PointCloud2 | LiDAR |
| `/perception/prompt` | std_msgs/String | live prompt change |

(All four configurable via `~param`s; rename for the real car.)

### 5.2 Publications

| Topic | Type | Frame | Purpose |
|---|---|---|---|
| `/perception/goal_pose` | geometry_msgs/PoseStamped | **map** | navigation goal in world |
| `/perception/goal_pose_base_link` | geometry_msgs/PoseStamped | **base_link** | goal relative to the car |
| `/perception/goal_is_estimated` | std_msgs/Bool | — | true while running on a 15 m ray estimate |
| `/perception/image_annotated` | sensor_msgs/Image | camera | bbox + mask + centroid overlay |
| `/perception/lidar_projected_image` | sensor_msgs/Image | camera | LiDAR points colored by depth (calibration check) |
| `/perception/object_cluster` | sensor_msgs/PointCloud2 | base_link | filtered/clustered points of the chosen object |
| `/perception/object_bbox_3d` | visualization_msgs/MarkerArray | base_link | 3D AABB cube + goal sphere |
| `/motion_image_with_goal` | sensor_msgs/Image | — | the GNSS BEV image with goal marker drawn on |

### 5.3 TF tree assumed

```
map ← base_link ← front_camera_link ← front_single_camera_link
                        └─ front_single_camera_optical_link
       └ top_rack_link ← ouster_base_link ← ouster
```

Frames are configurable; defaults match the GEM URDF.

The `map → base_link` link is **not** part of the URDF chain on the GEM —
it's broadcast by the helper `map_tf_broadcaster` from `/septentrio_gnss/insnavgeod`,
using a fixed lat/lon anchor as the map origin. On the real car, replace
that helper with whatever localization stack publishes `map → base_link`.

### 5.4 Live prompt change

```bash
rostopic pub -1 /perception/prompt std_msgs/String "red cone"     # ROS1
ros2 topic pub -1 /perception/prompt std_msgs/String "data: 'red cone'"  # ROS2
```

The detector caches the encoded text representation, so consecutive prompt
changes are cheap. The text encoding only re-runs when the string actually
differs from the cached one.

---

## 6. Parameters

All declared in `config/perception.yaml`. Tune for your platform.

| Param | Default | Meaning |
|---|---|---|
| `z_min_base` | 0.15 m | drop ground returns |
| `z_max_base` | 5.0 m | drop ceiling / sky |
| `dbscan_eps` | 0.4 m | cluster connectivity radius |
| `dbscan_min_samples` | 3 | minimum cluster size |
| `min_cluster_points` | 3 | reject tiny clusters |
| `estimated_goal_distance` | 15.0 m | length of the ray-cast estimate |
| `goal_hold_seconds` | 2.0 s | how long to keep the last goal after detection loss |
| `device` | `cuda` | torch device |
| `image_topic` / `camera_info_topic` / `lidar_topic` / `prompt_topic` | as listed above | rename for the real car |
| `camera_frame` / `lidar_frame` / `base_frame` / `map_frame` | as listed | TF frame names |
| `ref_lat` / `ref_lon` | 40.092722 / -88.236365 | sim map origin (highbay) |

---

## 7. Visualization

`rviz/perception.rviz` (loaded by the launch files) shows:

- 3D world: `OusterPoints` (full LiDAR), `ObjectCluster` (winning cluster
  highlighted in green), `ObjectBBox3D` (goal sphere + cube),
  `GoalPose (map)` and `GoalPose (base_link)` arrows.
- 2D images: `ImageAnnotated` (bbox/mask overlay),
  `LiDARProjected` (calibration check),
  `BEVGoal` (GNSS BEV with goal marker).

Color convention: **green = measured**, **yellow = estimated**.

---

## 8. Models and persistence

Weights are downloaded once into the host-mounted directory
`~/host/gem_perception_models/` (`~/gem_perception_models/` on the host),
so a docker container restart never re-downloads.

```bash
python3 scripts/download_models.py            # ROS1
ros2 run gem_perception_ros2 download_models  # ROS2
```

| File | ~Size | Used by |
|---|---|---|
| `yolov8s-worldv2.pt` | 26 MB | YOLO-World |
| `groundingdino_swint_ogc.pth` + `sam_vit_b_01ec64.pth` | 1.0 GB | SAM1 fallback (py3.8) |
| HuggingFace cache (LangSAM) | ~1 GB on first run | SAM2 (py≥3.10) |
| `~/.cache/clip/ViT-B-32.pt` | 338 MB | YOLO-World text encoder (auto-fetched on first use) |

---

## 9. Sim quickstart

```bash
# Terminal 1 — simulator (already wired up earlier in this project)
roslaunch gem_launch gem_init.launch world_name:="highbay_track.world" \
  x:=12.5 y:=-21 yaw:=3.1416 custom_scene:=true

# Terminal 2 — perception
docker exec -it ros-noetic-container bash
export DISPLAY=:1
cd ~/host/Downloads/temp/gem_simulation_ws
catkin_make --only-pkg-with-deps gem_perception   # first time only
source devel/setup.bash
roslaunch gem_perception perception_yolo.launch default_prompt:="red sign"

# Terminal 3 — change prompt anytime
rostopic pub -1 /perception/prompt std_msgs/String "red cone"
```

## 10. Real-car (jazzy) quickstart

```bash
cd ~/jazzy_ws/src
git clone git@github.com:keeesuke/gem_perception.git
ln -s gem_perception/ros2 gem_perception_ros2
cd ~/jazzy_ws
colcon build --symlink-install
source install/setup.bash

ros2 run gem_perception_ros2 download_models
ros2 launch gem_perception_ros2 perception_yolo.launch.py default_prompt:="red sign"
```

You'll need your localization stack to publish `map → base_link`; the
included `map_tf_broadcaster` is a fallback that uses GPS + IMU yaw.

---

## 11. Limitations and future work

- **No tracking across frames.** Top-1 each frame; if two equally-likely
  candidates flicker between frames the goal will jump. Adding a simple
  IoU-based tracker on the 2D side would stabilise this.
- **No depth-camera fallback.** When ouster is too far we ray-estimate at
  15 m. We could instead use `/oak/depth/points` for a denser short-range
  fallback before giving up to the ray estimate.
- **TF lookup uses message stamp.** If your sensors have severe time-sync
  drift on the real car, this can fail; consider falling back to
  `Time(0)` after a short timeout (already done for the worst case, but
  could be improved with a stamp-history buffer).
- **Sim has perfect calibration.** The TF chain is ground truth, so the
  "calibration check" overlay (`/perception/lidar_projected_image`) is
  trivially perfect in sim. On the real car this view is the primary
  visual diagnostic for camera↔LiDAR extrinsic quality.

---

## 12. Files at a glance

| File | Role |
|---|---|
| `src/gem_perception/geometry.py` | numpy projection, DBSCAN, SOR, cluster pick |
| `src/gem_perception/pipeline.py` | the orchestrator: detection → fusion → goal |
| `src/gem_perception/yolo_detector.py` | YOLO-World wrapper + the cuda/cpu workaround |
| `src/gem_perception/sam_detector.py` | LangSAM (py≥3.10) and GD+SAM1 (py3.8) backends |
| `src/gem_perception/ros_common.py` | image overlays, K parsing, GoalHold, framework-agnostic |
| `scripts/yolo_perception_node.py` | ROS1 wrapper around `pipeline.run_pipeline` |
| `scripts/sam_perception_node.py` | same, but with the SAM detector |
| `scripts/map_tf_broadcaster.py` | publishes map→base_link from GNSS |
| `scripts/bev_overlay_node.py` | overlays goal on `/motion_image` (GNSS BEV) |
| `scripts/download_models.py` | weight fetcher |
| `ros2/gem_perception_ros2/*.py` | ROS2 mirror of all of the above |
