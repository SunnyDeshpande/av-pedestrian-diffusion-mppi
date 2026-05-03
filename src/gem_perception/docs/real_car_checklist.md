# Real-car (GEM e4 / ROS2 humble) deployment checklist

Run these commands **on the real-car computer** (with sensors/drivers already
launched, e.g. via `ros2 launch basic_launch sensor_init.launch.py`). Paste the
output back so the perception config can be finalized for the real car.

---

## 1. ROS2 environment

```bash
# Confirm ROS2 distro
echo "ROS_DISTRO=$ROS_DISTRO"
ros2 --version

# Python in use
python3 --version
which python3
```

**Expected:** `ROS_DISTRO=humble`, Python 3.10.x.

---

## 2. Compute platform (matters for torch wheel selection)

```bash
# CPU & arch
uname -m                 # x86_64 → desktop GPU; aarch64 → Jetson
lscpu | grep "Model name"

# GPU
nvidia-smi 2>/dev/null | head -3 || echo "no nvidia-smi"
ls /etc/nv_tegra_release 2>/dev/null && echo "Jetson detected"

# CUDA visible to torch (if torch already installed)
python3 -c 'import torch; print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)' 2>&1 | head -3
```

---

## 3. Active topics (after sensors are running)

```bash
# All topics
ros2 topic list

# Front camera image
ros2 topic info /oak/rgb/image_raw 2>&1
ros2 topic info /oak/rgb/camera_info 2>&1
ros2 topic info /lucid/camera_fl/image_raw 2>&1
ros2 topic info /lucid/camera_fl/camera_info 2>&1

# Image rate + first message header (for stamp/format/encoding)
timeout 3 ros2 topic hz /oak/rgb/image_raw 2>&1 | tail -3
timeout 2 ros2 topic echo --once /oak/rgb/image_raw --field header 2>&1
timeout 2 ros2 topic echo --once /oak/rgb/image_raw --field encoding 2>&1
timeout 2 ros2 topic echo --once /oak/rgb/image_raw --field height 2>&1
timeout 2 ros2 topic echo --once /oak/rgb/image_raw --field width 2>&1

# Same for Lucid front-left (skip if you don't plan to use it)
timeout 3 ros2 topic hz /lucid/camera_fl/image_raw 2>&1 | tail -3
timeout 2 ros2 topic echo --once /lucid/camera_fl/image_raw --field header 2>&1
timeout 2 ros2 topic echo --once /lucid/camera_fl/image_raw --field encoding 2>&1

# LiDAR
ros2 topic info /ouster/points 2>&1
timeout 3 ros2 topic hz /ouster/points 2>&1 | tail -3
timeout 2 ros2 topic echo --once /ouster/points --field header 2>&1

# GNSS
timeout 2 ros2 topic echo --once /septentrio_gnss/insnavgeod --field header 2>&1
timeout 2 ros2 topic echo --once /septentrio_gnss/navsatfix --field header 2>&1
timeout 2 ros2 topic echo --once /septentrio_gnss/imu --field header 2>&1
```

**What I need from the output:**

- Which front camera you want perception to consume: OAK-D (`/oak/rgb/...`) or Lucid FL (`/lucid/camera_fl/...`) — answer in your reply.
- The `frame_id` printed in the chosen image's `header` (this is the camera optical frame to remap to).
- The `frame_id` printed in `/ouster/points` `header` (this becomes `lidar_frame`).
- The encoding of the chosen camera (`rgb8` or `bgr8` — affects color order in YOLO/SAM input).
- Image resolution.

---

## 4. CameraInfo (intrinsics + distortion)

```bash
# OAK
timeout 2 ros2 topic echo --once /oak/rgb/camera_info 2>&1 | head -50
# Lucid FL
timeout 2 ros2 topic echo --once /lucid/camera_fl/camera_info 2>&1 | head -50
```

**What I need:** confirm the K matrix is non-zero and whether `D` is populated (non-empty `[k1,k2,p1,p2,k3]`). If `D` has values > ~0.05, distortion correction matters.

---

## 5. TF tree

```bash
# Static + dynamic frames
ros2 run tf2_tools view_frames -o /tmp/real_e4_frames        # generates frames.pdf and .gv
ls -la /tmp/real_e4_frames* 2>&1

# Or text-list, easier to paste:
ros2 run tf2_ros tf2_echo map base_link 2>&1 | head -10      # FAILS = no map TF
ros2 run tf2_ros tf2_echo base_link os_lidar 2>&1 | head -10
ros2 run tf2_ros tf2_echo base_link front_single_camera_link 2>&1 | head -10
ros2 run tf2_ros tf2_echo base_link front_single_camera_optical_link 2>&1 | head -10
ros2 run tf2_ros tf2_echo base_link oak-d-base-frame 2>&1 | head -10
ros2 run tf2_ros tf2_echo base_link CAMERA_FL 2>&1 | head -10
```

**What I need:**

1. Does `map → base_link` exist? (If `tf2_echo map base_link` returns a transform, **yes**; if it errors with "Could not find a connection", **no** — perception will run in `base_link` only, or we wire up the helper.)
2. Does an **optical-frame** for the chosen camera exist? (`front_single_camera_optical_link` or any frame ending `_optical_link`)
3. Confirm the LiDAR cloud's frame matches what TF publishes (probably `os_lidar` or `os_sensor`).

If `map` doesn't exist *and* you want goals in map: I'll wire up the helper against `/septentrio_gnss/navsatfix` + `/septentrio_gnss/imu`. Confirm those topics are present from §3.

---

## 6. GPU + Python deps prerequisites

```bash
# These will be needed for inference. Just check they import/run:
python3 -c "import torch, torchvision, ultralytics, cv2, numpy, sklearn; print('OK')" 2>&1 | tail -5
python3 -c "import lang_sam" 2>&1 | tail -3
python3 -c "import groundingdino, segment_anything" 2>&1 | tail -3
```

If any of those fail, paste the errors. If `torch` reports `cuda:False`, GPU isn't visible to torch and we need to fix that before deployment (Jetson: re-install JetPack torch; x86: check CUDA driver).

---

## 7. (Optional) Network / DDS

Real-car DDS may require a specific RMW. Check:

```bash
echo "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
```

If multiple computers participate, ensure same `RMW_IMPLEMENTATION` and domain ID across them.

---

## What to paste back

A short reply containing:

1. ROS distro + Python version
2. Compute platform + GPU info
3. Chosen front camera (OAK or Lucid_FL)
4. From §3: encoding, resolution, frame_id of chosen image; frame_id of /ouster/points
5. From §4: whether D distortion vector is non-zero
6. From §5: yes/no on `map → base_link`; yes/no on optical-link frame existing
7. From §6: did all imports succeed?

That's enough to finalize `config/perception_real_e4.yaml` and (if the optical-link is missing) write the static TF launch line.
