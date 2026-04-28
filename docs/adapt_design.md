---
title: "Adapt — Design Document"
subtitle: "MPPI Motion Planner for the UIUC Polaris GEM e4"
author: "Sunny Deshpande — UIUC AVSE"
date: "2026-04-22"
geometry: margin=1in
fontsize: 11pt
documentclass: article
colorlinks: true
linkcolor: NavyBlue
urlcolor: NavyBlue
toc: true
toc-depth: 2
numbersections: true
header-includes:
  - \usepackage{booktabs}
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{mathtools}
---

\newpage

# Executive Summary

**Adapt** replaces AutoShield's Stanley lateral controller on the UIUC
Polaris GEM e4 with a **Model Predictive Path Integral (MPPI)** motion
planner. MPPI samples hundreds of candidate control sequences per tick,
weighs them by a cost function, and emits a softmax-averaged update. It
reasons jointly about steering **and** acceleration, and natively accepts
non-differentiable costs — notably, obstacle clearance costs derived from
AutoShield's pedestrian prediction pipeline.

Current status (2026-04-22):

- **Phase 1 — MPPI controller: complete.** Standalone numerical test:
  mean |lateral error| = 0.007 m, max 0.024 m, 100 % of post-warmup steps
  within the 0.5 m spec.
- **Unified ROS2 workspace: complete.** Adapt is now a self-contained
  colcon workspace with AutoShield's pipeline, the GEM e4 vehicle
  description, and the `pacmod2_msgs` / `septentrio_gnss_driver` /
  `nmea_msgs` message packages all vendored as siblings under
  `cs_588_g10/src/`. 11 packages build cleanly.
- **AutoShield-native controller: complete.** `adapt_mppi_node`
  speaks AutoShield's topic contract (NavSatFix + INSNavGeod +
  VehicleSpeedRpt + `/pacmod/enabled` inputs; PACMod2 commands outputs)
  and consumes `/fusion_pedestrian_position` (polar ego-frame
  `Int32MultiArray` — the existing AutoShield pedestrian tracker
  output) as the MPPI obstacle source.
- **Sim-bridge infrastructure: scaffolded, not yet run closed-loop.** A
  thin ROS1 bridge exposes the pure-numpy MPPI to the POLARIS GEM Gazebo
  simulator (Noetic, Docker).
- **Next:** first closed-loop run in `track1.world` on the E4, then
  pedestrian-prediction integration (shape change of the obstacle cost
  from static `(M, 2)` to time-indexed `(M, H, 2)` — the aspirational
  diffusion-policy research track).

\newpage

# Project Background

## What Adapt is

A UIUC AVSE research project retrofitting AutoShield's autonomy stack to
evaluate sampling-based MPC (MPPI) as the production motion planner on
the Polaris GEM e4. Deliverables are both a working controller and a
reproducible evaluation pipeline (standalone test → Gazebo sim → real
vehicle).

A parallel research track explores replacing AutoShield's current
pedestrian-motion predictor with a **diffusion policy** that generates
multi-modal future trajectories. Whether this lands in the current
project window is uncertain; the MPPI controller is designed so the
prediction source can be swapped without touching the planner — the
obstacle-cost tensor just changes shape from $(M, 2)$ to $(M, H, 2)$
(see §Next design phase).

## Stack position

```
LiDAR / RGB-D --> Sensor Fusion --> Pedestrian Prediction --> High-Level Decision
                                                                     |
                                                                     v
                                                 Pedestrian-Aware Path Planner
                                                                     |
                                                                     v
                                                  MPPI (Adapt)  <--  replaces Stanley
                                                                     |
                                                                     v
                                                         PACMod  steering / throttle
```

Adapt touches only the controller node. Upstream (perception, prediction,
path planner) and downstream (PACMod command topics) are preserved.

## Why MPPI (and not Stanley / QP-MPC / iLQR)

Stanley is reactive (no prediction), steering-only (longitudinal is a
decoupled PID), and every new safety consideration requires re-deriving
gains. Linear MPC needs convex costs, which rules out the clipped
quadratic clearance penalty used for obstacles. iLQR is on the edge of
real-time at $H{=}30$ on CPU.

MPPI wins because:

1. **Non-smooth costs are trivial.** Obstacle cost is zero outside the
   clearance radius and quadratic inside — perfectly fine for sampling.
2. **CPU-friendly at scale.** The same NumPy broadcast that rolls out
   $K{=}600$ candidates also evaluates their costs. Wall time per tick
   $\approx 15$ ms on a laptop CPU.
3. **Composable costs.** Future terms (time-indexed pedestrian
   predictions, uncertainty-weighted clearances) are absorbed without
   re-tuning anything — only weight *ratios* matter.

## Where the GEM Gazebo sim fits

Phase 1 was verified with a standalone numerical test (no physics engine,
no ROS). Before real-vehicle deployment, we need closed-loop evaluation
under a full rigid-body simulator with realistic actuator lag and sensor
timing. The **POLARIS GEM Simulator** (UIUC, ROS1 Noetic + Gazebo, shipped
as a Docker image) fills that role. A thin bridge package exposes the
pure-numpy MPPI to the ROS1 sim.

\newpage

# System Architecture

## Pipeline (data flow)

```
┌─────────────────────────── SENSORS ─────────────────────────────────┐
│   Ouster OS1-128     Livox HAP      OAK-D LR      Septentrio GNSS   │
│   (top LiDAR)        (front LiDAR)  (stereo)      + INS             │
└───────┬──────────────────┬───────────────┬─────────────┬────────────┘
        │ PointCloud2      │ PointCloud2   │ Image+Depth │ NavSatFix
        │                  │               │             │ INSNavGeod
        ▼                  ▼               ▼             │
 ┌───────────────────┐           ┌───────────────────┐   │
 │ adapt_full / │           │ yolo_person_      │   │
 │ lidar_processing  │           │ detector          │   │
 └─────────┬─────────┘           └──────────┬────────┘   │
           │ Int32MultiArray                │            │
           │ /lidar_pedestrian_position     │            │
           │                                │            │
           ▼              Int32MultiArray   ▼            │
        ┌─────────────────────────────┐                  │
        │ adapt_full /           │                  │
        │ lidar_camera_fusion         │                  │
        └─────────────────────────────┘                  │
                    │                                    │
                    │ Int32MultiArray                    │
                    │ /fusion_pedestrian_position        │
                    │ [dist_m, bearing_deg, …]           │
                    │ (ego-frame polar)                  │
                    ▼                                    ▼
  PACMod      ┌────────────────────────────────────────────┐
  /vehicle_   │           adapt_mppi_node                  │
  speed_rpt──►│  1. decode polar → world-frame (M, 2)      │
  /enabled ──►│  2. load waypoints CSV → ReferencePath     │
              │  3. MPPI.update(state, path, obstacles)    │
              │     K=600 samples, H=30 @ 10 Hz            │
              │  4. (δ, a) → steering-wheel + throttle PID │
              └────────────────────────────────────────────┘
                         │
                         ▼  pacmod2_msgs
                  /pacmod/{steering,accel,brake,global,shift,turn}_cmd
                         │
                         ▼
                      PACMod2  →  GEM e4 actuators
```

Key invariants:

* **Pedestrian tracking / prediction is unchanged from AutoShield.** The
  MPPI consumes whatever `/fusion_pedestrian_position` emits; it treats
  each detection as a static per-tick obstacle in world frame. A future
  diffusion-policy predictor can slot in here by republishing on the
  same topic (or upgrading to a `(M, H, 2)` variant — see next-phase
  section).
* **MPPI replaces only the controller.** Stanley, the pedestrian-aware
  path planner, and the safety state machine remain in the workspace
  as optional fallbacks — not wired into `adapt_mppi_launch.py`.
* **Commands are PACMod2-native.** No bridge nodes needed between MPPI
  and the drive-by-wire — `adapt_mppi_node` publishes the full six-topic
  PACMod2 suite exactly as Stanley did.


## ROS 2 topic contract — canonical `adapt_mppi_node`

AutoShield-native contract. Same I/O as `adapt_stanley_controller`
so it's a true drop-in replacement, plus one new pedestrian
subscription.

| Direction | Topic | Type | Notes |
|---|---|---|---|
| in | `/navsatfix` | `sensor_msgs/NavSatFix` | GPS lat/lon from Septentrio |
| in | `/insnavgeod` | `septentrio_gnss_driver/INSNavGeod` | INS heading |
| in | `/pacmod/enabled` | `std_msgs/Bool` | PACMod enable gate |
| in | `/pacmod/vehicle_speed_rpt` | `pacmod2_msgs/VehicleSpeedRpt` | Filtered speed |
| in | `/fusion_pedestrian_position` | `std_msgs/Int32MultiArray` | Flat `[dist_m, bearing_deg, …]`, ego polar, from `adapt_lidar_camera_fusion` |
| out | `/pacmod/global_cmd` | `pacmod2_msgs/GlobalCmd` | Enable + clear-override handshake |
| out | `/pacmod/shift_cmd` | `pacmod2_msgs/SystemCmdInt` | `command=3` = FORWARD |
| out | `/pacmod/brake_cmd` | `pacmod2_msgs/SystemCmdFloat` | Zero in cruise, non-zero if MPPI decides to brake |
| out | `/pacmod/accel_cmd` | `pacmod2_msgs/SystemCmdFloat` | Throttle (0–`max_acceleration`) |
| out | `/pacmod/turn_cmd` | `pacmod2_msgs/SystemCmdInt` | `command=1` = no signal |
| out | `/pacmod/steering_cmd` | `pacmod2_msgs/PositionWithSpeed` | Steering-wheel angle (radians), `angular_velocity_limit=4.0` |

Pedestrian-decode pipeline: ego polar `(dist, bearing_deg)` →
ego Cartesian `(d cos θ, d sin θ)` → world frame via the ego pose
(GPS → ENU + compass-heading → yaw). The resulting `(M, 2)` tensor is
fed straight into `mppi.update(state, ref_path, obstacles)`.

## ROS 2 topic contract — generic `adapt_mppi_generic_node` (test harness)

Use this when running against sim / rosbag with generic ROS types.

| Direction | Topic | Type |
|---|---|---|
| in | `/odom` | `nav_msgs/Odometry` |
| in | `/adapt/reference_path` | `nav_msgs/Path` |
| in | `/obstacles` | `geometry_msgs/PolygonStamped` |
| out | `/pacmod/steering_cmd` | `std_msgs/Float64` (front-wheel rad) |
| out | `/pacmod/accel_cmd` | `std_msgs/Float64` (m/s²) |

## Vehicle model

Kinematic bicycle with state $\mathbf{x} = (x, y, \psi, v)$ and input
$\mathbf{u} = (\delta, a)$. Euler integration at $\Delta t = 0.1$ s,
wheelbase $L = 1.75$ m:

$$
\begin{aligned}
x_{k+1}   &= x_k + v_k \cos\psi_k \, \Delta t \\
y_{k+1}   &= y_k + v_k \sin\psi_k \, \Delta t \\
\psi_{k+1}&= \mathrm{wrap}_\pi\!\big(\psi_k + (v_k/L)\tan\delta_k \, \Delta t\big) \\
v_{k+1}   &= \mathrm{clip}\!\big(v_k + a_k\,\Delta t,\; 0,\; 8\big).
\end{aligned}
$$

`bicycle_model.step(states, controls)` accepts $(K, 4)$ and $(K, 2)$
arrays and returns $(K, 4)$ in one vectorized call — no Python-level
loop over samples. Kinematic (not dynamic) is justified at 0–8 m/s where
tire slip is < 2°; AutoShield's Stanley also assumes kinematic.

## Reference path

`ReferencePath` consumes waypoints $(x_i, y_i)_{i=0}^{N-1}$ and computes:

- cumulative arc length $s_i$,
- per-segment heading $\theta_i = \mathrm{atan2}(\Delta y_i, \Delta x_i)$,
- a vectorized `nearest_point(p)` that projects a query point onto
  **every** segment at once and picks the closest.

The signed lateral error is
$d = (p_x - \hat p_x)(-\sin\theta) + (p_y - \hat p_y)\cos\theta$,
used directly as the MPPI lane-deviation cost term.

## MPPI backend — adapted from the adapt repo

As of 2026-04-22 the MPPI core is a torch-backed implementation ported
from **the adapt repo** (upstream remote:
`github.com/het915/cs_588_g10`), branch `feature/mppi`, file
`src/vehicle_drivers/gem_mppi_control/mppi_t.py`. Lives at
`src/vehicle_drivers/mppi_controller/mppi_controller/mppi.py`.
`update(state, ref_path, obstacles)` signature is preserved so the ROS2
node and the ROS1 sim bridge keep working without changes.

Running cost (replaces the original lane/heading/speed/obstacle/ctrl
terms):

1. **goal position error** — L2 distance to a look-ahead point
   (default 8 m ahead along the reference path).
2. **velocity error** — `|v - v_ref|`.
3. **stability penalty** — `|δ| · v` discourages high-curvature
   commands at high speed, yielding smoother steering.
4. **pedestrian obstacle cost** — per-pedestrian **temporal
   confidence-growth**: predicted position at `t_eff` (estimated time
   along this rollout sample) with a 2-D uncertainty ellipse that
   widens as confidence decays; Gaussian repulsion. Plus a hard
   clearance step and exponential falloff anchor to keep static
   detections (vx=vy=0, conf=1.0) well-handled.

The controller uses the `pytorch_mppi` library (pip-installable) to
run K=600 rollouts at H=30, dt=0.1 s, λ=0.1 in torch, with the
kinematic bicycle matching AutoShield's wheelbase (2.57 m).

Runtime dep: `torch`, `pytorch_mppi`. Reproduce via
`adapt_requirements.text` at the workspace root (conda env only — no
apt / no system-Python edits). The Humble dev host's system Python 3.10 does **not** currently have
torch, so `ros2 launch` on the host would fail until torch is added
there — dev on host therefore flows through the standalone phase-1
test in `cs588` + the ROS1 Gazebo sim bridge.

## MPPI algorithm

Per-tick update (maintained state: nominal control sequence
$U \in \mathbb{R}^{H \times 2}$):

1. **Sample noise** $\varepsilon_{k,h} \sim \mathcal{N}(0, \Sigma)$,
   form $V = U + \varepsilon$, clip to actuator limits.
2. **Rollout** $V$ through the batched bicycle step for $H$ sequential
   calls, producing trajectory tensor of shape $(K, H, 4)$.
3. **Cost** $\mathcal{J}_k$ (five terms, see below), fully broadcast.
4. **Weights** (stable softmax):
   $\beta = \min_k \mathcal{J}_k,\; w_k = \exp(-(\mathcal{J}_k - \beta)/\lambda) / Z$.
5. **Update** $U \gets U + \sum_k w_k \varepsilon_k$
   (implemented as `np.einsum('k,kht->ht', w, noise)`).
6. **Emit** $u_0 = U_0$; receding-horizon shift $U \gets \mathrm{roll}(U, -1)$.

The warm-start $U$ is what makes sampling MPC feasible at 10 Hz with only
600 samples; a cold-start MPPI needs ~10× more.

## Cost function

$$
\mathcal{J}_k = \sum_{h=0}^{H-1}\Big[
w_\text{lat}\, d_{k,h}^2
+ w_\text{head}\, \tilde\psi_{k,h}^2
+ w_\text{spd}\, (v_{k,h} - v_\text{ref})^2
+ w_\text{obs}\, C^{\text{obs}}_{k,h}
+ w_\text{ctrl}\, \|u_{k,h}\|^2
\Big]
$$

with
$C^{\text{obs}}_{k,h} = \sum_m \big(\max(r_c - \|p_{k,h} - o_m\|, 0)\big)^2$
— quadratic-inside-clearance, zero outside. Computed as a single 4-D
broadcast `diff = traj[:, :, None, :2] - obs[None, None, :, :]`
(shape $(K, H, M, 2)$) — vectorized across all rollouts **and** all
obstacles simultaneously.

## Default parameters

| Symbol | Default | Meaning |
|---|---|---|
| $K$ | 600 | sample count |
| $H$ | 30 | horizon (3.0 s lookahead) |
| $\Delta t$ | 0.1 s | rollout step |
| $\sigma_\delta$ | 0.05 rad | steering noise std |
| $\sigma_a$ | 0.80 m/s² | accel noise std |
| $\lambda$ | 1.0 | softmax temperature |
| $v_\text{ref}$ | 3.0 m/s | cruise target |
| $w_\text{lat}, w_\text{head}, w_\text{spd}, w_\text{obs}, w_\text{ctrl}$ | 10, 5, 2, 1000, 0.1 | cost weights |
| $r_c$ | 3.0 m | obstacle clearance radius |
| $\delta_{\max}$ | 0.61 rad | steering limit |
| $a_{\min}, a_{\max}$ | $-3, +2$ m/s² | accel limits |

Only weight *ratios* matter — scaling every $w$ by a constant is absorbed
by $\lambda$.

## Software layout (consolidated workspace)

```
~/UIUC/AVSE/
├── cs_588_g10/                      ← THE workspace (adapt repo clone;
│   │                                 origin = github.com/het915/cs_588_g10)
│   ├── docs/adapt_design.md         ← this document
│   ├── docs/understand.md           ← canonical PACMod2 topic map
│   ├── adapt_requirements.text      ← conda-env recipe
│   ├── run_tmux.sh                  ← tmux multi-pane launcher
│   └── src/
│       ├── basic_launch/            ← sensor bringup (per-vehicle YAML)
│       ├── adapt_full/              ← perception / fusion / safety
│       │                              (pedestrian tracker pipeline)
│       ├── yolo_person_detector/    ← YOLOv11 RGB-D detector
│       ├── utilities/               ← CAN / radar / highbay shells
│       ├── vehicle_drivers/
│       │   ├── gem_gnss_control/    ← pure-pursuit fallback controller
│       │   ├── gem_visualization/   ← URDF / RViz / GNSS image
│       │   └── mppi_controller/     ← MPPI controller (adapt_mppi_node)
│       │       └── mppi_controller/
│       │           ├── bicycle_model.py
│       │           ├── reference_path.py
│       │           ├── mppi.py            (torch MPPI, adapt-repo port)
│       │           ├── adapt_mppi_node.py (canonical ROS2 node)
│       │           └── adapt_mppi_generic_node.py (sim/rosbag harness)
│       └── hardware_drivers/3rd_drivers/
│           ├── ouster-ros-ros2/, lucid_vision_driver/, …
│           ├── septentrio_gnss_driver/
│           ├── pacmod2_msgs/
│           └── nmea_msgs/
├── POLARIS_GEM_Simulator/           ← upstream sim repo (ROS 1 Noetic,
│   └── vehicle_drivers/gem_mppi_sim/  Docker) + ROS1 bridge package
└── CS588-SP26/                      ← coursework, unrelated
```

`mppi.py`, `reference_path.py`, `bicycle_model.py` are pure NumPy — no
ROS imports — so the sim bridge (ROS 1) reuses them unchanged by inserting
`cs_588_g10/src/vehicle_drivers/mppi_controller` into `sys.path` at startup.

\newpage

# Current Progress

## Phase 1 — MPPI controller (complete, verified)

**Standalone test** (`test/test_phase1.py`) — 50 m radius arc, two static
obstacles, 200 steps at $\Delta t = 0.1$ s. Runs without ROS.

| Metric | Value |
|---|---|
| mean $|d|$ (post-warmup) | **0.007 m** |
| max $|d|$ (post-warmup) | **0.024 m** |
| fraction within 0.5 m | **100 %** |
| final speed | **3.01 m/s** (tracks $v_\text{ref} = 3$) |

Pass criterion (80 % within 0.5 m after the first 20 warmup steps) is
comfortably met. Vectorization invariants hold: the one `for h in range(H)`
loop calls the batched bicycle step on a $(K, 4)$ batch once; cost
evaluation and weight update have zero Python loops over $K$.

## Sim-bridge infrastructure (scaffolded 2026-04-17)

ROS 1 catkin package `gem_mppi_sim` at
`POLARIS_GEM_Simulator/vehicle_drivers/gem_mppi_sim/`.

- **State in:** `/gazebo/get_model_state` (model name `gem_e4`);
  fallback `/odom` via `use_odom:=true`.
- **Reference path:** CSV → `ReferencePath`. Default
  `waypoints/wps.csv` (track1 coordinates, copied from the pure-pursuit
  demo; vehicle-agnostic).
- **Control out:** `/ackermann_cmd` (`ackermann_msgs/AckermannDrive`).
  MPPI returns $(\delta, a)$; bridge integrates to a speed setpoint
  $v_\text{cmd} = \mathrm{clip}(v + a \cdot dt, 0, 8)$ because the sim's
  existing `gem_control.py` expects a speed, not an acceleration.
- **Rate:** default 20 Hz.
- **Layout decision:** repo is **symlinked** into
  `~/gem_simulation_ws/src/` rather than moved, preserving the
  `~/UIUC/AVSE/{cs_588_g10, POLARIS_GEM_Simulator, CS588-SP26}` sibling tree.

**Pending:** first closed-loop run on `track1.world`. Host-side one-time
prep (workspace symlink + Docker image build) has not yet been executed.

## Key paths (host vs. container)

| What | Host | Container |
|---|---|---|
| Sim repo | `~/UIUC/AVSE/POLARIS_GEM_Simulator` | `~/host/UIUC/AVSE/POLARIS_GEM_Simulator` |
| Catkin workspace (symlinked) | `~/gem_simulation_ws/` | `~/host/gem_simulation_ws/` |
| Adapt MPPI | `~/UIUC/AVSE/cs_588_g10/src/vehicle_drivers/mppi_controller` | `~/host/UIUC/AVSE/cs_588_g10/src/vehicle_drivers/mppi_controller` |
| Bridge package | `POLARIS_GEM_Simulator/vehicle_drivers/gem_mppi_sim/` | same, under `~/host/...` |

Host–container path difference comes from the
`${HOME}:/home/${USER}/host` volume mount in
`POLARIS_GEM_Simulator/setup/docker-compose.yaml`.

\newpage

# How to Run

Three modes, fastest to most realistic. See the pipeline diagram
(§System Architecture) for context on what each terminal produces /
consumes.

## Prerequisite — conda env + ROS version compatibility

The MPPI node requires `torch` + `pytorch_mppi` at runtime. These live
in a **conda env only** — no system-Python edits, no `apt`, no env-var
changes on the vehicle. The env is reproducible from
`adapt_requirements.text` at the workspace root:

```bash
conda create -n adapt python=3.12 -y
conda activate adapt
pip install -r ~/UIUC/AVSE/cs_588_g10/adapt_requirements.text
```

This gives Python 3.12 + torch 2.5.1+cu121 + pytorch_mppi + numpy +
colcon build deps (empy 3.3.4, lark, catkin_pkg). Matches the Jazzy
vehicle's Python version; on the Humble dev host (py3.10) create a
parallel env with `python=3.10` and the same `pip install -r`.

**Runtime-Python rule**:

| Task | Shell |
|---|---|
| `colcon build`, standalone MPPI test, lint | `conda activate adapt` |
| `ros2 launch` on Humble host (py3.10) | `conda activate adapt-py310` (py3.10 variant) |
| `ros2 launch` on Jazzy vehicle (py3.12) | `conda activate adapt` (py3.12) |

ROS 2 itself (Humble / Jazzy) is assumed to be preinstalled on the
host — `source /opt/ros/<distro>/setup.bash` as usual. We never
apt-install anything.

## GPU selection

The MPPI's torch device is chosen in this order:

1. ROS 2 param `mppi/device` — string; `cuda:0`, `cpu`, or `''` for
   auto. Overridable at launch via `device:=cuda:0`.
2. If param is empty (default): `cuda if torch.cuda.is_available()
   else cpu`.

On startup the node logs the chosen device, e.g.:

```
MPPI device: cuda:0 (NVIDIA GeForce RTX 3060 Laptop GPU, 6.0 GiB VRAM)
```

Smoke check before launch:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.free --format=csv,noheader
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

## Mode A — standalone MPPI numerical test (1 terminal, no ROS)

Fast regression after editing cost terms, rollout math, or dynamics.

**Terminal 1** (cs588 OK):
```bash
cd ~/UIUC/AVSE/cs_588_g10
python3 src/vehicle_drivers/mppi_controller/test/test_phase1.py
```

Expected: `PHASE 1 TEST PASSED`, mean |lateral error| ≈ 0.007 m,
writes `src/vehicle_drivers/mppi_controller/test/phase1_result.png`.

## Mode B — rosbag replay (3 terminals)

Testing MPPI against recorded sensor data. No live sensors, no vehicle.

**Terminal 1 — build (one-time; re-run after each code edit). cs588 OK:**
```bash
cd ~/UIUC/AVSE/cs_588_g10
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

**Terminal 2 — replay the rosbag. NO conda:**
```bash
conda deactivate
source /opt/ros/humble/setup.bash
ros2 bag play /path/to/your/bag --clock
```

The bag must publish: `/navsatfix`, `/insnavgeod`,
`/pacmod/vehicle_speed_rpt`, `/pacmod/enabled`, and either
`/fusion_pedestrian_position` directly, **or**
`/lidar_pedestrian_position` + `/rgbd_pedestrian_position` (then keep
`enable_fusion:=true` in Terminal 3).

**Terminal 3 — MPPI + fusion. NO conda:**
```bash
conda deactivate
source /opt/ros/humble/setup.bash
source ~/UIUC/AVSE/cs_588_g10/install/setup.bash
ros2 launch adapt_full adapt_mppi_launch.py \
    vehicle_name:=e4 \
    desired_speed:=2.0 \
    enable_mppi:=true \
    enable_fusion:=true \
    enable_lidar:=false \
    enable_safety:=false
```

Expected log from `adapt_mppi_node` (throttled to 1 Hz):
```
MPPI | pos=(12.34, 5.67) yaw=45.0deg v=1.95 -> v_cmd=2.00 thr=0.28
sw=2.1deg obs=1 ESS/K=0.18
```

**Optional Terminal 4 — RViz. NO conda:**
```bash
conda deactivate
source /opt/ros/humble/setup.bash
source ~/UIUC/AVSE/cs_588_g10/install/setup.bash
rviz2 -d src/gem_rviz_display/rviz/gem_e4.rviz
```

## Mode C — live vehicle (4+ terminals)

Real GEM e4 with sensors + PACMod2 bringup. Every run-terminal is
`conda deactivate` first.

**Terminal 1 — build:**
```bash
cd ~/UIUC/AVSE/cs_588_g10
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

**Terminal 2 — sensor bringup (Ouster + Septentrio + cameras + PACMod):**
```bash
conda deactivate && source /opt/ros/humble/setup.bash
source ~/UIUC/AVSE/cs_588_g10/install/setup.bash
ros2 launch basic_launch sensor_init.launch.py vehicle_name:=e4
```

**Terminal 3 — perception (LiDAR person detector + YOLO RGB-D):**
```bash
conda deactivate && source /opt/ros/humble/setup.bash
source ~/UIUC/AVSE/cs_588_g10/install/setup.bash
# LiDAR-side pedestrian detector:
ros2 run adapt_full lidar_processing &
# YOLO RGB-D detector:
ros2 run yolo_person_detector yolo_person_detector
```

**Terminal 4 — MPPI + fusion + safety:**
```bash
conda deactivate && source /opt/ros/humble/setup.bash
source ~/UIUC/AVSE/cs_588_g10/install/setup.bash
ros2 launch adapt_full adapt_mppi_launch.py \
    vehicle_name:=e4 \
    desired_speed:=2.0 \
    enable_mppi:=true \
    enable_fusion:=true \
    enable_safety:=true \
    enable_lidar:=false
```

The MPPI node waits for `/pacmod/enabled = true` and then auto-primes
PACMod (enable + FORWARD gear + zero brake/accel). Stanley's pygame
L+R bumper handshake is **not** replicated — use the normal PACMod
enable flow from the driver side.

**Optional Terminal 5 — monitoring / recording:**
```bash
ros2 topic hz /pacmod/steering_cmd         # expect ~10 Hz
ros2 topic echo /fusion_pedestrian_position --once
ros2 bag record -a -o my_run
```

## Runtime parameter overrides

Any MPPI / PID param is overridable:

```bash
ros2 launch adapt_full adapt_mppi_launch.py \
    desired_speed:=3.0
# or directly on ros2 run:
ros2 run adapt_mppi adapt_mppi_node --ros-args \
    -p desired_speed:=3.0 \
    -p mppi/K:=800 -p mppi/sigma_steer:=0.08 \
    -p mppi/w_obs:=2000.0 -p mppi/clearance:=3.5 \
    -p require_pacmod_enable:=False
```

`waypoints_csv` defaults to
`<install>/adapt_full/share/adapt_full/waypoints/track.csv`
(via `ament_index_python`). Override:

```bash
-p waypoints_csv:=/abs/path/to/other_track.csv
```

`require_pacmod_enable:=False` lets the loop run on the first GPS fix
instead of waiting for `/pacmod/enabled = true` — useful for
bench / rosbag testing where PACMod isn't live.

## Sanity test (any run mode)

With the install sourced in a spare terminal:

```bash
# Inject a fake pedestrian 10 m straight ahead:
ros2 topic pub --once /fusion_pedestrian_position \
    std_msgs/msg/Int32MultiArray "{data: [10, 0]}"

# Watch MPPI react: obs=1 should show in the log line, thr should drop:
ros2 topic echo /pacmod/accel_cmd
```

If `obs=1` never appears, the pedestrian-decode gate is probably
blocked on missing GPS (`lat==0 && lon==0`). Either publish a NavSatFix
+ INSNavGeod pair, or set `require_pacmod_enable:=False` and publish
any non-zero `/navsatfix`.

which lets the loop run on the first GPS fix instead of waiting for
`/pacmod/enabled = True`.

## GEM Gazebo sim + MPPI bridge (Docker Noetic)

The end-to-end closed-loop path. The host (Ubuntu 22.04 + ROS 2 Humble)
cannot run ROS 1 Noetic natively, so sim-side work happens inside the
container.

### Host one-time prep

```bash
mkdir -p ~/gem_simulation_ws/src
ln -s ~/UIUC/AVSE/POLARIS_GEM_Simulator ~/gem_simulation_ws/src/POLARIS_GEM_Simulator
cd ~/UIUC/AVSE/POLARIS_GEM_Simulator
bash setup/build_docker_image.sh
```

### Launch the sim (container terminal A)

```bash
cd ~/UIUC/AVSE/POLARIS_GEM_Simulator
bash run_docker_container.sh
# --- inside the container ---
cd ~/host/gem_simulation_ws
catkin_make
source devel/setup.bash
roslaunch gem_launch gem_init.launch world_name:=track1.world vehicle_name:=e4
```

### Launch the MPPI bridge (container terminal B)

Re-run `run_docker_container.sh` on the host to attach a second shell:

```bash
bash run_docker_container.sh
# --- inside the container ---
cd ~/host/gem_simulation_ws
source devel/setup.bash
roslaunch gem_mppi_sim mppi_sim.launch vehicle_name:=e4
```

Expected log line (1 Hz throttled):

```
MPPI | x=... y=... v=... -> delta=... v_cmd=... a=...
```

### Sanity checks

```bash
rostopic hz /ackermann_cmd            # expect ~20 Hz
rostopic echo -n 1 /ackermann_cmd     # speed, steering_angle non-zero
rosservice call /gazebo/get_model_state "{model_name: 'gem_e4'}"
```

### Debug fall-throughs

- **ImportError: adapt_mppi** — pass the container-side path explicitly:
  `roslaunch gem_mppi_sim mppi_sim.launch adapt_src:=/home/$USER/host/UIUC/AVSE/cs_588_g10/src/vehicle_drivers/mppi_controller`.
- **"model not found"** from `get_model_state` — list spawned models via
  `rosservice call /gazebo/get_world_properties` and adjust `vehicle_name`.
- **Vehicle doesn't move but `/ackermann_cmd` is publishing** — initial
  $v \approx 0$ clips the integrated speed to 0; raise `v_ref` (e.g.
  `v_ref:=4.0`).

\newpage

# What to Do Next

Prioritized, near-term to long-term.

## Near term (finish the closed-loop path)

1. **Execute the host-side prep** (symlink + Docker image build) and
   run the first closed-loop lap on `track1.world`.
2. **Tuning pass.** Sweep $K$, $H$, $\sigma_\delta$, $\sigma_a$, and the
   $w_*$ cost weights with the first run in hand. Watch effective sample
   count (`MPPI.effective_sample_count`) — keep
   $\mathrm{ESS}/K \in [0.05, 0.5]$ in steady state.
3. **Diagnostic topic.** Publish per-tick cost, ESS, and selected
   $(\delta, a)$ from the bridge for offline analysis and an RViz overlay.

## Medium term

4. **Live obstacles.** Subscribe to `/front_laser_points`
   (`sensor_msgs/LaserScan`), cluster into centroids, feed
   `self.obstacles`. `MPPI._obstacle_cost` is already wired — only the
   producer is missing.
5. **Waypoints for other worlds** (`parking.world`,
   `highbay_track.world`) via `POLARIS_GEM_Simulator/utils/generate_waypoints.py`;
   expose via the bridge's `waypoints_csv` launch arg.
6. **Dynamics-parity check.** Log Gazebo ground truth vs. kinematic
   rollout to quantify divergence under realistic steering and speed lag.

## Next design phase (Phase 2 proper)

7. **Pedestrian-prediction integration.** Swap single-snapshot obstacles
   for a time-indexed prediction tensor of shape $(M, H, 2)$. Obstacle
   cost becomes
   $C^{\text{obs}}_{k,h} = \sum_m \big(\max(r_c - \|p_{k,h} - o_{m,h}\|, 0)\big)^2$
   — same NumPy broadcast, different shape. MPPI will then delay,
   swerve, or creep based on *predicted* pedestrian occupancy at each
   rollout step, rather than reacting to a snapshot. This is the
   capability Stanley cannot express.

## Long term

8. **Uncertainty-weighted obstacle costs** using prediction covariance —
   per-obstacle, per-step $r_c$ scaling.
9. **Safety state-machine integration** with AutoShield's high-level
   decision module (emergency-brake override).
10. **GPU rollouts** (CuPy / torch backend) if $K$ scales beyond
    $\sim 4000$ and CPU vectorization becomes the bottleneck.

## Aspirational research track

- **Diffusion policy for pedestrian motion prediction.** Replace
  AutoShield's current predictor with a conditional diffusion model that
  samples multi-modal futures, then feed the samples directly into
  MPPI's $(M, H, 2)$ obstacle tensor (one prediction sample per obstacle,
  or a per-mode weighted cost). Stretch goal — may not land in the
  current project window. Timeline risk is tracked separately from the
  controller deliverable; MPPI ships regardless of prediction model.

## Known risks to revisit

- **Symlink + catkin.** Noetic handles symlinks in `src/` in practice,
  but a physical move into `~/gem_simulation_ws/src/` is the fallback if
  `rospack` or `catkin_make` misbehaves.
- **Acceleration → speed integration $\Delta t$.** The bridge uses
  `mppi.dt = 0.1 s` when integrating $a$ into $v_\text{cmd}$, not its
  own loop period. Mismatch is small at `rate_hz=20`, grows if the
  bridge rate drops.
- **Frame assumptions.** MPPI state and waypoints both live in the
  Gazebo world frame. If the reference path source ever switches to
  GPS/geodetic coordinates, insert an explicit transform.

## Tuning guide

Do this in order. Only move on when the previous step passes cleanly.

### Step 0 — Verify with the standalone test

`python3 src/vehicle_drivers/mppi_controller/test/test_phase1.py` must pass with mean
|lateral error| < 0.05 m before you tune anything else. If it doesn't,
the problem is in the MPPI class or cost function, not in params.

### Step 1 — Tracking (no obstacles, empty `/fusion_pedestrian_position`)

Watch the log line:
```
MPPI | pos=(x,y) yaw=… v=… -> v_cmd=… thr=… sw=…deg obs=0 ESS/K=…
```

Targets on a straight / gentle-curve run:

* |lateral error| steady-state < 0.5 m (from RViz cross-track overlay
  when added, or from offline replay).
* `v` converges to `desired_speed` within 3–5 s.
* `ESS/K ∈ [0.05, 0.5]` — effective sample count fraction.

Knobs in order of leverage:

| Symptom | Knob | Direction |
|---|---|---|
| Lateral error bouncing around target | `mppi/w_lat` | **up** (20 → 40) |
| Heading lag on curves | `mppi/w_head` | **up** (5 → 15) |
| Jerky steering | `mppi/sigma_steer` | **down** (0.05 → 0.03); `mppi/w_ctrl` up |
| Sluggish speed response | `pid/kp` | **up** (0.6 → 0.9) |
| ESS/K < 0.05 | `mppi/lambda_` | **up** (1.0 → 2.0) — softmax too sharp |
| ESS/K saturating at 1.0 | `mppi/sigma_steer`, `mppi/sigma_accel` | **up** — exploration too narrow |
| ESS/K flat / cost landscape useless | `mppi/K` | **up** (600 → 1000) |

### Step 2 — Obstacle response

Drop a fake pedestrian onto the graph:
```bash
ros2 topic pub --once /fusion_pedestrian_position \
    std_msgs/msg/Int32MultiArray \
    "{data: [10, 0]}"     # 10 m ahead, 0° bearing (straight forward)
```

Check:
* `obs=1` shows up in the log line.
* Vehicle slows or swerves by the time clearance < `mppi/clearance` (3 m default).

Knobs:

| Symptom | Knob |
|---|---|
| Plows through pedestrian | `mppi/w_obs` **up** (1000 → 2000); `mppi/clearance` **up** (3.0 → 4.0) |
| Freezes before reaching obstacle | `mppi/w_obs` **down** or widen path via new waypoints; check `mppi/clearance` isn't > lane half-width |
| Swerves violently | `mppi/sigma_steer` **down** and `mppi/w_ctrl` **up** |

### Step 3 — Real pedestrian detections

Point at a rosbag with `/lidar_pedestrian_position` +
`/rgbd_pedestrian_position`. The fusion node produces
`/fusion_pedestrian_position` every tick; MPPI treats each detection as
a static-per-tick obstacle in world frame (transform applied in
`_ped_cb`).

Gotchas to watch for:

* Intermittent detections (false negatives) cause the obstacle to
  flicker in and out. MPPI re-plans every tick, so the vehicle "locks
  back onto path" between hits. If this is too twitchy, buffer the last
  N obstacle snapshots and union them.
* Very noisy bearing estimates from YOLO at distance cause apparent
  position jumps. Tighten `matching_threshold` in `sensor_fusion_params.yaml`
  or downstream-filter.
* The pedestrian position is rounded to integer metres/degrees by
  AutoShield's fusion node — expect ~0.5–1 m quantization noise, which
  sits well inside MPPI's 3 m clearance radius so it's fine.

### Full parameter reference

All params are overridable at `ros2 launch` or `ros2 run` time via
`--ros-args -p name:=value`. Only ratios among the cost weights
matter — if you scale all `w_*` by the same factor, `lambda_` absorbs
it.

**MPPI sampling**

| Param | Default | Typical range | Effect | Tune-up symptom |
|---|---|---|---|---|
| `mppi/K` | 600 | 200–2000 | # rollout samples per tick | ESS/K saturating at 1 → raise K |
| `mppi/H` | 30 | 20–50 | horizon steps (at dt=0.1 → 3 s) | Planner short-sighted around obstacles → raise H |
| `mppi/dt` | 0.1 | 0.05–0.2 | per-step integrator dt | Rarely tuned; must match control rate |
| `mppi/sigma_steer` | 0.15 | 0.05–0.30 | steering noise stdev (rad) | Exploration too narrow (ESS/K ~1) → raise σ |
| `mppi/sigma_accel` | 0.5 | 0.2–1.0 | accel noise stdev (m/s²) | Jerky throttle → lower σ |
| `mppi/lambda_` | 0.1 | 0.01–2.0 | softmax temperature | ESS/K < 0.05 (one-sample greedy) → raise λ |
| `mppi/device` | `''` | `cuda:0`, `cpu`, `''` | torch device; `''` auto-detects | Force CPU during debug profiling |

**Tracking cost** (higher = tracks waypoints tighter; trade-off: slower response to obstacles)

| Param | Default | Effect |
|---|---|---|
| `mppi/w_pos` | 15.0 | goal position error weight |
| `mppi/w_vel` | 5.0 | velocity-tracking weight toward `v_ref` |
| `mppi/w_curv` | 2.0 | stability penalty `|δ|·v` — punishes high-steer-at-high-speed |
| `mppi/lookahead_m` | 8.0 | look-ahead distance on ref-path for goal selection (lower → tighter tracking, higher overshoot; upper → smoother but wider corner cuts) |

**Obstacle / pedestrian cost**

| Param | Default | Effect |
|---|---|---|
| `mppi/w_obs` | 150.0 | peak Gaussian repulsion per pedestrian (temporal confidence-growth term) |
| `mppi/w_obs_hard` | 250.0 | step penalty inside `clearance` radius |
| `mppi/w_obs_soft` | 40.0 | exponential-decay repulsion around each ped |
| `mppi/clearance` | 3.0 | hard-clearance radius (m) |

**Actuator limits / longitudinal PID**

| Param | Default | Effect |
|---|---|---|
| `desired_speed` | 2.0 | cruise target (m/s); hard-capped at 5.0 in node |
| `max_acceleration` | 0.5 | throttle cap in PACMod units, hard-capped at 2.0 |
| `pid/kp`, `pid/ki`, `pid/kd`, `pid/wg` | 0.6, 0.0, 0.1, 10 | speed-loop PID gains + anti-windup |
| `filter/cutoff`, `filter/fs`, `filter/order` | 1.2, 30, 4 | EMA on incoming speed-report |
| `wheelbase` | 1.75 | GEM e4 kinematic bicycle wheelbase |
| `offset` | 1.26 | GPS-antenna to rear-axle offset |
| `origin_lat`, `origin_lon` | 40.0927422, -88.2359639 | ENU origin for GPS→local conversion |

**Operational**

| Param | Default | Effect |
|---|---|---|
| `rate_hz` | 10.0 | control-loop frequency |
| `require_pacmod_enable` | `True` | if `False`, skip waiting for `/pacmod/enabled` gate (bench testing) |
| `waypoints_csv` | (auto) | resolves to `adapt_full/waypoints/track.csv` via ament-index |

**Visualization**

| Param | Default | Effect |
|---|---|---|
| `viz/num_samples` | 19 | # top-weighted sampled rollouts to publish as MarkerArray |
| `viz/frame_id` | `map` | frame for all MPPI viz markers |

---

# Attribution & repo relationship

- **"The adapt repo"** refers to this repository
  (remote `github.com/het915/cs_588_g10`). Everywhere in prose below
  it's called *the adapt repo*; we do not refer to upstream
  maintainers by handle in docs.
- The torch MPPI algorithmic core was ported from the adapt repo's
  `feature/mppi` branch (`src/vehicle_drivers/gem_mppi_control/`).
- We do not append `Co-Authored-By:` Claude / Anthropic lines to any
  commit in this tree. The committer is the human contributor.
- `.claude/` and `CLAUDE.md` are git-ignored to keep local agent state
  out of the tree.
