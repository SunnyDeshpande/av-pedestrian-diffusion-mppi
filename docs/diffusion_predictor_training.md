# Diffusion-policy pedestrian-motion predictor — training & integration plan

This document is **self-contained**. It captures everything a fresh
engineer (or a fresh Claude session after `git pull`) needs to train a
diffusion model that replaces the current heuristic pedestrian-motion
predictor in this repo. Read it top-to-bottom before touching code.

The work is research-engineering — multi-week. It is sequenced so the
heuristic predictor and MPPI keep working at every step; the new node
ships as a drop-in replacement first, then the MPPI obstacle tensor
upgrades to consume time-indexed predictions in the final phase.

---

## 1. Goal

Replace `src/yolo_person_detector/yolo_person_detector/pedestrian_behaviour_predictor.py`
with a learned, multi-modal trajectory model. The current node is a
constant-velocity extrapolator on top of a smoothing tracker; the new
node is a conditional diffusion model that emits K=20 candidate 5 s
futures per pedestrian.

Detection stays as it is: YOLOv11 + LiDAR/RGBD fusion already produces
metric ego-polar pedestrian positions on `/fusion_pedestrian_position`
at ~10 Hz. The diffusion model only replaces the **predict** stage.

The MPPI controller (`src/vehicle_drivers/mppi_controller/`) currently
treats pedestrians as static-per-tick obstacles `(M, 2)`. The design
doc (`docs/adapt_design.md`, "Aspirational research track") commits to
upgrading that tensor to `(M, H, 2)` once a trajectory-prediction
model exists. That upgrade is Phase D of this plan.

## 2. Locked-in decisions

| Decision | Choice |
|---|---|
| Dataset path | **Argoverse 2 pretrain → small GEM-rosbag finetune** |
| Output | **K=20 multi-modal samples**, collapse to best mode for the runtime `(M, H, 2)` tensor; keep all 20 for offline minADE-Top-K eval |
| Scope | **Prediction only** — keep YOLOv11 + LiDAR/RGBD fusion as the detection front-end |
| Horizon | **5.0 s, 20 steps, dt=0.25 s** — exact match to today's RViz marker so visualization keeps working |
| Inference budget | **< 30 ms per tick** for M=8 peds × K=20 samples on the vehicle GPU; hard ceiling 50 ms (10 Hz control loop) |

## 3. Drop-in I/O contract

The new node MUST publish/subscribe identical topic names and message
types to the current predictor so `adapt_high_level_command.py`, the
RViz config, and downstream nodes work unchanged. The only addition
is a new `(M, H, 2)` tensor topic for the Phase-D MPPI upgrade.

| Direction | Topic | Type | Notes |
|---|---|---|---|
| Sub | `/fusion_pedestrian_position` | `std_msgs/Int32MultiArray` | ego-polar `[dist_m, dir_deg, ...]`, ~10 Hz |
| Sub | `/vehicle_rpt` | `pacmod2_msgs/VehicleSpeedRpt` | for TTC gating |
| Pub | `/person_prediction` | `visualization_msgs/Marker` | LINE_STRIP, 20 pts in `base_link` (x forward, y left) |
| Pub | `/pedestrian_motion` | `geometry_msgs/Twist` | selected ped position; `linear.x/y` in `base_link` |
| Pub | `/pedestrian_ttc` | `std_msgs/Float64` | seconds; `inf` if no collision predicted |
| Pub *(new)* | `/pedestrian_predictions_tensor` | `std_msgs/Float32MultiArray` | shape header `(M, 20, 2)` for Phase D |

Reuse the existing tracker (greedy ID association with 2 m gate,
median spike removal, 7-pt moving average, 15-pt non-uniform-dt
velocity estimate, EMA α=0.6) verbatim. Only the **predict** call is
replaced with the diffusion forward pass.

## 4. Model

**MID-style Transformer denoiser** (Motion Indeterminacy Diffusion).

- Input per agent: history `(T_hist=20, 4)` of `[x, y, vx, vy]` in ego
  frame at t=0; pad short tracks with zero plus a presence mask.
- Conditioning: ego linear and angular velocity at t=0 (2 scalars,
  AdaLN-injected) and the diffusion-step embedding.
- Backbone: 4-layer Transformer encoder, 4 heads, d_model=128, MLP
  ε-prediction head emitting `(T_fut=20, 2)` Cartesian offsets.
- Diffusion: DDPM with 100 train steps, cosine β-schedule.
  **Inference: DDIM with 10 steps.**
- Parameters target: ~3-4 M (~15 MB checkpoint).
- Day-1: per-agent independent — no social attention. Add cross-agent
  attention only if Phase C A/B collision-rate metrics demand it.

Why this and not heavier models: AV2 alone gives ~150 k pedestrian
sequences at the right cadence; the GEM e4 has no HD map, so models
that condition on map rasters (Trajectron++, AgentFormer) bring no
benefit. MID is the simplest published diffusion baseline that fits
trajectory-only inputs and the RTX 3060 Laptop's ~6 GB usable VRAM.

**Best-mode selection** for the runtime `(M, H, 2)` tensor:

1. Pick the mode closest to the mean of all K samples (stable; no
   extra classifier head needed).
2. Add sticky temporal selection per track-id: only flip mode when
   the current-mode cost exceeds 1.5× the alternative for ≥ 3
   consecutive ticks. Prevents TTC jitter from sample-to-sample mode
   reordering.

## 5. Dataset pipeline

### 5.1 AV2 pretrain (primary)

- **Source:** Argoverse 2 Motion Forecasting (CC-BY 4.0, ~60 GB
  compressed, 250 k scenarios, 5 s histories + 6 s futures at 10 Hz).
- **Filter:** keep tracks with `track_category == 'pedestrian'`,
  require ≥ 5 s history + 5 s future, drop sequences with > 0.2 s
  gaps.
- **Normalize:** ego frame at t=0 (translate so ego is origin, rotate
  so ego heading is +x). Output:
  - `history` — `(N≈150 k, T_hist=20, 4)` `[x, y, vx, vy]`
  - `future`  — `(N, T_fut=20, 2)` `[x, y]`
  - `ego_vel` — `(N, 2)` `[v_lin, v_ang]` for AdaLN conditioning
- **Augmentation:** random rotation ±15°, history dropout p=0.1
  (zeroes out random history steps to simulate detection gaps).
- **Splits:** AV2 train / val / test as published.

### 5.2 GEM finetune (deployment match)

- **Recording protocol:** 4-6 hours of rosbag at the highbay with
  confederate pedestrians scripted into:
  - **Crossing** scenarios (~150 events, 90° crossings at varying
    speeds and standoffs).
  - **Parallel-walk** scenarios (~80 events, ped walking same/opposite
    direction near the curb).
  - **Stationary** scenarios (~50 events, ped standing in/near the
    path).
  - **Group** scenarios (~20 events, 2-4 peds together).
- **Topics to record:** `/fusion_pedestrian_position`, `/vehicle_rpt`,
  `/tf`, `/tf_static`, plus any sensor topics convenient for replay.
- **Auto-labeling:** replay each bag through the existing fusion node;
  the recorded `/fusion_pedestrian_position` track histories ARE the
  labels. Apply the existing tracker offline to assemble
  history+future windows. **No manual annotation required.**
- **Finetune recipe:** start from AV2 pretrained weights, lr=2e-5,
  batch 32, 20 epochs, ~1 h on the RTX 3060 Laptop. Freeze the
  AdaLN/ego-velocity conditioning if data is sparse.

## 6. Training recipe

| Knob | Value |
|---|---|
| Optimizer | AdamW |
| Pretrain LR | 2e-4 → 1e-5 cosine over 200 epochs |
| Finetune LR | 2e-5, constant, 20 epochs |
| Batch size | 32 (fits 6 GB VRAM at d_model=128; drop to d_model=96 if OOM) |
| Loss | ε-prediction MSE (DDPM standard) |
| EMA | model weights, decay=0.999 |
| Pretrain walltime | ~12 h on RTX 3060 Laptop |
| Finetune walltime | ~1 h |
| Logging | TensorBoard → `logs/diffusion/<timestamp>/` |
| Checkpointing | every epoch + final EMA snapshot |

`adapt_requirements.text` already pins `torch==2.5.1+cu121`, so the
existing `adapt` conda env (Python 3.12) trains and serves the model.
Bring-up on a fresh machine:

```bash
conda create -n adapt python=3.12 -y
conda activate adapt
pip install -r ~/UIUC/AVSE/cs_588_g10/adapt_requirements.text
```

## 7. New ROS package

Create `src/pedestrian_diffusion_predictor/` as a sibling of
`src/yolo_person_detector/`. Colcon-buildable, ament_python.

```
src/pedestrian_diffusion_predictor/
├── package.xml
├── setup.py
├── setup.cfg
├── resource/pedestrian_diffusion_predictor
└── pedestrian_diffusion_predictor/
    ├── __init__.py
    ├── dataset.py        # AV2 + GEM loaders, normalization, augmentation
    ├── model.py          # transformer denoiser + DDPM/DDIM scheduler
    ├── train.py          # AV2 pretrain entrypoint (200 epochs)
    ├── finetune.py       # GEM finetune entrypoint (20 epochs)
    ├── eval.py           # ADE/FDE, minADE-20, minFDE-20, miss@2m, latency
    ├── infer_node.py     # drop-in ROS 2 node, same publishers as today
    └── tracker.py        # copied from pedestrian_behaviour_predictor.py
```

The tracker is **copied**, not imported, to keep the new package
independent of `yolo_person_detector`.

## 8. Phased rollout

| Phase | Deliverable | Effort |
|---|---|---|
| **A** | AV2 pretrain. `dataset.py` + `model.py` + `train.py` + `eval.py` written, AV2 downloaded and preprocessed, training runs to completion, offline metrics meet targets | ~10 days |
| **B1** | Record GEM bags at the highbay per §5.2 protocol | ~3 days |
| **B2** | Auto-label, run `finetune.py`, evaluate on a UIUC bag holdout | ~3 days |
| **C1** | Deploy `infer_node` in **shadow mode** alongside the heuristic. Both run; new node publishes to remapped topics (`/person_prediction_diffusion`, etc.). Record A/B replay bags | ~4 days |
| **C2** | Flip subscribers: heuristic off, diffusion on the canonical `/person_prediction`, `/pedestrian_motion`, `/pedestrian_ttc` topics. MPPI still consumes `/fusion_pedestrian_position` directly (unchanged) | ~2 days |
| **D** | MPPI `(M, H, 2)` upgrade. New node also publishes `/pedestrian_predictions_tensor`. MPPI subscribes and time-indexes the obstacle cost | ~3 days |

Total: ~5-6 weeks, single engineer, with buffer.

## 9. Phase D — MPPI obstacle-cost upgrade

Files to modify (do not touch in Phases A-C):

- `src/vehicle_drivers/mppi_controller/mppi_controller/mppi.py` —
  obstacle term in the running cost picks `(M, H, 2)` indexed by
  rollout time-step instead of static `(M, 2)`. Keep the existing
  soft Gaussian repulsion (`max(r_c - dist, 0)^2`); only the
  obstacle position becomes time-dependent. Cost broadcast becomes
  `(K, H, M, 2)` after subtraction.
- `src/vehicle_drivers/mppi_controller/mppi_controller/adapt_mppi_node.py` —
  add `/pedestrian_predictions_tensor` subscriber that decodes the
  `Float32MultiArray` shape header back to `(M, H, 2)`. Keep the
  `/fusion_pedestrian_position` subscriber as a fallback (used when
  the diffusion node is offline or `M_tensor != M_fusion`).

The shape change is the only breaking modification on the controller
side; the existing reference path, bicycle model, and PID/cost
weights are unchanged.

## 10. Evaluation targets

**Offline (AV2 val, then UIUC bag holdout):**

| Metric | Target |
|---|---|
| minFDE-20 @ 5 s | ≤ 1.0 m (literature: MID ~0.96 m) |
| minADE-20 | ≤ 0.5 m |
| Miss-rate @ 2 m | ≤ 0.20 |
| UIUC vs AV2 minFDE ratio | ≤ 1.5× (else escalate Phase B) |

**Latency** (vehicle GPU, M=8 peds, K=20 samples, DDIM=10):
- < 30 ms per inference (target).
- < 50 ms per inference (hard ceiling — 10 Hz loop budget).

If latency exceeds 50 ms: drop DDIM to 5 steps, then d_model to 96,
then K to 10 (in that order).

**Closed-loop proxy:** count near-miss events
(ego–pedestrian distance < 1.5 m) under MPPI on a fixed bag set with
each predictor. Diffusion (Phase D) should reduce near-misses vs the
heuristic baseline.

## 11. Verification

End-to-end checks the next agent should run:

1. **Pretrain pass:**
   ```bash
   python -m pedestrian_diffusion_predictor.train \
       --data <av2_path> --epochs 200
   ```
   Final checkpoint logs minFDE-20 ≤ 1.0 m, minADE-20 ≤ 0.5 m on AV2 val.

2. **Finetune pass:**
   ```bash
   python -m pedestrian_diffusion_predictor.finetune \
       --bags <gem_bag_dir> --pretrained <ckpt>
   ```
   UIUC holdout minFDE-20 within 1.5× AV2 val.

3. **Latency bench:**
   ```bash
   python -m pedestrian_diffusion_predictor.eval \
       --benchmark --M 8 --K 20
   ```
   Vehicle GPU < 30 ms per inference.

4. **Drop-in replacement smoke test:**
   ```bash
   colcon build --packages-select pedestrian_diffusion_predictor
   ros2 run pedestrian_diffusion_predictor infer_node
   ```
   Confirm `/person_prediction`, `/pedestrian_motion`,
   `/pedestrian_ttc` publish with the same types and
   `adapt_high_level_command` sees them unchanged.

5. **Shadow A/B on a recorded bag:**
   ```bash
   ros2 bag play <uiuc_bag> --clock
   ```
   With both predictors running on remapped topics, compare ADE @ 5 s
   on shared track ids.

6. **Phase D MPPI integration:**
   ```bash
   ros2 launch adapt_full adapt_mppi_launch.py \
       enable_fusion:=true device:=cuda:0
   ```
   Confirm obstacle cost uses `(M, H, 2)` and near-miss rate drops vs
   the heuristic baseline on the same bag set.

## 12. Reuse / do-not-duplicate

- **Tracker:** copy from
  `src/yolo_person_detector/yolo_person_detector/pedestrian_behaviour_predictor.py`
  into `pedestrian_diffusion_predictor/tracker.py`. Do not import
  across packages — keep the new package independent.
- **Fusion node** (`src/adapt_full/adapt_full/adapt_lidar_camera_fusion.py`)
  and detection front-ends (`adapt_lidar_processing.py`,
  `rgbd_pedestrian_detector.py`) remain untouched.
- **MPPI core** (`mppi.py`, `bicycle_model.py`, `reference_path.py`)
  is touched only in Phase D, and only the obstacle-cost branch.
- **Conda env** (`adapt`, py3.12) and the
  `adapt_requirements.text` pin (`torch==2.5.1+cu121`) are reused
  unchanged for both training and inference.

## 13. Risk register

- **Distribution shift (AV2 → UIUC).** Phase B finetune is the
  primary mitigation. Phase C1 shadow A/B is the gate; if minFDE on
  UIUC bags is > 1.5× AV2 val, escalate to a longer GEM collection
  (target 12-15 hours of bag, ~600 crossing events).
- **10 Hz inference budget under M > 1.** Batch all M × K in one
  forward pass. Model is small enough that M=8, K=20 (effective
  batch 160) fits 6 GB. If still tight, K=10 at runtime, K=20 only
  for offline eval.
- **Mode-flip jitter.** Sticky temporal selection (§4) is the
  mitigation. Tune the sticky-cost ratio if needed.
- **MPPI cost numerics with time-indexed obstacles.** Keep the
  existing soft Gaussian repulsion in `mppi.py`; only the obstacle
  position becomes time-dependent. Hard step costs at clearance
  radius are explicitly avoided.
- **DDIM frame-to-frame jitter.** Fix the DDIM noise schedule with a
  constant seed per-tick OR average two independent samples.
- **Dataset license.** AV2 is CC-BY 4.0 — fine for academic use; add
  attribution in `pedestrian_diffusion_predictor/package.xml` and a
  one-line credit in `docs/adapt_design.md`.

## 14. Repo hygiene (do this BEFORE the first training run)

Add these patterns to `cs_588_g10/.gitignore` so weights, datasets,
and TensorBoard logs cannot be accidentally committed:

```
data/
models/
*.pt
*.ckpt
logs/
```

Verify with `git status` after staging that no `.pt`, `.ckpt`, or
`data/*.parquet` files appear.

## 15. Files this plan will create or modify

**Created during execution (do not exist yet):**

- `src/pedestrian_diffusion_predictor/package.xml`
- `src/pedestrian_diffusion_predictor/setup.py`
- `src/pedestrian_diffusion_predictor/setup.cfg`
- `src/pedestrian_diffusion_predictor/resource/pedestrian_diffusion_predictor`
- `src/pedestrian_diffusion_predictor/pedestrian_diffusion_predictor/__init__.py`
- `src/pedestrian_diffusion_predictor/pedestrian_diffusion_predictor/dataset.py`
- `src/pedestrian_diffusion_predictor/pedestrian_diffusion_predictor/model.py`
- `src/pedestrian_diffusion_predictor/pedestrian_diffusion_predictor/train.py`
- `src/pedestrian_diffusion_predictor/pedestrian_diffusion_predictor/finetune.py`
- `src/pedestrian_diffusion_predictor/pedestrian_diffusion_predictor/eval.py`
- `src/pedestrian_diffusion_predictor/pedestrian_diffusion_predictor/infer_node.py`
- `src/pedestrian_diffusion_predictor/pedestrian_diffusion_predictor/tracker.py`

**Modified (Phase D only):**

- `src/vehicle_drivers/mppi_controller/mppi_controller/mppi.py`
- `src/vehicle_drivers/mppi_controller/mppi_controller/adapt_mppi_node.py`

**Modified once at start (repo hygiene):**

- `.gitignore` — add `data/`, `models/`, `*.pt`, `*.ckpt`, `logs/`.

## 16. What to read before starting

To pick this up cold, the next engineer should read, in order:

1. This document.
2. `docs/adapt_design.md` — full design context, MPPI cost structure,
   pipeline diagram, and the "Aspirational research track" section
   that motivates this work.
3. `src/yolo_person_detector/yolo_person_detector/pedestrian_behaviour_predictor.py`
   — the node being replaced; understand the tracker, smoothing, and
   prediction logic before reimplementing.
4. `src/vehicle_drivers/mppi_controller/mppi_controller/mppi.py` and
   `adapt_mppi_node.py` — only needed before Phase D, but skim early
   to internalize the obstacle-cost shape.
5. The MID paper (Gu et al., 2022, *Stochastic Trajectory Prediction
   via Motion Indeterminacy Diffusion*) — the architectural template.
6. The Argoverse 2 Motion Forecasting documentation — dataset format,
   API, splits.
