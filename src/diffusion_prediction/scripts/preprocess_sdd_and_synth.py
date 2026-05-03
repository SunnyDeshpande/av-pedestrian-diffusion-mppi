#!/usr/bin/env python3
"""Preprocess Stanford Drone Dataset + generate synthetic arc trajectories.

SDD annotations: track_id xmin ymin xmax ymax frame_id lost occluded generated label
Video FPS: 30. We subsample to 4 Hz (every 7.5 frames, interpolated).

Synthetic arcs: parametric curves with controlled curvature distribution
to fill gaps in real data for tight turns.

Usage:
    python scripts/preprocess_sdd_and_synth.py \
        --sdd-dir data/sdd/raw/annotations \
        --output data/sdd_synth_processed \
        --mode both
"""

import argparse
import os
import pathlib

import numpy as np
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SDD_FPS = 30.0
TARGET_FPS = 4.0
TARGET_DT = 1.0 / TARGET_FPS  # 0.25 s
T_HIST = 20
T_FUT = 20
T_TOTAL = T_HIST + T_FUT

SHARD_SIZE_SINGLE = 8192
SHARD_SIZE_JOINT = 4096
MAX_AGENTS_JOINT = 16

# SDD pixel-to-meter scales (estimated from OpenTraj)
# These are approximate; good enough for learning motion dynamics
SCENE_SCALES = {
    "bookstore": 0.038,
    "coupa": 0.030,
    "deathCircle": 0.028,
    "gates": 0.035,
    "hyang": 0.030,
    "little": 0.030,
    "nexus": 0.032,
    "quad": 0.035,
}
DEFAULT_SCALE = 0.032


# ---------------------------------------------------------------------------
# SDD Parsing
# ---------------------------------------------------------------------------

def load_sdd_scene(filepath: str, scale: float, include_bikers: bool = True):
    """Load an SDD annotation file, return trajectories in meters.

    Returns dict: {track_id: [(frame, x_m, y_m), ...]}
    Only keeps Pedestrian (and optionally Biker) tracks that are not lost.
    """
    trajectories = {}
    valid_labels = {'"Pedestrian"'}
    if include_bikers:
        valid_labels.add('"Biker"')

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue

            track_id = int(parts[0])
            xmin, ymin, xmax, ymax = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            frame = int(parts[5])
            lost = int(parts[6])
            label = parts[9]

            if label not in valid_labels:
                continue
            if lost == 1:
                continue

            # Center of bounding box -> meters
            cx = (xmin + xmax) / 2.0 * scale
            cy = (ymin + ymax) / 2.0 * scale

            if track_id not in trajectories:
                trajectories[track_id] = []
            trajectories[track_id].append((frame, cx, cy))

    for tid in trajectories:
        trajectories[tid].sort(key=lambda r: r[0])

    return trajectories


def interpolate_to_4hz(frames, positions):
    """Interpolate SDD 30fps annotations to 4Hz."""
    times = np.array(frames, dtype=np.float64) / SDD_FPS

    if len(times) < 4:
        kind = "linear"
    else:
        kind = "cubic"

    # Remove duplicate timestamps
    _, unique_idx = np.unique(times, return_index=True)
    times = times[unique_idx]
    positions = positions[unique_idx]

    if len(times) < 2:
        return None, None

    fx = interp1d(times, positions[:, 0], kind=kind, fill_value="extrapolate")
    fy = interp1d(times, positions[:, 1], kind=kind, fill_value="extrapolate")

    t_start = times[0]
    t_end = times[-1]
    n_steps = int((t_end - t_start) * TARGET_FPS) + 1
    times_4hz = np.linspace(t_start, t_start + (n_steps - 1) * TARGET_DT, n_steps)
    times_4hz = times_4hz[times_4hz <= t_end + 1e-6]

    if len(times_4hz) < T_TOTAL:
        return None, None

    positions_4hz = np.stack([fx(times_4hz), fy(times_4hz)], axis=-1)
    return times_4hz, positions_4hz


def extract_windows_from_traj(pos_4hz, stride=4):
    """Extract sliding windows from a 4Hz trajectory."""
    windows = []
    vel = np.zeros_like(pos_4hz)
    vel[1:] = (pos_4hz[1:] - pos_4hz[:-1]) / TARGET_DT
    vel[0] = vel[1]

    for start in range(0, len(pos_4hz) - T_TOTAL + 1, stride):
        hist_pos = pos_4hz[start:start + T_HIST]
        hist_vel = vel[start:start + T_HIST]
        fut_pos = pos_4hz[start + T_HIST:start + T_TOTAL]

        origin = hist_pos[-1].copy()
        hist_pos_norm = hist_pos - origin
        fut_pos_norm = fut_pos - origin

        history = np.concatenate([hist_pos_norm, hist_vel], axis=-1).astype(np.float32)
        history_mask = np.ones(T_HIST, dtype=np.uint8)
        future = fut_pos_norm.astype(np.float32)
        ego_vel = np.zeros(2, dtype=np.float32)

        windows.append({
            "history": history,
            "history_mask": history_mask,
            "future": future,
            "ego_vel": ego_vel,
        })

    return windows


# ---------------------------------------------------------------------------
# Synthetic Arc Generation
# ---------------------------------------------------------------------------

def generate_synthetic_arcs(n_samples=10000, seed=42):
    """Generate diverse synthetic arc trajectories.

    Covers:
    - Tight to wide arcs (R = 1.5 to 15 m)
    - Left and right turns
    - S-bends, U-turns, spirals
    - Varying speeds (0.5 to 2.5 m/s)
    """
    rng = np.random.default_rng(seed)
    windows = []
    t = np.arange(T_TOTAL) * TARGET_DT  # 0 to 9.75s

    for _ in range(n_samples):
        traj_type = rng.choice(["arc", "s_bend", "spiral", "uturn", "accel_arc"], p=[0.35, 0.2, 0.15, 0.15, 0.15])

        if traj_type == "arc":
            # Constant-radius arc
            R = rng.uniform(1.5, 15.0)
            speed = rng.uniform(0.5, 2.5)
            omega = speed / R * rng.choice([-1, 1])
            theta = omega * t
            x = R * np.sin(theta)
            y = R * (1 - np.cos(theta))

        elif traj_type == "s_bend":
            speed = rng.uniform(0.8, 2.0)
            omega = rng.uniform(0.15, 0.5)
            mid = t[-1] / 2
            theta = np.where(t < mid, omega * t, omega * mid - omega * (t - mid))
            x = np.cumsum(speed * np.cos(theta)) * TARGET_DT
            y = np.cumsum(speed * np.sin(theta)) * TARGET_DT

        elif traj_type == "spiral":
            R0 = rng.uniform(2.0, 8.0)
            speed = rng.uniform(0.6, 1.8)
            omega = speed / R0 * rng.choice([-1, 1])
            # Radius grows/shrinks linearly
            R_rate = rng.uniform(-0.3, 0.5)
            R_t = R0 + R_rate * t
            R_t = np.clip(R_t, 1.0, 20.0)
            theta = np.cumsum(speed / R_t) * TARGET_DT
            x = R_t * np.sin(theta)
            y = R_t * (1 - np.cos(theta))

        elif traj_type == "uturn":
            R = rng.uniform(1.5, 5.0)
            speed = rng.uniform(0.6, 1.5)
            omega = speed / R * rng.choice([-1, 1])
            theta = omega * t
            x = R * np.sin(theta)
            y = R * (1 - np.cos(theta))

        elif traj_type == "accel_arc":
            R = rng.uniform(3.0, 12.0)
            speed0 = rng.uniform(0.5, 1.5)
            accel = rng.uniform(0.05, 0.3) * rng.choice([-1, 1])
            speed_t = np.clip(speed0 + accel * t, 0.3, 3.0)
            omega = speed_t / R * rng.choice([-1, 1])
            theta = np.cumsum(omega) * TARGET_DT
            x = np.cumsum(speed_t * np.cos(theta)) * TARGET_DT
            y = np.cumsum(speed_t * np.sin(theta)) * TARGET_DT

        pos = np.stack([x, y], axis=-1)

        # Random rotation
        angle = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(angle), np.sin(angle)
        R_mat = np.array([[c, -s], [s, c]])
        pos = pos @ R_mat.T

        # Random offset (doesn't matter after ego-normalization, but adds variety to velocities)
        pos += rng.uniform(-5, 5, size=2)

        # Add slight noise for realism
        pos += rng.normal(0, 0.02, size=pos.shape)

        # Extract window
        hist_pos = pos[:T_HIST]
        fut_pos = pos[T_HIST:]

        origin = hist_pos[-1].copy()
        hist_pos_norm = hist_pos - origin
        fut_pos_norm = fut_pos - origin

        vel = np.zeros_like(hist_pos_norm)
        vel[1:] = (hist_pos_norm[1:] - hist_pos_norm[:-1]) / TARGET_DT
        vel[0] = vel[1]

        history = np.concatenate([hist_pos_norm, vel], axis=-1).astype(np.float32)
        history_mask = np.ones(T_HIST, dtype=np.uint8)
        future = fut_pos_norm.astype(np.float32)
        ego_vel = np.zeros(2, dtype=np.float32)

        windows.append({
            "history": history,
            "history_mask": history_mask,
            "future": future,
            "ego_vel": ego_vel,
        })

    return windows


# ---------------------------------------------------------------------------
# Shard saving (reuse from ETH/UCY script)
# ---------------------------------------------------------------------------

def save_single_shards(windows, output_dir, prefix, shard_size=SHARD_SIZE_SINGLE):
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(windows)
    if n == 0:
        print(f"  [WARN] No windows for {prefix}")
        return

    rng = np.random.default_rng(42)
    indices = rng.permutation(n)

    shard_idx = 0
    for start in range(0, n, shard_size):
        end = min(start + shard_size, n)
        batch_idx = indices[start:end]

        np.savez_compressed(
            output_dir / f"{prefix}_shard_{shard_idx:04d}.npz",
            history=np.stack([windows[i]["history"] for i in batch_idx]),
            history_mask=np.stack([windows[i]["history_mask"] for i in batch_idx]),
            future=np.stack([windows[i]["future"] for i in batch_idx]),
            ego_vel=np.stack([windows[i]["ego_vel"] for i in batch_idx]),
        )
        print(f"  Saved {prefix}_shard_{shard_idx:04d}.npz: {len(batch_idx)} samples")
        shard_idx += 1

    print(f"  Total: {n} windows in {shard_idx} shards")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess SDD + synthetic arcs")
    parser.add_argument("--sdd-dir", type=str, default="data/sdd/raw/annotations")
    parser.add_argument("--output", type=str, default="data/sdd_synth_processed")
    parser.add_argument("--synth-train", type=int, default=15000,
                        help="Number of synthetic arc training samples")
    parser.add_argument("--synth-val", type=int, default=2000,
                        help="Number of synthetic arc validation samples")
    parser.add_argument("--val-scenes", type=str, nargs="+",
                        default=["hyang", "coupa"],
                        help="SDD scenes to hold out for validation")
    parser.add_argument("--include-bikers", action="store_true", default=True)
    args = parser.parse_args()

    sdd_dir = pathlib.Path(args.sdd_dir)
    output_dir = pathlib.Path(args.output)

    print("=" * 60)
    print("SDD + Synthetic Arc Preprocessing")
    print("=" * 60)

    # --- Process SDD ---
    train_windows = []
    val_windows = []

    if sdd_dir.exists():
        print(f"\n[SDD] Scanning {sdd_dir}...")
        scene_dirs = sorted([d for d in sdd_dir.iterdir() if d.is_dir()])

        for scene_dir in scene_dirs:
            scene_name = scene_dir.name
            scale = SCENE_SCALES.get(scene_name, DEFAULT_SCALE)
            is_val = scene_name in args.val_scenes

            video_dirs = sorted([d for d in scene_dir.iterdir() if d.is_dir()])
            scene_windows = []

            for video_dir in video_dirs:
                ann_file = video_dir / "annotations.txt"
                if not ann_file.exists():
                    continue

                trajectories = load_sdd_scene(str(ann_file), scale, args.include_bikers)

                for tid, raw_traj in trajectories.items():
                    frames = np.array([r[0] for r in raw_traj])
                    positions = np.array([[r[1], r[2]] for r in raw_traj])

                    if len(frames) < 30:  # need ~1s at 30fps minimum
                        continue

                    times_4hz, pos_4hz = interpolate_to_4hz(frames, positions)
                    if times_4hz is None:
                        continue

                    wins = extract_windows_from_traj(pos_4hz, stride=4)
                    scene_windows.extend(wins)

            print(f"  {scene_name}: {len(scene_windows)} windows ({'val' if is_val else 'train'})")
            if is_val:
                val_windows.extend(scene_windows)
            else:
                train_windows.extend(scene_windows)

        print(f"\n[SDD] Total: {len(train_windows)} train, {len(val_windows)} val")
    else:
        print(f"[WARN] SDD dir not found: {sdd_dir}")

    # --- Generate synthetic arcs ---
    print(f"\n[Synthetic] Generating {args.synth_train} train + {args.synth_val} val arcs...")
    synth_train = generate_synthetic_arcs(n_samples=args.synth_train, seed=42)
    synth_val = generate_synthetic_arcs(n_samples=args.synth_val, seed=123)
    print(f"  Generated {len(synth_train)} train, {len(synth_val)} val synthetic arcs")

    # --- Combine ---
    all_train = train_windows + synth_train
    all_val = val_windows + synth_val
    print(f"\n[Combined] {len(all_train)} train, {len(all_val)} val")

    # --- Save ---
    print(f"\nSaving to {output_dir}...")
    save_single_shards(all_train, output_dir / "train", "sdd_synth")
    save_single_shards(all_val, output_dir / "val", "sdd_synth")

    print("\nDone!")


if __name__ == "__main__":
    main()
