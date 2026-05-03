#!/usr/bin/env python3
"""Preprocess ETH/UCY pedestrian trajectory data for diffusion training.

Converts the standard ETH/UCY annotation format (frame, ped_id, x, y at 2.5 Hz)
into .npz shards compatible with our diffusion model's TrajectoryDataset and
JointTrajectoryDataset.

Key steps:
  1. Parse tab-separated trajectory files
  2. Interpolate 2.5 Hz -> 4 Hz (our model's native rate)
  3. Extract sliding windows: 20-step history (5s) + 20-step future (5s)
  4. Ego-normalize: last observed position at origin, compute velocities
  5. Save as .npz shards (single-agent and joint)

Usage:
    python scripts/preprocess_eth_ucy.py \
        --input data/eth_ucy/raw \
        --output data/eth_ucy_processed \
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

NATIVE_FPS = 2.5        # ETH/UCY annotation rate
TARGET_FPS = 4.0        # Our model's expected rate
TARGET_DT = 1.0 / TARGET_FPS  # 0.25 s
FRAME_STEP = 10         # Frame increment in annotation files (25fps video / 2.5Hz = 10)

T_HIST = 20             # History steps (5 s at 4 Hz)
T_FUT = 20              # Future steps (5 s at 4 Hz)
T_TOTAL = T_HIST + T_FUT  # 40 steps = 10 s total window

SHARD_SIZE_SINGLE = 8192
SHARD_SIZE_JOINT = 4096
MAX_AGENTS_JOINT = 16

MIN_TRAJ_LENGTH = 20    # Minimum frames (at 2.5Hz) = 8s for a usable window


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def load_scene(filepath: str) -> dict:
    """Load an ETH/UCY annotation file.

    Returns dict: {ped_id: [(frame, x, y), ...]} sorted by frame.
    """
    data = np.loadtxt(filepath, delimiter="\t")
    # Columns: frame, ped_id, x, y
    trajectories = {}
    for row in data:
        frame, pid, x, y = row[0], int(row[1]), row[2], row[3]
        if pid not in trajectories:
            trajectories[pid] = []
        trajectories[pid].append((frame, x, y))

    # Sort each trajectory by frame
    for pid in trajectories:
        trajectories[pid].sort(key=lambda r: r[0])

    return trajectories


def interpolate_trajectory(frames, positions, target_dt=TARGET_DT):
    """Interpolate trajectory from 2.5Hz to 4Hz using cubic spline.

    Parameters
    ----------
    frames : array of frame numbers
    positions : (N, 2) array of (x, y)
    target_dt : target time step

    Returns
    -------
    times_4hz : (M,) array of time stamps at 4Hz
    positions_4hz : (M, 2) array of interpolated positions
    """
    # Convert frames to seconds
    times = np.array(frames, dtype=np.float64) / (NATIVE_FPS * FRAME_STEP)

    if len(times) < 4:
        # Not enough points for cubic, use linear
        kind = "linear"
    else:
        kind = "cubic"

    # Interpolate x and y separately
    fx = interp1d(times, positions[:, 0], kind=kind, fill_value="extrapolate")
    fy = interp1d(times, positions[:, 1], kind=kind, fill_value="extrapolate")

    # Generate target timestamps at 4Hz
    t_start = times[0]
    t_end = times[-1]
    n_steps = int((t_end - t_start) * TARGET_FPS) + 1
    times_4hz = np.linspace(t_start, t_start + (n_steps - 1) * target_dt, n_steps)

    # Only keep times within original range (no extrapolation)
    times_4hz = times_4hz[times_4hz <= t_end + 1e-6]

    positions_4hz = np.stack([fx(times_4hz), fy(times_4hz)], axis=-1)
    return times_4hz, positions_4hz


# ---------------------------------------------------------------------------
# Window extraction (single-agent)
# ---------------------------------------------------------------------------

def extract_single_windows(trajectories: dict) -> list:
    """Extract all valid (history, future) windows from a scene.

    Returns list of dicts with keys:
        history      : (20, 4) float32 - [x, y, vx, vy]
        history_mask : (20,)   uint8
        future       : (20, 2) float32 - [x, y]
        ego_vel      : (2,)    float32 - [0, 0] (no ego vehicle)
    """
    windows = []

    for pid, raw_traj in trajectories.items():
        frames = np.array([r[0] for r in raw_traj])
        positions = np.array([[r[1], r[2]] for r in raw_traj])

        if len(frames) < MIN_TRAJ_LENGTH:
            continue

        # Interpolate to 4Hz
        times_4hz, pos_4hz = interpolate_trajectory(frames, positions)

        if len(pos_4hz) < T_TOTAL:
            continue

        # Compute velocities via finite differences
        vel_4hz = np.zeros_like(pos_4hz)
        vel_4hz[1:] = (pos_4hz[1:] - pos_4hz[:-1]) / TARGET_DT
        vel_4hz[0] = vel_4hz[1]  # copy first

        # Sliding window with stride 4 (1 second)
        stride = 4
        for start in range(0, len(pos_4hz) - T_TOTAL + 1, stride):
            hist_pos = pos_4hz[start:start + T_HIST]          # (20, 2)
            hist_vel = vel_4hz[start:start + T_HIST]          # (20, 2)
            fut_pos = pos_4hz[start + T_HIST:start + T_TOTAL] # (20, 2)

            # Ego-normalize: last observed position at origin
            origin = hist_pos[-1].copy()
            hist_pos_norm = hist_pos - origin
            fut_pos_norm = fut_pos - origin

            # History: [x, y, vx, vy]
            history = np.concatenate([hist_pos_norm, hist_vel], axis=-1).astype(np.float32)
            history_mask = np.ones(T_HIST, dtype=np.uint8)
            future = fut_pos_norm.astype(np.float32)
            ego_vel = np.zeros(2, dtype=np.float32)  # No ego vehicle in ETH/UCY

            windows.append({
                "history": history,
                "history_mask": history_mask,
                "future": future,
                "ego_vel": ego_vel,
            })

    return windows


# ---------------------------------------------------------------------------
# Window extraction (joint multi-agent)
# ---------------------------------------------------------------------------

def extract_joint_windows(trajectories: dict) -> list:
    """Extract joint multi-agent windows where multiple pedestrians overlap in time.

    Returns list of dicts with keys:
        histories      : (M, 20, 4) float32
        history_masks  : (M, 20)    uint8
        futures        : (M, 20, 2) float32
        agent_masks    : (M,)       uint8
        ego_vels       : (2,)       float32
    where M = MAX_AGENTS_JOINT (padded)
    """
    # First, interpolate all trajectories to 4Hz and store with time info
    interp_trajs = {}
    for pid, raw_traj in trajectories.items():
        frames = np.array([r[0] for r in raw_traj])
        positions = np.array([[r[1], r[2]] for r in raw_traj])

        if len(frames) < MIN_TRAJ_LENGTH:
            continue

        times_4hz, pos_4hz = interpolate_trajectory(frames, positions)
        if len(pos_4hz) < T_TOTAL:
            continue

        # Velocities
        vel_4hz = np.zeros_like(pos_4hz)
        vel_4hz[1:] = (pos_4hz[1:] - pos_4hz[:-1]) / TARGET_DT
        vel_4hz[0] = vel_4hz[1]

        interp_trajs[pid] = {
            "times": times_4hz,
            "pos": pos_4hz,
            "vel": vel_4hz,
        }

    if len(interp_trajs) < 2:
        return []

    # Find global time range
    all_times = np.concatenate([t["times"] for t in interp_trajs.values()])
    t_min = all_times.min()
    t_max = all_times.max()

    # Create a global time grid at 4Hz
    n_global = int((t_max - t_min) * TARGET_FPS) + 1
    global_times = np.linspace(t_min, t_min + (n_global - 1) * TARGET_DT, n_global)

    # For each pedestrian, map to global time indices
    ped_data = {}
    for pid, traj in interp_trajs.items():
        # Find which global indices this pedestrian spans
        start_idx = int(round((traj["times"][0] - t_min) * TARGET_FPS))
        end_idx = start_idx + len(traj["pos"])
        end_idx = min(end_idx, n_global)
        actual_len = end_idx - start_idx

        ped_data[pid] = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "pos": traj["pos"][:actual_len],
            "vel": traj["vel"][:actual_len],
        }

    # Extract joint windows with sliding window
    windows = []
    stride = 8  # 2 seconds between windows (less overlap for joint)

    for win_start in range(0, n_global - T_TOTAL + 1, stride):
        win_end = win_start + T_TOTAL
        hist_end = win_start + T_HIST

        # Find agents present for the full window
        present_pids = []
        for pid, pd in ped_data.items():
            if pd["start_idx"] <= win_start and pd["end_idx"] >= win_end:
                present_pids.append(pid)

        if len(present_pids) < 2:
            continue

        # Cap at MAX_AGENTS_JOINT
        if len(present_pids) > MAX_AGENTS_JOINT:
            present_pids = present_pids[:MAX_AGENTS_JOINT]

        M = MAX_AGENTS_JOINT
        histories = np.zeros((M, T_HIST, 4), dtype=np.float32)
        history_masks = np.zeros((M, T_HIST), dtype=np.uint8)
        futures = np.zeros((M, T_FUT, 2), dtype=np.float32)
        agent_masks = np.zeros(M, dtype=np.uint8)

        # Use first agent's last observed position as shared origin
        first_pid = present_pids[0]
        first_pd = ped_data[first_pid]
        local_hist_end = hist_end - first_pd["start_idx"]
        origin = first_pd["pos"][local_hist_end - 1].copy()

        for i, pid in enumerate(present_pids):
            pd = ped_data[pid]
            local_start = win_start - pd["start_idx"]
            local_hist_end_i = local_start + T_HIST
            local_fut_end_i = local_start + T_TOTAL

            pos_hist = pd["pos"][local_start:local_hist_end_i] - origin
            vel_hist = pd["vel"][local_start:local_hist_end_i]
            pos_fut = pd["pos"][local_hist_end_i:local_fut_end_i] - origin

            histories[i] = np.concatenate([pos_hist, vel_hist], axis=-1)
            history_masks[i] = 1
            futures[i] = pos_fut
            agent_masks[i] = 1

        windows.append({
            "histories": histories,
            "history_masks": history_masks,
            "futures": futures,
            "agent_masks": agent_masks,
            "ego_vels": np.zeros(2, dtype=np.float32),
        })

    return windows


# ---------------------------------------------------------------------------
# Shard writing
# ---------------------------------------------------------------------------

def save_single_shards(windows: list, output_dir: pathlib.Path, prefix: str):
    """Save single-agent windows as .npz shards."""
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(windows)
    if n == 0:
        print(f"  [WARN] No windows for {prefix}")
        return

    # Shuffle
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)

    shard_idx = 0
    for start in range(0, n, SHARD_SIZE_SINGLE):
        end = min(start + SHARD_SIZE_SINGLE, n)
        batch_idx = indices[start:end]

        history = np.stack([windows[i]["history"] for i in batch_idx])
        history_mask = np.stack([windows[i]["history_mask"] for i in batch_idx])
        future = np.stack([windows[i]["future"] for i in batch_idx])
        ego_vel = np.stack([windows[i]["ego_vel"] for i in batch_idx])

        shard_path = output_dir / f"{prefix}_shard_{shard_idx:04d}.npz"
        np.savez_compressed(
            shard_path,
            history=history,
            history_mask=history_mask,
            future=future,
            ego_vel=ego_vel,
        )
        print(f"  Saved {shard_path.name}: {len(batch_idx)} samples")
        shard_idx += 1

    print(f"  Total: {n} single-agent windows in {shard_idx} shards")


def save_joint_shards(windows: list, output_dir: pathlib.Path, prefix: str):
    """Save joint multi-agent windows as .npz shards."""
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(windows)
    if n == 0:
        print(f"  [WARN] No joint windows for {prefix}")
        return

    rng = np.random.default_rng(42)
    indices = rng.permutation(n)

    shard_idx = 0
    for start in range(0, n, SHARD_SIZE_JOINT):
        end = min(start + SHARD_SIZE_JOINT, n)
        batch_idx = indices[start:end]

        histories = np.stack([windows[i]["histories"] for i in batch_idx])
        history_masks = np.stack([windows[i]["history_masks"] for i in batch_idx])
        futures = np.stack([windows[i]["futures"] for i in batch_idx])
        agent_masks = np.stack([windows[i]["agent_masks"] for i in batch_idx])
        ego_vels = np.stack([windows[i]["ego_vels"] for i in batch_idx])

        shard_path = output_dir / f"{prefix}_shard_{shard_idx:04d}.npz"
        np.savez_compressed(
            shard_path,
            histories=histories,
            history_masks=history_masks,
            futures=futures,
            agent_masks=agent_masks,
            ego_vels=ego_vels,
        )
        print(f"  Saved {shard_path.name}: {len(batch_idx)} scenes")
        shard_idx += 1

    print(f"  Total: {n} joint windows in {shard_idx} shards")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SCENE_FILES = {
    "eth": "eth/biwi_eth.txt",
    "hotel": "hotel/biwi_hotel.txt",
    "univ": "univ/students003.txt",
    "zara1": "zara1/crowds_zara01.txt",
    "zara2": "zara2/crowds_zara02.txt",
}


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ETH/UCY for diffusion trajectory prediction"
    )
    parser.add_argument("--input", type=str, default="data/eth_ucy/raw",
                        help="Directory containing raw scene files")
    parser.add_argument("--output", type=str, default="data/eth_ucy_processed",
                        help="Output directory for processed shards")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["single", "joint", "both"],
                        help="Generate single-agent, joint, or both")
    parser.add_argument("--val-scene", type=str, default="hotel",
                        choices=list(SCENE_FILES.keys()),
                        help="Scene to hold out for validation (leave-one-out)")
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output)

    print(f"ETH/UCY Preprocessing")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Mode:   {args.mode}")
    print(f"  Val scene: {args.val_scene}")
    print()

    # Load and process all scenes
    train_single = []
    val_single = []
    train_joint = []
    val_joint = []

    for scene_name, rel_path in SCENE_FILES.items():
        filepath = input_dir / rel_path
        if not filepath.exists():
            print(f"  [SKIP] {filepath} not found")
            continue

        print(f"Processing {scene_name} ({filepath})...")
        trajectories = load_scene(str(filepath))
        print(f"  Loaded {len(trajectories)} pedestrians")

        is_val = (scene_name == args.val_scene)

        if args.mode in ("single", "both"):
            sw = extract_single_windows(trajectories)
            print(f"  Single-agent windows: {len(sw)}")
            if is_val:
                val_single.extend(sw)
            else:
                train_single.extend(sw)

        if args.mode in ("joint", "both"):
            jw = extract_joint_windows(trajectories)
            print(f"  Joint windows: {len(jw)}")
            if is_val:
                val_joint.extend(jw)
            else:
                train_joint.extend(jw)

        print()

    # Save shards
    print("=" * 50)
    print("Saving shards...")

    if args.mode in ("single", "both"):
        print(f"\n[Single-agent TRAIN] {len(train_single)} windows")
        save_single_shards(train_single, output_dir / "train", "eth_ucy")
        print(f"\n[Single-agent VAL] {len(val_single)} windows")
        save_single_shards(val_single, output_dir / "val", "eth_ucy")

    if args.mode in ("joint", "both"):
        print(f"\n[Joint TRAIN] {len(train_joint)} windows")
        save_joint_shards(train_joint, output_dir / "train_joint", "eth_ucy")
        print(f"\n[Joint VAL] {len(val_joint)} windows")
        save_joint_shards(val_joint, output_dir / "val_joint", "eth_ucy")

    print("\nDone!")


if __name__ == "__main__":
    main()
