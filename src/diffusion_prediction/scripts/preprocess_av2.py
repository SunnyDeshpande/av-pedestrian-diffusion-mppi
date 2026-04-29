#!/usr/bin/env python3
"""Preprocess Argoverse 2 Motion Forecasting dataset for diffusion training.

Extracts pedestrian trajectories, normalises to ego frame, and saves
as .npz shards compatible with diffusion_prediction.dataset.TrajectoryDataset.

Usage:
    python scripts/preprocess_av2.py \
        --input data/av2 \
        --output data/av2_processed \
        --workers 4
"""

import argparse
import math
import os
import pathlib
from multiprocessing import Pool

import numpy as np

# AV2 imports (pip install av2)
try:
    from av2.datasets.motion_forecasting.scenario_serialization import (
        load_argoverse_scenario_parquet,
    )
    from av2.datasets.motion_forecasting.data_schema import ObjectType
except ImportError:
    raise ImportError(
        "av2 package required: pip install av2\n"
        "See https://argoverse.github.io/user-guide/datasets/motion_forecasting.html"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DT = 0.25          # target time step (seconds)
AV2_HZ = 10        # AV2 native sample rate
T_HIST = 20        # history steps (5 s / 0.25 s)
T_FUT = 20         # future steps (5 s / 0.25 s)
SHARD_SIZE = 8192  # rows per shard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def interp_positions(timestamps, positions, target_times):
    """Linearly interpolate (x, y) positions at target_times.

    Parameters
    ----------
    timestamps : array-like, shape (N,)
    positions  : array-like, shape (N, 2)
    target_times : array-like, shape (M,)

    Returns
    -------
    np.ndarray, shape (M, 2)
    """
    xs = np.interp(target_times, timestamps, positions[:, 0])
    ys = np.interp(target_times, timestamps, positions[:, 1])
    return np.stack([xs, ys], axis=-1)


def ego_frame_transform(positions, ego_xy, ego_heading):
    """Transform world positions into ego frame at t=0.

    Parameters
    ----------
    positions   : (N, 2) in world frame
    ego_xy      : (2,) ego position at t=0 in world
    ego_heading : float, ego heading at t=0 in radians

    Returns
    -------
    (N, 2) in ego frame (x = forward, y = left)
    """
    translated = positions - ego_xy
    c = math.cos(-ego_heading)
    s = math.sin(-ego_heading)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return translated @ R.T


def compute_heading(positions):
    """Estimate heading from consecutive positions."""
    if len(positions) < 2:
        return 0.0
    dx = positions[-1, 0] - positions[-2, 0]
    dy = positions[-1, 1] - positions[-2, 1]
    return math.atan2(dy, dx)


# ---------------------------------------------------------------------------
# Process a single scenario
# ---------------------------------------------------------------------------

def process_scenario(scenario_path: str) -> list:
    """Extract pedestrian windows from one AV2 scenario.

    Returns list of dicts with keys:
        history       : (20, 4) float32
        history_mask  : (20,)   uint8
        future        : (20, 2) float32
        ego_vel       : (2,)    float32
    """
    scenario_dir = pathlib.Path(scenario_path)
    parquet = scenario_dir / f"scenario_{scenario_dir.name}.parquet"
    if not parquet.exists():
        # Try finding any parquet
        parquets = list(scenario_dir.glob("*.parquet"))
        if not parquets:
            return []
        parquet = parquets[0]

    try:
        scenario = load_argoverse_scenario_parquet(parquet)
    except Exception as e:
        print(f"[WARN] failed to load {scenario_dir.name}: {e}")
        return []

    # Build timestep index -> nanosecond timestamp mapping
    # scenario.timestamps_ns is an array of nanosecond timestamps (length 110 at 10Hz)
    ts_ns = np.array(scenario.timestamps_ns, dtype=np.float64)

    # Find AV (focal) track for ego frame
    av_track = None
    for track in scenario.tracks:
        if track.track_id == scenario.focal_track_id:
            av_track = track
            break
    if av_track is None:
        return []

    # Build AV position array indexed by timestep
    av_states = {s.timestep: s for s in av_track.object_states}
    av_timesteps = sorted(av_states.keys())
    av_pos = np.array(
        [[av_states[t].position[0], av_states[t].position[1]] for t in av_timesteps],
        dtype=np.float64,
    )
    av_ts_ns = np.array([ts_ns[t] for t in av_timesteps], dtype=np.float64)

    # AV state at t=0 (the prediction moment: timestep 49 in AV2)
    # Find the closest AV state to timestep 49
    t0_step = 49
    t0_ns = ts_ns[t0_step] if t0_step < len(ts_ns) else ts_ns[-1]

    # Interpolate AV position at t0
    ego_xy = interp_positions(av_ts_ns, av_pos, np.array([t0_ns]))[0]
    ego_heading = compute_heading(
        interp_positions(av_ts_ns, av_pos, np.array([t0_ns - 0.1e9, t0_ns]))
    )

    # Ego velocity at t0 from AV track velocity field
    v_ego = 0.0
    if t0_step in av_states and av_states[t0_step].velocity is not None:
        vx, vy = av_states[t0_step].velocity[0], av_states[t0_step].velocity[1]
        v_ego = math.sqrt(vx ** 2 + vy ** 2)
    elif len(av_pos) >= 2:
        # Fallback: finite difference
        dt_s = (av_ts_ns[-1] - av_ts_ns[-2]) / 1e9
        if dt_s > 0:
            v_ego = np.linalg.norm(av_pos[-1] - av_pos[-2]) / dt_s

    ego_vel = np.array([v_ego, 0.0], dtype=np.float32)

    # Define target time grids (in nanoseconds)
    hist_times = t0_ns + np.arange(-T_HIST + 1, 1) * DT * 1e9
    fut_times = t0_ns + np.arange(1, T_FUT + 1) * DT * 1e9

    results = []

    for track in scenario.tracks:
        # Filter to pedestrians only
        if track.object_type != ObjectType.PEDESTRIAN:
            continue

        states = track.object_states
        if len(states) < 2:
            continue

        # Get timestamps (nanoseconds) and positions for this track
        track_timesteps = [s.timestep for s in states]
        track_ts_ns = np.array([ts_ns[t] for t in track_timesteps], dtype=np.float64)
        track_pos = np.array(
            [[s.position[0], s.position[1]] for s in states],
            dtype=np.float64,
        )

        # Check temporal coverage
        t_min, t_max = track_ts_ns.min(), track_ts_ns.max()
        if t_min > hist_times[0] or t_max < fut_times[-1]:
            continue

        # Interpolate to target dt=0.25s
        hist_xy = interp_positions(track_ts_ns, track_pos, hist_times)  # (20, 2)
        fut_xy = interp_positions(track_ts_ns, track_pos, fut_times)    # (20, 2)

        # Transform to ego frame
        hist_ego = ego_frame_transform(hist_xy, ego_xy, ego_heading).astype(np.float32)
        fut_ego = ego_frame_transform(fut_xy, ego_xy, ego_heading).astype(np.float32)

        # Compute velocities (first-difference)
        hist_vel = np.zeros_like(hist_ego)
        hist_vel[1:] = (hist_ego[1:] - hist_ego[:-1]) / DT

        history = np.concatenate([hist_ego, hist_vel], axis=-1)  # (20, 4)
        history_mask = np.ones(T_HIST, dtype=np.uint8)

        results.append({
            "history": history,
            "history_mask": history_mask,
            "future": fut_ego,
            "ego_vel": ego_vel,
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_shard(records, output_dir, shard_idx):
    """Save a list of records as a .npz shard."""
    if not records:
        return
    np.savez_compressed(
        os.path.join(output_dir, f"shard_{shard_idx:05d}.npz"),
        history=np.stack([r["history"] for r in records]),
        history_mask=np.stack([r["history_mask"] for r in records]),
        future=np.stack([r["future"] for r in records]),
        ego_vel=np.stack([r["ego_vel"] for r in records]),
    )


def main():
    parser = argparse.ArgumentParser(description="Preprocess AV2 for diffusion training")
    parser.add_argument("--input", type=str, required=True, help="Path to raw AV2 dataset")
    parser.add_argument("--output", type=str, required=True, help="Path for processed shards")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    # Find scenario directories
    input_path = pathlib.Path(args.input)

    for split in ["train", "val"]:
        split_dir = input_path / split
        if not split_dir.exists():
            print(f"[WARN] split directory not found: {split_dir}")
            continue

        out_dir = os.path.join(args.output, split)
        os.makedirs(out_dir, exist_ok=True)

        scenario_dirs = sorted([
            str(d) for d in split_dir.iterdir()
            if d.is_dir()
        ])
        print(f"[{split}] found {len(scenario_dirs)} scenarios")

        # Process in parallel
        all_records = []
        with Pool(args.workers) as pool:
            for i, records in enumerate(pool.imap_unordered(process_scenario, scenario_dirs)):
                all_records.extend(records)
                if (i + 1) % 500 == 0:
                    print(f"  processed {i+1}/{len(scenario_dirs)} scenarios, "
                          f"{len(all_records)} windows so far")

        print(f"[{split}] total pedestrian windows: {len(all_records)}")

        # Save shards
        for shard_idx in range(0, len(all_records), SHARD_SIZE):
            chunk = all_records[shard_idx : shard_idx + SHARD_SIZE]
            save_shard(chunk, out_dir, shard_idx // SHARD_SIZE)

        print(f"[{split}] saved {math.ceil(len(all_records) / SHARD_SIZE)} shards to {out_dir}")


if __name__ == "__main__":
    main()
