#!/usr/bin/env python3
"""Preprocess Argoverse 2 for joint multi-agent diffusion training.

Groups all pedestrians per scenario into a single scene-level sample,
padded to max_agents. Output .npz shards are compatible with
diffusion_prediction.dataset_joint.JointTrajectoryDataset.

Usage:
    python scripts/preprocess_av2_joint.py \
        --input data/av2 \
        --output data/av2_processed_joint \
        --workers 4 --max-agents 16
"""

import argparse
import math
import os
import pathlib
from multiprocessing import Pool

import numpy as np

try:
    from av2.datasets.motion_forecasting.scenario_serialization import (
        load_argoverse_scenario_parquet,
    )
    from av2.datasets.motion_forecasting.data_schema import ObjectType
except ImportError:
    raise ImportError("av2 package required: pip install av2")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DT = 0.25
T_HIST = 20
T_FUT = 20
SHARD_SIZE = 4096
MAX_AGENTS = 16  # default, overridden by CLI


# ---------------------------------------------------------------------------
# Helpers (same as single-agent preprocessor)
# ---------------------------------------------------------------------------

def interp_positions(timestamps, positions, target_times):
    xs = np.interp(target_times, timestamps, positions[:, 0])
    ys = np.interp(target_times, timestamps, positions[:, 1])
    return np.stack([xs, ys], axis=-1)


def ego_frame_transform(positions, ego_xy, ego_heading):
    translated = positions - ego_xy
    c = math.cos(-ego_heading)
    s = math.sin(-ego_heading)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return translated @ R.T


def compute_heading(positions):
    if len(positions) < 2:
        return 0.0
    dx = positions[-1, 0] - positions[-2, 0]
    dy = positions[-1, 1] - positions[-2, 1]
    return math.atan2(dy, dx)


# ---------------------------------------------------------------------------
# Process a single scenario → one scene-level sample
# ---------------------------------------------------------------------------

# Module-level variable set by initializer
_max_agents = MAX_AGENTS


def _init_worker(max_agents):
    global _max_agents
    _max_agents = max_agents


def process_scenario(scenario_path: str) -> list:
    """Extract a scene-level sample from one AV2 scenario.

    Returns a list with 0 or 1 dict. Each dict has:
        histories    : (M, 20, 4) float32
        history_masks: (M, 20)    uint8
        futures      : (M, 20, 2) float32
        agent_mask   : (M,)       uint8
        ego_vel      : (2,)       float32
    where M = _max_agents, padded with zeros for missing agents.
    """
    max_agents = _max_agents
    scenario_dir = pathlib.Path(scenario_path)
    parquet = scenario_dir / f"scenario_{scenario_dir.name}.parquet"
    if not parquet.exists():
        parquets = list(scenario_dir.glob("*.parquet"))
        if not parquets:
            return []
        parquet = parquets[0]

    try:
        scenario = load_argoverse_scenario_parquet(parquet)
    except Exception:
        return []

    ts_ns = np.array(scenario.timestamps_ns, dtype=np.float64)

    # Find AV track
    av_track = None
    for track in scenario.tracks:
        if track.track_id == scenario.focal_track_id:
            av_track = track
            break
    if av_track is None:
        return []

    av_states = {s.timestep: s for s in av_track.object_states}
    av_timesteps = sorted(av_states.keys())
    av_pos = np.array(
        [[av_states[t].position[0], av_states[t].position[1]] for t in av_timesteps],
        dtype=np.float64,
    )
    av_ts_ns = np.array([ts_ns[t] for t in av_timesteps], dtype=np.float64)

    t0_step = 49
    t0_ns = ts_ns[t0_step] if t0_step < len(ts_ns) else ts_ns[-1]
    ego_xy = interp_positions(av_ts_ns, av_pos, np.array([t0_ns]))[0]
    ego_heading = compute_heading(
        interp_positions(av_ts_ns, av_pos, np.array([t0_ns - 0.1e9, t0_ns]))
    )

    v_ego = 0.0
    if t0_step in av_states and av_states[t0_step].velocity is not None:
        vx, vy = av_states[t0_step].velocity[0], av_states[t0_step].velocity[1]
        v_ego = math.sqrt(vx ** 2 + vy ** 2)

    ego_vel = np.array([v_ego, 0.0], dtype=np.float32)

    hist_times = t0_ns + np.arange(-T_HIST + 1, 1) * DT * 1e9
    fut_times = t0_ns + np.arange(1, T_FUT + 1) * DT * 1e9

    # Collect all valid pedestrians
    ped_histories = []
    ped_masks = []
    ped_futures = []

    for track in scenario.tracks:
        if track.object_type != ObjectType.PEDESTRIAN:
            continue

        states = track.object_states
        if len(states) < 2:
            continue

        track_timesteps = [s.timestep for s in states]
        track_ts_ns = np.array([ts_ns[t] for t in track_timesteps], dtype=np.float64)
        track_pos = np.array(
            [[s.position[0], s.position[1]] for s in states], dtype=np.float64,
        )

        t_min, t_max = track_ts_ns.min(), track_ts_ns.max()
        if t_min > hist_times[0] or t_max < fut_times[-1]:
            continue

        hist_xy = interp_positions(track_ts_ns, track_pos, hist_times)
        fut_xy = interp_positions(track_ts_ns, track_pos, fut_times)

        hist_ego = ego_frame_transform(hist_xy, ego_xy, ego_heading).astype(np.float32)
        fut_ego = ego_frame_transform(fut_xy, ego_xy, ego_heading).astype(np.float32)

        hist_vel = np.zeros_like(hist_ego)
        hist_vel[1:] = (hist_ego[1:] - hist_ego[:-1]) / DT

        history = np.concatenate([hist_ego, hist_vel], axis=-1)  # (20, 4)
        ped_histories.append(history)
        ped_masks.append(np.ones(T_HIST, dtype=np.uint8))
        ped_futures.append(fut_ego)

    if len(ped_histories) == 0:
        return []

    # Truncate to max_agents, pad if fewer
    n_peds = min(len(ped_histories), max_agents)

    histories = np.zeros((max_agents, T_HIST, 4), dtype=np.float32)
    history_masks = np.zeros((max_agents, T_HIST), dtype=np.uint8)
    futures = np.zeros((max_agents, T_FUT, 2), dtype=np.float32)
    agent_mask = np.zeros(max_agents, dtype=np.uint8)

    for i in range(n_peds):
        histories[i] = ped_histories[i]
        history_masks[i] = ped_masks[i]
        futures[i] = ped_futures[i]
        agent_mask[i] = 1

    return [{
        "histories": histories,
        "history_masks": history_masks,
        "futures": futures,
        "agent_mask": agent_mask,
        "ego_vel": ego_vel,
    }]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_shard(records, output_dir, shard_idx):
    if not records:
        return
    np.savez_compressed(
        os.path.join(output_dir, f"shard_{shard_idx:05d}.npz"),
        histories=np.stack([r["histories"] for r in records]),
        history_masks=np.stack([r["history_masks"] for r in records]),
        futures=np.stack([r["futures"] for r in records]),
        agent_masks=np.stack([r["agent_mask"] for r in records]),
        ego_vels=np.stack([r["ego_vel"] for r in records]),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess AV2 for joint multi-agent diffusion training"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-agents", type=int, default=16)
    args = parser.parse_args()

    input_path = pathlib.Path(args.input)

    for split in ["train", "val"]:
        split_dir = input_path / split
        if not split_dir.exists():
            print(f"[WARN] split directory not found: {split_dir}")
            continue

        out_dir = os.path.join(args.output, split)
        os.makedirs(out_dir, exist_ok=True)

        scenario_dirs = sorted([str(d) for d in split_dir.iterdir() if d.is_dir()])
        print(f"[{split}] found {len(scenario_dirs)} scenarios")

        all_records = []
        with Pool(
            args.workers,
            initializer=_init_worker,
            initargs=(args.max_agents,),
        ) as pool:
            for i, records in enumerate(
                pool.imap_unordered(process_scenario, scenario_dirs)
            ):
                all_records.extend(records)
                if (i + 1) % 500 == 0:
                    n_with_peds = len(all_records)
                    print(
                        f"  processed {i+1}/{len(scenario_dirs)} scenarios, "
                        f"{n_with_peds} scenes with pedestrians"
                    )

        print(f"[{split}] total scenes with pedestrians: {len(all_records)}")

        for shard_idx in range(0, max(len(all_records), 1), SHARD_SIZE):
            chunk = all_records[shard_idx : shard_idx + SHARD_SIZE]
            if chunk:
                save_shard(chunk, out_dir, shard_idx // SHARD_SIZE)

        n_shards = math.ceil(max(len(all_records), 1) / SHARD_SIZE)
        print(f"[{split}] saved {n_shards} shards to {out_dir}")


if __name__ == "__main__":
    main()
