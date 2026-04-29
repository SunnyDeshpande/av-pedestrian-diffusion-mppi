#!/usr/bin/env python3
"""Extract pedestrian trajectory windows from recorded GEM rosbags.

Replays /fusion_pedestrian_position offline, runs the tracker, and emits
(history, future, ego_vel) windows in the same .npz format as the AV2
preprocessor.

Usage:
    python scripts/extract_gem_windows.py \
        --bags data/gem_bags/ \
        --output data/gem_processed \
        --holdout-ratio 0.2
"""

import argparse
import math
import os
import pathlib

import numpy as np

# rosbag2 reader
try:
    from rosbags.rosbag2 import Reader
    from rosbags.serde import deserialize_cdr
except ImportError:
    raise ImportError(
        "rosbags package required: pip install rosbags\n"
        "Used for offline rosbag replay without needing a full ROS 2 install."
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DT = 0.25
T_HIST = 20
T_FUT = 20
SHARD_SIZE = 4096


# ---------------------------------------------------------------------------
# Minimal tracker (same logic as diffusion_prediction.tracker)
# ---------------------------------------------------------------------------

def polar_to_cartesian(dist, deg):
    theta = math.radians(deg)
    x = dist * math.sin(theta)
    y = -dist * math.cos(theta)
    return x, y


class SimpleTrack:
    def __init__(self, tid, x, y, t):
        self.tid = tid
        self.x = x
        self.y = y
        self.path = [(x, y)]
        self.times = [t]
        self.missed = 0

    def update(self, x, y, t, alpha=0.6):
        self.x = alpha * x + (1 - alpha) * self.x
        self.y = alpha * y + (1 - alpha) * self.y
        self.path.append((self.x, self.y))
        self.times.append(t)
        if len(self.path) > 200:
            self.path.pop(0)
            self.times.pop(0)
        self.missed = 0


def greedy_associate(tracks, detections, t_now, max_dist=2.0, alpha=0.6):
    """Greedy 2-m association; returns (next_id_offset, list_of_deleted_tids)."""
    used = set()
    for det_x, det_y in detections:
        best_tid = None
        best_d2 = max_dist ** 2
        for tid, tr in tracks.items():
            if tid in used:
                continue
            d2 = (det_x - tr.x) ** 2 + (det_y - tr.y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_tid = tid
        if best_tid is not None:
            tracks[best_tid].update(det_x, det_y, t_now, alpha)
            used.add(best_tid)
        # New tracks are handled by caller

    # Handle unmatched detections (create new tracks externally)
    unmatched_dets = []
    for i, (dx, dy) in enumerate(detections):
        matched = False
        for tid in used:
            tr = tracks[tid]
            # Check if this detection was matched to this track
            # (simplified: we already matched above)
            pass
        # Actually, let's redo: track which dets got matched
    # Re-approach: mark matched dets
    matched_det_indices = set()
    for det_idx, (det_x, det_y) in enumerate(detections):
        for tid in used:
            tr = tracks[tid]
            # Already updated, check proximity
            if (det_x - tr.x) ** 2 + (det_y - tr.y) ** 2 < 0.01:
                matched_det_indices.add(det_idx)
                break

    deleted = []
    for tid in list(tracks.keys()):
        if tid not in used:
            tracks[tid].missed += 1
            if tracks[tid].missed > 10:
                deleted.append(tid)
                del tracks[tid]

    return [i for i in range(len(detections)) if i not in matched_det_indices], deleted


# ---------------------------------------------------------------------------
# Extract windows from a single bag
# ---------------------------------------------------------------------------

def extract_from_bag(bag_path: str) -> list:
    """Extract trajectory windows from one rosbag.

    Returns list of dicts with keys: history, history_mask, future, ego_vel.
    """
    records = []

    with Reader(bag_path) as reader:
        # Collect all messages by topic
        fusion_msgs = []
        speed_msgs = []

        connections = {c.topic: c for c in reader.connections}

        for connection, timestamp, rawdata in reader.messages():
            topic = connection.topic
            t_sec = timestamp / 1e9

            if topic == "/fusion_pedestrian_position":
                msg = deserialize_cdr(rawdata, connection.msgtype)
                fusion_msgs.append((t_sec, list(msg.data)))
            elif topic in ("/vehicle_rpt", "/pacmod/vehicle_speed_rpt"):
                msg = deserialize_cdr(rawdata, connection.msgtype)
                speed_msgs.append((t_sec, float(msg.vehicle_speed)))

    if not fusion_msgs:
        return records

    # Build speed lookup (nearest-neighbor in time)
    speed_times = np.array([s[0] for s in speed_msgs]) if speed_msgs else np.array([0.0])
    speed_vals = np.array([s[1] for s in speed_msgs]) if speed_msgs else np.array([0.0])

    def get_speed(t):
        if len(speed_times) == 0:
            return 0.0
        idx = np.searchsorted(speed_times, t)
        idx = min(idx, len(speed_vals) - 1)
        return float(speed_vals[idx])

    # Run tracker over all fusion messages
    tracks = {}
    next_id = 0

    for t_sec, data in fusion_msgs:
        if len(data) < 2 or len(data) % 2 != 0:
            continue

        dets = []
        for i in range(0, len(data), 2):
            x, y = polar_to_cartesian(float(data[i]), float(data[i + 1]))
            dets.append((x, y))

        # Simple greedy association
        used_tids = set()
        for dx, dy in dets:
            best_tid = None
            best_d2 = 4.0  # 2.0^2
            for tid, tr in tracks.items():
                if tid in used_tids:
                    continue
                d2 = (dx - tr.x) ** 2 + (dy - tr.y) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_tid = tid

            if best_tid is not None:
                tracks[best_tid].update(dx, dy, t_sec)
                used_tids.add(best_tid)
            else:
                tracks[next_id] = SimpleTrack(next_id, dx, dy, t_sec)
                used_tids.add(next_id)
                next_id += 1

        for tid in list(tracks.keys()):
            if tid not in used_tids:
                tracks[tid].missed += 1
                if tracks[tid].missed > 10:
                    del tracks[tid]

    # Extract windows from completed tracks
    for tid, tr in tracks.items():
        path = np.array(tr.path)   # (N, 2)
        times = np.array(tr.times) # (N,)
        N = len(path)

        # Need at least T_HIST + T_FUT observations
        if N < T_HIST + T_FUT:
            continue

        # Slide a window
        for start in range(0, N - T_HIST - T_FUT + 1, T_FUT // 2):
            hist_slice = path[start : start + T_HIST]
            fut_slice = path[start + T_HIST : start + T_HIST + T_FUT]
            t0 = times[start + T_HIST - 1]

            # Check time span roughly matches expected duration
            hist_dt = times[start + T_HIST - 1] - times[start]
            if hist_dt < 2.0:  # less than 2 seconds of history = too compressed
                continue

            # Normalize: center at last history position
            ref = hist_slice[-1].copy()
            hist_xy = (hist_slice - ref).astype(np.float32)
            fut_xy = (fut_slice - ref).astype(np.float32)

            # Velocities
            hist_vel = np.zeros_like(hist_xy)
            for i in range(1, T_HIST):
                dt = times[start + i] - times[start + i - 1]
                if dt > 0:
                    hist_vel[i] = (hist_xy[i] - hist_xy[i - 1]) / dt

            history = np.concatenate([hist_xy, hist_vel], axis=-1)  # (20, 4)
            history_mask = np.ones(T_HIST, dtype=np.uint8)

            ego_vel = np.array([get_speed(t0), 0.0], dtype=np.float32)

            records.append({
                "history": history,
                "history_mask": history_mask,
                "future": fut_xy,
                "ego_vel": ego_vel,
            })

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_shard(records, output_dir, shard_idx):
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
    parser = argparse.ArgumentParser(description="Extract GEM rosbag windows")
    parser.add_argument("--bags", type=str, required=True, help="Directory containing rosbags")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--holdout-ratio", type=float, default=0.2, help="Fraction for validation")
    args = parser.parse_args()

    bags_dir = pathlib.Path(args.bags)
    bag_paths = sorted([
        str(d) for d in bags_dir.iterdir()
        if d.is_dir() and (d / "metadata.yaml").exists()
    ])

    if not bag_paths:
        # Try .db3 files directly
        bag_paths = sorted([str(f) for f in bags_dir.glob("*.db3")])

    print(f"Found {len(bag_paths)} bags")

    # Split bags into train/val by holdout ratio
    n_val = max(1, int(len(bag_paths) * args.holdout_ratio))
    val_bags = set(bag_paths[-n_val:])
    train_bags = [b for b in bag_paths if b not in val_bags]

    for split, split_bags in [("train", train_bags), ("val", list(val_bags))]:
        out_dir = os.path.join(args.output, split)
        os.makedirs(out_dir, exist_ok=True)

        all_records = []
        for bag_path in split_bags:
            print(f"  [{split}] processing {bag_path}")
            records = extract_from_bag(bag_path)
            all_records.extend(records)
            print(f"    -> {len(records)} windows")

        print(f"[{split}] total windows: {len(all_records)}")

        for shard_idx in range(0, max(1, len(all_records)), SHARD_SIZE):
            chunk = all_records[shard_idx : shard_idx + SHARD_SIZE]
            if chunk:
                save_shard(chunk, out_dir, shard_idx // SHARD_SIZE)

        n_shards = math.ceil(max(len(all_records), 1) / SHARD_SIZE)
        print(f"[{split}] saved {n_shards} shards to {out_dir}")


if __name__ == "__main__":
    main()
