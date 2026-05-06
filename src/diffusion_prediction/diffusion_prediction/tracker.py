#!/usr/bin/env python3
"""Pedestrian tracker with greedy association and smoothing pipeline.

Ported from the adapt pedestrian behaviour predictor. Provides per-track
history suitable for feeding into the diffusion denoiser.
"""

import numpy as np
from collections import deque


class Track:
    """Single pedestrian track."""

    __slots__ = (
        "track_id", "x", "y", "z",
        "path_3d", "times", "smoothed_path", "predicted_path",
        "missed", "_vel_x", "_vel_y",
    )

    def __init__(self, track_id: int, x: float, y: float, z: float, t: float):
        self.track_id = track_id
        self.x = x
        self.y = y
        self.z = z
        self.path_3d = deque(maxlen=100)
        self.path_3d.append((x, y, z))
        self.times = deque(maxlen=100)
        self.times.append(t)
        self.smoothed_path = [(x, y, z)]
        self.predicted_path = []
        self.missed = 0
        self._vel_x = 0.0
        self._vel_y = 0.0


class Tracker:
    """Greedy Euclidean tracker with smoothing pipeline.

    Parameters
    ----------
    max_dist : float
        Maximum association distance in metres.
    max_missing : int
        Number of unmatched frames before a track is deleted.
    smooth_alpha : float
        EMA blending factor for position updates.
    """

    def __init__(
        self,
        max_dist: float = 2.0,
        max_missing: int = 10,
        smooth_alpha: float = 0.6,
    ):
        self.max_dist = max_dist
        self.max_dist_sq = max_dist ** 2
        self.max_missing = max_missing
        self.smooth_alpha = smooth_alpha
        self.tracks: dict[int, Track] = {}
        self._next_id = 0

    def update(self, detections: np.ndarray, t_now: float) -> list[int]:
        """Update tracks with new detections.

        Parameters
        ----------
        detections : np.ndarray, shape (M, 2) — [x, y] in base_link.
        t_now      : float, current timestamp in seconds.

        Returns
        -------
        deleted_ids : list of track ids that were removed.
        """
        used = set()
        M = len(detections)

        for det_idx in range(M):
            X = float(detections[det_idx, 0])
            Y = float(detections[det_idx, 1])
            Z = 0.0

            best_id = None
            best_d2 = self.max_dist_sq

            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                dx = X - tr.x
                dy = Y - tr.y
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_id = tid

            if best_id is None:
                # New track
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = Track(tid, X, Y, Z, t_now)
                used.add(tid)
            else:
                # Update existing track with EMA
                tr = self.tracks[best_id]
                a = self.smooth_alpha
                tr.x = a * X + (1.0 - a) * tr.x
                tr.y = a * Y + (1.0 - a) * tr.y
                tr.z = a * Z + (1.0 - a) * tr.z

                tr.path_3d.append((tr.x, tr.y, tr.z))
                tr.times.append(t_now)

                smoothed, predicted, vel = self._smooth_and_predict(tr)
                tr.smoothed_path = smoothed
                tr.predicted_path = predicted
                tr._vel_x = vel[0]
                tr._vel_y = vel[1]
                tr.missed = 0
                used.add(best_id)

        # Increment missed counter; delete stale tracks
        deleted = []
        for tid in list(self.tracks.keys()):
            if tid not in used:
                self.tracks[tid].missed += 1
                if self.tracks[tid].missed > self.max_missing:
                    deleted.append(tid)
                    del self.tracks[tid]

        return deleted

    def get_history(
        self,
        track_id: int,
        T_hist: int = 20,
        dt: float = 0.25,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract history tensor for a track, suitable for the denoiser.

        Returns
        -------
        history : np.ndarray (T_hist, 4)  [x, y, vx, vy], pedestrian-centric:
                                          positions are normalized so the most
                                          recent position is the origin (matches
                                          the training-side inference spec).
                                          Velocities stay in real m/s.
        mask    : np.ndarray (T_hist,)     1.0 = valid, 0.0 = padding
        """
        tr = self.tracks[track_id]
        path = list(tr.path_3d)
        times = list(tr.times)
        n = len(path)

        history = np.zeros((T_hist, 4), dtype=np.float32)
        mask = np.zeros(T_hist, dtype=np.float32)

        # Fill from the end (most recent = last slot)
        fill = min(n, T_hist)
        for i in range(fill):
            src = n - fill + i
            dst = T_hist - fill + i
            history[dst, 0] = path[src][0]  # x
            history[dst, 1] = path[src][1]  # y
            mask[dst] = 1.0

        # Compute velocities via first-difference (real m/s)
        for i in range(1, T_hist):
            if mask[i] > 0 and mask[i - 1] > 0:
                src_curr = n - fill + (i - (T_hist - fill))
                src_prev = src_curr - 1
                if 0 <= src_prev < len(times) and 0 <= src_curr < len(times):
                    dt_actual = times[src_curr] - times[src_prev]
                    if dt_actual > 0:
                        history[i, 2] = (history[i, 0] - history[i - 1, 0]) / dt_actual
                        history[i, 3] = (history[i, 1] - history[i - 1, 1]) / dt_actual

        # Pedestrian-centric: shift positions so the most recent point is at the
        # origin. The model's training distribution requires history[-1] = (0, 0).
        if mask[-1] > 0:
            ref_x = history[-1, 0]
            ref_y = history[-1, 1]
            for i in range(T_hist):
                if mask[i] > 0:
                    history[i, 0] -= ref_x
                    history[i, 1] -= ref_y

        return history, mask

    def _smooth_and_predict(self, tr: Track) -> tuple:
        """Smoothing pipeline + constant-velocity prediction (fallback).

        Returns (smoothed_path, predicted_path, avg_velocity).
        """
        path = list(tr.path_3d)
        times = list(tr.times)

        if len(path) < 3:
            return path, [], np.zeros(3)

        path_arr = np.array(path)

        # 1. Spike removal (median filter, threshold 0.5 m)
        filtered = [path_arr[0]]
        for i in range(1, len(path_arr) - 1):
            prev_pt = path_arr[i - 1]
            curr_pt = path_arr[i]
            next_pt = path_arr[i + 1]
            expected = (prev_pt + next_pt) / 2.0
            deviation = np.linalg.norm(curr_pt - expected)
            if deviation > 0.5:
                filtered.append(np.median([prev_pt, curr_pt, next_pt], axis=0))
            else:
                filtered.append(curr_pt)
        filtered.append(path_arr[-1])
        filtered_arr = np.array(filtered)

        # 2. Moving average (window 7)
        window = min(7, len(filtered_arr))
        smoothed = []
        for i in range(len(filtered_arr)):
            start = max(0, i - window // 2)
            end = min(len(filtered_arr), i + window // 2 + 1)
            smoothed.append(np.mean(filtered_arr[start:end], axis=0))

        # 3. Velocity estimation (last 15 points, 2-sigma outlier rejection)
        N = len(path)
        start_idx = max(1, N - 15)
        velocities = []
        for i in range(start_idx, N):
            p_prev = np.array(path[i - 1])
            p_curr = np.array(path[i])
            dt = times[i] - times[i - 1]
            if dt <= 0.0:
                continue
            velocities.append((p_curr - p_prev) / dt)

        if len(velocities) == 0:
            return [tuple(s) for s in smoothed], [], np.zeros(3)

        vel_arr = np.array(velocities)
        mean_vel = np.mean(vel_arr, axis=0)
        std_vel = np.std(vel_arr, axis=0) + 1e-6

        filt_vel = [v for v in vel_arr if np.all(np.abs(v - mean_vel) < 2 * std_vel)]
        avg_vel = np.mean(filt_vel, axis=0) if filt_vel else mean_vel

        # 4. Z-flatten
        avg_vel[2] = 0.0

        # Clamp speed to 3.0 m/s
        speed = np.linalg.norm(avg_vel)
        if speed > 3.0:
            avg_vel *= 3.0 / speed

        # 5. Constant-velocity prediction (20 points, 5 s)
        last_pos = np.array(smoothed[-1])
        last_pos[2] = 0.0
        predicted = []
        for i in range(1, 21):
            t = i * 0.25
            pred = last_pos + avg_vel * t
            pred[2] = 0.0
            dist = np.linalg.norm(pred[:2] - last_pos[:2])
            if dist > 15.0:
                direction = (pred - last_pos) / (dist + 1e-9)
                pred = last_pos + direction * 15.0
                pred[2] = 0.0
                predicted.append(tuple(pred))
                break
            predicted.append(tuple(pred))

        return [tuple(s) for s in smoothed], predicted, avg_vel
