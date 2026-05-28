"""ROS-agnostic helpers used by both ROS1 and ROS2 wrappers.

This module deliberately avoids importing rospy / rclpy - it only uses numpy
and standard library so both framework-specific wrappers can reuse it.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def K_from_camera_info(K_list) -> np.ndarray:
    """Convert CameraInfo.K (length-9 row-major) to (3,3) numpy array."""
    return np.asarray(K_list, dtype=float).reshape(3, 3)


def D_from_camera_info(D_list) -> Optional[np.ndarray]:
    """Convert CameraInfo.D to a 1-D numpy array, or ``None`` if not populated.

    plumb_bob layout: ``[k1, k2, p1, p2, k3]`` (length 5). Returns ``None``
    when D is empty or all-zero so callers can skip the distortion code path.
    """
    if D_list is None:
        return None
    arr = np.asarray(D_list, dtype=float).reshape(-1)
    if arr.size == 0 or not np.any(np.abs(arr) > 1e-6):
        return None
    return arr


def transform_to_matrix(translation: Tuple[float, float, float],
                        rotation_xyzw: Tuple[float, float, float, float]) -> np.ndarray:
    """(x,y,z) + (qx,qy,qz,qw) → 4x4 homogeneous transform."""
    qx, qy, qz, qw = rotation_xyzw
    x, y, z = translation
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    R = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
    ], dtype=float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def matrix_inverse(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def draw_detection_overlay(
    image_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    mask: Optional[np.ndarray],
    pixel_centroid: Tuple[float, float],
    is_estimated: bool,
    prompt: str,
    score: float,
) -> np.ndarray:
    out = image_bgr.copy()
    color_real = (0, 255, 0)
    color_est = (0, 255, 255)
    color = color_est if is_estimated else color_real

    if mask is not None:
        overlay = out.copy()
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array(color)).astype(np.uint8)
        out = overlay

    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    cx, cy = int(round(pixel_centroid[0])), int(round(pixel_centroid[1]))
    cv2.circle(out, (cx, cy), 6, (0, 0, 255), -1)

    label = f"{prompt} ({score:.2f}){' [est]' if is_estimated else ''}"
    cv2.putText(out, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA)
    return out


def draw_lidar_projection(
    image_bgr: np.ndarray,
    uv: np.ndarray,
    depths: np.ndarray,
    max_depth: float = 30.0,
) -> np.ndarray:
    out = image_bgr.copy()
    if uv.size == 0:
        return out
    H, W = out.shape[:2]
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    within = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depths > 0.1)
    u, v, d = u[within], v[within], depths[within]
    # Colorize: near = red, far = green/blue
    d_norm = np.clip(d / max_depth, 0, 1)
    colors = np.zeros((len(d_norm), 3), dtype=np.uint8)
    colors[:, 2] = ((1.0 - d_norm) * 255).astype(np.uint8)   # B (BGR → "far = red" visually? swap to taste)
    colors[:, 1] = (d_norm * 255).astype(np.uint8)
    for i in range(len(u)):
        cv2.circle(out, (u[i], v[i]), 2, tuple(int(c) for c in colors[i]), -1)
    return out


class GoalHold:
    """Keeps the last goal active for a short timeout after detection is lost."""

    def __init__(self, hold_seconds: float = 2.0):
        self.hold_seconds = hold_seconds
        self._last_time: Optional[float] = None
        self._last_goal: Optional[Tuple[np.ndarray, bool]] = None

    def update(self, now: float, goal_base: Optional[np.ndarray], is_estimated: bool) -> Optional[Tuple[np.ndarray, bool]]:
        if goal_base is not None:
            self._last_time = now
            self._last_goal = (goal_base.copy(), is_estimated)
            return self._last_goal
        if self._last_goal is None or self._last_time is None:
            return None
        if now - self._last_time > self.hold_seconds:
            self._last_goal = None
            return None
        return self._last_goal
