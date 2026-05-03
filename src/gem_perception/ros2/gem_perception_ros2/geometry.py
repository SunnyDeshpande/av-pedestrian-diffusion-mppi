"""LiDAR/camera geometry helpers used by ROS1 and ROS2 perception nodes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to an (N, 3) point array."""
    if points.size == 0:
        return points
    homog = np.hstack([points, np.ones((points.shape[0], 1), dtype=points.dtype)])
    return (T @ homog.T).T[:, :3]


def project_to_image(points_cam: np.ndarray, K: np.ndarray,
                     D: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Project (N,3) camera-frame points to (u,v) + keep mask (Z>0 only).

    If ``D`` is given (length 4 or 5, plumb_bob: ``[k1,k2,p1,p2[,k3]]`` and
    has any non-zero element, points are projected through OpenCV's
    ``cv2.projectPoints`` so radial/tangential distortion is applied. If
    ``D`` is None or all-zero the cheap pinhole path is used.

    Returns ``uv (M,2)`` float array and the index array into the original points.
    """
    z = points_cam[:, 2]
    valid = z > 0.05
    p = points_cam[valid]
    if p.shape[0] == 0:
        return np.empty((0, 2), dtype=float), np.where(valid)[0]
    if D is not None and np.any(np.abs(D) > 1e-6):
        import cv2
        # cv2.projectPoints expects (N,1,3) object points + zero rvec/tvec
        pts = p.reshape(-1, 1, 3).astype(np.float64)
        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.zeros(3, dtype=np.float64)
        uv, _ = cv2.projectPoints(pts, rvec, tvec,
                                  K.astype(np.float64),
                                  np.asarray(D, dtype=np.float64).reshape(-1))
        return uv.reshape(-1, 2), np.where(valid)[0]
    uv = (K @ p.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv, np.where(valid)[0]


def clip_by_mask(points: np.ndarray, uv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep points whose projected pixels (uv) fall inside `mask` (HxW bool)."""
    if uv.size == 0:
        return np.empty((0, 3), dtype=points.dtype)
    H, W = mask.shape
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    inside = np.zeros(uv.shape[0], dtype=bool)
    inside[in_img] = mask[v[in_img], u[in_img]]
    return points[inside]


def z_axis_filter(points_base: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    """Keep base_link points with z_min < z < z_max (simple ground filter)."""
    if points_base.size == 0:
        return points_base
    z = points_base[:, 2]
    keep = (z > z_min) & (z < z_max)
    return points_base[keep]


def statistical_outlier_filter(points: np.ndarray, k: int = 8, std_mul: float = 2.0) -> np.ndarray:
    """Very small SOR: drops points whose k-NN mean distance exceeds mean + std_mul * std."""
    n = points.shape[0]
    if n < max(k + 1, 10):
        return points
    # Brute-force k-NN (small n after mask clip, so it's fine)
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    d.sort(axis=1)
    mean_k = d[:, 1:k + 1].mean(axis=1)
    thr = mean_k.mean() + std_mul * mean_k.std()
    keep = mean_k < thr
    return points[keep]


def dbscan(points: np.ndarray, eps: float = 0.3, min_samples: int = 5) -> np.ndarray:
    """Return cluster labels (-1 = noise). Small pure-numpy DBSCAN."""
    n = points.shape[0]
    labels = np.full(n, -1, dtype=int)
    if n == 0:
        return labels
    visited = np.zeros(n, dtype=bool)
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    neighbors = [np.where(d[i] < eps)[0] for i in range(n)]
    cid = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        if len(neighbors[i]) < min_samples:
            continue
        labels[i] = cid
        seeds = list(neighbors[i])
        while seeds:
            j = seeds.pop()
            if not visited[j]:
                visited[j] = True
                if len(neighbors[j]) >= min_samples:
                    for k in neighbors[j]:
                        if k not in seeds and labels[k] == -1:
                            seeds.append(k)
            if labels[j] == -1:
                labels[j] = cid
        cid += 1
    return labels


def choose_cluster(
    clusters_cam: list[np.ndarray],
    pixel_centroid: Tuple[float, float],
    K: np.ndarray,
    D: Optional[np.ndarray] = None,
) -> int:
    """Index of the cluster whose 3D centroid projects closest to the 2D target pixel.

    When ``D`` is given and non-zero the centroids are projected through the
    distortion model so the comparison stays in image-pixel space.
    """
    best_i, best_d = -1, float("inf")
    for i, c in enumerate(clusters_cam):
        if c.size == 0:
            continue
        cent = c.mean(axis=0)
        if cent[2] <= 0:
            continue
        uv2, _ = project_to_image(cent.reshape(1, 3), K, D)
        if uv2.shape[0] == 0:
            continue
        d = float(np.hypot(uv2[0, 0] - pixel_centroid[0], uv2[0, 1] - pixel_centroid[1]))
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def pixel_to_ray(pixel: Tuple[float, float], K: np.ndarray,
                 D: Optional[np.ndarray] = None) -> np.ndarray:
    """Unit ray in camera optical frame (Z-forward) through the given pixel.

    When ``D`` is given and non-zero, ``cv2.undistortPoints`` is used so the
    pixel is first un-distorted before being lifted to a 3D ray.
    """
    u, v = pixel
    if D is not None and np.any(np.abs(D) > 1e-6):
        import cv2
        pts = np.array([[[u, v]]], dtype=np.float64)
        und = cv2.undistortPoints(pts, K.astype(np.float64),
                                  np.asarray(D, dtype=np.float64).reshape(-1))
        x, y = und[0, 0]
        r = np.array([x, y, 1.0], dtype=float)
        return r / np.linalg.norm(r)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    r = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=float)
    return r / np.linalg.norm(r)


@dataclass
class ClusterResult:
    points_base: np.ndarray          # (M,3) in base_link
    centroid_base: np.ndarray        # (3,)
    bbox_min_base: np.ndarray        # (3,) axis-aligned
    bbox_max_base: np.ndarray        # (3,)

    @classmethod
    def from_points(cls, pts: np.ndarray) -> "ClusterResult":
        return cls(
            points_base=pts,
            centroid_base=pts.mean(axis=0),
            bbox_min_base=pts.min(axis=0),
            bbox_max_base=pts.max(axis=0),
        )
