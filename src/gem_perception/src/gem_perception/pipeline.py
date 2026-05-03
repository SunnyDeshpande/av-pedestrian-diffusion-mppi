"""End-to-end detection → LiDAR fusion → goal pose.

Framework-agnostic: ROS1 and ROS2 wrappers supply inputs (image, cloud, TFs, K)
and call :func:`run_pipeline`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .geometry import (
    ClusterResult,
    choose_cluster,
    dbscan,
    pixel_to_ray,
    project_to_image,
    statistical_outlier_filter,
    transform_points,
    z_axis_filter,
)
from .yolo_detector import Detection2D


@dataclass
class PipelineParams:
    z_min_base: float = 0.15
    z_max_base: float = 5.0
    sor_k: int = 8
    sor_std_mul: float = 2.0
    dbscan_eps: float = 0.4
    dbscan_min_samples: int = 3
    min_cluster_points: int = 3
    estimated_goal_distance: float = 15.0


@dataclass
class PerceptionResult:
    detection: Detection2D
    pixel_centroid: Tuple[float, float]
    cluster: Optional[ClusterResult]
    goal_base: np.ndarray
    is_estimated: bool
    cluster_cloud_base: Optional[np.ndarray]


def bbox_center(bbox_xyxy: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)


def _lidar_mask_to_base(
    points_lidar: np.ndarray,
    K: np.ndarray,
    T_cam_from_lidar: np.ndarray,
    T_base_from_lidar: np.ndarray,
    image_mask: np.ndarray,
    D: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return base_link-frame subset of points_lidar whose projection falls in image_mask."""
    if points_lidar.size == 0:
        return np.empty((0, 3), dtype=points_lidar.dtype)
    pts_cam = transform_points(points_lidar, T_cam_from_lidar)
    uv, idx = project_to_image(pts_cam, K, D)
    H, W = image_mask.shape
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    within = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    inside = np.zeros(uv.shape[0], dtype=bool)
    inside[within] = image_mask[v[within], u[within]]
    full_idx = idx[inside]
    pts_base = transform_points(points_lidar[full_idx], T_base_from_lidar)
    return pts_base


def run_pipeline(
    detection: Detection2D,
    points_lidar: np.ndarray,
    K: np.ndarray,
    T_cam_from_lidar: np.ndarray,
    T_base_from_lidar: np.ndarray,
    T_base_from_cam: np.ndarray,
    params: PipelineParams,
    D: Optional[np.ndarray] = None,
) -> PerceptionResult:
    pixel_centroid = bbox_center(detection.bbox_xyxy)

    # 1) Mask-clip + 2) z-axis ground filter + outlier filter, all in base_link
    pts_base_masked = _lidar_mask_to_base(
        points_lidar, K, T_cam_from_lidar, T_base_from_lidar, detection.mask, D
    )
    pts_base_filt = z_axis_filter(pts_base_masked, params.z_min_base, params.z_max_base)
    pts_base_filt = statistical_outlier_filter(
        pts_base_filt, k=params.sor_k, std_mul=params.sor_std_mul
    )

    # 3) DBSCAN in base_link; pick cluster whose projection is nearest to the 2D target pixel
    cluster: Optional[ClusterResult] = None
    if pts_base_filt.shape[0] >= params.min_cluster_points:
        labels = dbscan(pts_base_filt, eps=params.dbscan_eps,
                        min_samples=params.dbscan_min_samples)
        uniq = [l for l in np.unique(labels) if l >= 0]
        if uniq:
            T_cam_from_base = np.linalg.inv(T_base_from_cam)
            clusters_cam = []
            clusters_base = []
            for lid in uniq:
                cb = pts_base_filt[labels == lid]
                if cb.shape[0] < params.min_cluster_points:
                    continue
                clusters_cam.append(transform_points(cb, T_cam_from_base))
                clusters_base.append(cb)
            idx_best = choose_cluster(clusters_cam, pixel_centroid, K, D)
            if idx_best >= 0:
                cluster = ClusterResult.from_points(clusters_base[idx_best])

    # 4) Goal in base_link (real or estimated)
    if cluster is not None:
        goal_base = cluster.centroid_base
        is_estimated = False
        cluster_cloud_base = cluster.points_base
    else:
        ray_cam = pixel_to_ray(pixel_centroid, K, D)
        pt_cam = ray_cam * params.estimated_goal_distance
        goal_base = transform_points(pt_cam.reshape(1, 3), T_base_from_cam)[0]
        is_estimated = True
        cluster_cloud_base = None

    return PerceptionResult(
        detection=detection,
        pixel_centroid=pixel_centroid,
        cluster=cluster,
        goal_base=goal_base,
        is_estimated=is_estimated,
        cluster_cloud_base=cluster_cloud_base,
    )
