#!/usr/bin/env python3
"""Utility functions for the diffusion prediction package."""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def polar_to_cartesian(dist_m: float, dir_deg: float) -> tuple:
    """Convert polar (distance, direction) to Cartesian (x, y) in base_link.

    Convention (matches adapt fusion pipeline):
        0 deg   = right side of vehicle (-y)
        90 deg  = forward (+x)
        180 deg = left (+y)
        270 deg = backward (-x)

    Returns (x_forward, y_left).
    """
    theta = math.radians(dir_deg)
    x = dist_m * math.sin(theta)       # forward
    y = -dist_m * math.cos(theta)      # lateral (left positive)
    return (x, y)


def decode_fusion_msg(data: list) -> np.ndarray:
    """Decode a flat Int32MultiArray.data list into (M, 2) Cartesian array.

    Parameters
    ----------
    data : list[int]
        Flat list of [dist, dir, dist, dir, ...] pairs.

    Returns
    -------
    np.ndarray of shape (M, 2) with columns [x, y] in base_link metres.
    """
    if len(data) < 2 or len(data) % 2 != 0:
        return np.empty((0, 2), dtype=np.float64)

    M = len(data) // 2
    pts = np.empty((M, 2), dtype=np.float64)
    for i in range(M):
        dist = float(data[2 * i])
        deg = float(data[2 * i + 1])
        pts[i, 0], pts[i, 1] = polar_to_cartesian(dist, deg)
    return pts


# ---------------------------------------------------------------------------
# TTC (Time-to-Collision)
# ---------------------------------------------------------------------------

def compute_ttc(
    pred_traj: np.ndarray,
    vehicle_speed: float,
    dt: float = 0.25,
    collision_dist: float = 1.0,
) -> float:
    """Compute time-to-collision between ego and a predicted pedestrian trajectory.

    Parameters
    ----------
    pred_traj : np.ndarray, shape (N, 2)
        Predicted pedestrian positions [x, y] at future timesteps.
    vehicle_speed : float
        Current ego speed in m/s.
    dt : float
        Time step between predicted points (default 0.25 s).
    collision_dist : float
        Collision distance threshold in metres.

    Returns
    -------
    float
        Time to collision in seconds, or math.inf if no collision.
    """
    for i in range(len(pred_traj)):
        t = (i + 1) * dt
        ego_x = vehicle_speed * t
        dx = pred_traj[i, 0] - ego_x
        dy = pred_traj[i, 1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist <= collision_dist:
            return t
    return math.inf


# ---------------------------------------------------------------------------
# ROS message builders
# ---------------------------------------------------------------------------

def build_marker_msg(trajectory: np.ndarray, stamp, marker_id: int = 0):
    """Build a LINE_STRIP Marker for a predicted trajectory.

    Parameters
    ----------
    trajectory : np.ndarray, shape (N, 2)
        Predicted [x, y] points.
    stamp : builtin_interfaces.msg.Time
        Header timestamp.
    marker_id : int
        Marker id.

    Returns
    -------
    visualization_msgs.msg.Marker
    """
    from visualization_msgs.msg import Marker
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA

    m = Marker()
    m.header.frame_id = "base_link"
    m.header.stamp = stamp
    m.ns = "person_prediction"
    m.id = marker_id
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = 0.15
    m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
    m.lifetime.sec = 0
    m.lifetime.nanosec = 500_000_000  # 0.5 s

    m.points = [
        Point(x=float(trajectory[i, 0]), y=float(trajectory[i, 1]), z=0.0)
        for i in range(len(trajectory))
    ]
    return m


def build_twist_msg(x: float, y: float):
    """Build a Twist message with pedestrian position in linear.x/y."""
    from geometry_msgs.msg import Twist

    msg = Twist()
    msg.linear.x = float(x)
    msg.linear.y = float(y)
    msg.linear.z = 0.0
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = 0.0
    return msg


def build_predictions_tensor(trajectories: np.ndarray):
    """Build a Float32MultiArray for the (M, H, 2) predictions tensor.

    Parameters
    ----------
    trajectories : np.ndarray, shape (M, H, 2)
        Best-mode trajectories per pedestrian, in base_link Cartesian.

    Returns
    -------
    std_msgs.msg.Float32MultiArray
    """
    from std_msgs.msg import Float32MultiArray, MultiArrayDimension

    M, H, _ = trajectories.shape

    msg = Float32MultiArray()
    msg.layout.dim = [
        MultiArrayDimension(label="M", size=M, stride=M * H * 2),
        MultiArrayDimension(label="H", size=H, stride=H * 2),
        MultiArrayDimension(label="xy", size=2, stride=2),
    ]
    msg.data = trajectories.astype(np.float32).flatten().tolist()
    return msg


# ---------------------------------------------------------------------------
# Frame transforms
# ---------------------------------------------------------------------------

def rotation_matrix_2d(theta: float) -> np.ndarray:
    """Return a 2x2 rotation matrix for angle theta (radians)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)
