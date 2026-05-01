#!/usr/bin/env python3
"""Utility functions for the diffusion prediction package."""

import math
import numpy as np
from scipy.interpolate import UnivariateSpline


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


# ---------------------------------------------------------------------------
# Trajectory smoothing & physics-based filtering
# ---------------------------------------------------------------------------

def filter_and_smooth_trajectories(
    preds: np.ndarray,
    dt: float = 0.25,
    max_speed: float = 3.5,
    max_accel: float = 4.0,
    s_factor: float = 50.0,
) -> np.ndarray:
    """Physics-based outlier rejection + spline smoothing for diffusion samples.

    Pipeline:
    1. Per-sample physics check — reject trajectories with implausible
       speed (> max_speed m/s at any step) or acceleration (> max_accel m/s²).
    2. MAD-based statistical outlier rejection on the surviving samples.
    3. Replace rejected samples with median trajectory + small noise.
    4. Spline-smooth each trajectory.

    Parameters
    ----------
    preds : (K, T, 2) or (K, M, T, 2) numpy array
        Raw diffusion output (ego-normalized coordinates).
    dt : float
        Timestep between prediction points (default 0.25s = 4 Hz).
    max_speed : float
        Maximum plausible pedestrian speed in m/s. 3.5 m/s ≈ fast jog.
    max_accel : float
        Maximum plausible pedestrian acceleration in m/s².
    s_factor : float
        Spline smoothing factor (higher = smoother).

    Returns
    -------
    smoothed : same shape as preds
    """
    original_shape = preds.shape
    K = preds.shape[0]
    T = preds.shape[-2]
    rest_shape = preds.shape[1:]  # (T, 2) or (M, T, 2)

    # Flatten each sample to work on individual trajectories
    # For (K, M, T, 2), we check physics per-agent then combine
    if preds.ndim == 4:
        # Multi-agent: (K, M, T, 2)
        M = preds.shape[1]
        physics_valid = np.ones(K, dtype=bool)
        for m in range(M):
            agent_valid = _check_physics(preds[:, m, :, :], dt, max_speed, max_accel)
            physics_valid &= agent_valid
    else:
        # Single-agent: (K, T, 2)
        physics_valid = _check_physics(preds, dt, max_speed, max_accel)

    # --- Step 2: MAD-based statistical rejection on physics-valid samples ---
    flat_per_sample = preds.reshape(K, -1)

    # Compute median only from physics-valid samples if enough exist
    valid_indices = np.where(physics_valid)[0]
    if len(valid_indices) >= 3:
        median_sample = np.median(flat_per_sample[valid_indices], axis=0)
    else:
        median_sample = np.median(flat_per_sample, axis=0)

    dists = np.linalg.norm(flat_per_sample - median_sample[None], axis=-1)
    med_dist = np.median(dists[valid_indices]) if len(valid_indices) >= 3 else np.median(dists)
    mad = np.median(np.abs(dists - med_dist))
    threshold = med_dist + 2.5 * max(mad, 0.1)
    stat_valid = dists < threshold

    # Combined mask: must pass both physics and stats
    valid = physics_valid & stat_valid

    # --- Step 3: Replace invalid samples ---
    median_traj = median_sample.reshape(rest_shape)
    preds_clean = preds.copy()
    n_valid = valid.sum()

    if n_valid == 0:
        # All samples failed — just use median everywhere with noise
        for k in range(K):
            noise = np.random.randn(*rest_shape).astype(np.float32) * 0.02
            preds_clean[k] = median_traj + noise
    else:
        # Replace invalid samples with random valid sample + small noise
        valid_idx = np.where(valid)[0]
        for k in range(K):
            if not valid[k]:
                donor = preds[np.random.choice(valid_idx)]
                noise = np.random.randn(*rest_shape).astype(np.float32) * 0.03
                preds_clean[k] = donor + noise

    # --- Step 4: Spline smoothing ---
    t_axis = np.arange(T)
    flat = preds_clean.reshape(-1, T, 2)
    out = flat.copy()
    for i in range(len(flat)):
        for dim in range(2):
            try:
                spl = UnivariateSpline(t_axis, flat[i, :, dim], s=s_factor)
                out[i, :, dim] = spl(t_axis)
            except Exception:
                pass

    return out.reshape(original_shape)


def _check_physics(
    trajs: np.ndarray,
    dt: float,
    max_speed: float,
    max_accel: float,
) -> np.ndarray:
    """Check per-sample physics plausibility.

    Parameters
    ----------
    trajs : (K, T, 2)
    dt, max_speed, max_accel : physics bounds

    Returns
    -------
    valid : (K,) boolean mask
    """
    K, T, _ = trajs.shape

    # Velocities: (K, T-1, 2)
    displacements = np.diff(trajs, axis=1)
    velocities = displacements / dt
    speeds = np.linalg.norm(velocities, axis=-1)  # (K, T-1)

    # Accelerations: (K, T-2, 2)
    accels = np.diff(velocities, axis=1) / dt
    accel_mags = np.linalg.norm(accels, axis=-1)  # (K, T-2)

    # Check constraints
    max_speed_per_sample = speeds.max(axis=1)      # (K,)
    max_accel_per_sample = accel_mags.max(axis=1)   # (K,)

    valid = (max_speed_per_sample <= max_speed) & (max_accel_per_sample <= max_accel)
    return valid


def smooth_single_trajectory(
    traj: np.ndarray,
    s_factor: float = 50.0,
) -> np.ndarray:
    """Spline-smooth a single (T, 2) trajectory. Used for the infer_node best-mode."""
    T = traj.shape[0]
    t_axis = np.arange(T)
    out = traj.copy()
    for dim in range(2):
        try:
            spl = UnivariateSpline(t_axis, traj[:, dim], s=s_factor)
            out[:, dim] = spl(t_axis)
        except Exception:
            pass
    return out
