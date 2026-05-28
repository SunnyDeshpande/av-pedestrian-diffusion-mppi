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

def _extrapolate_from_history(hist: np.ndarray, T_fut: int = 20, dt: float = 0.25) -> np.ndarray:
    """Extrapolate a constant-curvature trajectory from observed history.

    Uses the last few history positions to estimate velocity and turning rate,
    then extrapolates forward. This serves as a prior for mode selection -
    during arcs, we prefer samples that continue the curve rather than
    samples closest to the (potentially misleading) sample mean.

    Parameters
    ----------
    hist : (T_hist, 4) array with [x, y, vx, vy] - pedestrian-centric, last
           position at origin (matches the training-side inference spec).
    T_fut : number of future steps to extrapolate
    dt : timestep

    Returns
    -------
    extrapolated : (T_fut, 2) future positions in pedestrian-centric frame
                   (consistent with the diffusion model's output, which also
                   starts implicitly near the origin)
    """
    # Use last velocity for speed/heading
    vx, vy = hist[-1, 2], hist[-1, 3]
    speed = math.sqrt(vx * vx + vy * vy)
    if speed < 0.05:
        # Essentially stationary - predict staying put at the origin
        return np.zeros((T_fut, 2), dtype=np.float32)

    heading = math.atan2(vy, vx)

    # Estimate turning rate from last few velocity vectors
    omega = 0.0
    n_vel = 0
    for i in range(max(0, len(hist) - 6), len(hist) - 1):
        v0x, v0y = hist[i, 2], hist[i, 3]
        v1x, v1y = hist[i + 1, 2], hist[i + 1, 3]
        s0 = math.sqrt(v0x * v0x + v0y * v0y)
        s1 = math.sqrt(v1x * v1x + v1y * v1y)
        if s0 > 0.05 and s1 > 0.05:
            h0 = math.atan2(v0y, v0x)
            h1 = math.atan2(v1y, v1x)
            dh = h1 - h0
            # Wrap to [-pi, pi]
            dh = (dh + math.pi) % (2 * math.pi) - math.pi
            omega += dh / dt
            n_vel += 1

    if n_vel > 0:
        omega /= n_vel

    # Clamp turning rate to plausible range (max ~90 deg/s)
    omega = max(-1.57, min(1.57, omega))

    # Extrapolate with constant speed + constant turning rate from the
    # origin (pedestrian-centric frame).
    extrap = np.zeros((T_fut, 2), dtype=np.float32)
    x, y = 0.0, 0.0
    h = heading
    for t in range(T_fut):
        h += omega * dt
        x += speed * math.cos(h) * dt
        y += speed * math.sin(h) * dt
        extrap[t, 0] = x
        extrap[t, 1] = y

    return extrap


def filter_and_smooth_trajectories(
    preds: np.ndarray,
    dt: float = 0.25,
    max_speed: float = 3.5,
    max_accel: float = 4.0,
    s_factor: float = 50.0,
    hist: np.ndarray = None,
) -> np.ndarray:
    """Physics-based outlier rejection + curvature-aware filtering + spline smoothing.

    Pipeline:
    1. Per-sample physics check - reject implausible speed/acceleration.
    2. If history is provided, compute a curvature-extrapolated reference
       trajectory and use it (blended with sample median) as the center
       for statistical outlier rejection. This prevents the arc problem
       where the sample mean drifts away from the true motion direction.
    3. Replace rejected samples with valid donors + noise.
    4. Spline-smooth each trajectory.

    Parameters
    ----------
    preds : (K, T, 2) or (K, M, T, 2) numpy array
    dt : float
    max_speed, max_accel : physics bounds
    s_factor : spline smoothing factor
    hist : optional (T_hist, 4) history for curvature-aware mode selection.
           Only used for single-agent (K, T, 2) predictions.

    Returns
    -------
    smoothed : same shape as preds
    """
    original_shape = preds.shape
    K = preds.shape[0]
    T = preds.shape[-2]
    rest_shape = preds.shape[1:]

    # --- Step 1: Physics rejection ---
    if preds.ndim == 4:
        M = preds.shape[1]
        physics_valid = np.ones(K, dtype=bool)
        for m in range(M):
            agent_valid = _check_physics(preds[:, m, :, :], dt, max_speed, max_accel)
            physics_valid &= agent_valid
    else:
        physics_valid = _check_physics(preds, dt, max_speed, max_accel)

    # --- Step 2: Curvature-aware + MAD rejection ---
    flat_per_sample = preds.reshape(K, -1)

    valid_indices = np.where(physics_valid)[0]
    if len(valid_indices) >= 3:
        median_sample = np.median(flat_per_sample[valid_indices], axis=0)
    else:
        median_sample = np.median(flat_per_sample, axis=0)

    # Blend median with curvature extrapolation for better arc handling
    if hist is not None and preds.ndim == 3:
        extrap = _extrapolate_from_history(hist, T_fut=T, dt=dt)  # (T, 2)
        extrap_flat = extrap.reshape(-1).astype(np.float32)
        # Blend: 60% curvature extrapolation, 40% sample median
        # This anchors the reference to the expected motion direction
        center = 0.6 * extrap_flat + 0.4 * median_sample
    else:
        center = median_sample

    dists = np.linalg.norm(flat_per_sample - center[None], axis=-1)
    med_dist = np.median(dists[valid_indices]) if len(valid_indices) >= 3 else np.median(dists)
    mad = np.median(np.abs(dists - med_dist))
    threshold = med_dist + 2.5 * max(mad, 0.1)
    stat_valid = dists < threshold

    valid = physics_valid & stat_valid

    # --- Step 3: Replace invalid samples ---
    center_traj = center.reshape(rest_shape)
    preds_clean = preds.copy()
    n_valid = valid.sum()

    if n_valid == 0:
        for k in range(K):
            noise = np.random.randn(*rest_shape).astype(np.float32) * 0.02
            preds_clean[k] = center_traj + noise
    else:
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
