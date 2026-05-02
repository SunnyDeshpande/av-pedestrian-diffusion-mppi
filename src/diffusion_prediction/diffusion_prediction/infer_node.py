#!/usr/bin/env python3
"""ROS 2 inference node — drop-in replacement for the constant-velocity predictor.

Subscribes to the same topics, publishes the same topics, and adds
/pedestrian_predictions_tensor for time-indexed MPPI obstacle cost.
"""

import math
import os

import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Int32MultiArray, Float64, Float32MultiArray
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker
from pacmod2_msgs.msg import VehicleSpeedRpt

from diffusion_prediction.tracker import Tracker
from diffusion_prediction.utils import (
    decode_fusion_msg,
    compute_ttc,
    build_marker_msg,
    build_twist_msg,
    build_predictions_tensor,
    smooth_single_trajectory,
    _check_physics,
    _extrapolate_from_history,
)


class DiffusionPredictorNode(Node):
    """Diffusion-based pedestrian trajectory prediction node."""

    def __init__(self):
        super().__init__("diffusion_predictor_node")

        # --------------- Parameters ---------------
        self.declare_parameter("weights", "")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("K", 20)
        self.declare_parameter("ddim_steps", 10)
        self.declare_parameter("min_history_count", 5)
        self.declare_parameter("prediction_time", 5.0)
        self.declare_parameter("prediction_points", 20)
        self.declare_parameter("collision_distance_threshold", 1.0)
        self.declare_parameter("latency_warn_ms", 80.0)
        self.declare_parameter("prediction_mode", "joint")  # "single" or "joint"
        self.declare_parameter("max_agents", 16)

        self.weights_path = self.get_parameter("weights").value
        self.device_str = self.get_parameter("device").value
        self.K = self.get_parameter("K").value
        self.ddim_steps = self.get_parameter("ddim_steps").value
        self.min_hist = self.get_parameter("min_history_count").value
        self.pred_time = self.get_parameter("prediction_time").value
        self.pred_pts = self.get_parameter("prediction_points").value
        self.collision_thresh = self.get_parameter("collision_distance_threshold").value
        self.latency_warn = self.get_parameter("latency_warn_ms").value
        self.prediction_mode = self.get_parameter("prediction_mode").value
        self.max_agents = self.get_parameter("max_agents").value

        # --------------- Subscribers ---------------
        self.sub_ped = self.create_subscription(
            Int32MultiArray, "fusion_pedestrian_position",
            self.pedestrian_cb, 10,
        )
        self.sub_vehicle = self.create_subscription(
            VehicleSpeedRpt, "vehicle_rpt",
            self.vehicle_cb, 10,
        )

        # --------------- Publishers ---------------
        self.pub_prediction = self.create_publisher(Marker, "person_prediction", 10)
        self.pub_motion = self.create_publisher(Twist, "pedestrian_motion", 10)
        self.pub_ttc = self.create_publisher(Float64, "pedestrian_ttc", 10)
        self.pub_tensor = self.create_publisher(
            Float32MultiArray, "pedestrian_predictions_tensor", 10,
        )

        # --------------- State ---------------
        self.tracker = Tracker(max_dist=2.0, max_missing=10, smooth_alpha=0.6)
        self.vehicle_speed = 0.0
        self.vehicle_speed_valid = False

        # Sticky mode selection state: track_id -> (prev_idx, consecutive_count)
        self._sticky_state: dict[int, tuple[int, int]] = {}

        # Temporal EMA state: track_id -> previous best trajectory (20, 2)
        self._prev_trajs: dict[int, np.ndarray] = {}
        self._temporal_alpha = 0.3  # blend: 0=all previous, 1=all current

        # --------------- Model ---------------
        self.model = None
        self.schedule = None
        self._load_model()

        self.get_logger().info("Diffusion predictor node ready.")

    def _load_model(self):
        """Load the diffusion model and schedule."""
        try:
            import torch
            from diffusion_prediction.ddpm import CosineSchedule

            device = torch.device(self.device_str if torch.cuda.is_available() else "cpu")
            self._torch_device = device

            if self.prediction_mode == "joint":
                from diffusion_prediction.model_joint import JointTrajectoryDenoiser
                self.model = JointTrajectoryDenoiser(
                    d=256, max_agents=self.max_agents,
                    nhead=8, num_enc_layers=6, num_dec_layers=4,
                    num_interaction_layers=3, dim_ff=512,
                ).to(device)
                self.get_logger().info(
                    f"Using joint multi-agent model (max_agents={self.max_agents})"
                )
            else:
                from diffusion_prediction.model import TrajectoryDenoiser
                self.model = TrajectoryDenoiser(
                    d=256, nhead=8, num_enc_layers=6,
                    num_dec_layers=4, dim_ff=512,
                ).to(device)
                self.get_logger().info("Using single-agent model")

            self.schedule = CosineSchedule(T=100).to(device)

            if self.weights_path and os.path.exists(self.weights_path):
                state = torch.load(self.weights_path, map_location=device, weights_only=True)
                if isinstance(state, dict) and "model_state" in state:
                    self.model.load_state_dict(state["model_state"])
                else:
                    self.model.load_state_dict(state)
                self.get_logger().info(f"Loaded weights from {self.weights_path}")
            else:
                self.get_logger().warn(
                    "No weights loaded — running with random weights (prediction will be noise). "
                    "Set the 'weights' parameter to a checkpoint path."
                )

            self.model.eval()
            self._torch = torch

        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            self.model = None

    def vehicle_cb(self, msg: VehicleSpeedRpt):
        if msg.vehicle_speed_valid:
            self.vehicle_speed = float(msg.vehicle_speed)
            self.vehicle_speed_valid = True
        else:
            self.vehicle_speed_valid = False

    def pedestrian_cb(self, msg: Int32MultiArray):
        """Main callback: track, predict, publish."""
        import time as _time

        t0 = _time.perf_counter()
        now = self.get_clock().now()
        t_now = float(now.nanoseconds) * 1e-9
        stamp = now.to_msg()

        # Decode polar -> Cartesian
        detections = decode_fusion_msg(list(msg.data))
        if detections.shape[0] == 0:
            return

        # Update tracker
        deleted = self.tracker.update(detections, t_now)

        # Clean up sticky state for deleted tracks
        for tid in deleted:
            self._sticky_state.pop(tid, None)

        # Collect tracks with enough history
        active_ids = []
        histories = []
        masks = []
        ego_vels = []

        for tid, tr in self.tracker.tracks.items():
            hist, mask = self.tracker.get_history(tid, T_hist=20)
            presence = mask.sum()
            if presence < self.min_hist:
                continue
            active_ids.append(tid)
            histories.append(hist)
            masks.append(mask)
            ego_vels.append(
                np.array([self.vehicle_speed, 0.0], dtype=np.float32)
            )

        if not active_ids or self.model is None:
            # Fallback: publish using tracker's constant-velocity prediction
            self._publish_fallback(stamp)
            return

        # Run diffusion inference
        torch = self._torch
        device = self._torch_device
        M = len(active_ids)

        if self.prediction_mode == "joint":
            from diffusion_prediction.ddpm import ddim_sample_loop_joint

            M_pad = self.max_agents
            # Pad to (1, M_pad, 20, 4) scene tensor
            hist_pad = np.zeros((M_pad, 20, 4), dtype=np.float32)
            mask_pad = np.zeros((M_pad, 20), dtype=np.float32)
            agent_mask = np.zeros(M_pad, dtype=np.float32)
            for i in range(M):
                hist_pad[i] = histories[i]
                mask_pad[i] = masks[i]
                agent_mask[i] = 1.0

            hist_t = torch.from_numpy(hist_pad).unsqueeze(0).to(device)   # (1, M_pad, 20, 4)
            mask_t = torch.from_numpy(mask_pad).unsqueeze(0).to(device)   # (1, M_pad, 20)
            amask_t = torch.from_numpy(agent_mask).unsqueeze(0).to(device) # (1, M_pad)
            ego_t = torch.from_numpy(ego_vels[0]).unsqueeze(0).to(device)  # (1, 2)

            # joint_futures: (1, K, M_pad, 20, 2)
            joint_futures = ddim_sample_loop_joint(
                self.model, self.schedule, hist_t, mask_t, amask_t, ego_t,
                K=self.K,
            )
            # Extract real agents: (M, K, 20, 2)
            futures = joint_futures[0, :, :M, :, :].permute(1, 0, 2, 3)
        else:
            from diffusion_prediction.ddpm import ddim_sample_loop

            hist_t = torch.from_numpy(np.stack(histories)).to(device)
            mask_t = torch.from_numpy(np.stack(masks)).to(device)
            ego_t = torch.from_numpy(np.stack(ego_vels)).to(device)

            # futures: (M, K, 20, 2)
            futures = ddim_sample_loop(
                self.model, self.schedule, hist_t, mask_t, ego_t, K=self.K,
            )

        # --- Physics-based filtering + best-mode selection per track ---
        best_trajs = np.zeros((M, 20, 2), dtype=np.float32)
        for m_idx in range(M):
            tid = active_ids[m_idx]
            samples_np = futures[m_idx].cpu().numpy()  # (K, 20, 2)

            # Physics check: reject implausible samples before mode selection
            valid = _check_physics(samples_np, dt=0.25, max_speed=3.5, max_accel=4.0)

            # Curvature-aware center: blend median with history extrapolation
            valid_idx = np.where(valid)[0]
            if len(valid_idx) >= 3:
                median_center = np.median(samples_np[valid_idx], axis=0)
            else:
                median_center = np.median(samples_np, axis=0)
                valid_idx = np.arange(len(samples_np))

            # Extrapolate from history curvature
            extrap = _extrapolate_from_history(histories[m_idx], T_fut=20, dt=0.25)
            center = 0.6 * extrap + 0.4 * median_center

            # Closest-to-center among valid samples
            cost = ((samples_np[valid_idx] - center[None]) ** 2).sum(axis=(1, 2))
            cand_local = cost.argmin()
            cand_idx = valid_idx[cand_local]
            cand_cost = cost[cand_local]

            # Sticky temporal: avoid jumping between modes
            prev = self._sticky_state.get(tid)
            if prev is not None:
                prev_idx, consec = prev
                if prev_idx in valid_idx:
                    prev_local = np.where(valid_idx == prev_idx)[0]
                    if len(prev_local) > 0:
                        prev_cost = cost[prev_local[0]]
                        if prev_cost > 1.5 * cand_cost:
                            consec += 1
                            if consec >= 3:
                                chosen_idx = cand_idx
                                self._sticky_state[tid] = (chosen_idx, 0)
                            else:
                                chosen_idx = prev_idx
                                self._sticky_state[tid] = (prev_idx, consec)
                        else:
                            chosen_idx = prev_idx
                            self._sticky_state[tid] = (prev_idx, 0)
                    else:
                        chosen_idx = cand_idx
                        self._sticky_state[tid] = (chosen_idx, 0)
                else:
                    # Previous choice is no longer physics-valid
                    chosen_idx = cand_idx
                    self._sticky_state[tid] = (chosen_idx, 0)
            else:
                chosen_idx = cand_idx
                self._sticky_state[tid] = (chosen_idx, 0)

            best_trajs[m_idx] = samples_np[chosen_idx]

        # --- Smooth predictions ---
        # 1) Spline smoothing per trajectory
        for m_idx in range(M):
            best_trajs[m_idx] = smooth_single_trajectory(best_trajs[m_idx], s_factor=50.0)

        # 2) Temporal EMA: blend with previous frame for stability
        for m_idx, tid in enumerate(active_ids):
            if tid in self._prev_trajs:
                best_trajs[m_idx] = (
                    self._temporal_alpha * best_trajs[m_idx]
                    + (1 - self._temporal_alpha) * self._prev_trajs[tid]
                )
            self._prev_trajs[tid] = best_trajs[m_idx].copy()

        # Clean up temporal state for deleted tracks
        active_set = set(active_ids)
        for tid in list(self._prev_trajs.keys()):
            if tid not in active_set and tid not in self.tracker.tracks:
                del self._prev_trajs[tid]

        # --- Select primary pedestrian ---
        # Transform back: add current position offset (undo the origin centering)
        for m_idx, tid in enumerate(active_ids):
            tr = self.tracker.tracks[tid]
            best_trajs[m_idx, :, 0] += tr.x
            best_trajs[m_idx, :, 1] += tr.y

        # Compute TTC for each pedestrian
        ttc_values = {}
        for m_idx, tid in enumerate(active_ids):
            ttc_values[tid] = compute_ttc(
                best_trajs[m_idx], self.vehicle_speed,
                dt=0.25, collision_dist=self.collision_thresh,
            )

        # Primary: smallest TTC, else closest
        primary_idx = None
        primary_ttc = math.inf

        finite_ttc = {tid: t for tid, t in ttc_values.items() if t < math.inf}
        if finite_ttc:
            best_tid = min(finite_ttc, key=finite_ttc.get)
            primary_idx = active_ids.index(best_tid)
            primary_ttc = finite_ttc[best_tid]
        else:
            min_dist = math.inf
            for m_idx, tid in enumerate(active_ids):
                tr = self.tracker.tracks[tid]
                d = math.sqrt(tr.x ** 2 + tr.y ** 2)
                if d < min_dist:
                    min_dist = d
                    primary_idx = m_idx

        # --- Publish ---
        if primary_idx is not None:
            # Marker
            marker = build_marker_msg(best_trajs[primary_idx], stamp)
            self.pub_prediction.publish(marker)

            # Twist
            tr = self.tracker.tracks[active_ids[primary_idx]]
            twist = build_twist_msg(tr.x, tr.y)
            self.pub_motion.publish(twist)

            # TTC
            ttc_msg = Float64()
            ttc_msg.data = float(primary_ttc)
            self.pub_ttc.publish(ttc_msg)

        # Tensor (all pedestrians)
        tensor_msg = build_predictions_tensor(best_trajs)
        self.pub_tensor.publish(tensor_msg)

        # Latency check
        elapsed_ms = (_time.perf_counter() - t0) * 1000.0
        if elapsed_ms > self.latency_warn:
            self.get_logger().warn(
                f"Inference cycle took {elapsed_ms:.1f} ms (> {self.latency_warn} ms)"
            )

    def _publish_fallback(self, stamp):
        """Fallback: use tracker's constant-velocity prediction when model is unavailable."""
        if not self.tracker.tracks:
            return

        # Find primary pedestrian (closest)
        min_dist = math.inf
        primary_tr = None
        for tid, tr in self.tracker.tracks.items():
            d = math.sqrt(tr.x ** 2 + tr.y ** 2)
            if d < min_dist:
                min_dist = d
                primary_tr = tr

        if primary_tr is None:
            return

        # Publish motion
        twist = build_twist_msg(primary_tr.x, primary_tr.y)
        self.pub_motion.publish(twist)

        # Publish prediction marker from tracker's CV prediction
        if primary_tr.predicted_path:
            pred_arr = np.array(primary_tr.predicted_path)[:, :2]  # (N, 2)
            marker = build_marker_msg(pred_arr, stamp)
            self.pub_prediction.publish(marker)

            ttc = compute_ttc(
                pred_arr, self.vehicle_speed,
                dt=0.25, collision_dist=self.collision_thresh,
            )
            ttc_msg = Float64()
            ttc_msg.data = float(ttc)
            self.pub_ttc.publish(ttc_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DiffusionPredictorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down diffusion predictor node")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
