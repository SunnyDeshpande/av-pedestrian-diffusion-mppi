"""MPPI controller — torch backend, adapted from the adapt repo.

Sources (adapt repo, branch feature/mppi):
    src/vehicle_drivers/gem_mppi_control/mppi_t.py  (simulation with
        pedestrian confidence-growth cost)
    src/vehicle_drivers/gem_mppi_control/mppi_ros.py (ROS2 node with
        velocity + stability costs)

This file keeps the `update(state, reference_path, obstacles)` API that
the rest of Adapt (adapt_mppi_node, adapt_mppi_generic_node, the ROS1
sim bridge in gem_mppi_sim) already calls, so the integration surface
stays identical — only the backend swaps from pure-NumPy to torch.

Cost structure (ported from the adapt repo):
  * goal position error  — drives the vehicle toward a look-ahead point
    on the reference path
  * velocity error       — tracks v_ref
  * stability penalty    — `|δ| · v` discourages high steering at high
    speed (smooth control under curvature)
  * obstacle cost        — per-pedestrian temporal confidence-growth:
    uncertainty ellipse around the predicted position at t_eff grows as
    confidence decays; a Gaussian repulsive term accumulates over the
    horizon.  Plus a hard clearance step + exponential falloff anchor
    for static detections where velocity / confidence is unknown.

Exposed for visualization (viz publishers in adapt_mppi_node):
  self.last_traj        (K, H, 4)  numpy
  self.last_weights     (K,)       numpy
  self.last_mean_traj   (H, 4)     numpy
"""
import math

import numpy as np
import torch

from pytorch_mppi import MPPI as _TorchMPPI

# Default GEM e4 wheelbase from the adapt repo. Kept in sync with the
# standalone sim + ROS node upstream.
L_DEFAULT = 2.57


class MPPI:
    def __init__(
        self,
        K=600,
        H=30,
        dt=0.1,
        sigma_steer=0.15,
        sigma_accel=0.5,
        lam=0.1,
        v_ref=2.0,
        w_lat=None,          # unused — kept for legacy constructor compat
        w_head=None,         # unused
        w_speed=None,        # unused
        w_obs=150.0,         # peak Gaussian repulsion per pedestrian
        w_ctrl=None,         # unused
        w_pos=15.0,          # goal position error weight
        w_vel=5.0,           # velocity tracking weight
        w_curv=2.0,          # stability: |δ|·v penalty
        w_obs_hard=250.0,    # step penalty inside clearance radius
        w_obs_soft=40.0,     # exponential falloff outside clearance
        clearance=3.0,
        delta_max=0.61,
        a_min=-1.0,
        a_max=2.0,
        wheelbase=L_DEFAULT,
        lookahead_m=8.0,     # look-ahead distance along ref path for goal
        seed=0,
        device=None,
    ):
        self.device = torch.device(device) if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.K = int(K)
        self.H = int(H)
        self.dt = float(dt)
        self.L = float(wheelbase)
        self.v_ref = float(v_ref)
        self.delta_max = float(delta_max)
        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self.lookahead_m = float(lookahead_m)

        self.w_pos = float(w_pos)
        self.w_vel = float(w_vel)
        self.w_curv = float(w_curv)
        self.w_obs = float(w_obs)
        self.w_obs_hard = float(w_obs_hard)
        self.w_obs_soft = float(w_obs_soft)
        self.clearance = float(clearance)
        self.lam = float(lam)

        # Per-tick inputs set in update() and read in running_cost.
        self._goal = torch.tensor([0.0, 0.0, 0.0, self.v_ref],
                                  dtype=torch.float32, device=self.device)
        self._ego = torch.zeros(4, dtype=torch.float32, device=self.device)
        self._peds = None  # (M, 5) tensor [x, y, vx, vy, conf] or None

        torch.manual_seed(seed)
        self.ctrl = _TorchMPPI(
            dynamics=self._dynamics,
            running_cost=self._running_cost,
            nx=4,
            num_samples=self.K,
            horizon=self.H,
            device=self.device,
            u_min=torch.tensor([self.a_min, -self.delta_max],
                               dtype=torch.float32, device=self.device),
            u_max=torch.tensor([self.a_max, self.delta_max],
                               dtype=torch.float32, device=self.device),
            noise_sigma=torch.tensor(
                [[sigma_accel ** 2, 0.0],
                 [0.0, sigma_steer ** 2]],
                dtype=torch.float32, device=self.device,
            ),
            lambda_=self.lam,
        )

        # Exposed for viz (populated after each update()).
        self.last_traj = None
        self.last_weights = None
        self.last_mean_traj = None
        self.last_costs = None

    # -------------------------------------------------------------- bicycle
    def _dynamics(self, state, u):
        """Kinematic bicycle (batched). state: (N,4)=[x,y,yaw,v], u: (N,2)=[a,δ]."""
        yaw, v = state[:, 2], state[:, 3]
        accel, delta = u[:, 0], u[:, 1]
        out = torch.zeros_like(state)
        out[:, 0] = state[:, 0] + v * torch.cos(yaw) * self.dt
        out[:, 1] = state[:, 1] + v * torch.sin(yaw) * self.dt
        out[:, 2] = state[:, 2] + (v / self.L) * torch.tan(delta) * self.dt
        out[:, 3] = state[:, 3] + accel * self.dt
        return out

    # ------------------------------------------------------------- cost fn
    def _running_cost(self, state, u):
        pos_err = torch.norm(state[:, :2] - self._goal[:2], dim=1)
        vel_err = torch.abs(state[:, 3] - self._goal[3])
        stability = torch.abs(u[:, 1]) * state[:, 3]
        cost = self.w_pos * pos_err + self.w_vel * vel_err + self.w_curv * stability

        if self._peds is not None:
            ped = self._peds                                       # (M, 5)
            px = ped[:, 0]; py = ped[:, 1]
            vx = ped[:, 2]; vy = ped[:, 3]
            conf0 = ped[:, 4]

            ego_xy = self._ego[:2]
            ego_v = self._ego[3] + 1e-3
            dist_from_ego = torch.norm(state[:, :2] - ego_xy, dim=1)  # (N,)
            t_eff = torch.clamp(dist_from_ego / ego_v, 0.0, self.H * self.dt)
            conf = torch.clamp(
                conf0[None, :] + (t_eff[:, None] / (self.H * self.dt))
                * (1.0 - conf0[None, :]),
                0.0, 1.0,
            )                                                       # (N, M)
            base_sigma = 1.0 + (1.0 - conf) * 3.0
            sigma_long = base_sigma * (1.0 + (1.0 - conf) * 2.0)
            sigma_lat = base_sigma
            mu_x = sigma_long * 1.5 * conf

            gt_x = px[None, :] + vx[None, :] * t_eff[:, None]
            gt_y = py[None, :] + vy[None, :] * t_eff[:, None]
            dx = state[:, 0:1] - gt_x
            dy = state[:, 1:2] - gt_y
            angle = torch.atan2(vy, vx)[None, :]
            dx_rot = dx * torch.cos(angle) + dy * torch.sin(angle)
            dy_rot = -dx * torch.sin(angle) + dy * torch.cos(angle)
            exponent = -((dx_rot - mu_x) ** 2 / (2 * sigma_long ** 2)
                         + dy_rot ** 2 / (2 * sigma_lat ** 2))
            soft_gauss = self.w_obs * torch.exp(exponent)

            static_dist = torch.sqrt(dx ** 2 + dy ** 2)
            hard_step = torch.where(
                static_dist < self.clearance,
                torch.full_like(static_dist, self.w_obs_hard),
                torch.zeros_like(static_dist),
            )
            soft_exp = self.w_obs_soft * torch.exp(-static_dist)

            cost = cost + soft_gauss.sum(dim=1) + hard_step.sum(dim=1) + soft_exp.sum(dim=1)

        return cost

    # -------------------------------------------------------------- update
    def update(self, state_np, reference_path, obstacles=None):
        """Drop-in replacement for the old numpy MPPI API.

        state_np: (4,) [x, y, yaw, v]  world-frame ENU
        reference_path: ReferencePath (exposes .xy)
        obstacles: (M, 2) world-frame xy, or (M, 5) [x, y, vx, vy, conf],
                   or None / empty.

        Returns (δ, a) as a length-2 numpy array — same order as the old
        numpy MPPI so adapt_mppi_node doesn't need to swap axes.
        """
        state = np.asarray(state_np, dtype=np.float32).reshape(4)
        self._ego = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        goal_xy, goal_yaw = self._lookahead_goal(reference_path, state[:2])
        self._goal = torch.tensor(
            [goal_xy[0], goal_xy[1], goal_yaw, self.v_ref],
            dtype=torch.float32, device=self.device,
        )

        if obstacles is None or len(obstacles) == 0:
            self._peds = None
        else:
            obs = np.asarray(obstacles, dtype=np.float32)
            if obs.ndim == 2 and obs.shape[1] == 2:
                peds = np.zeros((obs.shape[0], 5), dtype=np.float32)
                peds[:, 0:2] = obs
                peds[:, 4] = 1.0
            elif obs.ndim == 2 and obs.shape[1] == 5:
                peds = obs
            else:
                peds = None
            self._peds = (torch.as_tensor(peds, dtype=torch.float32,
                                          device=self.device)
                          if peds is not None else None)

        action = self.ctrl.command(self._ego)
        u = action.detach().cpu().numpy().astype(np.float64)
        # pytorch_mppi gives [a, δ]; legacy numpy API returned (δ, a).
        u_out = np.array([u[1], u[0]], dtype=np.float64)

        self._capture_viz_state()
        return u_out

    # --------------------------------------------------------- viz capture
    def _capture_viz_state(self):
        actions = getattr(self.ctrl, 'perturbed_action', None)
        if actions is None:
            return
        K_ = actions.shape[0]
        s = self._ego.unsqueeze(0).expand(K_, -1).contiguous().clone()
        traj = torch.zeros(K_, self.H, 4, dtype=torch.float32, device=self.device)
        for h in range(self.H):
            s = self._dynamics(s, actions[:, h, :])
            traj[:, h, :] = s

        costs = getattr(self.ctrl, 'cost_total', None)
        if costs is None:
            w = torch.ones(K_, device=self.device) / K_
        else:
            beta = costs.min()
            w = torch.exp(-(costs - beta) / self.ctrl.lambda_)
            w_sum = w.sum()
            w = (w / w_sum) if float(w_sum) > 1e-12 else torch.ones(K_, device=self.device) / K_
        mean_traj = torch.einsum('k,khd->hd', w, traj)

        self.last_traj = traj.detach().cpu().numpy()
        self.last_weights = w.detach().cpu().numpy()
        self.last_mean_traj = mean_traj.detach().cpu().numpy()
        self.last_costs = costs.detach().cpu().numpy() if costs is not None else None

    # ----------------------------------------------------------- helpers
    def _lookahead_goal(self, reference_path, ego_xy):
        """Pick a point on reference_path roughly `lookahead_m` ahead of
        the ego's nearest projection; heading = segment heading there."""
        xy = np.asarray(reference_path.xy, dtype=np.float32)
        if len(xy) < 2:
            return (float(ego_xy[0]), float(ego_xy[1])), 0.0
        d = np.linalg.norm(xy - ego_xy[None, :], axis=1)
        i0 = int(np.argmin(d))
        acc = 0.0
        i = i0
        while i + 1 < len(xy) and acc < self.lookahead_m:
            acc += float(np.linalg.norm(xy[i + 1] - xy[i]))
            i += 1
        tgt = xy[i]
        if i == 0:
            heading = math.atan2(float(xy[1, 1] - xy[0, 1]),
                                 float(xy[1, 0] - xy[0, 0]))
        else:
            heading = math.atan2(float(xy[i, 1] - xy[i - 1, 1]),
                                 float(xy[i, 0] - xy[i - 1, 0]))
        return (float(tgt[0]), float(tgt[1])), heading

    def effective_sample_count(self):
        if self.last_weights is None:
            return 0.0
        w = self.last_weights
        denom = float(np.sum(w * w))
        return (1.0 / denom) if denom > 0 else 0.0
