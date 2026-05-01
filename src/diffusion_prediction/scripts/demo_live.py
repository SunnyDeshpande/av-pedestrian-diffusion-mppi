#!/usr/bin/env python3
"""Live animated demo of diffusion pedestrian trajectory prediction.

Simulates pedestrians walking along trajectories and shows the diffusion
model predicting their future positions in real-time. No ROS or data needed.

Usage:
    # Single-agent (one pedestrian at a time)
    python scripts/demo_live.py \
        --model single \
        --weights ../../models/diffusion/av2_pretrain_v1/ema_best.pt \
        --output figures/demo_live_single.mp4

    # Joint multi-agent (multiple pedestrians together)
    python scripts/demo_live.py \
        --model joint \
        --weights ../../models/diffusion/av2_joint_v1/ema_best.pt \
        --output figures/demo_live_joint.mp4

    # Display live window instead of saving (needs X11):
    python scripts/demo_live.py --model single --weights ... --live
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diffusion_prediction.model import TrajectoryDenoiser
from diffusion_prediction.model_joint import JointTrajectoryDenoiser
from diffusion_prediction.ddpm import CosineSchedule, ddim_sample_loop, ddim_sample_loop_joint


# ---- Trajectory generators ----

def _trajectory_straight(t):
    return np.stack([1.4 * t, np.zeros_like(t)], axis=-1)

def _trajectory_curve(t):
    R, omega = 6.0, 0.25
    theta = omega * t
    return np.stack([R * np.sin(theta), R * (1 - np.cos(theta))], axis=-1)

def _trajectory_crossing_a(t):
    return np.stack([1.3 * t, 0.2 * t], axis=-1)

def _trajectory_crossing_b(t):
    return np.stack([0.3 * t + 8.0, 1.3 * t - 3.0], axis=-1)

def _trajectory_stopping(t):
    v0, decel = 1.5, 0.25
    vt = np.clip(v0 - decel * t, 0, None)
    x = np.cumsum(vt) * 0.25
    return np.stack([x, 0.3 * np.sin(0.4 * t)], axis=-1)

def _trajectory_swerve(t):
    return np.stack([1.2 * t, 2.0 * np.sin(0.35 * t)], axis=-1)

def _trajectory_parallel_a(t):
    return np.stack([1.1 * t, np.full_like(t, -1.5)], axis=-1)

def _trajectory_parallel_b(t):
    return np.stack([1.3 * t, np.full_like(t, 0.0)], axis=-1)

def _trajectory_parallel_c(t):
    return np.stack([1.0 * t, np.full_like(t, 1.5)], axis=-1)


SINGLE_SCENARIOS = [
    {"name": "Pedestrian walking straight",  "gen": _trajectory_straight,    "total_s": 15.0},
    {"name": "Pedestrian turning left",      "gen": _trajectory_curve,       "total_s": 15.0},
    {"name": "Pedestrian stopping",          "gen": _trajectory_stopping,    "total_s": 12.0},
    {"name": "Pedestrian swerving",          "gen": _trajectory_swerve,      "total_s": 15.0},
]

JOINT_SCENARIOS = [
    {
        "name": "Two pedestrians crossing",
        "gens": [_trajectory_crossing_a, _trajectory_crossing_b],
        "total_s": 14.0,
    },
    {
        "name": "Three pedestrians walking in parallel",
        "gens": [_trajectory_parallel_a, _trajectory_parallel_b, _trajectory_parallel_c],
        "total_s": 14.0,
    },
]


def build_history(positions, dt=0.25):
    """Convert (T, 2) absolute positions to (T, 4) ego-normalized history.

    Normalizes so the last position is at origin. Returns [x, y, vx, vy].
    """
    pos = positions.copy()
    origin = pos[-1].copy()
    pos -= origin

    vel = np.zeros_like(pos)
    if len(pos) > 1:
        vel[1:] = (pos[1:] - pos[:-1]) / dt
        vel[0] = vel[1]

    hist = np.zeros((len(pos), 4), dtype=np.float32)
    hist[:, :2] = pos
    hist[:, 2:] = vel
    return hist, origin


def parse_args():
    p = argparse.ArgumentParser(description="Live animated diffusion prediction demo")
    p.add_argument("--model", choices=["single", "joint"], required=True)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--K", type=int, default=20, help="Number of trajectory samples")
    p.add_argument("--output", type=str, default=None,
                   help="Save animation to file (.mp4 or .gif)")
    p.add_argument("--live", action="store_true",
                   help="Show live window instead of saving")
    p.add_argument("--fps", type=int, default=4, help="Animation FPS (matches 4 Hz sensor)")
    p.add_argument("--seed", type=int, default=42)
    # Architecture
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dim-ff", type=int, default=256)
    p.add_argument("--max-agents", type=int, default=16)
    p.add_argument("--num-enc-layers", type=int, default=4)
    p.add_argument("--num-interaction-layers", type=int, default=2)
    return p.parse_args()


def load_model(args, device):
    if args.model == "single":
        model = TrajectoryDenoiser(
            d=args.d_model, nhead=args.nhead,
            num_layers=args.num_layers, dim_ff=args.dim_ff,
        ).to(device)
    else:
        model = JointTrajectoryDenoiser(
            d=args.d_model, max_agents=args.max_agents,
            nhead=args.nhead, num_enc_layers=args.num_enc_layers,
            num_interaction_layers=args.num_interaction_layers,
            dim_ff=args.dim_ff,
        ).to(device)

    state = torch.load(args.weights, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.eval()
    print(f"Loaded {args.model} model from {args.weights}")
    return model


@torch.no_grad()
def predict_single(model, schedule, hist_np, device, K=20):
    """Run single-agent DDIM prediction.

    Parameters
    ----------
    hist_np : (T, 4) numpy array, ego-normalized history
    Returns: (K, 20, 2) predicted futures in ego-normalized frame
    """
    T = hist_np.shape[0]
    # Pad to 20 if shorter
    if T < 20:
        padded = np.zeros((20, 4), dtype=np.float32)
        padded[20 - T:] = hist_np
        mask = np.zeros(20, dtype=np.float32)
        mask[20 - T:] = 1.0
    else:
        padded = hist_np[-20:]
        mask = np.ones(20, dtype=np.float32)

    hist_t = torch.from_numpy(padded).unsqueeze(0).to(device)   # (1, 20, 4)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)     # (1, 20)
    ego_t = torch.zeros(1, 2, device=device)
    ego_t[0, 0] = 2.0

    futures = ddim_sample_loop(model, schedule, hist_t, mask_t, ego_t, K=K)
    return futures[0].cpu().numpy()  # (K, 20, 2)


@torch.no_grad()
def predict_joint(model, schedule, histories_np, masks_np, max_agents, device, K=20):
    """Run joint multi-agent DDIM prediction.

    Parameters
    ----------
    histories_np : list of (T_i, 4) arrays per agent
    masks_np     : list of (T_i,) arrays per agent
    Returns: (K, M_real, 20, 2) predicted futures
    """
    M_real = len(histories_np)
    hist_pad = np.zeros((max_agents, 20, 4), dtype=np.float32)
    mask_pad = np.zeros((max_agents, 20), dtype=np.float32)
    agent_mask = np.zeros(max_agents, dtype=np.float32)

    for m in range(M_real):
        T = histories_np[m].shape[0]
        if T < 20:
            hist_pad[m, 20 - T:] = histories_np[m]
            mask_pad[m, 20 - T:] = masks_np[m]
        else:
            hist_pad[m] = histories_np[m][-20:]
            mask_pad[m] = masks_np[m][-20:]
        agent_mask[m] = 1.0

    hist_t = torch.from_numpy(hist_pad).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask_pad).unsqueeze(0).to(device)
    amask_t = torch.from_numpy(agent_mask).unsqueeze(0).to(device)
    ego_t = torch.zeros(1, 2, device=device)
    ego_t[0, 0] = 2.0

    futures = ddim_sample_loop_joint(model, schedule, hist_t, mask_t, amask_t, ego_t, K=K)
    return futures[0, :, :M_real].cpu().numpy()  # (K, M_real, 20, 2)


def smooth_predictions(preds, window=5):
    """Smooth predicted trajectories with a moving average.

    The raw diffusion output can be jagged (noisy per-step positions).
    A light uniform filter produces visually clean, physically plausible
    trajectories without changing the overall shape.

    Parameters
    ----------
    preds : (..., T, 2) numpy array — last two dims are (timesteps, xy)
    window : int — smoothing window size

    Returns
    -------
    smoothed : same shape as preds
    """
    smoothed = np.copy(preds)
    # Timestep is the second-to-last axis: (K, T, 2) or (K, M, T, 2)
    t_axis = preds.ndim - 2
    smoothed[..., 0] = uniform_filter1d(preds[..., 0], size=window, axis=t_axis, mode="nearest")
    smoothed[..., 1] = uniform_filter1d(preds[..., 1], size=window, axis=t_axis, mode="nearest")
    return smoothed


def main():
    args = parse_args()

    if args.live:
        import matplotlib
        matplotlib.use("TkAgg")
    else:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import FancyArrowPatch

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = load_model(args, device)
    schedule = CosineSchedule(T=100).to(device)

    dt = 0.25  # 4 Hz
    T_hist_max = 20

    if args.model == "single":
        scenarios = SINGLE_SCENARIOS
    else:
        scenarios = JOINT_SCENARIOS

    # Pre-generate all positions for all scenarios
    all_frames = []  # list of frame dicts

    for sc_idx, sc in enumerate(scenarios):
        total_steps = int(sc["total_s"] / dt)

        if args.model == "single":
            t_all = np.arange(total_steps) * dt
            all_pos = sc["gen"](t_all)  # (total_steps, 2)

            # Need at least 5 history points before predicting
            for step in range(5, total_steps):
                hist_start = max(0, step - T_hist_max)
                positions = all_pos[hist_start:step + 1]  # up to 20 points

                hist_np, origin = build_history(positions, dt)
                mask_np = np.ones(len(hist_np), dtype=np.float32)

                # Ground truth future (if available)
                gt_future = None
                if step + 20 <= total_steps:
                    gt_abs = all_pos[step + 1:step + 21]  # next 20 steps
                    gt_future = gt_abs - origin  # ego-normalize

                all_frames.append({
                    "scenario_idx": sc_idx,
                    "scenario_name": sc["name"],
                    "step": step,
                    "hist_abs": all_pos[hist_start:step + 1].copy(),
                    "hist_ego": hist_np,
                    "mask": mask_np,
                    "origin": origin,
                    "gt_future_abs": (gt_abs.copy() if gt_future is not None else None),
                    "gt_future_ego": gt_future,
                })
        else:
            num_agents = len(sc["gens"])
            t_all = np.arange(total_steps) * dt
            all_pos_agents = [gen(t_all) for gen in sc["gens"]]

            for step in range(5, total_steps):
                hist_start = max(0, step - T_hist_max)
                agents_data = []

                for m in range(num_agents):
                    positions = all_pos_agents[m][hist_start:step + 1]
                    hist_np, origin = build_history(positions, dt)
                    mask_np = np.ones(len(hist_np), dtype=np.float32)

                    gt_future_abs = None
                    if step + 20 <= total_steps:
                        gt_future_abs = all_pos_agents[m][step + 1:step + 21].copy()

                    agents_data.append({
                        "hist_abs": all_pos_agents[m][hist_start:step + 1].copy(),
                        "hist_ego": hist_np,
                        "mask": mask_np,
                        "origin": origin,
                        "gt_future_abs": gt_future_abs,
                    })

                all_frames.append({
                    "scenario_idx": sc_idx,
                    "scenario_name": sc["name"],
                    "step": step,
                    "agents": agents_data,
                })

    print(f"Total frames: {len(all_frames)}")
    print("Running inference on all frames...")

    # Run inference for all frames
    t0 = time.perf_counter()
    predictions = []

    for fi, frame in enumerate(all_frames):
        if args.model == "single":
            preds = predict_single(model, schedule, frame["hist_ego"],
                                   device, K=args.K)  # (K, 20, 2)
            preds = smooth_predictions(preds, window=5)
            # Convert to absolute coordinates
            preds_abs = preds + frame["origin"]
            predictions.append(preds_abs)
        else:
            hists = [a["hist_ego"] for a in frame["agents"]]
            masks = [a["mask"] for a in frame["agents"]]
            preds = predict_joint(model, schedule, hists, masks,
                                  args.max_agents, device, K=args.K)
            preds = smooth_predictions(preds, window=5)
            # (K, M, 20, 2) — convert each agent to absolute
            preds_abs = preds.copy()
            for m in range(len(frame["agents"])):
                preds_abs[:, m] += frame["agents"][m]["origin"]
            predictions.append(preds_abs)

        if (fi + 1) % 20 == 0 or fi == len(all_frames) - 1:
            elapsed = time.perf_counter() - t0
            print(f"  [{fi+1}/{len(all_frames)}] {elapsed:.1f}s "
                  f"({(fi+1)/elapsed:.1f} frames/s)")

    print(f"Inference done in {time.perf_counter() - t0:.1f}s")

    # ---- Precompute fixed axis bounds per scenario ----
    # Use history + GT + median prediction (not outlier K samples) for tight bounds.
    scenario_pts = {}
    for fi, frame in enumerate(all_frames):
        sc_idx = frame["scenario_idx"]
        if sc_idx not in scenario_pts:
            scenario_pts[sc_idx] = []
        preds = predictions[fi]

        if args.model == "single":
            scenario_pts[sc_idx].append(frame["hist_abs"])
            # Use the mean trajectory instead of all K samples
            scenario_pts[sc_idx].append(preds.mean(axis=0))
            if frame["gt_future_abs"] is not None:
                scenario_pts[sc_idx].append(frame["gt_future_abs"])
        else:
            for m, agent in enumerate(frame["agents"]):
                scenario_pts[sc_idx].append(agent["hist_abs"])
                if agent["gt_future_abs"] is not None:
                    scenario_pts[sc_idx].append(agent["gt_future_abs"])
            # Mean across K for each agent
            scenario_pts[sc_idx].append(preds.mean(axis=0).reshape(-1, 2))

    scenario_bounds = {}
    for sc_idx, pts_list in scenario_pts.items():
        all_pts = np.concatenate(pts_list, axis=0)
        # Use percentile to exclude any remaining outliers
        xmin, xmax = np.percentile(all_pts[:, 0], [1, 99])
        ymin, ymax = np.percentile(all_pts[:, 1], [1, 99])

        pad = 3.0
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        half_range = max(xmax - xmin, ymax - ymin) / 2 + pad
        # Ensure minimum visible range of 8m
        half_range = max(half_range, 4.0)
        scenario_bounds[sc_idx] = {
            "xlim": (cx - half_range, cx + half_range),
            "ylim": (cy - half_range, cy + half_range),
        }

    # ---- Build animation ----
    COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.08)

    def animate(fi):
        ax.cla()
        frame = all_frames[fi]
        preds = predictions[fi]
        K = args.K
        sc_idx = frame["scenario_idx"]
        bounds = scenario_bounds[sc_idx]

        # Fixed axis limits — never changes within a scenario
        ax.set_xlim(bounds["xlim"])
        ax.set_ylim(bounds["ylim"])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_xlabel("x (m)", fontsize=11)
        ax.set_ylabel("y (m)", fontsize=11)

        ax.set_title(frame["scenario_name"], fontsize=14, fontweight="bold", pad=10)

        if args.model == "single":
            hist_abs = frame["hist_abs"]

            # Faded trail (older history points fade out)
            n_hist = len(hist_abs)
            for j in range(max(0, n_hist - 2), n_hist - 1):
                alpha = 0.3 + 0.7 * (j / max(n_hist - 1, 1))
                ax.plot(hist_abs[j:j+2, 0], hist_abs[j:j+2, 1],
                        "-", color=COLORS[0], linewidth=2.5, alpha=alpha, zorder=5)

            # History dots
            alphas = np.linspace(0.2, 1.0, n_hist)
            for j in range(n_hist):
                ax.plot(hist_abs[j, 0], hist_abs[j, 1], "o",
                        color=COLORS[0], markersize=3, alpha=alphas[j], zorder=5)

            # Current position marker
            ax.plot(hist_abs[-1, 0], hist_abs[-1, 1], "s",
                    color=COLORS[0], markersize=10, zorder=6, label="Current position")

            # Ground truth future
            if frame["gt_future_abs"] is not None:
                gt = frame["gt_future_abs"]
                trail = np.vstack([hist_abs[-1:], gt])
                ax.plot(trail[:, 0], trail[:, 1], "--",
                        color="#2ca02c", linewidth=2, alpha=0.7,
                        zorder=3, label="Ground truth future")

            # Filter outlier samples: only show those within 2 std of mean
            mean_traj = preds.mean(axis=0)
            dists = np.linalg.norm(preds - mean_traj[None], axis=-1).mean(axis=-1)
            dist_threshold = np.median(dists) + 2.0 * dists.std()
            inlier_mask = dists < dist_threshold

            # Predicted samples (fan) — only inliers
            for k in range(K):
                if not inlier_mask[k]:
                    continue
                trail = np.vstack([hist_abs[-1:], preds[k]])
                ax.plot(trail[:, 0], trail[:, 1],
                        color="#ff7f0e", alpha=0.15, linewidth=0.8, zorder=2)

            # Best mode (closest to mean among inliers)
            best_k = np.where(inlier_mask)[0][dists[inlier_mask].argmin()]
            trail = np.vstack([hist_abs[-1:], preds[best_k]])
            ax.plot(trail[:, 0], trail[:, 1],
                    color="#ff7f0e", linewidth=2.5, alpha=0.9,
                    zorder=4, label="Predicted (best mode)")

            # Endpoint scatter for inlier samples
            inlier_preds = preds[inlier_mask]
            ax.scatter(inlier_preds[:, -1, 0], inlier_preds[:, -1, 1],
                       c="#ff7f0e", s=15, alpha=0.4, zorder=3, edgecolors="none")

        else:
            for m, agent in enumerate(frame["agents"]):
                color = COLORS[m % len(COLORS)]
                hist_abs = agent["hist_abs"]
                label_prefix = f"Ped {chr(65 + m)}"
                n_hist = len(hist_abs)

                # Faded trail
                alphas = np.linspace(0.2, 1.0, n_hist)
                for j in range(n_hist):
                    ax.plot(hist_abs[j, 0], hist_abs[j, 1], "o",
                            color=color, markersize=3, alpha=alphas[j], zorder=5)
                if n_hist > 1:
                    ax.plot(hist_abs[:, 0], hist_abs[:, 1], "-",
                            color=color, linewidth=2, alpha=0.6, zorder=4)

                ax.plot(hist_abs[-1, 0], hist_abs[-1, 1], "s",
                        color=color, markersize=8, zorder=6,
                        label=f"{label_prefix}")

                if agent["gt_future_abs"] is not None:
                    gt = agent["gt_future_abs"]
                    trail = np.vstack([hist_abs[-1:], gt])
                    ax.plot(trail[:, 0], trail[:, 1], "--",
                            color=color, linewidth=1.5, alpha=0.5, zorder=3)

                agent_preds = preds[:, m]
                mean_traj = agent_preds.mean(axis=0)
                dists = np.linalg.norm(agent_preds - mean_traj[None], axis=-1).mean(axis=-1)
                dist_threshold = np.median(dists) + 2.0 * dists.std()
                inlier_mask = dists < dist_threshold

                for k in range(K):
                    if not inlier_mask[k]:
                        continue
                    trail = np.vstack([hist_abs[-1:], preds[k, m]])
                    ax.plot(trail[:, 0], trail[:, 1],
                            color=color, alpha=0.1, linewidth=0.6, zorder=2)

                best_k = np.where(inlier_mask)[0][dists[inlier_mask].argmin()]
                trail = np.vstack([hist_abs[-1:], preds[best_k, m]])
                ax.plot(trail[:, 0], trail[:, 1],
                        color=color, linewidth=2.5, alpha=0.9, zorder=4)

                inlier_agent = agent_preds[inlier_mask]
                ax.scatter(inlier_agent[:, -1, 0], inlier_agent[:, -1, 1],
                           c=color, s=12, alpha=0.3, zorder=3, edgecolors="none")

        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

        # Time indicator
        step = frame["step"]
        t_sec = step * dt
        ax.text(0.98, 0.02, f"t = {t_sec:.1f}s",
                transform=ax.transAxes, fontsize=13, fontweight="bold",
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

        # Model info
        model_name = "Single-Agent" if args.model == "single" else "Joint Multi-Agent"
        ax.text(0.98, 0.98,
                f"Diffusion {model_name}\nDDIM-10, K={K} samples",
                transform=ax.transAxes, fontsize=9, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.9))

    print("Building animation...")
    anim = animation.FuncAnimation(
        fig, animate, frames=len(all_frames),
        interval=1000 // args.fps, repeat=True,
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        ext = os.path.splitext(args.output)[1].lower()
        if ext == ".gif":
            writer = animation.PillowWriter(fps=args.fps)
        else:
            writer = animation.FFMpegWriter(fps=args.fps, bitrate=3000)
        print(f"Saving to {args.output} ...")
        anim.save(args.output, writer=writer)
        print(f"Saved: {args.output}")
    elif args.live:
        plt.show()
    else:
        out = "figures/demo_live.mp4"
        os.makedirs("figures", exist_ok=True)
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=3000)
        print(f"Saving to {out} ...")
        anim.save(out, writer=writer)
        print(f"Saved: {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
