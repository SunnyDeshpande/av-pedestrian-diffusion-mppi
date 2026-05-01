#!/usr/bin/env python3
"""Standalone visualization of diffusion trajectory predictions vs ground truth.

Loads a trained model, runs DDIM inference on test data, and generates
matplotlib trajectory plots showing multi-modal predictions.

Usage:
    # Single-agent
    python scripts/visualize.py \
        --model single \
        --weights ../../models/diffusion/av2_pretrain_v1/ema_best.pt \
        --data path/to/val_shards/ \
        --num-samples 16 --seed 42 \
        --output figures/single_vis.png

    # Joint multi-agent
    python scripts/visualize.py \
        --model joint \
        --weights ../../models/diffusion/av2_joint_v1/ema_best.pt \
        --data path/to/joint_val_shards/ \
        --num-samples 9 --seed 42 \
        --output figures/joint_vis.png
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running from repo root or scripts/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diffusion_prediction.model import TrajectoryDenoiser
from diffusion_prediction.model_joint import JointTrajectoryDenoiser
from diffusion_prediction.ddpm import CosineSchedule, ddim_sample_loop, ddim_sample_loop_joint
from diffusion_prediction.dataset import TrajectoryDataset, collate_fn
from diffusion_prediction.dataset_joint import JointTrajectoryDataset


def parse_args():
    p = argparse.ArgumentParser(description="Visualize diffusion trajectory predictions")
    p.add_argument("--model", choices=["single", "joint"], required=True)
    p.add_argument("--weights", type=str, required=True, help="Path to .pt checkpoint")
    p.add_argument("--data", type=str, default=None, help="Path to val .npz shard directory")
    p.add_argument("--demo", action="store_true",
                   help="Use synthetic pedestrian scenarios (no data needed)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-samples", type=int, default=16, help="Number of samples to visualize")
    p.add_argument("--K", type=int, default=20, help="Number of trajectory samples per pedestrian")
    p.add_argument("--output", type=str, default=None, help="Output path (default: plt.show())")
    p.add_argument("--seed", type=int, default=42)
    # Architecture args (defaults match training)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dim-ff", type=int, default=256)
    p.add_argument("--max-agents", type=int, default=16)
    p.add_argument("--num-enc-layers", type=int, default=4)
    p.add_argument("--num-interaction-layers", type=int, default=2)
    return p.parse_args()


def _make_history(positions, dt=0.25):
    """Build a (T, 4) history array [x, y, vx, vy] from (T, 2) positions.

    Velocities are computed via finite differences. The trajectory is
    normalized so that the last observed position is at the origin and
    the coordinate frame is ego-centric (heading = +x).
    """
    pos = np.array(positions, dtype=np.float32)
    T = pos.shape[0]

    # Normalize: translate so last point is origin
    origin = pos[-1].copy()
    pos = pos - origin

    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) / dt
    vel[0] = vel[1]

    hist = np.zeros((T, 4), dtype=np.float32)
    hist[:, :2] = pos
    hist[:, 2:] = vel
    return hist


def generate_synthetic_scenarios():
    """Create 9 diverse synthetic pedestrian motion scenarios.

    Returns list of dicts with keys: history (20,4), history_mask (20,),
    label (str). Each scenario is a realistic 5-second observed history
    at 4 Hz (20 timesteps).
    """
    dt = 0.25
    T = 20
    scenarios = []

    # 1. Walking straight (+x direction, ~1.4 m/s)
    t = np.arange(T) * dt
    pos = np.stack([1.4 * t, np.zeros(T)], axis=-1)
    scenarios.append({"label": "Straight walk", "positions": pos})

    # 2. Walking diagonally
    speed = 1.2
    angle = np.radians(35)
    pos = np.stack([speed * np.cos(angle) * t, speed * np.sin(angle) * t], axis=-1)
    scenarios.append({"label": "Diagonal walk", "positions": pos})

    # 3. Turning left (arc)
    radius = 4.0
    omega = 0.3  # rad/s
    theta = omega * t
    pos = np.stack([radius * np.sin(theta), radius * (1 - np.cos(theta))], axis=-1)
    scenarios.append({"label": "Left turn", "positions": pos})

    # 4. Turning right (arc)
    pos = np.stack([radius * np.sin(theta), -radius * (1 - np.cos(theta))], axis=-1)
    scenarios.append({"label": "Right turn", "positions": pos})

    # 5. Stopping (decelerating to standstill)
    v0 = 1.5
    decel_time = 3.0  # seconds to stop
    vt = np.clip(v0 - (v0 / decel_time) * t, 0, None)
    x = np.cumsum(vt) * dt
    pos = np.stack([x, np.zeros(T)], axis=-1)
    scenarios.append({"label": "Stopping", "positions": pos})

    # 6. Accelerating from standstill
    a = 0.4  # m/s^2
    x = 0.5 * a * t ** 2
    pos = np.stack([x, np.zeros(T)], axis=-1)
    scenarios.append({"label": "Accelerating", "positions": pos})

    # 7. S-curve (lane change)
    x = 1.3 * t
    y = 1.5 * np.sin(0.6 * t)
    pos = np.stack([x, y], axis=-1)
    scenarios.append({"label": "S-curve", "positions": pos})

    # 8. Standing still (stationary)
    pos = np.stack([np.full(T, 0.0), np.full(T, 0.0)], axis=-1)
    # Add slight jitter to simulate sensor noise
    pos += np.random.RandomState(123).randn(T, 2) * 0.03
    scenarios.append({"label": "Standing still", "positions": pos})

    # 9. Walking backward
    pos = np.stack([-1.0 * t, np.zeros(T)], axis=-1)
    scenarios.append({"label": "Walking backward", "positions": pos})

    results = []
    for sc in scenarios:
        hist = _make_history(sc["positions"])
        mask = np.ones(T, dtype=np.float32)
        results.append({
            "history": hist,
            "history_mask": mask,
            "label": sc["label"],
        })
    return results


def generate_synthetic_joint_scenarios(max_agents=16):
    """Create 4 multi-agent synthetic scenarios.

    Returns list of dicts with keys: histories (M,20,4), history_masks (M,20),
    agent_mask (M,), ego_vel (2,), label (str).
    """
    dt = 0.25
    T = 20
    t = np.arange(T) * dt
    scenarios = []

    # Scene 1: Two pedestrians crossing paths
    pos_a = np.stack([1.3 * t, 0.3 * t], axis=-1)
    pos_b = np.stack([0.2 * t, 1.4 * t], axis=-1) + np.array([3.0, -2.0])
    scenarios.append({"label": "Crossing paths", "agents": [pos_a, pos_b]})

    # Scene 2: Three pedestrians walking in parallel
    for offset_y in [-1.5, 0.0, 1.5]:
        if offset_y == -1.5:
            agents = []
        speed = 1.2 + 0.2 * offset_y / 1.5
        pos = np.stack([speed * t, np.full(T, offset_y)], axis=-1)
        agents.append(pos)
    scenarios.append({"label": "Parallel walkers", "agents": agents})

    # Scene 3: Group splitting (start together, diverge)
    base = np.stack([1.0 * t, np.zeros(T)], axis=-1)
    pos_a = base + np.stack([np.zeros(T), 0.15 * t ** 1.3], axis=-1)
    pos_b = base + np.stack([np.zeros(T), -0.15 * t ** 1.3], axis=-1)
    pos_c = base.copy()
    scenarios.append({"label": "Group splitting", "agents": [pos_a, pos_b, pos_c]})

    # Scene 4: Head-on approach
    pos_a = np.stack([1.3 * t, np.zeros(T)], axis=-1)
    pos_b = np.stack([8.0 - 1.1 * t, 0.5 * np.ones(T)], axis=-1)
    scenarios.append({"label": "Head-on approach", "agents": [pos_a, pos_b]})

    results = []
    for sc in scenarios:
        M_real = len(sc["agents"])
        histories = np.zeros((max_agents, T, 4), dtype=np.float32)
        history_masks = np.zeros((max_agents, T), dtype=np.float32)
        agent_mask = np.zeros(max_agents, dtype=np.float32)

        for m, pos in enumerate(sc["agents"]):
            histories[m] = _make_history(pos)
            history_masks[m] = 1.0
            agent_mask[m] = 1.0

        results.append({
            "histories": histories,
            "history_masks": history_masks,
            "agent_mask": agent_mask,
            "ego_vel": np.array([2.0, 0.0], dtype=np.float32),
            "label": sc["label"],
            "num_agents": M_real,
        })
    return results


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


def best_mode_index(predictions):
    """Pick the sample closest to the mean trajectory.

    Parameters
    ----------
    predictions : (K, T, 2)

    Returns
    -------
    int : index of best mode
    """
    mean_traj = predictions.mean(dim=0)  # (T, 2)
    dists = torch.norm(predictions - mean_traj.unsqueeze(0), dim=-1).mean(dim=-1)  # (K,)
    return dists.argmin().item()


def per_sample_metrics(preds, gt):
    """Compute minADE and minFDE for one sample.

    Parameters
    ----------
    preds : (K, T, 2)
    gt    : (T, 2)
    """
    gt_exp = gt.unsqueeze(0)  # (1, T, 2)
    ade = torch.norm(preds - gt_exp, dim=-1).mean(dim=-1)  # (K,)
    fde = torch.norm(preds[:, -1, :] - gt_exp[:, -1, :], dim=-1)  # (K,)
    return ade.min().item(), fde.min().item()


def run_single(model, schedule, dataset, args, device):
    """Run inference on single-agent samples."""
    N = min(args.num_samples, len(dataset))
    # Evenly spaced indices for diversity
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(dataset), size=N, replace=False)
    indices.sort()

    batch_items = [dataset[i] for i in indices]
    batch = collate_fn(batch_items)
    hist = batch["history"].to(device)        # (N, 20, 4)
    mask = batch["history_mask"].to(device)    # (N, 20)
    ego = batch["ego_vel"].to(device)          # (N, 2)
    gt = batch["future"]                       # (N, 20, 2)

    with torch.no_grad():
        preds = ddim_sample_loop(model, schedule, hist, mask, ego,
                                 K=args.K, seed=args.seed)  # (N, K, 20, 2)
    preds = preds.cpu()

    results = []
    for i in range(N):
        min_ade, min_fde = per_sample_metrics(preds[i], gt[i])
        best_k = best_mode_index(preds[i])
        results.append({
            "history": batch["history"][i].numpy(),       # (20, 4)
            "history_mask": batch["history_mask"][i].numpy(),  # (20,)
            "future_gt": gt[i].numpy(),                   # (20, 2)
            "predictions": preds[i].numpy(),              # (K, 20, 2)
            "best_k": best_k,
            "minADE": min_ade,
            "minFDE": min_fde,
        })
    return results


def run_joint(model, schedule, dataset, args, device):
    """Run inference on joint multi-agent scenes."""
    N = min(args.num_samples, len(dataset))
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(dataset), size=N, replace=False)
    indices.sort()

    results = []
    for idx in indices:
        sample = dataset[idx]
        hist = sample["history"].unsqueeze(0).to(device)        # (1, M, 20, 4)
        mask = sample["history_mask"].unsqueeze(0).to(device)   # (1, M, 20)
        amask = sample["agent_mask"].unsqueeze(0).to(device)    # (1, M)
        ego = sample["ego_vel"].unsqueeze(0).to(device)         # (1, 2)
        gt = sample["future"]                                   # (M, 20, 2)
        agent_mask_np = sample["agent_mask"].numpy()            # (M,)

        with torch.no_grad():
            preds = ddim_sample_loop_joint(model, schedule, hist, mask, amask, ego,
                                           K=args.K, seed=args.seed)  # (1, K, M, 20, 2)
        preds = preds[0].cpu()  # (K, M, 20, 2)

        num_real = int(agent_mask_np.sum())
        per_agent = []
        for m in range(num_real):
            agent_preds = preds[:, m, :, :]  # (K, 20, 2)
            agent_gt = gt[m]                 # (20, 2)
            min_ade, min_fde = per_sample_metrics(agent_preds, agent_gt)
            best_k = best_mode_index(agent_preds)
            per_agent.append({
                "minADE": min_ade,
                "minFDE": min_fde,
                "best_k": best_k,
            })

        results.append({
            "histories": sample["history"].numpy(),       # (M, 20, 4)
            "history_masks": sample["history_mask"].numpy(),  # (M, 20)
            "futures_gt": gt.numpy(),                     # (M, 20, 2)
            "predictions": preds.numpy(),                 # (K, M, 20, 2)
            "agent_mask": agent_mask_np,                  # (M,)
            "per_agent": per_agent,
        })
    return results


def run_demo_single(model, schedule, args, device):
    """Run inference on synthetic single-agent scenarios."""
    scenarios = generate_synthetic_scenarios()
    N = len(scenarios)

    hist_list = []
    mask_list = []
    for sc in scenarios:
        hist_list.append(torch.from_numpy(sc["history"]))
        mask_list.append(torch.from_numpy(sc["history_mask"]))

    hist = torch.stack(hist_list).to(device)   # (N, 20, 4)
    mask = torch.stack(mask_list).to(device)    # (N, 20)
    ego = torch.zeros(N, 2, device=device)
    ego[:, 0] = 2.0  # assume 2 m/s ego velocity

    with torch.no_grad():
        preds = ddim_sample_loop(model, schedule, hist, mask, ego,
                                 K=args.K, seed=args.seed)  # (N, K, 20, 2)
    preds = preds.cpu()

    results = []
    for i, sc in enumerate(scenarios):
        best_k = best_mode_index(preds[i])
        results.append({
            "history": sc["history"],
            "history_mask": sc["history_mask"],
            "future_gt": None,  # no ground truth in demo mode
            "predictions": preds[i].numpy(),
            "best_k": best_k,
            "label": sc["label"],
        })
    return results


def run_demo_joint(model, schedule, args, device):
    """Run inference on synthetic joint multi-agent scenarios."""
    scenarios = generate_synthetic_joint_scenarios(max_agents=args.max_agents)

    results = []
    for sc in scenarios:
        hist = torch.from_numpy(sc["histories"]).unsqueeze(0).to(device)
        mask = torch.from_numpy(sc["history_masks"]).unsqueeze(0).to(device)
        amask = torch.from_numpy(sc["agent_mask"]).unsqueeze(0).to(device)
        ego = torch.from_numpy(sc["ego_vel"]).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = ddim_sample_loop_joint(model, schedule, hist, mask, amask, ego,
                                           K=args.K, seed=args.seed)
        preds = preds[0].cpu().numpy()  # (K, M, 20, 2)

        num_real = sc["num_agents"]
        per_agent = []
        for m in range(num_real):
            agent_preds = torch.from_numpy(preds[:, m, :, :])
            best_k = best_mode_index(agent_preds)
            per_agent.append({"best_k": best_k})

        results.append({
            "histories": sc["histories"],
            "history_masks": sc["history_masks"],
            "futures_gt": None,
            "predictions": preds,
            "agent_mask": sc["agent_mask"],
            "per_agent": per_agent,
            "label": sc["label"],
            "num_agents": num_real,
        })
    return results


def plot_demo_single(results, output_path, K):
    """Plot demo single-agent results (no ground truth)."""
    N = len(results)
    num_cols = min(3, N)
    num_rows = math.ceil(N / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    if N == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i, res in enumerate(results):
        ax = axes[i // num_cols, i % num_cols]

        hist = res["history"][:, :2]
        mask = res["history_mask"]
        preds = res["predictions"]  # (K, 20, 2)
        best_k = res["best_k"]

        valid = mask > 0.5
        hist_valid = hist[valid]

        # History
        if len(hist_valid) > 0:
            ax.plot(hist_valid[:, 0], hist_valid[:, 1], "o-",
                    color="#1f77b4", linewidth=2.5, markersize=4,
                    label="Observed (5s)" if i == 0 else None, zorder=5)
            # Mark current position
            ax.plot(hist_valid[-1, 0], hist_valid[-1, 1], "s",
                    color="#1f77b4", markersize=8, zorder=6)

        # All K predicted samples
        for k in range(K):
            ax.plot(preds[k, :, 0], preds[k, :, 1],
                    color="#ff7f0e", alpha=0.15, linewidth=0.8, zorder=2)

        # Best mode
        ax.plot(preds[best_k, :, 0], preds[best_k, :, 1],
                color="#ff7f0e", alpha=0.9, linewidth=2.5,
                label="Predicted (5s)" if i == 0 else None, zorder=4)

        # Connect history to predictions
        if len(hist_valid) > 0:
            for k in range(K):
                ax.plot([hist_valid[-1, 0], preds[k, 0, 0]],
                        [hist_valid[-1, 1], preds[k, 0, 1]],
                        color="#ff7f0e", alpha=0.08, linewidth=0.5, zorder=2)

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(res["label"], fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.set_xlabel("x (m)", fontsize=9)
        ax.set_ylabel("y (m)", fontsize=9)

    for j in range(N, num_rows * num_cols):
        axes[j // num_cols, j % num_cols].set_visible(False)

    fig.suptitle("Diffusion Pedestrian Trajectory Prediction — Demo\n"
                 "Blue = observed history (5s)  |  Orange = predicted futures (5s, K=20 samples)",
                 fontsize=12, y=1.03)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved demo figure: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_demo_joint(results, output_path, K):
    """Plot demo joint multi-agent results."""
    N = len(results)
    num_cols = min(2, N)
    num_rows = math.ceil(N / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 6 * num_rows))
    if N == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    cmap = plt.cm.tab10
    agent_labels = ["Ped A", "Ped B", "Ped C", "Ped D"]

    for i, res in enumerate(results):
        ax = axes[i // num_cols, i % num_cols]
        num_real = res["num_agents"]
        preds = res["predictions"]   # (K, M, 20, 2)

        for m in range(num_real):
            color = cmap(m % 10)
            hist = res["histories"][m, :, :2]
            mask = res["history_masks"][m]
            best_k = res["per_agent"][m]["best_k"]
            label = agent_labels[m] if m < len(agent_labels) else f"Ped {m}"

            valid = mask > 0.5
            hist_valid = hist[valid]

            if len(hist_valid) > 0:
                ax.plot(hist_valid[:, 0], hist_valid[:, 1], "o-",
                        color=color, linewidth=2.5, markersize=3,
                        label=f"{label} history", zorder=5)
                ax.plot(hist_valid[-1, 0], hist_valid[-1, 1], "s",
                        color=color, markersize=7, zorder=6)

            for k in range(K):
                ax.plot(preds[k, m, :, 0], preds[k, m, :, 1],
                        color=color, alpha=0.1, linewidth=0.6, zorder=2)

            ax.plot(preds[best_k, m, :, 0], preds[best_k, m, :, 1],
                    color=color, alpha=0.9, linewidth=2.5,
                    label=f"{label} predicted", zorder=4)

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{res['label']} ({num_real} agents)", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.set_xlabel("x (m)", fontsize=9)
        ax.set_ylabel("y (m)", fontsize=9)
        ax.legend(fontsize=7, loc="best")

    for j in range(N, num_rows * num_cols):
        axes[j // num_cols, j % num_cols].set_visible(False)

    fig.suptitle("Joint Multi-Agent Diffusion Prediction — Demo\n"
                 "Solid = observed history  |  Thin = K=20 samples  |  Thick = best mode",
                 fontsize=12, y=1.03)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved demo figure: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_single_grid(results, output_path, K):
    N = len(results)
    num_cols = min(4, N)
    num_rows = math.ceil(N / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
    if N == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i, res in enumerate(results):
        ax = axes[i // num_cols, i % num_cols]

        hist = res["history"][:, :2]      # (20, 2)
        mask = res["history_mask"]        # (20,)
        gt = res["future_gt"]             # (20, 2)
        preds = res["predictions"]        # (K, 20, 2)
        best_k = res["best_k"]

        # Plot valid history points
        valid = mask > 0.5
        hist_valid = hist[valid]
        if len(hist_valid) > 0:
            ax.plot(hist_valid[:, 0], hist_valid[:, 1], "o-",
                    color="#1f77b4", linewidth=2, markersize=3,
                    label="Observed" if i == 0 else None, zorder=4)

        # Connect history to future
        if len(hist_valid) > 0:
            connect = np.vstack([hist_valid[-1:], gt[:1]])
            ax.plot(connect[:, 0], connect[:, 1], "--", color="#999999",
                    linewidth=0.8, zorder=2)

        # Ground truth future
        ax.plot(gt[:, 0], gt[:, 1], "--", color="#2ca02c", linewidth=2,
                label="Ground truth" if i == 0 else None, zorder=3)

        # All K predicted samples (semi-transparent fan)
        for k in range(K):
            ax.plot(preds[k, :, 0], preds[k, :, 1],
                    color="#ff7f0e", alpha=0.12, linewidth=0.7, zorder=2)

        # Best mode (thicker)
        ax.plot(preds[best_k, :, 0], preds[best_k, :, 1],
                color="#ff7f0e", alpha=0.9, linewidth=2,
                label="Best mode" if i == 0 else None, zorder=3)

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Sample {i}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.text(0.02, 0.98,
                f"minADE={res['minADE']:.2f}m\nminFDE={res['minFDE']:.2f}m",
                transform=ax.transAxes, fontsize=7, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Hide unused subplots
    for j in range(N, num_rows * num_cols):
        axes[j // num_cols, j % num_cols].set_visible(False)

    fig.suptitle("Diffusion Trajectory Prediction (Single-Agent)", fontsize=13, y=1.01)
    fig.legend(loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.995))
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved trajectory grid: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_joint_grid(results, output_path, K):
    N = len(results)
    num_cols = min(3, N)
    num_rows = math.ceil(N / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    if N == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    cmap = plt.cm.tab10

    for i, res in enumerate(results):
        ax = axes[i // num_cols, i % num_cols]
        agent_mask = res["agent_mask"]
        num_real = int(agent_mask.sum())
        preds = res["predictions"]   # (K, M, 20, 2)
        per_agent = res["per_agent"]

        for m in range(num_real):
            color = cmap(m % 10)
            hist = res["histories"][m, :, :2]       # (20, 2)
            mask = res["history_masks"][m]           # (20,)
            gt = res["futures_gt"][m]                # (20, 2)
            best_k = per_agent[m]["best_k"]

            valid = mask > 0.5
            hist_valid = hist[valid]
            if len(hist_valid) > 0:
                ax.plot(hist_valid[:, 0], hist_valid[:, 1], "o-",
                        color=color, linewidth=2, markersize=3, zorder=4)

            if len(hist_valid) > 0:
                connect = np.vstack([hist_valid[-1:], gt[:1]])
                ax.plot(connect[:, 0], connect[:, 1], "--", color="#999999",
                        linewidth=0.8, zorder=2)

            ax.plot(gt[:, 0], gt[:, 1], "--", color=color, linewidth=1.5,
                    alpha=0.6, zorder=3)

            for k in range(K):
                ax.plot(preds[k, m, :, 0], preds[k, m, :, 1],
                        color=color, alpha=0.08, linewidth=0.6, zorder=2)

            ax.plot(preds[best_k, m, :, 0], preds[best_k, m, :, 1],
                    color=color, alpha=0.9, linewidth=2, zorder=3)

        mean_ade = np.mean([a["minADE"] for a in per_agent])
        mean_fde = np.mean([a["minFDE"] for a in per_agent])

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Scene {i} ({num_real} agents)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.text(0.02, 0.98,
                f"minADE={mean_ade:.2f}m\nminFDE={mean_fde:.2f}m",
                transform=ax.transAxes, fontsize=7, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    for j in range(N, num_rows * num_cols):
        axes[j // num_cols, j % num_cols].set_visible(False)

    fig.suptitle("Diffusion Trajectory Prediction (Joint Multi-Agent)", fontsize=13, y=1.01)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved trajectory grid: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_fde_histogram(all_fde, output_path, model_type):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_fde, bins=50, color="#1f77b4", edgecolor="white", alpha=0.85)

    mean_fde = np.mean(all_fde)
    median_fde = np.median(all_fde)
    std_fde = np.std(all_fde)
    miss_rate = np.mean(np.array(all_fde) > 2.0)

    ax.axvline(mean_fde, color="#d62728", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_fde:.3f}m")
    ax.axvline(median_fde, color="#ff7f0e", linestyle="--", linewidth=1.5,
               label=f"Median: {median_fde:.3f}m")

    ax.set_xlabel("minFDE (m)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"minFDE Distribution ({model_type})", fontsize=13)
    ax.legend(fontsize=9)
    ax.text(0.98, 0.95,
            f"std: {std_fde:.3f}m\nmiss@2m: {miss_rate:.1%}\nn={len(all_fde)}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()

    if output_path:
        base, ext = os.path.splitext(output_path)
        hist_path = f"{base}_hist{ext}"
        fig.savefig(hist_path, dpi=150, bbox_inches="tight")
        print(f"Saved FDE histogram: {hist_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()

    if not args.demo and not args.data:
        print("Error: either --data or --demo is required")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = load_model(args, device)
    schedule = CosineSchedule(T=100).to(device)

    if args.demo:
        # ---- Demo mode: synthetic scenarios, no data needed ----
        print("Running demo with synthetic pedestrian scenarios...")
        if args.model == "single":
            results = run_demo_single(model, schedule, args, device)
            print(f"Generated {len(results)} synthetic scenarios")
            plot_demo_single(results, args.output, args.K)
        else:
            results = run_demo_joint(model, schedule, args, device)
            print(f"Generated {len(results)} synthetic multi-agent scenes")
            plot_demo_joint(results, args.output, args.K)
        print("Done!")
        return

    # ---- Data mode: load real val data ----
    if args.model == "single":
        dataset = TrajectoryDataset(args.data, augment=False)
        print(f"Loaded {len(dataset)} samples from {args.data}")

        results = run_single(model, schedule, dataset, args, device)
        all_fde = [r["minFDE"] for r in results]

        print(f"\n{'='*40}")
        print(f"  Samples: {len(results)}")
        print(f"  Mean minADE: {np.mean([r['minADE'] for r in results]):.4f} m")
        print(f"  Mean minFDE: {np.mean(all_fde):.4f} m")
        print(f"  Miss@2m:     {np.mean(np.array(all_fde) > 2.0):.4f}")
        print(f"{'='*40}\n")

        plot_single_grid(results, args.output, args.K)
        plot_fde_histogram(all_fde, args.output, "Single-Agent")

    else:
        dataset = JointTrajectoryDataset(args.data, augment=False,
                                         max_agents=args.max_agents)
        print(f"Loaded {len(dataset)} scenes from {args.data}")

        results = run_joint(model, schedule, dataset, args, device)
        all_fde = []
        for r in results:
            for a in r["per_agent"]:
                all_fde.append(a["minFDE"])

        print(f"\n{'='*40}")
        print(f"  Scenes: {len(results)}")
        print(f"  Total agents: {len(all_fde)}")
        print(f"  Mean minADE: {np.mean([a['minADE'] for r in results for a in r['per_agent']]):.4f} m")
        print(f"  Mean minFDE: {np.mean(all_fde):.4f} m")
        print(f"  Miss@2m:     {np.mean(np.array(all_fde) > 2.0):.4f}")
        print(f"{'='*40}\n")

        plot_joint_grid(results, args.output, args.K)
        plot_fde_histogram(all_fde, args.output, "Joint Multi-Agent")


if __name__ == "__main__":
    main()
