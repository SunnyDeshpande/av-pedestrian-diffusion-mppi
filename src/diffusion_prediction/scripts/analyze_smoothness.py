#!/usr/bin/env python3
"""Analyze prediction smoothness - compare raw diffusion output vs smoothed.

Generates diagnostic plots showing:
1. Frame-to-frame jitter (how much predictions jump between timesteps)
2. Per-trajectory curvature/jerkiness
3. Raw vs smoothed comparison side-by-side
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diffusion_prediction.model import TrajectoryDenoiser
from diffusion_prediction.ddpm import CosineSchedule, ddim_sample_loop
from diffusion_prediction.utils import filter_and_smooth_trajectories

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_history(positions, dt=0.25):
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


def smooth_predictions(preds, s_factor=50.0):
    return filter_and_smooth_trajectories(preds, s_factor=s_factor)


def compute_jerk(traj, dt=0.25):
    """Compute mean absolute jerk (3rd derivative) of a trajectory."""
    vel = np.diff(traj, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    return np.abs(jerk).mean()


def compute_frame_to_frame_shift(predictions_list):
    """How much the best-mode prediction shifts between consecutive frames."""
    shifts = []
    for i in range(1, len(predictions_list)):
        prev_mean = predictions_list[i-1].mean(axis=0)  # (20, 2)
        curr_mean = predictions_list[i].mean(axis=0)
        # Shift at each future timestep
        shift = np.linalg.norm(curr_mean - prev_mean, axis=-1)
        shifts.append(shift.mean())
    return np.array(shifts)


@torch.no_grad()
def predict(model, schedule, hist_np, device, K=20):
    T = hist_np.shape[0]
    if T < 20:
        padded = np.zeros((20, 4), dtype=np.float32)
        padded[20 - T:] = hist_np
        mask = np.zeros(20, dtype=np.float32)
        mask[20 - T:] = 1.0
    else:
        padded = hist_np[-20:]
        mask = np.ones(20, dtype=np.float32)

    hist_t = torch.from_numpy(padded).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
    ego_t = torch.zeros(1, 2, device=device)
    ego_t[0, 0] = 2.0
    futures = ddim_sample_loop(model, schedule, hist_t, mask_t, ego_t, K=K)
    return futures[0].cpu().numpy()  # (K, 20, 2)


def main():
    device = torch.device("cpu")
    weights = "../../models/diffusion/av2_pretrain_v1/ema_best.pt"

    model = TrajectoryDenoiser().to(device)
    state = torch.load(weights, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.eval()
    schedule = CosineSchedule(T=100).to(device)

    # Generate a straight-walking scenario - 60 frames
    dt = 0.25
    total_steps = 60
    t_all = np.arange(total_steps) * dt
    all_pos = np.stack([1.4 * t_all, np.zeros_like(t_all)], axis=-1)

    raw_preds = []
    spline_preds = []
    temporal_preds = []

    prev_temporal = None
    temporal_alpha = 0.25

    print("Running inference on 55 frames...")
    for step in range(5, total_steps):
        hist_start = max(0, step - 20)
        positions = all_pos[hist_start:step + 1]
        hist_np, origin = build_history(positions, dt)

        preds_raw = predict(model, schedule, hist_np, device, K=20)  # (K, 20, 2)
        preds_raw_abs = preds_raw + origin

        preds_spline = smooth_predictions(preds_raw.copy(), s_factor=12.0)
        preds_spline_abs = preds_spline + origin

        # Temporal EMA
        if prev_temporal is not None:
            preds_temporal_abs = temporal_alpha * preds_spline_abs + (1 - temporal_alpha) * prev_temporal
        else:
            preds_temporal_abs = preds_spline_abs.copy()
        prev_temporal = preds_temporal_abs.copy()

        raw_preds.append(preds_raw_abs)
        spline_preds.append(preds_spline_abs)
        temporal_preds.append(preds_temporal_abs)

        if (step - 4) % 10 == 0:
            print(f"  frame {step - 4}/55")

    print("Analyzing...")

    # --- Metric 1: Per-trajectory jerk ---
    raw_jerks = [compute_jerk(p.mean(axis=0)) for p in raw_preds]
    spline_jerks = [compute_jerk(p.mean(axis=0)) for p in spline_preds]
    temporal_jerks = [compute_jerk(p.mean(axis=0)) for p in temporal_preds]

    # --- Metric 2: Frame-to-frame shift of mean prediction ---
    raw_shifts = compute_frame_to_frame_shift(raw_preds)
    spline_shifts = compute_frame_to_frame_shift(spline_preds)
    temporal_shifts = compute_frame_to_frame_shift(temporal_preds)

    # --- Metric 3: Sample spread (std across K samples at each timestep) ---
    raw_spread = [p.std(axis=0).mean() for p in raw_preds]
    spline_spread = [p.std(axis=0).mean() for p in spline_preds]
    temporal_spread = [p.std(axis=0).mean() for p in temporal_preds]

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("SMOOTHNESS ANALYSIS (straight walk scenario, 55 frames)")
    print("=" * 60)
    print(f"\n{'Metric':<35} {'Raw':>8} {'Spline':>8} {'Temporal':>8}")
    print("-" * 60)
    print(f"{'Mean jerk (m/s³)':<35} {np.mean(raw_jerks):>8.3f} {np.mean(spline_jerks):>8.3f} {np.mean(temporal_jerks):>8.3f}")
    print(f"{'Mean frame-to-frame shift (m)':<35} {np.mean(raw_shifts):>8.3f} {np.mean(spline_shifts):>8.3f} {np.mean(temporal_shifts):>8.3f}")
    print(f"{'Mean sample spread (m)':<35} {np.mean(raw_spread):>8.3f} {np.mean(spline_spread):>8.3f} {np.mean(temporal_spread):>8.3f}")
    print(f"{'Max frame-to-frame shift (m)':<35} {np.max(raw_shifts):>8.3f} {np.max(spline_shifts):>8.3f} {np.max(temporal_shifts):>8.3f}")
    print()

    # --- Generate diagnostic plot ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Trajectory fans at frame 25 (middle of sequence)
    mid = 25
    for col, (preds, title) in enumerate([
        (raw_preds[mid], "Raw Diffusion Output"),
        (spline_preds[mid], f"Spline Smoothed (s=12)"),
        (temporal_preds[mid], "Spline + Temporal EMA"),
    ]):
        ax = axes[0, col]
        # Plot all K samples
        for k in range(preds.shape[0]):
            ax.plot(preds[k, :, 0], preds[k, :, 1], alpha=0.2, color="orange", linewidth=0.8)
        # Mean
        mean_traj = preds.mean(axis=0)
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], color="red", linewidth=2, label="Mean")
        # Ground truth (straight line)
        gt_x = all_pos[mid + 5 + 1:mid + 5 + 21, 0]
        gt_y = all_pos[mid + 5 + 1:mid + 5 + 21, 1]
        ax.plot(gt_x, gt_y, "--", color="green", linewidth=2, label="GT")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_aspect("equal")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    # Row 2: Time series metrics
    frames = np.arange(len(raw_jerks))

    ax = axes[1, 0]
    ax.plot(frames, raw_jerks, label="Raw", alpha=0.7)
    ax.plot(frames, spline_jerks, label="Spline", alpha=0.7)
    ax.plot(frames, temporal_jerks, label="Temporal", alpha=0.7)
    ax.set_title("Per-frame Jerk (mean trajectory)", fontweight="bold")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Jerk (m/s³)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(raw_shifts, label="Raw", alpha=0.7)
    ax.plot(spline_shifts, label="Spline", alpha=0.7)
    ax.plot(temporal_shifts, label="Temporal", alpha=0.7)
    ax.set_title("Frame-to-Frame Shift (mean pred)", fontweight="bold")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Shift (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(frames, raw_spread, label="Raw", alpha=0.7)
    ax.plot(frames, spline_spread, label="Spline", alpha=0.7)
    ax.plot(frames, temporal_spread, label="Temporal", alpha=0.7)
    ax.set_title("Sample Spread (std across K)", fontweight="bold")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Spread (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "../../figures/smoothness_analysis.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()

    # --- Also test higher smoothing values ---
    print("\n\nTesting different s_factor values on frame 25:")
    print(f"{'s_factor':<10} {'Mean Jerk':>12} {'Spread':>10}")
    print("-" * 35)
    preds_raw_25 = raw_preds[mid] - all_pos[mid + 5]  # back to ego frame for fair comparison
    for s in [5.0, 12.0, 25.0, 50.0, 100.0, 200.0]:
        smoothed = smooth_predictions(preds_raw_25.copy(), s_factor=s)
        j = compute_jerk(smoothed.mean(axis=0))
        sp = smoothed.std(axis=0).mean()
        print(f"{s:<10.0f} {j:>12.3f} {sp:>10.3f}")


if __name__ == "__main__":
    main()
