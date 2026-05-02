#!/usr/bin/env python3
"""Full diagnostic of diffusion model samples during visualization run.

Examines raw DDIM output before any smoothing to understand noise sources:
- Per-sample trajectory statistics (length, curvature, endpoint spread)
- Outlier frequency and magnitude
- Per-timestep noise distribution (which future steps are noisiest?)
- Cross-sample correlation (are samples diverse or clustered?)
- Frame-to-frame raw prediction stability
- Scenario-level breakdown (straight vs curve vs stopping vs swerve)
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diffusion_prediction.model import TrajectoryDenoiser
from diffusion_prediction.ddpm import CosineSchedule, ddim_sample_loop

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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


# --- Trajectory generators (same as demo_live.py) ---
def _traj_straight(t):
    return np.stack([1.4 * t, np.zeros_like(t)], axis=-1)

def _traj_curve(t):
    R, omega = 6.0, 0.25
    theta = omega * t
    return np.stack([R * np.sin(theta), R * (1 - np.cos(theta))], axis=-1)

def _traj_stopping(t):
    v0, decel = 1.5, 0.25
    vt = np.clip(v0 - decel * t, 0, None)
    x = np.cumsum(vt) * 0.25
    return np.stack([x, 0.3 * np.sin(0.4 * t)], axis=-1)

def _traj_swerve(t):
    return np.stack([1.2 * t, 2.0 * np.sin(0.35 * t)], axis=-1)


SCENARIOS = [
    {"name": "Straight walk",    "gen": _traj_straight,  "total_s": 15.0},
    {"name": "Turning left",     "gen": _traj_curve,     "total_s": 15.0},
    {"name": "Stopping",         "gen": _traj_stopping,  "total_s": 12.0},
    {"name": "Swerving",         "gen": _traj_swerve,    "total_s": 15.0},
]


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

    dt = 0.25
    K = 20

    # Collect all raw predictions per scenario
    all_data = {}  # scenario_name -> list of dicts per frame

    for sc in SCENARIOS:
        total_steps = int(sc["total_s"] / dt)
        t_all = np.arange(total_steps) * dt
        all_pos = sc["gen"](t_all)

        frames = []
        for step in range(5, total_steps):
            hist_start = max(0, step - 20)
            positions = all_pos[hist_start:step + 1]
            hist_np, origin = build_history(positions, dt)

            preds_ego = predict(model, schedule, hist_np, device, K=K)  # (K, 20, 2)

            # Ground truth future in ego frame
            gt_ego = None
            if step + 20 <= total_steps:
                gt_abs = all_pos[step + 1:step + 21]
                gt_ego = gt_abs - origin

            frames.append({
                "step": step,
                "preds_ego": preds_ego,
                "gt_ego": gt_ego,
                "origin": origin,
                "hist_ego": hist_np,
            })

        all_data[sc["name"]] = frames
        print(f"  {sc['name']}: {len(frames)} frames")

    print("\nAnalyzing samples...\n")

    # ================================================================
    # ANALYSIS 1: Per-sample statistics across all scenarios
    # ================================================================
    print("=" * 70)
    print("1. PER-SAMPLE TRAJECTORY STATISTICS")
    print("=" * 70)

    for sc_name, frames in all_data.items():
        all_preds = np.concatenate([f["preds_ego"] for f in frames], axis=0)  # (N*K, 20, 2)
        N = len(frames)

        # Trajectory length (sum of segment lengths)
        diffs = np.diff(all_preds, axis=1)  # (N*K, 19, 2)
        seg_lengths = np.linalg.norm(diffs, axis=-1)  # (N*K, 19)
        traj_lengths = seg_lengths.sum(axis=1)  # (N*K,)

        # Endpoint distance from origin
        endpoint_dist = np.linalg.norm(all_preds[:, -1, :], axis=-1)

        # Max lateral deviation (y)
        max_lateral = np.abs(all_preds[:, :, 1]).max(axis=1)

        # Curvature: mean absolute angle change
        angles = np.arctan2(diffs[:, :, 1], diffs[:, :, 0])  # (N*K, 19)
        angle_changes = np.abs(np.diff(angles, axis=1))  # (N*K, 18)
        angle_changes = np.minimum(angle_changes, 2 * np.pi - angle_changes)
        mean_curvature = angle_changes.mean(axis=1)

        print(f"\n  [{sc_name}] ({N} frames x {K} samples = {N*K} trajectories)")
        print(f"    Traj length  : mean={traj_lengths.mean():.2f}m  std={traj_lengths.std():.2f}  "
              f"min={traj_lengths.min():.2f}  max={traj_lengths.max():.2f}")
        print(f"    Endpoint dist: mean={endpoint_dist.mean():.2f}m  std={endpoint_dist.std():.2f}  "
              f"max={endpoint_dist.max():.2f}")
        print(f"    Max |y| dev  : mean={max_lateral.mean():.2f}m  std={max_lateral.std():.2f}  "
              f"max={max_lateral.max():.2f}")
        print(f"    Mean curv    : mean={mean_curvature.mean():.4f}rad  std={mean_curvature.std():.4f}  "
              f"max={mean_curvature.max():.4f}")

    # ================================================================
    # ANALYSIS 2: Outlier breakdown
    # ================================================================
    print("\n" + "=" * 70)
    print("2. OUTLIER ANALYSIS (per frame, MAD-based)")
    print("=" * 70)

    for sc_name, frames in all_data.items():
        outlier_counts = []
        outlier_magnitudes = []
        for f in frames:
            preds = f["preds_ego"]  # (K, 20, 2)
            flat = preds.reshape(K, -1)
            median = np.median(flat, axis=0)
            dists = np.linalg.norm(flat - median[None], axis=-1)
            med_dist = np.median(dists)
            mad = np.median(np.abs(dists - med_dist))
            threshold = med_dist + 3.0 * max(mad, 0.1)
            outliers = dists > threshold
            outlier_counts.append(outliers.sum())
            if outliers.any():
                outlier_magnitudes.extend(dists[outliers].tolist())

        outlier_counts = np.array(outlier_counts)
        print(f"\n  [{sc_name}]")
        print(f"    Frames with outliers: {(outlier_counts > 0).sum()}/{len(frames)} "
              f"({100*(outlier_counts > 0).mean():.0f}%)")
        print(f"    Outliers per frame : mean={outlier_counts.mean():.1f}  max={outlier_counts.max()}")
        if outlier_magnitudes:
            mags = np.array(outlier_magnitudes)
            print(f"    Outlier magnitude  : mean={mags.mean():.2f}  max={mags.max():.2f}")
        print(f"    Total outlier rate : {outlier_counts.sum()}/{len(frames)*K} "
              f"({100*outlier_counts.sum()/(len(frames)*K):.1f}%)")

    # ================================================================
    # ANALYSIS 3: Per-timestep noise (which future steps are noisiest?)
    # ================================================================
    print("\n" + "=" * 70)
    print("3. PER-TIMESTEP NOISE (std across K samples at each future step)")
    print("=" * 70)

    per_step_data = {}
    for sc_name, frames in all_data.items():
        # (N, K, 20, 2) -> std across K -> mean across N
        all_preds = np.stack([f["preds_ego"] for f in frames])  # (N, K, 20, 2)
        std_per_step = all_preds.std(axis=1)  # (N, 20, 2)
        mean_std = std_per_step.mean(axis=0)  # (20, 2)
        total_std = np.linalg.norm(mean_std, axis=-1)  # (20,)
        per_step_data[sc_name] = {"x": mean_std[:, 0], "y": mean_std[:, 1], "total": total_std}

        print(f"\n  [{sc_name}]")
        print(f"    Step   Std_x    Std_y    Total")
        for t in [0, 4, 9, 14, 19]:
            print(f"    t={t+1:2d}   {mean_std[t,0]:.3f}m   {mean_std[t,1]:.3f}m   {total_std[t]:.3f}m")

    # ================================================================
    # ANALYSIS 4: Cross-sample diversity (pairwise distances)
    # ================================================================
    print("\n" + "=" * 70)
    print("4. SAMPLE DIVERSITY (mean pairwise endpoint distance)")
    print("=" * 70)

    for sc_name, frames in all_data.items():
        pairwise_dists = []
        for f in frames:
            endpoints = f["preds_ego"][:, -1, :]  # (K, 2)
            # Pairwise distances
            for i in range(K):
                for j in range(i + 1, K):
                    pairwise_dists.append(np.linalg.norm(endpoints[i] - endpoints[j]))
        pd = np.array(pairwise_dists)
        print(f"  [{sc_name}] mean={pd.mean():.3f}m  std={pd.std():.3f}  "
              f"p25={np.percentile(pd, 25):.3f}  p75={np.percentile(pd, 75):.3f}  "
              f"max={pd.max():.3f}")

    # ================================================================
    # ANALYSIS 5: Frame-to-frame prediction stability (raw)
    # ================================================================
    print("\n" + "=" * 70)
    print("5. FRAME-TO-FRAME STABILITY (mean trajectory shift between frames)")
    print("=" * 70)

    stability_data = {}
    for sc_name, frames in all_data.items():
        shifts = []
        for i in range(1, len(frames)):
            prev_mean = frames[i-1]["preds_ego"].mean(axis=0)  # (20, 2)
            curr_mean = frames[i]["preds_ego"].mean(axis=0)
            # The frames share different origins, so compare in absolute coords
            prev_abs = prev_mean + frames[i-1]["origin"]
            curr_abs = curr_mean + frames[i]["origin"]
            shift = np.linalg.norm(curr_abs - prev_abs, axis=-1)  # (20,)
            shifts.append(shift)
        shifts = np.array(shifts)  # (N-1, 20)
        stability_data[sc_name] = shifts

        mean_shift = shifts.mean(axis=0)  # (20,) per-step mean shift
        print(f"\n  [{sc_name}]")
        print(f"    Overall mean shift: {shifts.mean():.3f}m  max: {shifts.max():.3f}m")
        print(f"    Step   Mean Shift")
        for t in [0, 4, 9, 14, 19]:
            print(f"    t={t+1:2d}   {mean_shift[t]:.3f}m")

    # ================================================================
    # ANALYSIS 6: Prediction accuracy vs GT
    # ================================================================
    print("\n" + "=" * 70)
    print("6. PREDICTION ACCURACY (ADE/FDE vs ground truth, raw samples)")
    print("=" * 70)

    for sc_name, frames in all_data.items():
        ade_all = []
        fde_all = []
        min_ade_all = []
        min_fde_all = []
        for f in frames:
            if f["gt_ego"] is None:
                continue
            gt = f["gt_ego"]  # (T_gt, 2)
            preds = f["preds_ego"]  # (K, 20, 2)
            T_gt = min(gt.shape[0], preds.shape[1])
            gt_crop = gt[:T_gt]
            preds_crop = preds[:, :T_gt]
            # Per-sample ADE and FDE
            ade = np.linalg.norm(preds_crop - gt_crop[None], axis=-1).mean(axis=1)  # (K,)
            fde = np.linalg.norm(preds_crop[:, -1] - gt_crop[-1], axis=-1)  # (K,)
            ade_all.extend(ade.tolist())
            fde_all.extend(fde.tolist())
            min_ade_all.append(ade.min())
            min_fde_all.append(fde.min())

        ade_arr = np.array(ade_all)
        fde_arr = np.array(fde_all)
        min_ade_arr = np.array(min_ade_all)
        min_fde_arr = np.array(min_fde_all)
        print(f"\n  [{sc_name}]")
        print(f"    All-sample ADE : mean={ade_arr.mean():.3f}m  std={ade_arr.std():.3f}  "
              f"median={np.median(ade_arr):.3f}")
        print(f"    All-sample FDE : mean={fde_arr.mean():.3f}m  std={fde_arr.std():.3f}  "
              f"median={np.median(fde_arr):.3f}")
        print(f"    minADE (best K): mean={min_ade_arr.mean():.3f}m")
        print(f"    minFDE (best K): mean={min_fde_arr.mean():.3f}m")

    # ================================================================
    # PLOT: Comprehensive diagnostic figure
    # ================================================================
    fig = plt.figure(figsize=(22, 20))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)

    sc_names = list(all_data.keys())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # --- Row 1: Sample fans at mid-frame for each scenario ---
    for col, sc_name in enumerate(sc_names):
        ax = fig.add_subplot(gs[0, col])
        frames = all_data[sc_name]
        mid = len(frames) // 2
        preds = frames[mid]["preds_ego"]
        gt = frames[mid]["gt_ego"]

        for k in range(K):
            ax.plot(preds[k, :, 0], preds[k, :, 1], alpha=0.25, color="orange", linewidth=0.7)
        mean_traj = preds.mean(axis=0)
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], color="red", linewidth=2, label="Mean")
        median_traj = np.median(preds, axis=0)
        ax.plot(median_traj[:, 0], median_traj[:, 1], color="purple", linewidth=2,
                linestyle="--", label="Median")
        if gt is not None:
            ax.plot(gt[:, 0], gt[:, 1], "--", color="green", linewidth=2, label="GT")
        ax.plot(0, 0, "ks", markersize=8)
        ax.set_title(f"{sc_name}\n(frame {mid})", fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x (m)", fontsize=8)
        ax.set_ylabel("y (m)", fontsize=8)
        if col == 0:
            ax.legend(fontsize=7)

    # --- Row 2: Per-timestep std (noise growth over prediction horizon) ---
    ax = fig.add_subplot(gs[1, 0:2])
    for i, sc_name in enumerate(sc_names):
        d = per_step_data[sc_name]
        ax.plot(range(1, 21), d["total"], color=colors[i], linewidth=2, label=sc_name)
    ax.set_title("Per-Timestep Prediction Std (noise growth)", fontweight="bold")
    ax.set_xlabel("Future timestep")
    ax.set_ylabel("Std across K samples (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # X vs Y noise breakdown
    ax = fig.add_subplot(gs[1, 2:4])
    for i, sc_name in enumerate(sc_names):
        d = per_step_data[sc_name]
        ax.plot(range(1, 21), d["x"], color=colors[i], linewidth=2, linestyle="-",
                label=f"{sc_name} x")
        ax.plot(range(1, 21), d["y"], color=colors[i], linewidth=2, linestyle="--",
                label=f"{sc_name} y")
    ax.set_title("X vs Y Noise per Timestep", fontweight="bold")
    ax.set_xlabel("Future timestep")
    ax.set_ylabel("Std (m)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Row 3: Frame-to-frame stability per step ---
    ax = fig.add_subplot(gs[2, 0:2])
    for i, sc_name in enumerate(sc_names):
        shifts = stability_data[sc_name]
        mean_shift = shifts.mean(axis=0)
        ax.plot(range(1, 21), mean_shift, color=colors[i], linewidth=2, label=sc_name)
    ax.set_title("Frame-to-Frame Shift per Future Timestep", fontweight="bold")
    ax.set_xlabel("Future timestep")
    ax.set_ylabel("Mean shift (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Endpoint scatter for each scenario at mid-frame
    for col, sc_name in enumerate(sc_names):
        if col >= 2:
            break
        ax = fig.add_subplot(gs[2, 2 + col])
        frames = all_data[sc_name]
        mid = len(frames) // 2
        preds = frames[mid]["preds_ego"]
        endpoints = preds[:, -1, :]  # (K, 2)
        gt = frames[mid]["gt_ego"]

        ax.scatter(endpoints[:, 0], endpoints[:, 1], c="orange", s=60,
                   edgecolors="darkorange", zorder=3, label="Sample endpoints")
        ax.plot(preds.mean(axis=0)[-1, 0], preds.mean(axis=0)[-1, 1],
                "r*", markersize=15, zorder=4, label="Mean endpoint")
        if gt is not None:
            ax.plot(gt[-1, 0], gt[-1, 1], "g^", markersize=12, zorder=4, label="GT endpoint")
        ax.set_title(f"Endpoint Scatter: {sc_name}", fontsize=10, fontweight="bold")
        ax.set_aspect("equal")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # --- Row 4: Histograms ---
    # ADE distribution
    ax = fig.add_subplot(gs[3, 0])
    for i, sc_name in enumerate(sc_names):
        frames = all_data[sc_name]
        ades = []
        for f in frames:
            if f["gt_ego"] is None:
                continue
            gt = f["gt_ego"]
            preds = f["preds_ego"]
            T_gt = min(gt.shape[0], preds.shape[1])
            ade = np.linalg.norm(preds[:, :T_gt] - gt[None, :T_gt], axis=-1).mean(axis=1)
            ades.extend(ade.tolist())
        ax.hist(ades, bins=40, alpha=0.5, color=colors[i], label=sc_name, density=True)
    ax.set_title("ADE Distribution (all samples)", fontweight="bold")
    ax.set_xlabel("ADE (m)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # FDE distribution
    ax = fig.add_subplot(gs[3, 1])
    for i, sc_name in enumerate(sc_names):
        frames = all_data[sc_name]
        fdes = []
        for f in frames:
            if f["gt_ego"] is None:
                continue
            gt = f["gt_ego"]
            preds = f["preds_ego"]
            T_gt = min(gt.shape[0], preds.shape[1])
            fde = np.linalg.norm(preds[:, T_gt-1] - gt[T_gt-1], axis=-1)
            fdes.extend(fde.tolist())
        ax.hist(fdes, bins=40, alpha=0.5, color=colors[i], label=sc_name, density=True)
    ax.set_title("FDE Distribution (all samples)", fontweight="bold")
    ax.set_xlabel("FDE (m)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Trajectory length distribution
    ax = fig.add_subplot(gs[3, 2])
    for i, sc_name in enumerate(sc_names):
        frames = all_data[sc_name]
        all_preds = np.concatenate([f["preds_ego"] for f in frames], axis=0)
        diffs = np.diff(all_preds, axis=1)
        traj_lengths = np.linalg.norm(diffs, axis=-1).sum(axis=1)
        ax.hist(traj_lengths, bins=40, alpha=0.5, color=colors[i], label=sc_name, density=True)
    ax.set_title("Trajectory Length Distribution", fontweight="bold")
    ax.set_xlabel("Length (m)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Per-sample max |y| deviation
    ax = fig.add_subplot(gs[3, 3])
    for i, sc_name in enumerate(sc_names):
        frames = all_data[sc_name]
        all_preds = np.concatenate([f["preds_ego"] for f in frames], axis=0)
        max_y = np.abs(all_preds[:, :, 1]).max(axis=1)
        ax.hist(max_y, bins=40, alpha=0.5, color=colors[i], label=sc_name, density=True)
    ax.set_title("Max |y| Deviation Distribution", fontweight="bold")
    ax.set_xlabel("|y| (m)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    out_path = "../../figures/sample_diagnosis.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
