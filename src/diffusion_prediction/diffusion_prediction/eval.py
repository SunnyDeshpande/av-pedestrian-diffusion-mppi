#!/usr/bin/env python3
"""Evaluation and latency benchmarking for the diffusion predictor."""

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_prediction.model import TrajectoryDenoiser
from diffusion_prediction.ddpm import CosineSchedule, ddim_sample_loop
from diffusion_prediction.dataset import TrajectoryDataset, collate_fn


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
) -> dict:
    """Compute minADE-K, minFDE-K, and miss-rate@2m.

    Parameters
    ----------
    predictions  : (N, K, T_fut, 2)
    ground_truth : (N, T_fut, 2)
    """
    gt = ground_truth.unsqueeze(1)  # (N, 1, T_fut, 2)

    # ADE per sample: mean over time of L2
    ade = torch.norm(predictions - gt, dim=-1).mean(dim=-1)  # (N, K)
    min_ade = ade.min(dim=1).values.mean().item()

    # FDE per sample: L2 at last timestep
    fde = torch.norm(predictions[:, :, -1, :] - gt[:, :, -1, :], dim=-1)  # (N, K)
    min_fde_vals = fde.min(dim=1).values
    min_fde = min_fde_vals.mean().item()

    # Miss rate: fraction where best FDE > 2.0 m
    miss = (min_fde_vals > 2.0).float().mean().item()

    return {
        "minADE_20": min_ade,
        "minFDE_20": min_fde,
        "miss_2m": miss,
    }


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def benchmark_latency(
    model: TrajectoryDenoiser,
    schedule: CosineSchedule,
    M: int = 8,
    K: int = 20,
    device: torch.device = torch.device("cuda"),
    num_iters: int = 100,
    warmup: int = 10,
) -> dict:
    """Profile DDIM inference latency.

    Parameters
    ----------
    M : number of pedestrians
    K : samples per pedestrian
    """
    model.eval()
    T_hist, T_fut = 20, 20

    # Dummy inputs
    hist = torch.randn(M, T_hist, 4, device=device)
    mask = torch.ones(M, T_hist, device=device)
    ego = torch.randn(M, 2, device=device)

    # Warmup
    for _ in range(warmup):
        _ = ddim_sample_loop(model, schedule, hist, mask, ego, K=K)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    latencies = []
    for _ in range(num_iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = ddim_sample_loop(model, schedule, hist, mask, ego, K=K)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    return {
        "mean_ms": latencies.mean(),
        "std_ms": latencies.std(),
        "p50_ms": np.median(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "max_ms": latencies.max(),
    }


# ---------------------------------------------------------------------------
# Eval on dataset
# ---------------------------------------------------------------------------

def evaluate_dataset(
    model: TrajectoryDenoiser,
    schedule: CosineSchedule,
    data_dir: str,
    K: int = 20,
    batch_size: int = 32,
    device: torch.device = torch.device("cuda"),
    num_workers: int = 4,
):
    ds = TrajectoryDataset(data_dir, augment=False)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
    )

    model.eval()
    all_preds = []
    all_gt = []

    with torch.no_grad():
        for batch in loader:
            hist = batch["history"].to(device)
            mask = batch["history_mask"].to(device)
            ego = batch["ego_vel"].to(device)
            gt = batch["future"].to(device)

            futures = ddim_sample_loop(model, schedule, hist, mask, ego, K=K)
            all_preds.append(futures.cpu())
            all_gt.append(gt.cpu())

    preds = torch.cat(all_preds, dim=0)
    gt = torch.cat(all_gt, dim=0)
    return compute_metrics(preds, gt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate diffusion predictor")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (EMA .pt)")
    parser.add_argument("--data", type=str, default=None, help="Path to val data shards")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-ff", type=int, default=256)

    # Benchmark mode
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark only")
    parser.add_argument("--M", type=int, default=8, help="Number of pedestrians for benchmark")
    parser.add_argument("--num-iters", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Build model
    model = TrajectoryDenoiser(
        d=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
    ).to(device)

    # Load checkpoint if provided
    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
        print(f"[eval] loaded checkpoint: {args.ckpt}")
    else:
        print("[eval] no checkpoint provided, using random weights")

    schedule = CosineSchedule(T=100).to(device)

    if args.benchmark:
        print(f"\n{'='*50}")
        print(f"Latency Benchmark: M={args.M}, K={args.K}, DDIM-10")
        print(f"Device: {device}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"{'='*50}\n")

        results = benchmark_latency(
            model, schedule,
            M=args.M, K=args.K, device=device,
            num_iters=args.num_iters,
        )

        for k, v in results.items():
            print(f"  {k:>10s}: {v:7.2f}")

        target = 30.0
        ceiling = 50.0
        status = "PASS" if results["mean_ms"] < target else (
            "WARN" if results["mean_ms"] < ceiling else "FAIL"
        )
        print(f"\n  Target < {target:.0f} ms: {status} (mean={results['mean_ms']:.2f} ms)")
        return

    # Dataset evaluation
    if not args.data:
        print("[eval] --data required for dataset evaluation (or use --benchmark)")
        return

    print(f"\n[eval] evaluating on {args.data} with K={args.K}")
    metrics = evaluate_dataset(
        model, schedule, args.data,
        K=args.K, batch_size=args.batch_size,
        device=device, num_workers=args.num_workers,
    )

    print(f"\n{'='*40}")
    print(f"  minADE-{args.K}:    {metrics['minADE_20']:.4f} m")
    print(f"  minFDE-{args.K}:    {metrics['minFDE_20']:.4f} m")
    print(f"  miss-rate@2m: {metrics['miss_2m']:.4f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
