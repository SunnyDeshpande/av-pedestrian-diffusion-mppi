#!/usr/bin/env python3
"""Standalone latency benchmark for the diffusion predictor.

Usage:
    python scripts/bench_latency.py --M 8 --K 20 --device cuda:0
    python scripts/bench_latency.py --ckpt models/diffusion/.../ema_final.pt
"""

import argparse
import sys
import os

# Allow running from repo root or scripts/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from diffusion_prediction.model import TrajectoryDenoiser
from diffusion_prediction.ddpm import CosineSchedule
from diffusion_prediction.eval import benchmark_latency


def main():
    parser = argparse.ArgumentParser(description="Benchmark diffusion predictor latency")
    parser.add_argument("--M", type=int, default=8, help="Number of pedestrians")
    parser.add_argument("--K", type=int, default=20, help="Samples per pedestrian")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (optional)")
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-ff", type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = TrajectoryDenoiser(
        d=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
    ).to(device)

    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.ckpt}")
    else:
        print("Using random weights (no checkpoint)")

    schedule = CosineSchedule(T=100).to(device)

    print(f"\nBenchmark: M={args.M}, K={args.K}, DDIM-10, {args.num_iters} iterations")
    print("-" * 40)

    results = benchmark_latency(
        model, schedule,
        M=args.M, K=args.K, device=device,
        num_iters=args.num_iters,
    )

    for k, v in results.items():
        print(f"  {k:>10s}: {v:7.2f} ms")

    target = 30.0
    ceiling = 50.0
    mean = results["mean_ms"]

    print()
    if mean < target:
        print(f"  PASS: {mean:.2f} ms < {target:.0f} ms target")
    elif mean < ceiling:
        print(f"  WARN: {mean:.2f} ms < {ceiling:.0f} ms ceiling but > {target:.0f} ms target")
        print("  Consider: reduce DDIM steps (10->5), d_model (128->96), or K (20->10)")
    else:
        print(f"  FAIL: {mean:.2f} ms > {ceiling:.0f} ms ceiling")
        print("  Must reduce: DDIM steps, d_model, or K")


if __name__ == "__main__":
    main()
