#!/usr/bin/env python3
"""Fine-tune the diffusion predictor on ETH/UCY data.

Uses pre-trained AV2 weights and fine-tunes with lower LR on ETH/UCY.

Usage:
    # Single-agent
    python scripts/finetune_eth_ucy.py \
        --data data/eth_ucy_processed \
        --ckpt models/diffusion/av2_pretrain_v2/ema_best.pt \
        --mode single --epochs 50

    # Joint
    python scripts/finetune_eth_ucy.py \
        --data data/eth_ucy_processed \
        --ckpt models/diffusion/av2_joint_v2/ema_best.pt \
        --mode joint --epochs 50
"""

import argparse
import os
import pathlib
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion_prediction.ddpm import CosineSchedule, ddim_sample_loop, ddim_sample_loop_joint
from diffusion_prediction.train import EMA


def load_model(args, device):
    """Load model architecture and pre-trained weights."""
    if args.mode == "joint":
        from diffusion_prediction.model_joint import JointTrajectoryDenoiser
        model = JointTrajectoryDenoiser(
            d=args.d_model, max_agents=args.max_agents,
            nhead=args.nhead, num_enc_layers=args.num_enc_layers,
            num_dec_layers=args.num_dec_layers,
            num_interaction_layers=args.num_interaction_layers,
            dim_ff=args.dim_ff,
        ).to(device)
    else:
        from diffusion_prediction.model import TrajectoryDenoiser
        model = TrajectoryDenoiser(
            d=args.d_model, nhead=args.nhead,
            num_enc_layers=args.num_enc_layers,
            num_dec_layers=args.num_dec_layers,
            dim_ff=args.dim_ff,
        ).to(device)

    if args.ckpt and os.path.exists(args.ckpt):
        state = torch.load(args.ckpt, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
        print(f"Loaded pre-trained weights: {args.ckpt}")
    else:
        print("[WARN] No pre-trained weights — training from scratch")

    return model


@torch.no_grad()
def validate_single(model, schedule, val_loader, K, device):
    """Validate single-agent model."""
    model.eval()
    all_ade, all_fde = [], []

    for batch in val_loader:
        hist = batch["history"].to(device)
        mask = batch["history_mask"].to(device)
        ego = batch["ego_vel"].to(device)
        gt = batch["future"].to(device)

        futures = ddim_sample_loop(model, schedule, hist, mask, ego, K=K)
        # futures: (B, K, 20, 2)
        gt_exp = gt.unsqueeze(1)  # (B, 1, 20, 2)

        ade = torch.norm(futures - gt_exp, dim=-1).mean(dim=-1)  # (B, K)
        fde = torch.norm(futures[:, :, -1, :] - gt_exp[:, :, -1, :], dim=-1)  # (B, K)

        all_ade.append(ade.min(dim=1).values)
        all_fde.append(fde.min(dim=1).values)

    model.train()
    return {
        "minADE_20": torch.cat(all_ade).mean().item(),
        "minFDE_20": torch.cat(all_fde).mean().item(),
    }


@torch.no_grad()
def validate_joint(model, schedule, val_loader, K, device):
    """Validate joint model."""
    model.eval()
    all_ade, all_fde = [], []

    for batch in val_loader:
        hist = batch["history"].to(device)
        mask = batch["history_mask"].to(device)
        amask = batch["agent_mask"].to(device)
        ego = batch["ego_vel"].to(device)
        gt = batch["future"].to(device)

        B, M = hist.shape[:2]
        futures = ddim_sample_loop_joint(model, schedule, hist, mask, amask, ego, K=K)
        # (B, K, M, 20, 2)

        gt_exp = gt.unsqueeze(1)  # (B, 1, M, 20, 2)
        ade = torch.norm(futures - gt_exp, dim=-1).mean(dim=-1)  # (B, K, M)
        fde = torch.norm(futures[:, :, :, -1, :] - gt_exp[:, :, :, -1, :], dim=-1)

        min_ade = ade.min(dim=1).values  # (B, M)
        min_fde = fde.min(dim=1).values

        for b in range(B):
            for m in range(M):
                if amask[b, m] > 0:
                    all_ade.append(min_ade[b, m].cpu())
                    all_fde.append(min_fde[b, m].cpu())

    model.train()
    if not all_ade:
        return {"minADE_20": float("inf"), "minFDE_20": float("inf")}
    return {
        "minADE_20": torch.stack(all_ade).mean().item(),
        "minFDE_20": torch.stack(all_fde).mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on ETH/UCY")
    parser.add_argument("--data", type=str, default="data/eth_ucy_processed")
    parser.add_argument("--ckpt", type=str, required=True, help="Pre-trained checkpoint")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "joint"])
    parser.add_argument("--output", type=str, default="models/diffusion")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5, help="Lower LR for fine-tuning")
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--diffusion-T", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--max-agents", type=int, default=16)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-enc-layers", type=int, default=6)
    parser.add_argument("--num-dec-layers", type=int, default=4)
    parser.add_argument("--num-interaction-layers", type=int, default=3)
    parser.add_argument("--dim-ff", type=int, default=512)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Fine-tuning ({args.mode}) on ETH/UCY")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    data_dir = pathlib.Path(args.data)
    if args.mode == "single":
        from diffusion_prediction.dataset import TrajectoryDataset
        train_ds = TrajectoryDataset(str(data_dir / "train"), augment=True)
        val_ds = TrajectoryDataset(str(data_dir / "val"), augment=False)
        collate = None
    else:
        from diffusion_prediction.dataset_joint import JointTrajectoryDataset, collate_fn_joint
        train_ds = JointTrajectoryDataset(
            str(data_dir / "train_joint"), augment=True, max_agents=args.max_agents,
        )
        val_ds = JointTrajectoryDataset(
            str(data_dir / "val_joint"), augment=False, max_agents=args.max_agents,
        )
        collate = collate_fn_joint

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate,
    )

    print(f"  Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_ds)} samples")

    # Model
    model = load_model(args, device)
    schedule = CosineSchedule(T=args.diffusion_T).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min,
    )
    ema = EMA(model, decay=args.ema_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    # Output
    run_name = f"eth_ucy_ft_{args.mode}"
    ckpt_dir = pathlib.Path(args.output) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_fde = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            if args.mode == "single":
                hist = batch["history"].to(device)
                mask = batch["history_mask"].to(device)
                ego = batch["ego_vel"].to(device)
                fut = batch["future"].to(device)

                B = hist.shape[0]
                t = torch.randint(0, args.diffusion_T, (B,), device=device)
                eps = torch.randn_like(fut)

                y_t = schedule.q_sample(fut, t, eps)

                with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                    eps_hat = model(hist, mask, ego, t, y_t)
                    loss = F.mse_loss(eps_hat, eps)
            else:
                hist = batch["history"].to(device)
                mask = batch["history_mask"].to(device)
                amask = batch["agent_mask"].to(device)
                ego = batch["ego_vel"].to(device)
                fut = batch["future"].to(device)

                B, M = hist.shape[:2]
                t = torch.randint(0, args.diffusion_T, (B,), device=device)
                eps = torch.randn_like(fut) * amask[:, :, None, None]

                y_t = schedule.q_sample(fut, t, eps)
                y_t = y_t * amask[:, :, None, None]

                with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                    eps_hat = model(hist, mask, amask, ego, t, y_t)
                    per_agent_loss = F.mse_loss(eps_hat, eps, reduction="none")
                    loss_mask = amask.float().unsqueeze(-1).unsqueeze(-1)
                    masked_loss = (per_agent_loss * loss_mask).sum()
                    n_real = loss_mask.sum() * 20 * 2
                    loss = masked_loss / n_real.clamp(min=1)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            epoch_loss += loss.item()

        lr_scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)
        elapsed = time.time() - t0

        print(
            f"[epoch {epoch:3d}/{args.epochs}] "
            f"loss={avg_loss:.5f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"time={elapsed:.1f}s"
        )

        # Validate
        if epoch % args.val_every == 0 or epoch == args.epochs:
            ema.apply(model)
            if args.mode == "single":
                metrics = validate_single(model, schedule, val_loader, K=20, device=device)
            else:
                metrics = validate_joint(model, schedule, val_loader, K=20, device=device)
            ema.restore(model)

            print(
                f"  [val] minADE-20={metrics['minADE_20']:.4f}  "
                f"minFDE-20={metrics['minFDE_20']:.4f}"
            )

            if metrics["minFDE_20"] < best_fde:
                best_fde = metrics["minFDE_20"]
                torch.save(ema.state_dict(), ckpt_dir / "ema_best.pt")
                print(f"  -> new best FDE={best_fde:.4f}, saved ema_best.pt")

    # Save final
    torch.save(ema.state_dict(), ckpt_dir / "ema_final.pt")
    print(f"\nDone! Best minFDE-20 = {best_fde:.4f}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
