#!/usr/bin/env python3
"""Training script for the joint multi-agent diffusion predictor."""

import argparse
import os
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion_prediction.model_joint import JointTrajectoryDenoiser
from diffusion_prediction.ddpm import CosineSchedule, ddim_sample_loop_joint
from diffusion_prediction.dataset_joint import JointTrajectoryDataset, collate_fn_joint
from diffusion_prediction.train import EMA


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_joint(
    model: JointTrajectoryDenoiser,
    schedule: CosineSchedule,
    val_loader: DataLoader,
    K: int = 20,
    device: torch.device = torch.device("cuda"),
) -> dict:
    """Compute per-agent minADE-K, minFDE-K, miss@2m on joint predictions."""
    model.eval()

    all_min_ade = []
    all_min_fde = []
    all_miss = []

    for batch in val_loader:
        hist = batch["history"].to(device)           # (B, M, 20, 4)
        mask = batch["history_mask"].to(device)       # (B, M, 20)
        amask = batch["agent_mask"].to(device)         # (B, M)
        ego = batch["ego_vel"].to(device)              # (B, 2)
        gt_fut = batch["future"].to(device)            # (B, M, 20, 2)

        B, M = hist.shape[:2]

        # (B, K, M, 20, 2)
        futures = ddim_sample_loop_joint(
            model, schedule, hist, mask, amask, ego, K=K,
        )

        # Compute metrics per real agent
        # gt: (B, 1, M, 20, 2)
        gt_exp = gt_fut.unsqueeze(1)

        # ADE: mean L2 over time, per agent, per sample
        ade = torch.norm(futures - gt_exp, dim=-1).mean(dim=-1)  # (B, K, M)
        min_ade = ade.min(dim=1).values                           # (B, M)

        # FDE: L2 at last step
        fde = torch.norm(
            futures[:, :, :, -1, :] - gt_exp[:, :, :, -1, :], dim=-1,
        )  # (B, K, M)
        min_fde = fde.min(dim=1).values  # (B, M)

        miss = (min_fde > 2.0).float()  # (B, M)

        # Mask out padding agents
        amask_f = amask.float()  # (B, M)
        for b in range(B):
            for m in range(M):
                if amask_f[b, m] > 0:
                    all_min_ade.append(min_ade[b, m].cpu())
                    all_min_fde.append(min_fde[b, m].cpu())
                    all_miss.append(miss[b, m].cpu())

    model.train()

    if not all_min_ade:
        return {"minADE_20": float("inf"), "minFDE_20": float("inf"), "miss_2m": 1.0}

    return {
        "minADE_20": torch.stack(all_min_ade).mean().item(),
        "minFDE_20": torch.stack(all_min_fde).mean().item(),
        "miss_2m": torch.stack(all_miss).mean().item(),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_dir = pathlib.Path(args.ckpt_dir) / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = pathlib.Path(args.log_dir) / args.run_name

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
    except ImportError:
        print("[train_joint] tensorboard not installed, skipping logging")

    # Data
    train_ds = JointTrajectoryDataset(
        os.path.join(args.data, "train"), augment=True, max_agents=args.max_agents,
    )
    val_ds = JointTrajectoryDataset(
        os.path.join(args.data, "val"), augment=False, max_agents=args.max_agents,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        collate_fn=collate_fn_joint, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        collate_fn=collate_fn_joint,
    )

    print(f"[train_joint] train scenes: {len(train_ds)}, val scenes: {len(val_ds)}")
    print(f"[train_joint] batches/epoch: {len(train_loader)}")

    # Model
    model = JointTrajectoryDenoiser(
        d=args.d_model,
        max_agents=args.max_agents,
        nhead=args.nhead,
        num_enc_layers=args.num_enc_layers,
        num_interaction_layers=args.num_interaction_layers,
        dim_ff=args.dim_ff,
    ).to(device)

    schedule = CosineSchedule(T=args.diffusion_T).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min,
    )

    ema = EMA(model, decay=args.ema_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and "cuda" in args.device))

    # Resume
    start_epoch = 1
    global_step = 0
    best_fde = float("inf")

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        ema.load_state_dict(ckpt["ema_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        lr_scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_fde = ckpt.get("best_fde", float("inf"))
        global_step = ckpt["epoch"] * len(train_loader)
        print(f"[train_joint] resumed from {args.resume}, epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            hist = batch["history"].to(device)           # (B, M, 20, 4)
            mask = batch["history_mask"].to(device)       # (B, M, 20)
            amask = batch["agent_mask"].to(device)         # (B, M)
            ego = batch["ego_vel"].to(device)              # (B, 2)
            fut = batch["future"].to(device)               # (B, M, 20, 2)

            B, M = hist.shape[:2]

            # Diffusion: sample timestep and noise
            t = torch.randint(0, args.diffusion_T, (B,), device=device)
            eps = torch.randn_like(fut)  # (B, M, 20, 2)

            # Zero noise for padding agents
            eps = eps * amask[:, :, None, None]

            # Forward diffusion
            y_t = schedule.q_sample(fut, t, eps)
            y_t = y_t * amask[:, :, None, None]  # zero padding

            # Predict noise
            with torch.amp.autocast("cuda", enabled=(args.amp and "cuda" in args.device)):
                eps_hat = model(hist, mask, amask, ego, t, y_t)

                # Masked MSE loss: only compute loss for real agents
                per_agent_loss = F.mse_loss(
                    eps_hat, eps, reduction="none",
                )  # (B, M, 20, 2)
                # Mask: (B, M, 1, 1)
                loss_mask = amask.float().unsqueeze(-1).unsqueeze(-1)
                masked_loss = (per_agent_loss * loss_mask).sum()
                n_real = loss_mask.sum() * 20 * 2  # total real elements
                loss = masked_loss / n_real.clamp(min=1)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)
            epoch_loss += loss.item()
            global_step += 1

            if writer:
                writer.add_scalar("loss/step", loss.item(), global_step)

        lr_scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)
        elapsed = time.time() - t0

        print(
            f"[epoch {epoch}/{args.epochs}] "
            f"loss={avg_loss:.5f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"time={elapsed:.1f}s"
        )

        if writer:
            writer.add_scalar("loss/epoch_mean", avg_loss, epoch)

        # Validation
        if epoch % args.val_every == 0 or epoch == args.epochs:
            ema.apply(model)
            metrics = validate_joint(model, schedule, val_loader, K=20, device=device)
            ema.restore(model)

            print(
                f"  [val] minADE-20={metrics['minADE_20']:.4f}  "
                f"minFDE-20={metrics['minFDE_20']:.4f}  "
                f"miss@2m={metrics['miss_2m']:.4f}"
            )

            if writer:
                for k, v in metrics.items():
                    writer.add_scalar(f"val/{k}", v, epoch)

            if metrics["minFDE_20"] < best_fde:
                best_fde = metrics["minFDE_20"]
                torch.save(ema.state_dict(), ckpt_dir / "ema_best.pt")
                print(f"  [val] new best minFDE-20={best_fde:.4f}, saved ema_best.pt")

        # Checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "ema_state": ema.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": lr_scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_fde": best_fde,
            },
            ckpt_dir / f"ckpt_{epoch:03d}.pt",
        )

    torch.save(ema.state_dict(), ckpt_dir / "ema_final.pt")
    print(f"[train_joint] done. best minFDE-20 = {best_fde:.4f}")
    print(f"[train_joint] checkpoints in {ckpt_dir}")

    if writer:
        writer.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Joint multi-agent diffusion predictor training"
    )
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--run-name", type=str, default="av2_joint_v1")
    parser.add_argument("--ckpt-dir", type=str, default="models/diffusion")
    parser.add_argument("--log-dir", type=str, default="logs/diffusion")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--diffusion-T", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--max-agents", type=int, default=16)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-enc-layers", type=int, default=4)
    parser.add_argument("--num-interaction-layers", type=int, default=2)
    parser.add_argument("--dim-ff", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
