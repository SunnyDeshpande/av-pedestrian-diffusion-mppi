#!/usr/bin/env python3
"""AV2 pretraining entrypoint for the diffusion pedestrian trajectory predictor."""

import argparse
import copy
import math
import os
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion_prediction.model import TrajectoryDenoiser
from diffusion_prediction.ddpm import CosineSchedule, ddim_sample_loop
from diffusion_prediction.dataset import TrajectoryDataset, collate_fn


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: p.data.clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply(self, model: torch.nn.Module):
        """Replace model params with EMA shadow. Save originals for restore."""
        self._backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self._backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module):
        """Restore original params after apply()."""
        for name, p in model.named_parameters():
            if p.requires_grad and name in self._backup:
                p.data.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self):
        return dict(self.shadow)

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: TrajectoryDenoiser,
    schedule: CosineSchedule,
    val_loader: DataLoader,
    K: int = 20,
    device: torch.device = torch.device("cuda"),
) -> dict:
    """Compute minADE-K, minFDE-K, and miss-rate@2m on a validation set."""
    model.eval()

    all_min_ade = []
    all_min_fde = []
    all_miss = []

    for batch in val_loader:
        hist = batch["history"].to(device)           # (B, 20, 4)
        mask = batch["history_mask"].to(device)       # (B, 20)
        ego = batch["ego_vel"].to(device)             # (B, 2)
        gt_fut = batch["future"].to(device)           # (B, 20, 2)

        B = hist.shape[0]

        # Generate K samples: (B, K, 20, 2)
        futures = ddim_sample_loop(model, schedule, hist, mask, ego, K=K)

        # gt_fut: (B, 1, 20, 2)
        gt_exp = gt_fut.unsqueeze(1)

        # ADE per sample: mean over time of L2
        ade = torch.norm(futures - gt_exp, dim=-1).mean(dim=-1)  # (B, K)
        min_ade = ade.min(dim=1).values                           # (B,)

        # FDE per sample: L2 at last step
        fde = torch.norm(futures[:, :, -1, :] - gt_exp[:, :, -1, :], dim=-1)  # (B, K)
        min_fde = fde.min(dim=1).values                                         # (B,)

        # Miss rate: all K samples have FDE > 2.0 m
        miss = (fde.min(dim=1).values > 2.0).float()  # (B,)

        all_min_ade.append(min_ade.cpu())
        all_min_fde.append(min_fde.cpu())
        all_miss.append(miss.cpu())

    model.train()

    return {
        "minADE_20": torch.cat(all_min_ade).mean().item(),
        "minFDE_20": torch.cat(all_min_fde).mean().item(),
        "miss_2m": torch.cat(all_miss).mean().item(),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Directories
    ckpt_dir = pathlib.Path(args.ckpt_dir) / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = pathlib.Path(args.log_dir) / args.run_name

    # TensorBoard (optional)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
    except ImportError:
        print("[train] tensorboard not installed, skipping logging")

    # Data
    train_ds = TrajectoryDataset(os.path.join(args.data, "train"), augment=True)
    val_ds = TrajectoryDataset(os.path.join(args.data, "val"), augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        collate_fn=collate_fn,
    )

    print(f"[train] train samples: {len(train_ds)}, val samples: {len(val_ds)}")
    print(f"[train] batches/epoch: {len(train_loader)}")

    # Model
    model = TrajectoryDenoiser(
        d=args.d_model,
        nhead=args.nhead,
        num_enc_layers=args.num_enc_layers,
        num_dec_layers=args.num_dec_layers,
        dim_ff=args.dim_ff,
    ).to(device)

    # Schedule
    schedule = CosineSchedule(T=args.diffusion_T).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min,
    )

    # EMA
    ema = EMA(model, decay=args.ema_decay)

    # AMP
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and "cuda" in args.device))

    # Resume from checkpoint
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
        print(f"[train] resumed from {args.resume}, epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            hist = batch["history"].to(device)
            mask = batch["history_mask"].to(device)
            ego = batch["ego_vel"].to(device)
            fut = batch["future"].to(device)  # (B, 20, 2)

            B = hist.shape[0]

            # Sample diffusion timestep and noise
            t = torch.randint(0, args.diffusion_T, (B,), device=device)
            eps = torch.randn_like(fut)

            # Forward diffusion
            y_t = schedule.q_sample(fut, t, eps)

            # Predict noise
            with torch.amp.autocast("cuda", enabled=(args.amp and "cuda" in args.device)):
                eps_hat = model(hist, mask, ego, t, y_t)
                loss = F.mse_loss(eps_hat, eps)

            # Backward
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
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        print(
            f"[epoch {epoch}/{args.epochs}] "
            f"loss={avg_loss:.5f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"time={elapsed:.1f}s"
        )

        if writer:
            writer.add_scalar("loss/epoch_mean", avg_loss, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # Validation
        if epoch % args.val_every == 0 or epoch == args.epochs:
            # Validate with EMA weights
            ema.apply(model)
            metrics = validate(model, schedule, val_loader, K=20, device=device)
            ema.restore(model)

            print(
                f"  [val] minADE-20={metrics['minADE_20']:.4f}  "
                f"minFDE-20={metrics['minFDE_20']:.4f}  "
                f"miss@2m={metrics['miss_2m']:.4f}"
            )

            if writer:
                for k, v in metrics.items():
                    writer.add_scalar(f"val/{k}", v, epoch)

            # Save best
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

    # Save final EMA
    torch.save(ema.state_dict(), ckpt_dir / "ema_final.pt")
    print(f"[train] done. best minFDE-20 = {best_fde:.4f}")
    print(f"[train] checkpoints in {ckpt_dir}")

    if writer:
        writer.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AV2 pretrain for diffusion predictor")
    parser.add_argument("--data", type=str, required=True, help="Path to processed AV2 shards")
    parser.add_argument("--run-name", type=str, default="av2_pretrain_v1")
    parser.add_argument("--ckpt-dir", type=str, default="models/diffusion")
    parser.add_argument("--log-dir", type=str, default="logs/diffusion")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--diffusion-T", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-enc-layers", type=int, default=6)
    parser.add_argument("--num-dec-layers", type=int, default=4)
    parser.add_argument("--dim-ff", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
