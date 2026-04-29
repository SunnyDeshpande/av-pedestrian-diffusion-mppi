#!/usr/bin/env python3
"""GEM rosbag finetune entrypoint for the diffusion pedestrian trajectory predictor.

Loads a pretrained checkpoint (from AV2 pretraining) and finetunes on
GEM-collected pedestrian trajectory data.
"""

import argparse
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
from diffusion_prediction.train import EMA, validate


def finetune(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Directories
    ckpt_dir = pathlib.Path(args.ckpt_dir) / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = pathlib.Path(args.log_dir) / args.run_name

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
    except ImportError:
        print("[finetune] tensorboard not installed, skipping logging")

    # Model
    model = TrajectoryDenoiser(
        d=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
    ).to(device)

    # Load pretrained EMA weights
    if not os.path.exists(args.pretrained):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained}")

    pretrained_state = torch.load(args.pretrained, map_location=device, weights_only=True)
    # EMA checkpoints are saved as {name: tensor} dicts
    if isinstance(pretrained_state, dict) and "model_state" in pretrained_state:
        model.load_state_dict(pretrained_state["model_state"])
    else:
        model.load_state_dict(pretrained_state)
    print(f"[finetune] loaded pretrained weights from {args.pretrained}")

    # Schedule
    schedule = CosineSchedule(T=args.diffusion_T).to(device)

    # Data
    train_ds = TrajectoryDataset(os.path.join(args.data, "train"), augment=True)
    val_ds = TrajectoryDataset(os.path.join(args.data, "val"), augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn,
    )

    print(f"[finetune] train samples: {len(train_ds)}, val samples: {len(val_ds)}")

    # Optionally freeze ego_in and t_embed for small datasets
    if args.freeze_conditioning and len(train_ds) < 5000:
        print("[finetune] freezing ego_in and t_embed (dataset < 5000 samples)")
        for name, p in model.named_parameters():
            if "ego_in" in name or "t_embed" in name:
                p.requires_grad = False

    # Optimizer — constant lr for finetune
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    ema = EMA(model, decay=args.ema_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and "cuda" in args.device))

    global_step = 0
    best_fde = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            hist = batch["history"].to(device)
            mask = batch["history_mask"].to(device)
            ego = batch["ego_vel"].to(device)
            fut = batch["future"].to(device)

            B = hist.shape[0]
            t = torch.randint(0, args.diffusion_T, (B,), device=device)
            eps = torch.randn_like(fut)
            y_t = schedule.q_sample(fut, t, eps)

            with torch.amp.autocast("cuda", enabled=(args.amp and "cuda" in args.device)):
                eps_hat = model(hist, mask, ego, t, y_t)
                loss = F.mse_loss(eps_hat, eps)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)
            epoch_loss += loss.item()
            global_step += 1

        avg_loss = epoch_loss / max(len(train_loader), 1)
        elapsed = time.time() - t0

        print(
            f"[finetune epoch {epoch}/{args.epochs}] "
            f"loss={avg_loss:.5f}  time={elapsed:.1f}s"
        )

        if writer:
            writer.add_scalar("finetune/loss", avg_loss, epoch)

        # Validate every epoch during finetune
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
                writer.add_scalar(f"finetune/val/{k}", v, epoch)

        if metrics["minFDE_20"] < best_fde:
            best_fde = metrics["minFDE_20"]
            torch.save(ema.state_dict(), ckpt_dir / "ema_best.pt")
            print(f"  [val] new best minFDE-20={best_fde:.4f}")

        # Checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "ema_state": ema.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            ckpt_dir / f"ckpt_{epoch:03d}.pt",
        )

    torch.save(ema.state_dict(), ckpt_dir / "ema_final.pt")
    print(f"[finetune] done. best minFDE-20 = {best_fde:.4f}")

    if writer:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description="GEM finetune for diffusion predictor")
    parser.add_argument("--pretrained", type=str, required=True, help="Path to pretrained EMA checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to processed GEM shards")
    parser.add_argument("--run-name", type=str, default="gem_finetune_v1")
    parser.add_argument("--ckpt-dir", type=str, default="models/diffusion")
    parser.add_argument("--log-dir", type=str, default="logs/diffusion")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--diffusion-T", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-ff", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--freeze-conditioning", action="store_true", default=True)
    parser.add_argument("--no-freeze-conditioning", dest="freeze_conditioning", action="store_false")
    args = parser.parse_args()
    finetune(args)


if __name__ == "__main__":
    main()
