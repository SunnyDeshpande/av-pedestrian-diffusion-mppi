#!/usr/bin/env python3
"""PyTorch datasets for AV2 and GEM pedestrian trajectory data."""

import math
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset


def _rotation_matrix_2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


class TrajectoryDataset(Dataset):
    """Dataset for pedestrian trajectory prediction (AV2 or GEM).

    Loads preprocessed Parquet shards with columns:
        history       : float32 (20, 4)   [x, y, vx, vy]
        history_mask  : uint8   (20,)     1 = valid
        future        : float32 (20, 2)   [x, y]
        ego_vel       : float32 (2,)      [v_lin, v_ang]
    """

    def __init__(self, shard_dir: str, augment: bool = True):
        shard_path = pathlib.Path(shard_dir)
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard directory not found: {shard_dir}")

        # Load all .npz shards
        shards = sorted(shard_path.glob("*.npz"))
        if not shards:
            raise FileNotFoundError(f"No .npz shards found in {shard_dir}")

        histories = []
        masks = []
        futures = []
        ego_vels = []

        for shard in shards:
            data = np.load(shard)
            histories.append(data["history"])
            masks.append(data["history_mask"])
            futures.append(data["future"])
            ego_vels.append(data["ego_vel"])

        self.history = np.concatenate(histories, axis=0).astype(np.float32)
        self.history_mask = np.concatenate(masks, axis=0).astype(np.float32)
        self.future = np.concatenate(futures, axis=0).astype(np.float32)
        self.ego_vel = np.concatenate(ego_vels, axis=0).astype(np.float32)

        self.augment = augment
        self.N = self.history.shape[0]

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> dict:
        hist = self.history[idx].copy()       # (20, 4)
        mask = self.history_mask[idx].copy()   # (20,)
        fut = self.future[idx].copy()          # (20, 2)
        ego = self.ego_vel[idx].copy()         # (2,)

        if self.augment:
            hist, fut, mask = self._apply_augmentations(hist, fut, mask)

        return {
            "history": torch.from_numpy(hist),
            "history_mask": torch.from_numpy(mask),
            "future": torch.from_numpy(fut),
            "ego_vel": torch.from_numpy(ego),
        }

    def _apply_augmentations(
        self,
        hist: np.ndarray,
        fut: np.ndarray,
        mask: np.ndarray,
    ) -> tuple:
        """Apply training-time augmentations.

        1. Random rotation +/- 15 degrees (positions and velocities together).
        2. Random history dropout: zero out 0-20% of history steps.
        3. Random translation jitter: +/- 0.1 m offset at t=0.
        """
        # 1. Random rotation
        angle = np.random.uniform(-math.radians(15), math.radians(15))
        R = _rotation_matrix_2d(angle)

        hist[:, :2] = hist[:, :2] @ R.T     # rotate positions
        hist[:, 2:4] = hist[:, 2:4] @ R.T   # rotate velocities
        fut[:, :2] = fut[:, :2] @ R.T       # rotate future positions

        # 2. Random history dropout (0-20% of valid steps)
        valid_count = int(mask.sum())
        if valid_count > 2:
            drop_frac = np.random.uniform(0.0, 0.2)
            n_drop = max(0, int(valid_count * drop_frac))
            if n_drop > 0:
                valid_idxs = np.where(mask > 0)[0]
                drop_idxs = np.random.choice(valid_idxs, size=n_drop, replace=False)
                hist[drop_idxs] = 0.0
                mask[drop_idxs] = 0.0

        # 3. Translation jitter
        jitter = np.random.uniform(-0.1, 0.1, size=(1, 2)).astype(np.float32)
        hist[:, :2] += jitter
        fut[:, :2] += jitter

        return hist, fut, mask


def collate_fn(batch: list) -> dict:
    """Stack batch items into tensors."""
    return {
        "history": torch.stack([b["history"] for b in batch]),
        "history_mask": torch.stack([b["history_mask"] for b in batch]),
        "future": torch.stack([b["future"] for b in batch]),
        "ego_vel": torch.stack([b["ego_vel"] for b in batch]),
    }
