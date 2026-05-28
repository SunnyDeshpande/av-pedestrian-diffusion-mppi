#!/usr/bin/env python3
"""Scene-level trajectory dataset for joint multi-agent prediction.

Each sample contains ALL pedestrians in a scene, padded to max_agents.
Shard format (.npz):
    histories    : (N, M, 20, 4) float32
    history_masks: (N, M, 20)    uint8
    futures      : (N, M, 20, 2) float32
    agent_masks  : (N, M)        uint8    - 1 = real agent, 0 = padding
    ego_vels     : (N, 2)        float32
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion_prediction.utils import rotation_matrix_2d


class JointTrajectoryDataset(Dataset):
    """Scene-level dataset for joint multi-agent training."""

    def __init__(self, shard_dir: str, augment: bool = True, max_agents: int = 16):
        self.augment = augment
        self.max_agents = max_agents

        self.histories = []
        self.history_masks = []
        self.futures = []
        self.agent_masks = []
        self.ego_vels = []

        shard_files = sorted(
            f for f in os.listdir(shard_dir) if f.endswith(".npz")
        )

        for sf in shard_files:
            data = np.load(os.path.join(shard_dir, sf))
            self.histories.append(data["histories"])
            self.history_masks.append(data["history_masks"])
            self.futures.append(data["futures"])
            self.agent_masks.append(data["agent_masks"])
            self.ego_vels.append(data["ego_vels"])

        if self.histories:
            self.histories = np.concatenate(self.histories, axis=0)
            self.history_masks = np.concatenate(self.history_masks, axis=0)
            self.futures = np.concatenate(self.futures, axis=0)
            self.agent_masks = np.concatenate(self.agent_masks, axis=0)
            self.ego_vels = np.concatenate(self.ego_vels, axis=0)
        else:
            self.histories = np.zeros((0, max_agents, 20, 4), dtype=np.float32)
            self.history_masks = np.zeros((0, max_agents, 20), dtype=np.uint8)
            self.futures = np.zeros((0, max_agents, 20, 2), dtype=np.float32)
            self.agent_masks = np.zeros((0, max_agents), dtype=np.uint8)
            self.ego_vels = np.zeros((0, 2), dtype=np.float32)

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, idx):
        hist = self.histories[idx].copy()       # (M, 20, 4)
        mask = self.history_masks[idx].copy()    # (M, 20)
        fut = self.futures[idx].copy()           # (M, 20, 2)
        agent_mask = self.agent_masks[idx].copy()  # (M,)
        ego = self.ego_vels[idx].copy()          # (2,)

        if self.augment:
            # Random rotation ±15° - apply same rotation to all agents
            theta = np.random.uniform(-np.pi / 12, np.pi / 12)
            R = rotation_matrix_2d(theta)

            for m in range(len(agent_mask)):
                if agent_mask[m] == 0:
                    continue
                # Rotate positions
                hist[m, :, :2] = hist[m, :, :2] @ R.T
                hist[m, :, 2:] = hist[m, :, 2:] @ R.T
                fut[m] = fut[m] @ R.T

            # History dropout: randomly zero 0-20% of history steps (per agent)
            for m in range(len(agent_mask)):
                if agent_mask[m] == 0:
                    continue
                drop_rate = np.random.uniform(0, 0.2)
                drop_mask = np.random.rand(20) < drop_rate
                # Don't drop the last step (current position)
                drop_mask[-1] = False
                hist[m, drop_mask] = 0.0
                mask[m, drop_mask] = 0

            # Translation jitter ±0.1m - same offset for all agents
            jitter = np.random.uniform(-0.1, 0.1, size=(1, 1, 2)).astype(np.float32)
            for m in range(len(agent_mask)):
                if agent_mask[m] == 0:
                    continue
                hist[m, :, :2] += jitter[0, 0]
                fut[m] += jitter[0, 0]

        return {
            "history": torch.from_numpy(hist),
            "history_mask": torch.from_numpy(mask),
            "future": torch.from_numpy(fut),
            "agent_mask": torch.from_numpy(agent_mask),
            "ego_vel": torch.from_numpy(ego),
        }


def collate_fn_joint(batch):
    """Stack scene-level samples (all already padded to max_agents)."""
    return {
        "history": torch.stack([b["history"] for b in batch]),
        "history_mask": torch.stack([b["history_mask"] for b in batch]),
        "future": torch.stack([b["future"] for b in batch]),
        "agent_mask": torch.stack([b["agent_mask"] for b in batch]),
        "ego_vel": torch.stack([b["ego_vel"] for b in batch]),
    }
