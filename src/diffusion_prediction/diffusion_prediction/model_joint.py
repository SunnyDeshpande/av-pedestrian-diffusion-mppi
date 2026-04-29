#!/usr/bin/env python3
"""Joint multi-agent Transformer denoiser for pedestrian trajectory prediction.

Extends the single-agent TrajectoryDenoiser with cross-agent attention
so that each pedestrian's prediction is informed by the trajectories of
all other pedestrians in the scene.

Architecture:
    1. Per-agent encoding (shared weights): history → TransformerEncoder
    2. Agent interaction: mean-pool → cross-agent TransformerEncoder
    3. Per-agent decoding: noisy future cross-attends to enriched history
"""

import math
import torch
import torch.nn as nn

from diffusion_prediction.model import SinusoidalPosEmb


class JointTrajectoryDenoiser(nn.Module):
    """Joint multi-agent conditional trajectory denoiser.

    Parameters
    ----------
    d           : model dimension
    T_hist      : history length (timesteps)
    T_fut       : future length (timesteps)
    max_agents  : maximum number of agents per scene (for learned embeddings)
    nhead       : attention heads
    num_enc_layers       : per-agent encoder layers
    num_interaction_layers : cross-agent interaction layers
    dim_ff      : feed-forward dimension
    dropout     : dropout rate
    """

    def __init__(
        self,
        d: int = 128,
        T_hist: int = 20,
        T_fut: int = 20,
        max_agents: int = 16,
        nhead: int = 4,
        num_enc_layers: int = 4,
        num_interaction_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d = d
        self.T_hist = T_hist
        self.T_fut = T_fut
        self.max_agents = max_agents

        # ---- Per-agent encoding (same as single-agent model) ----
        self.hist_in = nn.Linear(4, d)
        self.hist_pos = nn.Parameter(torch.randn(T_hist, d) * 0.02)

        self.ego_in = nn.Linear(2, d)

        self.t_embed = nn.Sequential(
            SinusoidalPosEmb(d),
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

        # Per-agent transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        # ---- Cross-agent interaction ----
        self.agent_pool_norm = nn.LayerNorm(d)
        interaction_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.interaction = nn.TransformerEncoder(
            interaction_layer, num_layers=num_interaction_layers,
        )
        self.interaction_proj = nn.Linear(d, d)

        # ---- Per-agent decoder ----
        self.noisy_fut_in = nn.Linear(2, d)
        self.fut_pos = nn.Parameter(torch.randn(T_fut, d) * 0.02)

        self.dec = nn.MultiheadAttention(d, nhead, batch_first=True, dropout=dropout)
        self.dec_norm_q = nn.LayerNorm(d)
        self.dec_norm_kv = nn.LayerNorm(d)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, 2),
        )

        self._log_param_count()

    def _log_param_count(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[JointTrajectoryDenoiser] trainable params: {n:,} ({n/1e6:.2f}M)")

    def forward(
        self,
        hist: torch.Tensor,
        hist_mask: torch.Tensor,
        agent_mask: torch.Tensor,
        ego_vel: torch.Tensor,
        t: torch.Tensor,
        y_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hist       : (B, M, T_hist, 4)  per-agent history [x, y, vx, vy]
        hist_mask  : (B, M, T_hist)      1 = real observation, 0 = padding
        agent_mask : (B, M)              1 = real agent, 0 = padding agent
        ego_vel    : (B, 2)              [v_lin, v_ang]
        t          : (B,)                diffusion timestep
        y_t        : (B, M, T_fut, 2)   noised future trajectories

        Returns
        -------
        eps_pred   : (B, M, T_fut, 2)   predicted noise per agent
        """
        B, M, T_h, _ = hist.shape
        T_f = y_t.shape[2]

        # ---- 1. Per-agent encoding ----
        # Flatten agents into batch: (B*M, T_h, 4)
        h = hist.reshape(B * M, T_h, 4)
        h = self.hist_in(h) + self.hist_pos  # (B*M, T_h, d)

        # Conditioning: ego velocity + diffusion timestep (broadcast to all agents)
        ego = self.ego_in(ego_vel)                              # (B, d)
        ego_exp = ego.unsqueeze(1).expand(-1, M, -1)            # (B, M, d)
        ego_exp = ego_exp.reshape(B * M, 1, self.d)             # (B*M, 1, d)

        tt = self.t_embed(t)                                    # (B, d)
        tt_exp = tt.unsqueeze(1).expand(-1, M, -1)              # (B, M, d)
        tt_exp = tt_exp.reshape(B * M, 1, self.d)               # (B*M, 1, d)

        h = h + ego_exp + tt_exp

        # Per-agent history mask
        src_pad = (hist_mask.reshape(B * M, T_h) == 0)          # True = ignore

        # For padding agents (all positions masked), unmask the last position
        # to prevent NaN from softmax of all -inf in transformer attention
        all_masked = src_pad.all(dim=1)                          # (B*M,)
        src_pad = src_pad.clone()
        src_pad[all_masked, -1] = False

        # Per-agent transformer encoder
        z = self.enc(h, src_key_padding_mask=src_pad)            # (B*M, T_h, d)

        # ---- 2. Cross-agent interaction ----
        # Mean-pool over valid timesteps to get agent-level tokens
        valid = hist_mask.reshape(B * M, T_h, 1).float()        # (B*M, T_h, 1)
        agent_tokens = (z * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)  # (B*M, d)
        agent_tokens = self.agent_pool_norm(agent_tokens)
        agent_tokens = agent_tokens.reshape(B, M, self.d)       # (B, M, d)

        # Cross-agent attention (agents attend to each other)
        agent_pad = (agent_mask == 0)                            # (B, M), True = ignore
        agent_tokens = self.interaction(
            agent_tokens, src_key_padding_mask=agent_pad,
        )                                                        # (B, M, d)

        # Broadcast interaction back to per-timestep representation
        interaction_out = self.interaction_proj(agent_tokens)     # (B, M, d)
        interaction_out = interaction_out.reshape(B * M, 1, self.d).expand(-1, T_h, -1)
        z = z + interaction_out                                  # enriched history

        # ---- 3. Per-agent decoding ----
        yt_flat = y_t.reshape(B * M, T_f, 2)
        q = self.noisy_fut_in(yt_flat) + self.fut_pos            # (B*M, T_f, d)
        q = q + ego_exp + tt_exp                                 # condition queries too

        q_norm = self.dec_norm_q(q)
        z_norm = self.dec_norm_kv(z)
        dec_out, _ = self.dec(
            q_norm, z_norm, z_norm, key_padding_mask=src_pad,
        )
        dec_out = dec_out + q                                    # residual

        # ---- 4. Output ----
        eps_pred = self.head(dec_out)                            # (B*M, T_f, 2)
        return eps_pred.reshape(B, M, T_f, 2)
