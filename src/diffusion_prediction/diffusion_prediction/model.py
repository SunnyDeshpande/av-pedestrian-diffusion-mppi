#!/usr/bin/env python3
"""MID-style Transformer denoiser for pedestrian trajectory prediction.

v2: Upgraded architecture with multi-layer cross-attention decoder,
wider dimensions, and more capacity for capturing non-linear motion.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : (B,) long tensor of timesteps

        Returns
        -------
        emb : (B, dim)
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)   # (B, dim)


class TrajectoryDenoiser(nn.Module):
    """MID-style conditional trajectory denoiser.

    Encodes observed history + ego velocity + diffusion timestep,
    then a multi-layer cross-attention decoder attends from noisy-future
    queries to produce an epsilon (noise) prediction of shape (B, T_fut, 2).

    Parameters
    ----------
    d           : model dimension
    T_hist      : history length (timesteps)
    T_fut       : future/prediction length (timesteps)
    nhead       : attention heads
    num_enc_layers : encoder transformer layers
    num_dec_layers : decoder cross-attention layers
    dim_ff      : feed-forward hidden dimension
    dropout     : dropout rate
    """

    def __init__(
        self,
        d: int = 256,
        T_hist: int = 20,
        T_fut: int = 20,
        nhead: int = 8,
        num_enc_layers: int = 6,
        num_dec_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d = d
        self.T_hist = T_hist
        self.T_fut = T_fut

        # History encoder input
        self.hist_in = nn.Linear(4, d)
        self.hist_pos = nn.Parameter(torch.randn(T_hist, d) * 0.02)

        # Ego velocity
        self.ego_in = nn.Linear(2, d)

        # Diffusion timestep
        self.t_embed = nn.Sequential(
            SinusoidalPosEmb(d),
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

        # Noisy future input (MID-style: decoder queries come from noised traj)
        self.noisy_fut_in = nn.Linear(2, d)
        self.fut_pos = nn.Parameter(torch.randn(T_fut, d) * 0.02)

        # Transformer encoder over history tokens
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        # Multi-layer cross-attention decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.dec = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)

        # Output head: per-token MLP -> (x, y) noise prediction
        self.head = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, 2),
        )

        self._log_param_count()

    def _log_param_count(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[TrajectoryDenoiser] trainable params: {n:,} ({n/1e6:.2f}M)")

    def forward(
        self,
        hist: torch.Tensor,
        hist_mask: torch.Tensor,
        ego_vel: torch.Tensor,
        t: torch.Tensor,
        y_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hist      : (B, T_hist, 4)  [x, y, vx, vy]
        hist_mask : (B, T_hist)      1 = real observation, 0 = padding
        ego_vel   : (B, 2)           [v_lin, v_ang]
        t         : (B,)             diffusion timestep (long)
        y_t       : (B, T_fut, 2)    noised future trajectory

        Returns
        -------
        eps_pred  : (B, T_fut, 2)    predicted noise
        """
        B = hist.shape[0]

        # Encode history
        h = self.hist_in(hist) + self.hist_pos  # (B, T_hist, d)

        # Add ego + timestep conditioning (AdaLN-lite: additive)
        ego = self.ego_in(ego_vel).unsqueeze(1)    # (B, 1, d)
        tt = self.t_embed(t).unsqueeze(1)          # (B, 1, d)
        h = h + ego + tt

        # Mask invalid history steps for transformer
        src_key_padding_mask = (hist_mask == 0)    # (B, T_hist), True = ignore

        # Transformer encoder
        z = self.enc(h, src_key_padding_mask=src_key_padding_mask)  # (B, T_hist, d)

        # Build decoder queries from noisy future (MID-style)
        q = self.noisy_fut_in(y_t) + self.fut_pos  # (B, T_fut, d)
        q = q + ego + tt  # also condition queries

        # Multi-layer cross-attention decoder
        dec_out = self.dec(
            q, z,
            memory_key_padding_mask=src_key_padding_mask,
        )  # (B, T_fut, d)

        # Project to noise prediction
        eps_pred = self.head(dec_out)  # (B, T_fut, 2)
        return eps_pred
