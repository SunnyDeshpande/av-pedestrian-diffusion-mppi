#!/usr/bin/env python3
"""Cosine noise schedule, forward diffusion, and DDIM sampling."""

import math
import torch

# 10-step DDIM schedule (linearly spaced from 99 down to 0)
DDIM_TAUS = [99, 88, 77, 66, 55, 44, 33, 22, 11, 0]


class CosineSchedule:
    """Cosine beta schedule (Nichol & Dhariwal, 2021).

    Precomputes alpha_bar for T diffusion timesteps.
    """

    def __init__(self, T: int = 100, s: float = 0.008):
        self.T = T
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((steps / T + s) / (1.0 + s) * math.pi / 2.0) ** 2
        alpha_bar = (f / f[0]).float().clamp(min=1e-5)  # (T+1,), clamped to avoid div-by-zero
        self.register(alpha_bar)

    def register(self, alpha_bar: torch.Tensor):
        self.alpha_bar = alpha_bar              # (T+1,)  index 0..T
        self.alphas = alpha_bar[1:] / alpha_bar[:-1]
        self.betas = (1.0 - self.alphas).clamp(0.0, 0.999)

    def to(self, device: torch.device) -> "CosineSchedule":
        """Move schedule tensors to a device."""
        self.alpha_bar = self.alpha_bar.to(device)
        self.alphas = self.alphas.to(device)
        self.betas = self.betas.to(device)
        return self

    # ------------------------------------------------------------------
    # Forward diffusion  q(x_t | x_0)
    # ------------------------------------------------------------------
    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """Sample x_t given x_0 and noise eps.

        Parameters
        ----------
        x0  : (B, ..., 2)  clean trajectory (any number of middle dims)
        t   : (B,)          integer timesteps in [0, T-1]
        eps : same shape as x0

        Returns
        -------
        x_t : same shape as x0
        """
        # Reshape alpha_bar to broadcast with arbitrary middle dims
        ab = self.alpha_bar[t + 1]  # (B,)
        shape = [-1] + [1] * (x0.dim() - 1)
        ab = ab.view(*shape)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps

    # ------------------------------------------------------------------
    # DDIM deterministic step
    # ------------------------------------------------------------------
    def ddim_step(
        self,
        xt: torch.Tensor,
        eps_hat: torch.Tensor,
        t_now: int,
        t_next: int,
    ) -> torch.Tensor:
        """One deterministic DDIM step from t_now -> t_next.

        Parameters
        ----------
        xt      : (B, T_fut, 2)
        eps_hat : (B, T_fut, 2)  predicted noise
        t_now   : int            current timestep
        t_next  : int            target timestep (< t_now)

        Returns
        -------
        x_{t_next} : (B, T_fut, 2)
        """
        ab_now = self.alpha_bar[t_now + 1].view(1, 1, 1)
        ab_next = self.alpha_bar[t_next + 1].view(1, 1, 1)

        # Predict x_0
        x0_pred = (xt - (1.0 - ab_now).sqrt() * eps_hat) / ab_now.sqrt()

        # Step to t_next
        return ab_next.sqrt() * x0_pred + (1.0 - ab_next).sqrt() * eps_hat


@torch.no_grad()
def ddim_sample_loop(
    model,
    schedule: CosineSchedule,
    hist: torch.Tensor,
    hist_mask: torch.Tensor,
    ego_vel: torch.Tensor,
    K: int = 20,
    taus: list = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Run DDIM sampling to generate K trajectory samples per pedestrian.

    Parameters
    ----------
    model     : TrajectoryDenoiser
    schedule  : CosineSchedule
    hist      : (M, T_hist, 4)
    hist_mask : (M, T_hist)
    ego_vel   : (M, 2)
    K         : number of samples per pedestrian
    taus      : DDIM timestep schedule (default: DDIM_TAUS)
    seed      : optional RNG seed for reproducibility

    Returns
    -------
    futures : (M, K, T_fut, 2) predicted trajectories
    """
    if taus is None:
        taus = DDIM_TAUS

    device = hist.device
    M = hist.shape[0]
    T_fut = 20

    # Tile inputs by K:  (M, ...) -> (M*K, ...)
    hist_k = hist.unsqueeze(1).expand(-1, K, -1, -1).reshape(M * K, -1, 4)
    mask_k = hist_mask.unsqueeze(1).expand(-1, K, -1).reshape(M * K, -1)
    ego_k = ego_vel.unsqueeze(1).expand(-1, K, -1).reshape(M * K, 2)

    # Initialize from noise
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        yt = torch.randn(M * K, T_fut, 2, device=device, generator=gen)
    else:
        yt = torch.randn(M * K, T_fut, 2, device=device)

    # DDIM denoising loop
    for i in range(len(taus) - 1):
        t_now = taus[i]
        t_next = taus[i + 1]

        t_batch = torch.full((M * K,), t_now, device=device, dtype=torch.long)
        eps_hat = model(hist_k, mask_k, ego_k, t_batch, yt)
        yt = schedule.ddim_step(yt, eps_hat, t_now, t_next)

    # Final step: predict clean from t=taus[-1]
    t_batch = torch.full((M * K,), taus[-1], device=device, dtype=torch.long)
    eps_hat = model(hist_k, mask_k, ego_k, t_batch, yt)
    ab_final = schedule.alpha_bar[taus[-1] + 1].view(1, 1, 1)
    y0 = (yt - (1.0 - ab_final).sqrt() * eps_hat) / ab_final.sqrt()

    return y0.reshape(M, K, T_fut, 2)


# ----------------------------------------------------------------------
# Joint multi-agent DDIM sampling
# ----------------------------------------------------------------------

@torch.no_grad()
def ddim_sample_loop_joint(
    model,
    schedule: CosineSchedule,
    hist: torch.Tensor,
    hist_mask: torch.Tensor,
    agent_mask: torch.Tensor,
    ego_vel: torch.Tensor,
    K: int = 20,
    taus: list = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Run joint DDIM sampling for multi-agent prediction.

    Parameters
    ----------
    model      : JointTrajectoryDenoiser
    schedule   : CosineSchedule
    hist       : (B, M, T_hist, 4)
    hist_mask  : (B, M, T_hist)
    agent_mask : (B, M)          1 = real agent, 0 = padding
    ego_vel    : (B, 2)
    K          : number of joint samples
    taus       : DDIM timestep schedule
    seed       : optional RNG seed

    Returns
    -------
    futures : (B, K, M, T_fut, 2) predicted trajectories
    """
    if taus is None:
        taus = DDIM_TAUS

    device = hist.device
    B, M = hist.shape[:2]
    T_fut = 20

    # Tile inputs by K: (B, ...) -> (B*K, ...)
    hist_k = hist.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(B * K, M, -1, 4)
    hmask_k = hist_mask.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, M, -1)
    amask_k = agent_mask.unsqueeze(1).expand(-1, K, -1).reshape(B * K, M)
    ego_k = ego_vel.unsqueeze(1).expand(-1, K, -1).reshape(B * K, 2)

    # Initialize joint noise: (B*K, M, T_fut, 2)
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        yt = torch.randn(B * K, M, T_fut, 2, device=device, generator=gen)
    else:
        yt = torch.randn(B * K, M, T_fut, 2, device=device)

    # Zero out padding agents
    yt = yt * amask_k[:, :, None, None]

    def _ddim_step_4d(xt, eps_hat, t_now, t_next):
        """DDIM step for (B, M, T_fut, 2) shaped tensors."""
        ab_now = schedule.alpha_bar[t_now + 1].view(1, 1, 1, 1)
        ab_next = schedule.alpha_bar[t_next + 1].view(1, 1, 1, 1)
        x0_pred = (xt - (1.0 - ab_now).sqrt() * eps_hat) / ab_now.sqrt()
        return ab_next.sqrt() * x0_pred + (1.0 - ab_next).sqrt() * eps_hat

    # DDIM denoising loop
    for i in range(len(taus) - 1):
        t_now = taus[i]
        t_next = taus[i + 1]

        t_batch = torch.full((B * K,), t_now, device=device, dtype=torch.long)
        eps_hat = model(hist_k, hmask_k, amask_k, ego_k, t_batch, yt)
        eps_hat = eps_hat * amask_k[:, :, None, None]  # zero padding agents
        yt = _ddim_step_4d(yt, eps_hat, t_now, t_next)

    # Final step
    t_batch = torch.full((B * K,), taus[-1], device=device, dtype=torch.long)
    eps_hat = model(hist_k, hmask_k, amask_k, ego_k, t_batch, yt)
    eps_hat = eps_hat * amask_k[:, :, None, None]
    ab_final = schedule.alpha_bar[taus[-1] + 1].view(1, 1, 1, 1)
    y0 = (yt - (1.0 - ab_final).sqrt() * eps_hat) / ab_final.sqrt()

    return y0.reshape(B, K, M, T_fut, 2)
