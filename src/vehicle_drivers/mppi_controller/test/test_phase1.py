"""Standalone Phase 1 MPPI test. No ROS2.

Simulates a GEM e4 on a 50 m radius curved path with two static obstacles.
Runs 200 MPPI steps, plots results, and asserts the lateral tracking spec:
after the first 20 steps, lateral error must be <= 0.5 m for at least 80%
of remaining steps.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mppi_controller.bicycle_model import step
from mppi_controller.reference_path import ReferencePath
from mppi_controller.mppi import MPPI


def build_curved_path(R=50.0, arc_deg=90.0, n=200):
    theta = np.linspace(0.0, np.deg2rad(arc_deg), n)
    # start tangent pointing +x at origin: center at (0, R)
    x = R * np.sin(theta)
    y = R * (1.0 - np.cos(theta))
    return np.stack([x, y], axis=1)


def run():
    waypoints = build_curved_path()
    ref = ReferencePath(waypoints)

    # Placed ~4 m off the reference curve so MPPI avoids without braking to 0.
    obstacles = np.array([
        [12.0, -2.8],
        [28.0, 3.8],
    ])

    mppi = MPPI(K=600, H=30, dt=0.1, sigma_steer=0.05, sigma_accel=0.8,
                lam=1.0, v_ref=3.0, seed=1)

    state = np.array([0.0, 0.0, 0.0, 0.0])
    N = 200

    xs = np.zeros((N, 4))
    lat_errs = np.zeros(N)
    ess = np.zeros(N)

    for i in range(N):
        u = mppi.update(state, ref, obstacles)
        state = step(state[None, :], u[None, :])[0]
        xs[i] = state
        _, _, _, d = ref.nearest_point(state[:2])
        lat_errs[i] = d
        ess[i] = mppi.effective_sample_count()

    tail = np.abs(lat_errs[20:])
    frac_ok = float(np.mean(tail <= 0.5))
    print(f"mean |lat err| (post-20): {tail.mean():.3f} m")
    print(f"max  |lat err| (post-20): {tail.max():.3f} m")
    print(f"fraction within 0.5 m:    {frac_ok:.2%}")
    print(f"final speed:              {xs[-1, 3]:.2f} m/s")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]
        ax.plot(ref.xy[:, 0], ref.xy[:, 1], 'k--', label='reference')
        ax.plot(xs[:, 0], xs[:, 1], 'b-', label='vehicle')
        ax.scatter(obstacles[:, 0], obstacles[:, 1], c='r', s=120,
                   marker='X', label='obstacles')
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title('MPPI trajectory')
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')

        ax = axes[1]
        ax.plot(ess, label='effective sample count')
        ax.axhline(mppi.K, color='k', ls=':', label=f'K={mppi.K}')
        ax.set_title('ESS = 1 / sum(w^2)')
        ax.set_xlabel('step'); ax.legend()

        out = os.path.join(os.path.dirname(__file__), 'phase1_result.png')
        fig.tight_layout(); fig.savefig(out, dpi=120)
        print(f"saved plot -> {out}")
    except Exception as e:
        print(f"(plotting skipped: {e})")

    assert frac_ok >= 0.80, (
        f"Phase 1 spec failed: only {frac_ok:.2%} of post-20 steps "
        f"within 0.5 m lateral error"
    )
    print("PHASE 1 TEST PASSED")


if __name__ == '__main__':
    run()
