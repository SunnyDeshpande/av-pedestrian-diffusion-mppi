"""Batched kinematic bicycle model for the GEM e4."""
import numpy as np

L = 1.75
DT = 0.1
V_MIN = 0.0
V_MAX = 8.0


def wrap_to_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def step(states, controls, dt=DT, L=L):
    """Advance a batch of states by one dt.

    states:   (K, 4)  columns = x, y, psi, v
    controls: (K, 2)  columns = delta, a
    returns:  (K, 4)
    """
    x, y, psi, v = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    delta, a = controls[:, 0], controls[:, 1]

    x_next = x + v * np.cos(psi) * dt
    y_next = y + v * np.sin(psi) * dt
    psi_next = wrap_to_pi(psi + (v / L) * np.tan(delta) * dt)
    v_next = np.clip(v + a * dt, V_MIN, V_MAX)

    out = np.empty_like(states)
    out[:, 0] = x_next
    out[:, 1] = y_next
    out[:, 2] = psi_next
    out[:, 3] = v_next
    return out
