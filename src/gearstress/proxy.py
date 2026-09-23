"""Analytic proxy fields for software smoke tests only.

These fields are not a gear-contact FEM solution and must not be used as evidence.
They are derived from smooth Airy potentials so the unmasked stress field is
approximately divergence-free under finite differences.
"""

from __future__ import annotations

import numpy as np


def _second_derivatives(phi: np.ndarray, spacing: float) -> tuple[np.ndarray, ...]:
    dphi_dy, dphi_dx = np.gradient(phi, spacing, spacing, edge_order=2)
    d2phi_dy2 = np.gradient(dphi_dy, spacing, axis=0, edge_order=2)
    d2phi_dx2 = np.gradient(dphi_dx, spacing, axis=1, edge_order=2)
    d2phi_dxdy = np.gradient(dphi_dx, spacing, axis=0, edge_order=2)
    return d2phi_dy2, d2phi_dx2, d2phi_dxdy


def make_proxy_dataset(
    trajectories: int = 32,
    frames_per_trajectory: int = 4,
    grid_size: int = 64,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Create deterministic, grouped proxy samples with transient conditions."""
    rng = np.random.default_rng(seed)
    axis = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    spacing = float(axis[1] - axis[0])
    # A smooth tooth-like domain: tapered body with a rounded root cutout.
    half_width = 0.34 + 0.16 * (yy + 1.0) / 2.0
    body = (np.abs(xx) <= half_width) & (yy >= -0.82) & (yy <= 0.9)
    root_notch = (xx**2 + (yy + 0.86) ** 2) < 0.11**2
    mask = (body & ~root_notch).astype(np.float32)

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    ids: list[int] = []
    conditions: list[np.ndarray] = []
    for trajectory in range(trajectories):
        torque0 = rng.uniform(0.55, 1.15)
        speed0 = rng.uniform(0.45, 1.25)
        friction = rng.uniform(0.02, 0.16)
        phase0 = rng.uniform(0.0, 2.0 * np.pi)
        for frame in range(frames_per_trajectory):
            t = frame / max(frames_per_trajectory - 1, 1)
            phase = phase0 + 2.0 * np.pi * t
            torque = torque0 * (1.0 + 0.18 * np.sin(phase))
            speed = speed0 * (1.0 + 0.12 * np.cos(phase))
            contact_y = 0.15 + 0.52 * t
            contact_x = 0.28 + 0.05 * np.sin(phase)
            r_contact = (xx - contact_x) ** 2 / 0.055 + (yy - contact_y) ** 2 / 0.018
            r_root = xx**2 / 0.11 + (yy + 0.67) ** 2 / 0.035
            phi = torque * (np.exp(-r_contact) + 0.55 * np.exp(-r_root))
            phi += friction * torque * xx * np.exp(-r_contact)
            sxx, syy, sxym = _second_derivatives(phi, spacing)
            stress = np.stack((sxx, syy, -sxym), axis=0).astype(np.float32)
            scale = max(float(np.max(np.abs(stress))), 1e-6)
            stress = stress / scale
            cond = np.array([torque, speed, friction, np.sin(phase), np.cos(phase)], dtype=np.float32)
            cond_maps = np.broadcast_to(cond[:, None, None], (5, grid_size, grid_size))
            features = np.concatenate(
                (mask[None], xx[None], yy[None], cond_maps), axis=0
            ).astype(np.float32)
            inputs.append(features)
            targets.append(stress)
            ids.append(trajectory)
            conditions.append(cond)
    return {
        "inputs": np.stack(inputs),
        "targets": np.stack(targets),
        "trajectory_ids": np.asarray(ids, dtype=np.int32),
        "conditions": np.stack(conditions),
        "mask": mask,
    }

