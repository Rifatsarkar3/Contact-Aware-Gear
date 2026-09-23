"""Field, hotspot, and equilibrium losses."""

from __future__ import annotations

import torch


def relative_l2(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    numerator = torch.linalg.vector_norm((prediction - target).flatten(1), dim=1)
    denominator = torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(eps)
    return (numerator / denominator).mean()


def hotspot_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantile: float = 0.95,
) -> torch.Tensor:
    magnitude = torch.sqrt((target.square()).sum(dim=1) + 1e-8)
    threshold = torch.quantile(magnitude.flatten(1), quantile, dim=1)
    hot = magnitude >= threshold[:, None, None]
    error = torch.abs(prediction - target).mean(dim=1)
    return (error * hot).sum() / hot.sum().clamp_min(1)


def equilibrium_residual(
    stress: torch.Tensor,
    mask: torch.Tensor | None = None,
    dx: float = 1.0,
    dy: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Dimensionless interior equilibrium residual for [sxx, syy, sxy].

    Central differences use physical grid spacing.  A five-point eroded mask
    excludes stencils that cross a free/contact boundary, where a strong-form
    divergence computed after grid interpolation is not valid.  Normalization by
    each sample's stress RMS and domain length makes the loss independent of the
    global stress scaling used during training.
    """
    if stress.shape[1] != 3:
        raise ValueError("stress must have exactly three channels")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be positive")
    sxx, syy, sxy = stress[:, 0], stress[:, 1], stress[:, 2]
    dsxx_dx = (sxx[:, 1:-1, 2:] - sxx[:, 1:-1, :-2]) / (2.0 * dx)
    dsxy_dy = (sxy[:, 2:, 1:-1] - sxy[:, :-2, 1:-1]) / (2.0 * dy)
    dsxy_dx = (sxy[:, 1:-1, 2:] - sxy[:, 1:-1, :-2]) / (2.0 * dx)
    dsyy_dy = (syy[:, 2:, 1:-1] - syy[:, :-2, 1:-1]) / (2.0 * dy)
    residual = (dsxx_dx + dsxy_dy).square() + (dsxy_dx + dsyy_dy).square()
    if mask is not None:
        material = mask[:, 0]
        interior = (
            material[:, 1:-1, 1:-1]
            * material[:, 1:-1, 2:]
            * material[:, 1:-1, :-2]
            * material[:, 2:, 1:-1]
            * material[:, :-2, 1:-1]
        )
    else:
        interior = torch.ones_like(residual)
        material = torch.ones_like(sxx)
    interior_count = interior.sum(dim=(1, 2)).clamp_min(1.0)
    residual_mean = (residual * interior).sum(dim=(1, 2)) / interior_count
    material_count = material.sum(dim=(1, 2)).clamp_min(1.0)
    stress_rms_sq = (stress.square() * material[:, None]).sum(dim=(1, 2, 3)) / (3.0 * material_count)
    height, width = stress.shape[-2:]
    length_sq = ((width - 1) * dx) * ((height - 1) * dy)
    return (length_sq * residual_mean / stress_rms_sq.clamp_min(eps)).mean()


def combined_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    hotspot_weight: float,
    physics_weight: float,
    hotspot_quantile: float,
    dx: float = 1.0,
    dy: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    field = relative_l2(prediction, target)
    hotspot = hotspot_mae(prediction, target, hotspot_quantile)
    physics_raw = equilibrium_residual(prediction, mask, dx=dx, dy=dy)
    # Nodal FEM stresses are discontinuous before projection to the operator
    # grid.  Normalize by the projected reference residual so the term measures
    # relative equilibrium quality instead of treating interpolation noise as a
    # zero-residual target.  The denominator is a scale only; no gradient flows
    # through the reference field.
    reference_physics = equilibrium_residual(target, mask, dx=dx, dy=dy).detach().clamp_min(1e-8)
    physics = physics_raw / reference_physics
    total = field + hotspot_weight * hotspot + physics_weight * physics
    return total, {"field": field, "hotspot": hotspot, "physics": physics}
