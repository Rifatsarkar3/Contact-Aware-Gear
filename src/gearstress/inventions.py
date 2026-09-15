"""Experimental stress operators for the Paper 01 scientific smoke gate.

These models are isolated from the accepted baselines in ``models.py``.  They
test whether discrete equilibrium can be introduced through the output space
without adding another scalar penalty to the training loss.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import nn

from .models import SpectralConv2d


def _dct_matrix(n: int) -> torch.Tensor:
    """Orthonormal DCT-II basis matrix (rows are basis vectors, ``C @ C.T = I``).

    DCT-II implicitly assumes an even/reflective (Neumann-like) boundary
    extension, unlike the FFT's periodic (wrap-around) assumption -- a better
    match for a bounded elastic domain with free/masked edges.
    """
    spatial = torch.arange(n, dtype=torch.float32).unsqueeze(0)
    freq = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    basis = torch.cos(math.pi / n * (spatial + 0.5) * freq)
    basis = basis * math.sqrt(2.0 / n)
    basis[0, :] = basis[0, :] / math.sqrt(2.0)
    return basis


class DCTSpectralConv2d(nn.Module):
    """Truncated spectral convolution using DCT-II instead of the FFT.

    Structurally mirrors ``SpectralConv2d`` (truncate to the lowest modes,
    mix channels per mode, transform back), but the DCT's reflective boundary
    assumption matches a bounded, non-periodic domain instead of wrapping the
    field around its edges. Real-valued throughout, so no positive/negative
    frequency split is needed the way ``rfft2`` requires.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_y: int,
        grid_h: int = 64,
        grid_w: int = 64,
    ):
        super().__init__()
        if modes_y > grid_h or modes_x > grid_w:
            raise ValueError("modes cannot exceed the grid size")
        self.register_buffer("basis_h", _dct_matrix(grid_h)[:modes_y, :])
        self.register_buffer("basis_w", _dct_matrix(grid_w)[:modes_x, :])
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_y, modes_x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeff = torch.einsum("ph,bchw->bcpw", self.basis_h, x)
        coeff = torch.einsum("qw,bcpw->bcpq", self.basis_w, coeff)
        mixed = torch.einsum("bcyx,coyx->boyx", coeff, self.weight)
        out = torch.einsum("ph,bopq->bohq", self.basis_h, mixed)
        out = torch.einsum("qw,bohq->bohw", self.basis_w, out)
        return out


class DCTFNO(nn.Module):
    """Boundary-consistent spectral operator (DCT-FNO).

    Identical to ``FNO2d`` (same lift/local/activation/project structure) with
    the FFT-based ``SpectralConv2d`` replaced by ``DCTSpectralConv2d``,
    isolating the periodic-vs-reflective boundary assumption as the single
    experimental variable against the accepted vanilla FNO baseline.
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 3,
        width: int = 24,
        modes_x: int = 12,
        modes_y: int = 12,
        layers: int = 4,
        grid_h: int = 64,
        grid_w: int = 64,
    ):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spectral = nn.ModuleList(
            DCTSpectralConv2d(width, width, modes_x, modes_y, grid_h=grid_h, grid_w=grid_w)
            for _ in range(layers)
        )
        self.local = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.activation = nn.GELU()
        self.project = nn.Sequential(
            nn.Conv2d(width, width * 2, 1), nn.GELU(), nn.Conv2d(width * 2, out_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for spectral, local in zip(self.spectral, self.local):
            x = self.activation(spectral(x) + local(x))
        return self.project(x)


def _central_difference(field: torch.Tensor, spacing: float, dim: int) -> torch.Tensor:
    """Periodic central difference used consistently by the compatible head."""
    return (torch.roll(field, -1, dims=dim) - torch.roll(field, 1, dims=dim)) / (2.0 * spacing)


def airy_stress(potential: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Map a scalar potential to a symmetric, central-difference-compatible stress."""
    if potential.ndim != 4 or potential.shape[1] != 1:
        raise ValueError("potential must have shape [batch, 1, height, width]")
    phi = potential[:, 0]
    dphi_dx = _central_difference(phi, dx, dim=-1)
    dphi_dy = _central_difference(phi, dy, dim=-2)
    sxx = _central_difference(dphi_dy, dy, dim=-2)
    syy = _central_difference(dphi_dx, dx, dim=-1)
    sxy = -_central_difference(dphi_dx, dy, dim=-2)
    return torch.stack((sxx, syy, sxy), dim=1)


def spectral_equilibrium_projection(stress: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Project [sxx, syy, sxy] onto the discrete central-difference equilibrium space.

    For every Fourier mode this solves the minimum-distance problem subject to
    ``Dx(sxx) + Dy(sxy) = 0`` and ``Dx(sxy) + Dy(syy) = 0``.  The derivative
    symbols match the central differences used by the held-out equilibrium
    metric, rather than assuming continuous spectral derivatives.
    """
    if stress.ndim != 4 or stress.shape[1] != 3:
        raise ValueError("stress must have shape [batch, 3, height, width]")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be positive")
    height, width = stress.shape[-2:]
    stress_ft = torch.fft.rfft2(stress, norm="ortho")
    fy = torch.fft.fftfreq(height, device=stress.device, dtype=stress.dtype)
    fx = torch.fft.rfftfreq(width, device=stress.device, dtype=stress.dtype)
    qy = torch.sin(2.0 * math.pi * fy)[:, None] / dy
    qx = torch.sin(2.0 * math.pi * fx)[None, :] / dx
    qx = qx.expand(height, -1)
    qy = qy.expand(-1, width // 2 + 1)

    sxx, syy, sxy = stress_ft[:, 0], stress_ft[:, 1], stress_ft[:, 2]
    residual_x = qx * sxx + qy * sxy
    residual_y = qy * syy + qx * sxy
    diagonal = qx.square() + qy.square()
    off_diagonal = qx * qy
    determinant = diagonal.square() - off_diagonal.square()
    active = determinant > torch.finfo(stress.dtype).eps
    safe_determinant = torch.where(active, determinant, torch.ones_like(determinant))
    multiplier_x = (diagonal * residual_x - off_diagonal * residual_y) / safe_determinant
    multiplier_y = (-off_diagonal * residual_x + diagonal * residual_y) / safe_determinant
    multiplier_x = torch.where(active, multiplier_x, torch.zeros_like(multiplier_x))
    multiplier_y = torch.where(active, multiplier_y, torch.zeros_like(multiplier_y))

    projected_ft = torch.stack(
        (
            sxx - qx * multiplier_x,
            syy - qy * multiplier_y,
            sxy - qy * multiplier_x - qx * multiplier_y,
        ),
        dim=1,
    )
    return torch.fft.irfft2(projected_ft, s=(height, width), norm="ortho")


class _FNOFeatureBackbone(nn.Module):
    """The accepted FNO feature extractor, without its output projection."""

    def __init__(self, in_channels: int, width: int, modes_x: int, modes_y: int, layers: int):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spectral = nn.ModuleList(
            SpectralConv2d(width, width, modes_x, modes_y) for _ in range(layers)
        )
        self.local = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.lift(inputs)
        for spectral, local in zip(self.spectral, self.local):
            features = self.activation(spectral(features) + local(features))
        return features


class _ContactPotentialHead(nn.Module):
    """Localized multiscale potential correction; zero-initialized for stable entry."""

    def __init__(self, width: int):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2),
            nn.GELU(),
        )
        self.potential = nn.Conv2d(width, 1, 1)
        self.gate = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.potential.weight)
        nn.init.zeros_(self.potential.bias)

    def forward(self, features: torch.Tensor, contact_map: torch.Tensor) -> torch.Tensor:
        local = self.refine(features)
        # The learned gate may extend the supplied contact prior toward the root,
        # but the prior ensures that the branch starts focused on physical contact.
        gate = torch.sigmoid(self.gate(local) + 2.0 * contact_map)
        return gate * self.potential(local)


class AdaptiveEquilibriumExchangeFNO(nn.Module):
    """AEE-FNO: learned exchange between empirical and admissible stress spaces.

    A sample-level gate retains raw stress modes when demanded by the labels and
    exchanges the remainder for their closest discrete-equilibrium counterpart.
    A localized Airy-potential correction can restore sharp compatible detail.
    """

    def __init__(
        self,
        in_channels: int = 8,
        width: int = 24,
        modes_x: int = 12,
        modes_y: int = 12,
        layers: int = 4,
        dx: float = 1.0,
        dy: float = 1.0,
    ):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.backbone = _FNOFeatureBackbone(in_channels, width, modes_x, modes_y, layers)
        self.raw_head = nn.Sequential(
            nn.Conv2d(width, width * 2, 1), nn.GELU(), nn.Conv2d(width * 2, 3, 1)
        )
        self.exchange_gate = nn.Linear(width, 1)
        nn.init.zeros_(self.exchange_gate.weight)
        nn.init.zeros_(self.exchange_gate.bias)
        self.contact_potential = _ContactPotentialHead(width)
        self.last_exchange: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        raw = self.raw_head(features)
        admissible = spectral_equilibrium_projection(raw, self.dx, self.dy)
        exchange = torch.sigmoid(self.exchange_gate(features.mean(dim=(-2, -1))))[:, :, None, None]
        potential = self.contact_potential(features, inputs[:, 7:8])
        compatible_correction = airy_stress(potential, self.dx, self.dy)
        self.last_exchange = exchange.detach()
        return raw + exchange * (admissible - raw) + compatible_correction


class CompatibleContactPotentialFNO(nn.Module):
    """CCP-FNO: fully compatible projected stress plus contact-potential detail."""

    def __init__(
        self,
        in_channels: int = 8,
        width: int = 24,
        modes_x: int = 12,
        modes_y: int = 12,
        layers: int = 4,
        dx: float = 1.0,
        dy: float = 1.0,
    ):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.backbone = _FNOFeatureBackbone(in_channels, width, modes_x, modes_y, layers)
        self.raw_head = nn.Sequential(
            nn.Conv2d(width, width * 2, 1), nn.GELU(), nn.Conv2d(width * 2, 3, 1)
        )
        self.contact_potential = _ContactPotentialHead(width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        admissible = spectral_equilibrium_projection(self.raw_head(features), self.dx, self.dy)
        potential = self.contact_potential(features, inputs[:, 7:8])
        return admissible + airy_stress(potential, self.dx, self.dy)


class ResidualCompatibleRepairFNO(nn.Module):
    """RCR-FNO: repair a frozen empirical field with a compatible residual student.

    A fixed exchange between raw and projected stress removes a prescribed
    fraction of the discrete non-equilibrium amplitude.  A compact potential operator then learns the
    missing admissible detail from the input, raw field, projected field, and
    removed component.  Since the repair is potential-derived, it cannot restore
    the discarded non-equilibrium component.
    """

    def __init__(
        self,
        baseline: nn.Module,
        width: int = 12,
        modes_x: int = 8,
        modes_y: int = 8,
        layers: int = 2,
        dx: float = 1.0,
        dy: float = 1.0,
        repair_fraction: float = 0.5,
    ):
        super().__init__()
        if not 0.0 <= repair_fraction <= 1.0:
            raise ValueError("repair_fraction must lie in [0, 1]")
        self.dx = dx
        self.dy = dy
        self.repair_fraction = repair_fraction
        self.baseline = baseline
        self.baseline.requires_grad_(False)
        self.baseline.eval()
        self.repair_backbone = _FNOFeatureBackbone(17, width, modes_x, modes_y, layers)
        self.global_potential = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, 1, 1)
        )
        nn.init.zeros_(self.global_potential[-1].weight)
        nn.init.zeros_(self.global_potential[-1].bias)
        self.contact_potential = _ContactPotentialHead(width)

    def train(self, mode: bool = True) -> "ResidualCompatibleRepairFNO":
        super().train(mode)
        self.baseline.eval()
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            raw = self.baseline(inputs)
            admissible = spectral_equilibrium_projection(raw, self.dx, self.dy)
        removed = raw - admissible
        repair_inputs = torch.cat((inputs, raw, admissible, removed), dim=1)
        features = self.repair_backbone(repair_inputs)
        potential = self.global_potential(features)
        potential = potential + self.contact_potential(features, inputs[:, 7:8])
        repaired_base = raw + self.repair_fraction * (admissible - raw)
        return repaired_base + airy_stress(potential, self.dx, self.dy)


class LoadPathModulatedFNO(nn.Module):
    """LPM-FNO: residual spectral transport modulated by operating conditions.

    Channels 3:7 encode indentation, friction, elastic modulus, and mesh phase.
    Their spatial means form a load-path context that applies feature-wise gain
    and bias at every operator layer.  Geometry and contact remain in the full
    spatial stream, avoiding the need to rediscover global condition scalars at
    every grid point.
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 3,
        width: int = 24,
        modes_x: int = 12,
        modes_y: int = 12,
        layers: int = 4,
    ):
        super().__init__()
        self.layers = layers
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spectral = nn.ModuleList(
            SpectralConv2d(width, width, modes_x, modes_y) for _ in range(layers)
        )
        self.local = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1, groups=width),
                nn.Conv2d(width, width, 1),
            )
            for _ in range(layers)
        )
        self.context = nn.Sequential(
            nn.Linear(4, width),
            nn.GELU(),
            nn.Linear(width, 2 * width * layers),
        )
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)
        self.activation = nn.GELU()
        self.project = nn.Sequential(
            nn.Conv2d(width, width * 2, 1), nn.GELU(), nn.Conv2d(width * 2, out_channels, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.lift(inputs)
        operating_context = inputs[:, 3:7].mean(dim=(-2, -1))
        modulation = self.context(operating_context).view(
            inputs.shape[0], self.layers, 2, features.shape[1], 1, 1
        )
        for layer_index, (spectral, local) in enumerate(zip(self.spectral, self.local)):
            gain = 1.0 + 0.1 * torch.tanh(modulation[:, layer_index, 0])
            bias = modulation[:, layer_index, 1]
            update = gain * (spectral(features) + local(features)) + bias
            features = features + self.activation(update)
        return self.project(features)


def _masked_divergence(
    stress: torch.Tensor,
    interior: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    sxx, syy, sxy = stress[:, 0], stress[:, 1], stress[:, 2]
    residual_x = (
        (sxx[:, 1:-1, 2:] - sxx[:, 1:-1, :-2]) / (2.0 * dx)
        + (sxy[:, 2:, 1:-1] - sxy[:, :-2, 1:-1]) / (2.0 * dy)
    )
    residual_y = (
        (sxy[:, 1:-1, 2:] - sxy[:, 1:-1, :-2]) / (2.0 * dx)
        + (syy[:, 2:, 1:-1] - syy[:, :-2, 1:-1]) / (2.0 * dy)
    )
    return torch.stack((residual_x * interior, residual_y * interior), dim=1)


def _masked_divergence_adjoint(
    multiplier: torch.Tensor,
    interior: torch.Tensor,
    height: int,
    width: int,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """Adjoint of ``_masked_divergence`` under the Euclidean grid inner product."""
    multiplier_x = multiplier[:, 0] * interior
    multiplier_y = multiplier[:, 1] * interior
    result = multiplier.new_zeros((multiplier.shape[0], 3, height, width))
    result[:, 0, 1:-1, 2:] += multiplier_x / (2.0 * dx)
    result[:, 0, 1:-1, :-2] -= multiplier_x / (2.0 * dx)
    result[:, 2, 2:, 1:-1] += multiplier_x / (2.0 * dy)
    result[:, 2, :-2, 1:-1] -= multiplier_x / (2.0 * dy)
    result[:, 2, 1:-1, 2:] += multiplier_y / (2.0 * dx)
    result[:, 2, 1:-1, :-2] -= multiplier_y / (2.0 * dx)
    result[:, 1, 2:, 1:-1] += multiplier_y / (2.0 * dy)
    result[:, 1, :-2, 1:-1] -= multiplier_y / (2.0 * dy)
    return result


def _eroded_material_mask(mask: torch.Tensor) -> torch.Tensor:
    material = mask[:, 0]
    return (
        material[:, 1:-1, 1:-1]
        * material[:, 1:-1, 2:]
        * material[:, 1:-1, :-2]
        * material[:, 2:, 1:-1]
        * material[:, :-2, 1:-1]
    )


def _equilibrium_residual_per_sample(
    stress: torch.Tensor,
    mask: torch.Tensor,
    dx: float,
    dy: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    material = mask[:, 0]
    interior = _eroded_material_mask(mask)
    residual = _masked_divergence(stress, interior, dx, dy).square().sum(dim=1)
    interior_count = interior.sum(dim=(1, 2)).clamp_min(1.0)
    residual_mean = residual.sum(dim=(1, 2)) / interior_count
    material_count = material.sum(dim=(1, 2)).clamp_min(1.0)
    stress_rms_sq = (stress.square() * material[:, None]).sum(dim=(1, 2, 3)) / (
        3.0 * material_count
    )
    height, width = stress.shape[-2:]
    length_sq = ((width - 1) * dx) * ((height - 1) * dy)
    return length_sq * residual_mean / stress_rms_sq.clamp_min(eps)


def contact_hotspot_weights(
    stress: torch.Tensor,
    contact_map: torch.Tensor,
    contact_weight: float,
    hotspot_weight: float,
    quantile: float = 0.95,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Positive metric weights that protect contact and predicted critical regions."""
    if contact_weight < 0 or hotspot_weight < 0:
        raise ValueError("preservation weights must be non-negative")
    magnitude = torch.sqrt(stress.square().sum(dim=1) + 1e-8)
    threshold = torch.quantile(magnitude.flatten(1), quantile, dim=1)[:, None, None]
    scale = magnitude.flatten(1).std(dim=1)[:, None, None] * temperature + 1e-6
    critical = torch.sigmoid((magnitude - threshold) / scale)
    spatial_weight = 1.0 + contact_weight * contact_map[:, 0] + hotspot_weight * critical
    return spatial_weight[:, None].expand(-1, 3, -1, -1)


def weighted_equilibrium_projection(
    stress: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    dx: float,
    dy: float,
    iterations: int = 80,
    damping: float = 1e-7,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Closest discrete-equilibrium stress in a positive diagonal metric.

    Solves ``min 0.5 ||W^(1/2)(s-y)||^2`` subject to the masked central-
    difference equilibrium equations.  The KKT Schur complement is evaluated
    matrix-free with conjugate gradients.
    """
    if stress.shape != weight.shape:
        raise ValueError("stress and weight must have identical shapes")
    if torch.any(weight <= 0):
        raise ValueError("weight must be strictly positive")
    interior = _eroded_material_mask(mask)
    height, width = stress.shape[-2:]
    inverse_weight = weight.reciprocal()
    right_hand_side = _masked_divergence(stress, interior, dx, dy)

    def schur(multiplier: torch.Tensor) -> torch.Tensor:
        adjoint = _masked_divergence_adjoint(
            multiplier, interior, height, width, dx, dy
        )
        return _masked_divergence(inverse_weight * adjoint, interior, dx, dy) + damping * multiplier

    multiplier = torch.zeros_like(right_hand_side)
    residual = right_hand_side.clone()
    direction = residual.clone()
    residual_norm = residual.square().sum(dim=(1, 2, 3), keepdim=True)
    for _ in range(iterations):
        image = schur(direction)
        denominator = (direction * image).sum(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
        step = residual_norm / denominator
        multiplier = multiplier + step * direction
        new_residual = residual - step * image
        new_norm = new_residual.square().sum(dim=(1, 2, 3), keepdim=True)
        coefficient = new_norm / residual_norm.clamp_min(eps)
        direction = new_residual + coefficient * direction
        residual = new_residual
        residual_norm = new_norm
    correction = inverse_weight * _masked_divergence_adjoint(
        multiplier, interior, height, width, dx, dy
    )
    return stress - correction


class MinimumInterventionContactEquilibrium(nn.Module):
    """MICE: contact-aware repair using the smallest budget-satisfying exchange.

    The layer first finds the closest equilibrium field in a metric that protects
    contact and predicted hotspot regions.  It then uses bisection to apply only
    the smallest fraction of that correction required to meet an equilibrium
    residual budget.  The operation is deterministic and needs no test labels.
    """

    def __init__(
        self,
        dx: float,
        dy: float,
        equilibrium_budget: float = 100.0,
        contact_weight: float = 4.0,
        hotspot_weight: float = 4.0,
        projection_iterations: int = 80,
        bisection_iterations: int = 16,
    ):
        super().__init__()
        if equilibrium_budget <= 0:
            raise ValueError("equilibrium_budget must be positive")
        self.dx = dx
        self.dy = dy
        self.equilibrium_budget = equilibrium_budget
        self.contact_weight = contact_weight
        self.hotspot_weight = hotspot_weight
        self.projection_iterations = projection_iterations
        self.bisection_iterations = bisection_iterations
        self.last_repair_fraction: torch.Tensor | None = None

    def forward(
        self,
        stress: torch.Tensor,
        mask: torch.Tensor,
        contact_map: torch.Tensor,
    ) -> torch.Tensor:
        weight = contact_hotspot_weights(
            stress,
            contact_map,
            contact_weight=self.contact_weight,
            hotspot_weight=self.hotspot_weight,
        )
        equilibrium = weighted_equilibrium_projection(
            stress,
            mask,
            weight,
            self.dx,
            self.dy,
            iterations=self.projection_iterations,
        )
        low = torch.zeros(stress.shape[0], device=stress.device, dtype=stress.dtype)
        high = torch.ones_like(low)
        raw_residual = _equilibrium_residual_per_sample(stress, mask, self.dx, self.dy)
        already_valid = raw_residual <= self.equilibrium_budget
        for _ in range(self.bisection_iterations):
            middle = 0.5 * (low + high)
            candidate = stress + middle[:, None, None, None] * (equilibrium - stress)
            residual = _equilibrium_residual_per_sample(candidate, mask, self.dx, self.dy)
            meets_budget = residual <= self.equilibrium_budget
            high = torch.where(meets_budget, middle, high)
            low = torch.where(meets_budget, low, middle)
        fraction = torch.where(already_valid, torch.zeros_like(high), high)
        self.last_repair_fraction = fraction.detach()
        return stress + fraction[:, None, None, None] * (equilibrium - stress)


def contact_centroid(
    inputs: torch.Tensor,
    contact_channel: int = 7,
    x_channel: int = 1,
    y_channel: int = 2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Weighted centroid of the contact-location channel, in the x/y channels' own units."""
    weight = inputs[:, contact_channel]
    x = inputs[:, x_channel]
    y = inputs[:, y_channel]
    total = weight.sum(dim=(-2, -1)).clamp_min(eps)
    cx = (x * weight).sum(dim=(-2, -1)) / total
    cy = (y * weight).sum(dim=(-2, -1)) / total
    return torch.stack((cx, cy), dim=1)


def _local_sampling_grid(center: torch.Tensor, window: float, size: int, coord_extent: float) -> torch.Tensor:
    """``grid_sample`` grid that gathers a ``size x size`` patch centered at ``center``."""
    offsets = torch.linspace(-window / 2, window / 2, size, device=center.device, dtype=center.dtype)
    grid_y, grid_x = torch.meshgrid(offsets, offsets, indexing="ij")
    grid_x = grid_x.unsqueeze(0) + center[:, 0].view(-1, 1, 1)
    grid_y = grid_y.unsqueeze(0) + center[:, 1].view(-1, 1, 1)
    grid = torch.stack((grid_x / coord_extent, grid_y / coord_extent), dim=-1)
    return grid.clamp(-1.0, 1.0)


def _inverse_sampling_grid(
    center: torch.Tensor, window: float, full_size: int, coord_extent: float
) -> torch.Tensor:
    """``grid_sample`` grid that scatters a local patch back onto the full ``full_size`` canvas."""
    coords = torch.linspace(-coord_extent, coord_extent, full_size, device=center.device, dtype=center.dtype)
    full_y, full_x = torch.meshgrid(coords, coords, indexing="ij")
    full_x = full_x.unsqueeze(0)
    full_y = full_y.unsqueeze(0)
    relative_x = (full_x - center[:, 0].view(-1, 1, 1)) / (window / 2)
    relative_y = (full_y - center[:, 1].view(-1, 1, 1)) / (window / 2)
    return torch.stack((relative_x, relative_y), dim=-1)


class ContactCenteredMultiscaleFNO(nn.Module):
    """CCM-FNO: an untruncated local branch recovers the high-frequency detail
    that FNO's spectral mode truncation structurally low-pass filters away.

    ``SpectralConv2d`` retains only the lowest ``modes_x x modes_y`` Fourier
    coefficients -- an explicit spatial low-pass filter with cutoff
    wavelength roughly ``grid_size / (2 * modes)``. The contact hotspot is,
    by definition, the sharpest feature in the field: exactly what a
    low-mode truncation smooths. This mechanism does not touch the
    equilibrium constraint, the boundary/periodicity assumption, or load
    conditioning -- every other invention in this module does one of those
    three things instead. A plain (non-spectral) CNN processes a small
    window gathered around the contact centroid at full input resolution,
    with no mode truncation, and its zero-initialized output is blended back
    at full resolution through a smooth Gaussian gate so training starts
    exactly at the global-only baseline. The global branch also receives two
    extra contact-relative coordinate channels, so every layer can read
    "distance and direction from contact" directly instead of reconstructing
    it from a diffuse Gaussian map with only ~4 examples per phase position.
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 3,
        width: int = 24,
        modes_x: int = 12,
        modes_y: int = 12,
        layers: int = 4,
        local_width: int = 16,
        local_window: float = 0.375,
        local_size: int = 16,
        gate_sigma_fraction: float = 0.25,
        contact_channel: int = 7,
        x_channel: int = 1,
        y_channel: int = 2,
        coord_extent: float = 0.5,
    ):
        super().__init__()
        self.contact_channel = contact_channel
        self.x_channel = x_channel
        self.y_channel = y_channel
        self.coord_extent = coord_extent
        self.local_window = local_window
        self.local_size = local_size
        self.gate_sigma = max(gate_sigma_fraction * local_window, 1e-6)

        self.global_lift = nn.Conv2d(in_channels + 2, width, 1)
        self.global_spectral = nn.ModuleList(
            SpectralConv2d(width, width, modes_x, modes_y) for _ in range(layers)
        )
        self.global_local = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.global_activation = nn.GELU()
        self.global_project = nn.Sequential(
            nn.Conv2d(width, width * 2, 1), nn.GELU(), nn.Conv2d(width * 2, out_channels, 1)
        )

        self.local_net = nn.Sequential(
            nn.Conv2d(in_channels, local_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(local_width, local_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(local_width, out_channels, 1),
        )
        nn.init.zeros_(self.local_net[-1].weight)
        nn.init.zeros_(self.local_net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, _ = x.shape
        center = contact_centroid(x, self.contact_channel, self.x_channel, self.y_channel)

        relative_x = (x[:, self.x_channel] - center[:, 0].view(-1, 1, 1)) / self.coord_extent
        relative_y = (x[:, self.y_channel] - center[:, 1].view(-1, 1, 1)) / self.coord_extent
        global_input = torch.cat((x, relative_x.unsqueeze(1), relative_y.unsqueeze(1)), dim=1)

        features = self.global_lift(global_input)
        for spectral, local in zip(self.global_spectral, self.global_local):
            features = self.global_activation(spectral(features) + local(features))
        global_out = self.global_project(features)

        forward_grid = _local_sampling_grid(center, self.local_window, self.local_size, self.coord_extent)
        patch = functional.grid_sample(x, forward_grid, mode="bilinear", align_corners=True, padding_mode="border")
        delta = self.local_net(patch)

        inverse_grid = _inverse_sampling_grid(center, self.local_window, height, self.coord_extent)
        local_correction = functional.grid_sample(
            delta, inverse_grid, mode="bilinear", align_corners=True, padding_mode="zeros"
        )

        xs = x[:, self.x_channel]
        ys = x[:, self.y_channel]
        distance_sq = (xs - center[:, 0].view(-1, 1, 1)).square() + (ys - center[:, 1].view(-1, 1, 1)).square()
        gate = torch.exp(-distance_sq / (2.0 * self.gate_sigma**2)).unsqueeze(1)

        return global_out + gate * local_correction


def _von_mises_like(stress: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    sxx, syy, sxy = stress[:, 0], stress[:, 1], stress[:, 2]
    return torch.sqrt((sxx.square() - sxx * syy + syy.square() + 3.0 * sxy.square()).clamp_min(eps))


def _soft_argmax_centroid(
    magnitude: torch.Tensor, x: torch.Tensor, y: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Differentiable soft-argmax centroid of ``magnitude``, weighted by a softmax over its values."""
    batch = magnitude.shape[0]
    weights = torch.softmax(magnitude.reshape(batch, -1) / temperature, dim=1).view_as(magnitude)
    cx = (x * weights).sum(dim=(-2, -1))
    cy = (y * weights).sum(dim=(-2, -1))
    return torch.stack((cx, cy), dim=1)


class HotspotGuidedMultiscaleFNO(nn.Module):
    """HGM-FNO: local high-frequency refinement centered on the model's own
    predicted hotspot, not the geometric contact point.

    ``ContactCenteredMultiscaleFNO`` centered its local refinement window on
    the input contact-location channel's centroid, on the assumption that
    the field's sharpest feature occurs at the contact point. Measured
    directly on this dataset, that assumption is false: the true top-5%
    von-Mises stress centroid sits on average 0.26 coordinate units from the
    contact centroid, in a domain that only spans 1.0 unit -- every single
    sample exceeds a 0.15-unit offset (`docs/CCM_FNO_STUDY_2026-07-24.md`).
    The stress concentration is almost certainly at the tooth-root fillet, a
    location physically distinct from the Hertzian contact point, not a bug
    in the contact-map channel.

    HGM-FNO fixes this by computing the local window's center from a
    differentiable soft-argmax over the *global branch's own predicted*
    von-Mises-like magnitude field, so the window follows wherever the model
    currently believes the peak is (and can move as training improves that
    belief), instead of a fixed geometric assumption that is measurably
    wrong on this dataset.
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 3,
        width: int = 24,
        modes_x: int = 12,
        modes_y: int = 12,
        layers: int = 4,
        local_width: int = 16,
        local_window: float = 0.375,
        local_size: int = 16,
        gate_sigma_fraction: float = 0.25,
        softmax_temperature: float = 0.1,
        contact_channel: int = 7,
        x_channel: int = 1,
        y_channel: int = 2,
        coord_extent: float = 0.5,
    ):
        super().__init__()
        self.contact_channel = contact_channel
        self.x_channel = x_channel
        self.y_channel = y_channel
        self.coord_extent = coord_extent
        self.local_window = local_window
        self.local_size = local_size
        self.gate_sigma = max(gate_sigma_fraction * local_window, 1e-6)
        self.softmax_temperature = softmax_temperature

        self.global_lift = nn.Conv2d(in_channels + 2, width, 1)
        self.global_spectral = nn.ModuleList(
            SpectralConv2d(width, width, modes_x, modes_y) for _ in range(layers)
        )
        self.global_local = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.global_activation = nn.GELU()
        self.global_project = nn.Sequential(
            nn.Conv2d(width, width * 2, 1), nn.GELU(), nn.Conv2d(width * 2, out_channels, 1)
        )

        self.local_net = nn.Sequential(
            nn.Conv2d(in_channels, local_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(local_width, local_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(local_width, out_channels, 1),
        )
        nn.init.zeros_(self.local_net[-1].weight)
        nn.init.zeros_(self.local_net[-1].bias)
        self.last_hotspot_center: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, _ = x.shape
        contact_center = contact_centroid(x, self.contact_channel, self.x_channel, self.y_channel)

        relative_x = (x[:, self.x_channel] - contact_center[:, 0].view(-1, 1, 1)) / self.coord_extent
        relative_y = (x[:, self.y_channel] - contact_center[:, 1].view(-1, 1, 1)) / self.coord_extent
        global_input = torch.cat((x, relative_x.unsqueeze(1), relative_y.unsqueeze(1)), dim=1)

        features = self.global_lift(global_input)
        for spectral, local in zip(self.global_spectral, self.global_local):
            features = self.global_activation(spectral(features) + local(features))
        global_out = self.global_project(features)

        magnitude = _von_mises_like(global_out)
        hotspot_center = _soft_argmax_centroid(
            magnitude, x[:, self.x_channel], x[:, self.y_channel], self.softmax_temperature
        )
        self.last_hotspot_center = hotspot_center.detach()

        forward_grid = _local_sampling_grid(hotspot_center, self.local_window, self.local_size, self.coord_extent)
        patch = functional.grid_sample(x, forward_grid, mode="bilinear", align_corners=True, padding_mode="border")
        delta = self.local_net(patch)

        inverse_grid = _inverse_sampling_grid(hotspot_center, self.local_window, height, self.coord_extent)
        local_correction = functional.grid_sample(
            delta, inverse_grid, mode="bilinear", align_corners=True, padding_mode="zeros"
        )

        xs = x[:, self.x_channel]
        ys = x[:, self.y_channel]
        distance_sq = (xs - hotspot_center[:, 0].view(-1, 1, 1)).square() + (
            ys - hotspot_center[:, 1].view(-1, 1, 1)
        ).square()
        gate = torch.exp(-distance_sq / (2.0 * self.gate_sigma**2)).unsqueeze(1)

        return global_out + gate * local_correction


class DualWindowMultiscaleFNO(nn.Module):
    """DWM-FNO: both local-refinement windows from CCM-FNO and HGM-FNO at once.

    CCM-FNO (contact-centered) and HGM-FNO (hotspot-guided) individually
    gave clean null results, but they attend to two physically distinct
    locations -- the Hertzian contact point and the (measurably different,
    on this dataset ~0.26 coordinate units away on average) tooth-root
    bending-stress concentration. Neither mechanism alone helped, but that
    does not establish that both signals together are redundant; this
    composes both, independently zero-initialized, so the combination can
    only add signal beyond either used alone, never conflict with it at
    initialization. Built entirely from the same, already-unit-tested
    primitives as CCM-FNO/HGM-FNO (``contact_centroid``,
    ``_local_sampling_grid``, ``_inverse_sampling_grid``, ``_von_mises_like``,
    ``_soft_argmax_centroid``).
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 3,
        width: int = 24,
        modes_x: int = 12,
        modes_y: int = 12,
        layers: int = 4,
        local_width: int = 16,
        local_window: float = 0.375,
        local_size: int = 16,
        gate_sigma_fraction: float = 0.25,
        softmax_temperature: float = 0.1,
        contact_channel: int = 7,
        x_channel: int = 1,
        y_channel: int = 2,
        coord_extent: float = 0.5,
    ):
        super().__init__()
        self.contact_channel = contact_channel
        self.x_channel = x_channel
        self.y_channel = y_channel
        self.coord_extent = coord_extent
        self.local_window = local_window
        self.local_size = local_size
        self.gate_sigma = max(gate_sigma_fraction * local_window, 1e-6)
        self.softmax_temperature = softmax_temperature

        self.global_lift = nn.Conv2d(in_channels + 2, width, 1)
        self.global_spectral = nn.ModuleList(
            SpectralConv2d(width, width, modes_x, modes_y) for _ in range(layers)
        )
        self.global_local = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.global_activation = nn.GELU()
        self.global_project = nn.Sequential(
            nn.Conv2d(width, width * 2, 1), nn.GELU(), nn.Conv2d(width * 2, out_channels, 1)
        )

        def _make_local_net() -> nn.Sequential:
            net = nn.Sequential(
                nn.Conv2d(in_channels, local_width, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(local_width, local_width, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(local_width, out_channels, 1),
            )
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            return net

        self.contact_local_net = _make_local_net()
        self.hotspot_local_net = _make_local_net()
        self.last_contact_center: torch.Tensor | None = None
        self.last_hotspot_center: torch.Tensor | None = None

    def _refine(self, x: torch.Tensor, center: torch.Tensor, local_net: nn.Module, height: int) -> torch.Tensor:
        forward_grid = _local_sampling_grid(center, self.local_window, self.local_size, self.coord_extent)
        patch = functional.grid_sample(x, forward_grid, mode="bilinear", align_corners=True, padding_mode="border")
        delta = local_net(patch)
        inverse_grid = _inverse_sampling_grid(center, self.local_window, height, self.coord_extent)
        correction = functional.grid_sample(delta, inverse_grid, mode="bilinear", align_corners=True, padding_mode="zeros")
        xs = x[:, self.x_channel]
        ys = x[:, self.y_channel]
        distance_sq = (xs - center[:, 0].view(-1, 1, 1)).square() + (ys - center[:, 1].view(-1, 1, 1)).square()
        gate = torch.exp(-distance_sq / (2.0 * self.gate_sigma**2)).unsqueeze(1)
        return gate * correction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, _ = x.shape
        contact_center = contact_centroid(x, self.contact_channel, self.x_channel, self.y_channel)
        self.last_contact_center = contact_center.detach()

        relative_x = (x[:, self.x_channel] - contact_center[:, 0].view(-1, 1, 1)) / self.coord_extent
        relative_y = (x[:, self.y_channel] - contact_center[:, 1].view(-1, 1, 1)) / self.coord_extent
        global_input = torch.cat((x, relative_x.unsqueeze(1), relative_y.unsqueeze(1)), dim=1)

        features = self.global_lift(global_input)
        for spectral, local in zip(self.global_spectral, self.global_local):
            features = self.global_activation(spectral(features) + local(features))
        global_out = self.global_project(features)

        magnitude = _von_mises_like(global_out)
        hotspot_center = _soft_argmax_centroid(
            magnitude, x[:, self.x_channel], x[:, self.y_channel], self.softmax_temperature
        )
        self.last_hotspot_center = hotspot_center.detach()

        contact_correction = self._refine(x, contact_center, self.contact_local_net, height)
        hotspot_correction = self._refine(x, hotspot_center, self.hotspot_local_net, height)

        return global_out + contact_correction + hotspot_correction
