"""Compact 2D baselines for the feasibility study."""

from __future__ import annotations

import torch
from torch import nn


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int):
        super().__init__()
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1.0 / max(1, in_channels * out_channels)
        shape = (in_channels, out_channels, modes_x, modes_y)
        self.weight_pos = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_neg = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def _multiply(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        modes_x = min(self.modes_x, height // 2)
        modes_y = min(self.modes_y, width // 2 + 1)
        out_ft = torch.zeros(
            batch,
            self.weight_pos.shape[1],
            height,
            width // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, :modes_x, :modes_y] = self._multiply(
            x_ft[:, :, :modes_x, :modes_y], self.weight_pos[:, :, :modes_x, :modes_y]
        )
        out_ft[:, :, -modes_x:, :modes_y] = self._multiply(
            x_ft[:, :, -modes_x:, :modes_y], self.weight_neg[:, :, :modes_x, :modes_y]
        )
        return torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")


class FNO2d(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 3,
        width: int = 24,
        modes_x: int = 12,
        modes_y: int = 12,
        layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spectral = nn.ModuleList(
            SpectralConv2d(width, width, modes_x, modes_y) for _ in range(layers)
        )
        self.local = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.activation = nn.GELU()
        # dropout=0.0 is a no-op (scale factor 1/(1-p)=1, nothing zeroed), so this
        # is exactly backward compatible with every existing FNO2d(...) call site.
        self.dropout_layers = nn.ModuleList(nn.Dropout(dropout) for _ in range(layers))
        self.project = nn.Sequential(nn.Conv2d(width, width * 2, 1), nn.GELU(), nn.Conv2d(width * 2, out_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for spectral, local, drop in zip(self.spectral, self.local, self.dropout_layers):
            x = drop(self.activation(spectral(x) + local(x)))
        return self.project(x)


class UNetSmall(nn.Module):
    """Small matched-purpose convolutional field baseline."""

    def __init__(self, in_channels: int = 8, out_channels: int = 3, width: int = 24):
        super().__init__()
        self.enc1 = self._block(in_channels, width)
        self.enc2 = self._block(width, width * 2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._block(width * 2, width * 4)
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.dec2 = self._block(width * 4, width * 2)
        self.up1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.dec1 = self._block(width * 2, width)
        self.out = nn.Conv2d(width, out_channels, 1)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat((self.up2(b), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        return self.out(d1)


class DeepONet(nn.Module):
    """Branch-trunk operator baseline (multi-output DeepONet, Lu et al. 2021).

    Branch net is a small CNN over the *full* 8-channel input -- the same
    information every other model in this project sees, so the comparison
    isn't handicapped by starving the branch of the contact-map/condition
    channels FNO and UNetSmall both get. Trunk net is an MLP over query
    coordinates, read directly from the input's own x_norm/y_norm channels
    (indices 1-2) rather than a hardcoded grid, so this works for any H, W --
    not just the real dataset's 64x64. Trunk is shared across the 3 output
    channels; the branch produces a separate coefficient vector per channel
    (the standard multi-output extension), combined via a per-point dot
    product plus a learned per-channel bias.
    """

    def __init__(self, in_channels: int = 8, out_channels: int = 3, width: int = 32, basis_dim: int = 64):
        super().__init__()
        self.out_channels = out_channels
        self.basis_dim = basis_dim
        self.branch_cnn = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 2, 3, padding=1),
            nn.GELU(),
        )
        self.branch_pool = nn.AdaptiveAvgPool2d(1)
        self.branch_head = nn.Sequential(
            nn.Linear(width * 2, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, basis_dim * out_channels),
        )
        self.trunk = nn.Sequential(
            nn.Linear(2, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, basis_dim),
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        branch_feat = self.branch_pool(self.branch_cnn(x)).flatten(1)
        branch_out = self.branch_head(branch_feat).view(batch, self.out_channels, self.basis_dim)

        coords = x[:, 1:3, :, :].permute(0, 2, 3, 1).reshape(batch, height * width, 2)
        trunk_out = self.trunk(coords)

        out = torch.einsum("bop,bhp->boh", branch_out, trunk_out)
        out = out + self.bias.view(1, self.out_channels, 1)
        return out.reshape(batch, self.out_channels, height, width)


class PODDeepONet(nn.Module):
    """DeepONet whose trunk is fixed to the POD modes of the training fields (POD-DeepONet, Lu et al. 2022).

    The branch is an MLP over the four scalar input channels (indices 3-6: approach, friction
    coefficient, elastic modulus and contact position, each uniform over the grid), standardized
    with training statistics. For one gear geometry the other input channels (mask, coordinates and
    the analytically placed contact map) follow from these four, so the model sees the same
    information as the others. The branch returns ``rank`` coefficients in units of each mode's
    standard deviation over the training fields; the prediction is the training mean field plus
    the coefficient-weighted modes. The basis and the standardization are buffers set by
    ``fit_basis`` on the training cases, so a saved state dict restores them.
    """

    SCALARS = slice(3, 7)

    def __init__(self, rank: int = 32, width: int = 64, out_channels: int = 3, height: int = 64, width_px: int = 64):
        super().__init__()
        self.field_shape = (out_channels, height, width_px)
        n = out_channels * height * width_px
        self.branch = nn.Sequential(
            nn.Linear(4, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, rank),
        )
        self.register_buffer("scalar_mean", torch.zeros(4))
        self.register_buffer("scalar_std", torch.ones(4))
        self.register_buffer("field_mean", torch.zeros(n))
        self.register_buffer("modes", torch.zeros(rank, n))
        self.register_buffer("mode_std", torch.ones(rank))

    @torch.no_grad()
    def fit_basis(self, inputs, targets) -> "PODDeepONet":
        """Set the scalar standardization and the POD basis from the training inputs and (scaled) targets."""
        import numpy as np

        rank = self.modes.shape[0]
        scal = np.asarray(inputs, dtype=np.float64)[:, self.SCALARS].mean(axis=(2, 3))
        std = scal.std(0)
        std[std < 1e-12] = 1.0
        fields = np.asarray(targets, dtype=np.float64).reshape(len(targets), -1)
        if rank > len(fields) - 1:
            raise ValueError(f"POD rank {rank} needs at least {rank + 1} training cases, got {len(fields)}")
        mean = fields.mean(0)
        _, sv, vt = np.linalg.svd(fields - mean, full_matrices=False)
        self.scalar_mean.copy_(torch.from_numpy(scal.mean(0)))
        self.scalar_std.copy_(torch.from_numpy(std))
        self.field_mean.copy_(torch.from_numpy(mean))
        self.modes.copy_(torch.from_numpy(vt[:rank]))
        self.mode_std.copy_(torch.from_numpy(sv[:rank] / np.sqrt(len(fields))))
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x[:, self.SCALARS].mean(dim=(2, 3)) - self.scalar_mean) / self.scalar_std
        coeff = self.branch(z) * self.mode_std
        out = self.field_mean + coeff @ self.modes
        return out.reshape(x.shape[0], *self.field_shape)

