import numpy as np
import torch

from gearstress.models import PODDeepONet


class _Fixed(torch.nn.Module):
    """Stands in for the branch and returns given coefficients, to test the reconstruction path alone."""

    def __init__(self, out: torch.Tensor):
        super().__init__()
        self.out = out

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out


def _data(n=12, h=8, w=8, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8, h, w)).astype(np.float32)
    x[:, 3:7] = rng.normal(size=(n, 4, 1, 1))          # scalar channels, uniform over the grid
    y = rng.normal(size=(n, 3, h, w)).astype(np.float32)
    return x, y


def test_pod_deeponet_basis_reproduces_training_fields_at_full_rank():
    x, y = _data()
    n = len(x)
    model = PODDeepONet(rank=n - 1, width=16, height=8, width_px=8).fit_basis(x, y)
    flat = torch.from_numpy(y.reshape(n, -1)).double()
    coeff = (flat - model.field_mean.double()) @ model.modes.double().T / model.mode_std.double()
    model.branch = _Fixed(coeff.float())
    assert torch.allclose(model(torch.from_numpy(x)), torch.from_numpy(y), atol=1e-4)


def test_pod_deeponet_reads_only_the_scalar_channels():
    x, y = _data()
    model = PODDeepONet(rank=4, width=16, height=8, width_px=8).fit_basis(x, y).eval()
    x2 = x.copy()
    x2[:, [0, 1, 2, 7]] = 0.0
    with torch.no_grad():
        assert torch.equal(model(torch.from_numpy(x)), model(torch.from_numpy(x2)))


def test_pod_deeponet_state_dict_restores_the_fixed_basis():
    x, y = _data()
    model = PODDeepONet(rank=4, width=16, height=8, width_px=8).fit_basis(x, y).eval()
    fresh = PODDeepONet(rank=4, width=16, height=8, width_px=8).eval()
    fresh.load_state_dict(model.state_dict())
    with torch.no_grad():
        out = fresh(torch.from_numpy(x))
        assert out.shape == (len(x), 3, 8, 8)
        assert torch.equal(out, model(torch.from_numpy(x)))
    trainable = dict(model.named_parameters())
    assert all(name.startswith("branch.") for name in trainable)


def test_pod_deeponet_rejects_a_rank_above_the_training_cases():
    x, y = _data(n=5)
    try:
        PODDeepONet(rank=5, width=8, height=8, width_px=8).fit_basis(x, y)
    except ValueError:
        return
    raise AssertionError("rank 5 from 5 cases should be rejected")
