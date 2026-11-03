import math

import torch
from torch import nn

from transtimegrad.modules.flows import Flow


class Identity(nn.Module):
    def forward(self, x, cond):
        return x, torch.zeros_like(x)

    def inverse(self, x, cond):
        return x, torch.zeros_like(x)


def test_scaled_density_matches_analytic_normal_and_preserves_input():
    flow = Flow(2)
    flow.net = Identity()
    flow.scale = torch.tensor([2.0, 3.0])
    values = torch.tensor([[0.0, 1.0], [2.0, -3.0]], requires_grad=True)
    original = values.detach().clone()
    actual = flow.log_prob(values, None)
    expected = torch.distributions.Normal(torch.zeros(2), flow.scale).log_prob(values).sum(-1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(values, original)
    actual.sum().backward()
    assert torch.isfinite(values.grad).all()


def test_scaled_flow_jacobians_cancel():
    flow = Flow(2)
    flow.net = Identity()
    flow.scale = torch.tensor([2.0, 3.0])
    values = torch.tensor([[2.0, -3.0]])
    latent, forward_jac = flow(values, None)
    restored, inverse_jac = flow.inverse(latent, None)
    torch.testing.assert_close(restored, values)
    torch.testing.assert_close(forward_jac + inverse_jac, torch.zeros_like(values))
    torch.testing.assert_close(forward_jac.sum(-1), torch.tensor([-math.log(6)]))
