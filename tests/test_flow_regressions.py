import copy

import pytest
import torch

from transtimegrad.modules.distribution_output import FlowOutput
from transtimegrad.modules.flows import MADE, MAF, BatchNorm, RealNVP


@pytest.mark.parametrize("dim", [1, 4])
@pytest.mark.parametrize("order", ["sequential", "random"])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_made_inverse_and_jacobian(dim, order, seed):
    torch.manual_seed(seed)
    model = MADE(dim, 12, 2, input_order=order, cond_label_size=2).double().eval()
    cond = torch.randn(2, dtype=torch.double)
    value = torch.randn(dim, dtype=torch.double)
    latent, forward_jac = model(value, cond)
    recovered, inverse_jac = model.inverse(latent, cond)
    torch.testing.assert_close(value, recovered)
    torch.testing.assert_close(forward_jac + inverse_jac, torch.zeros_like(value))
    actual = torch.autograd.functional.jacobian(lambda v: model(v, cond)[0], value)
    torch.testing.assert_close(torch.linalg.slogdet(actual).logabsdet, forward_jac.sum())
    # Inverse must remain differentiable, without in-place version-counter errors.
    model.inverse(latent, cond)[0].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("flow_cls", [MAF, RealNVP])
def test_flow_adapter_captures_independent_conditioning(flow_cls):
    torch.manual_seed(9)
    flow = flow_cls(2, 2, 8, 1, cond_label_size=3, batch_norm=False).eval()
    output = FlowOutput(flow, input_size=2, cond_size=3)
    cond = torch.randn(2, 4, 3, requires_grad=True)
    loc, scale = torch.ones(2, 1, 2), torch.full((2, 1, 2), 2.0)
    first = output.distribution((cond,), loc=loc, scale=scale)
    values = torch.randn(2, 4, 2)
    before = first.log_prob(values)
    output.distribution((cond + 10,), loc=torch.zeros_like(loc), scale=torch.ones_like(scale))
    with torch.no_grad():
        loc.add_(10)
        scale.add_(10)
    torch.testing.assert_close(first.log_prob(values), before)
    assert first.sample((3, 5)).shape == (3, 5, 2, 4, 2)
    assert first.log_prob(first.sample((3,))).shape == (3, 2, 4)
    (-before.mean()).backward()
    assert torch.isfinite(cond.grad).all()
    assert flow.scale is None and flow.loc is None


def test_random_order_state_dict_roundtrip():
    torch.manual_seed(3)
    first = MAF(2, 4, 12, 1, input_order="random", batch_norm=False).eval()
    torch.manual_seed(18)
    second = MAF(2, 4, 12, 1, input_order="random", batch_norm=False).eval()
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    value = torch.randn(2, 4)
    latent, _ = first(value)
    torch.testing.assert_close(second.inverse(latent)[0], value)


@pytest.mark.parametrize("count", [1, 4])
def test_batchnorm_training_and_eval_are_finite_and_invertible(count):
    layer = BatchNorm(2)
    value = torch.randn(count, 2, requires_grad=True)
    mapped, jac = layer(value)
    recovered, inv_jac = layer.inverse(mapped)
    torch.testing.assert_close(recovered, value)
    torch.testing.assert_close(jac + inv_jac, torch.zeros_like(value))
    mapped.sum().backward()
    assert torch.isfinite(value.grad).all() and torch.isfinite(layer.running_var).all()
    layer.eval()
    mapped, _ = layer(value)
    torch.testing.assert_close(layer.inverse(mapped)[0], value)


def test_first_autoregressive_coordinate_can_use_conditioning():
    model = MADE(1, 8, 1, cond_label_size=2)
    cond = torch.randn(2, requires_grad=True)
    model(torch.zeros(1), cond)[0].sum().backward()
    assert cond.grad.abs().sum() > 0


def test_flow_output_parameters_are_registered_and_trained_in_transformer():
    from test_model_logic import make_batch

    from transtimegrad.model.transformer import TransformerModel

    flow = RealNVP(2, 2, 8, 1, cond_label_size=3, batch_norm=False)
    output = FlowOutput(flow, input_size=2, cond_size=3)
    model = TransformerModel(
        freq="D",
        context_length=3,
        prediction_length=3,
        input_size=2,
        d_model=8,
        nhead=2,
        lags_seq=[1],
        distr_output=output,
    )
    parameter_ids = {id(p) for p in model.parameters()}
    assert all(id(p) in parameter_ids for p in flow.parameters())
    assert any(key.startswith("distr_output.flow.") for key in model.state_dict())
    model.loss(**make_batch(model)).mean().backward()
    gradients = [p.grad for p in flow.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0
    model.double()
    assert all(p.dtype == torch.float64 for p in flow.parameters())


def test_direct_flow_affine_defaults_move_with_module_and_accept_scalar_scale():
    flow = RealNVP(1, 2, 8, 1, batch_norm=False).eval()
    flow.scale, flow.loc = 2.0, 3.0
    flow.double()
    assert flow.scale.dtype == torch.float64 and flow.loc.dtype == torch.float64
    values = torch.randn(2, 2, dtype=torch.float64)
    latent, jac = flow(values)
    recovered, inv_jac = flow.inverse(latent)
    torch.testing.assert_close(values, recovered)
    torch.testing.assert_close(jac + inv_jac, torch.zeros_like(values))


def test_training_batchnorm_logdet_matches_finite_difference_with_frozen_prior_state():
    layer = BatchNorm(2).double()
    layer.running_mean.copy_(torch.tensor([0.2, -0.1], dtype=torch.double))
    layer.running_var.copy_(torch.tensor([0.5, 1.3], dtype=torch.double))
    value = torch.tensor([[0.3, 0.7], [-0.5, 0.1]], dtype=torch.double)
    prior = copy.deepcopy(layer.state_dict())
    mapped, jac = layer(value)
    recovered, _ = layer.inverse(mapped)
    torch.testing.assert_close(recovered, value)
    # Reset the prior state for each evaluation; current observations cannot
    # influence the affine transform that is applied to those observations.
    epsilon = 1e-5
    columns = []
    for index in range(value.numel()):
        perturbation = torch.zeros_like(value).reshape(-1)
        perturbation[index] = epsilon
        perturbation = perturbation.reshape_as(value)
        layer.load_state_dict(prior)
        plus = layer(value + perturbation)[0]
        layer.load_state_dict(prior)
        minus = layer(value - perturbation)[0]
        columns.append(((plus - minus) / (2 * epsilon)).reshape(-1))
    finite_difference = torch.stack(columns, dim=1)
    torch.testing.assert_close(torch.linalg.slogdet(finite_difference).logabsdet, jac.sum())


@pytest.mark.parametrize("flow_cls", [RealNVP, MAF])
def test_invalid_flow_configuration_fails_early(flow_cls):
    with pytest.raises(ValueError):
        flow_cls(-1, 2, 8, 1)
    flow = flow_cls(1, 2, 8, 1)
    with pytest.raises(ValueError, match="event dimension"):
        FlowOutput(flow, input_size=3, cond_size=2)
