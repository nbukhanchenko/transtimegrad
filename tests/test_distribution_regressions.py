import pytest
import torch

from transtimegrad.distributions import ZeroInflatedPoisson
from transtimegrad.modules.distribution_output import (
    BetaOutput,
    LowRankMultivariateNormalOutput,
    MultivariateNormalOutput,
    NegativeBinomialOutput,
    NormalMixtureOutput,
    NormalOutput,
    PoissonOutput,
    StudentTMixtureOutput,
    StudentTOutput,
    ZeroInflatedNegativeBinomialOutput,
    ZeroInflatedPoissonOutput,
    squareplus,
)

OUTPUTS = [
    NormalOutput,
    BetaOutput,
    PoissonOutput,
    NegativeBinomialOutput,
    ZeroInflatedPoissonOutput,
    ZeroInflatedNegativeBinomialOutput,
    StudentTOutput,
]


@pytest.mark.parametrize("output_cls", OUTPUTS)
@pytest.mark.parametrize("dim", [None, 1, 3])
def test_distribution_shapes_and_gradient(output_cls, dim):
    output = output_cls(dim=dim)
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    args = output.get_args_proj(8)(hidden)
    dist = output.distribution(args)
    event = () if dim is None else (dim,)
    assert dist.batch_shape == (2, 4)
    assert dist.event_shape == event
    values = torch.full((2, 4) + event, output.value_in_support)
    log_prob = dist.log_prob(values)
    assert log_prob.shape == (2, 4)
    assert torch.isfinite(log_prob).all()
    (-log_prob.mean()).backward()
    assert torch.isfinite(hidden.grad).all()
    assert dist.sample((5,)).shape == (5, 2, 4) + event


@pytest.mark.parametrize("output_cls", [NormalMixtureOutput, StudentTMixtureOutput])
@pytest.mark.parametrize("components", [1, 3])
def test_mixture_component_dimension(output_cls, components):
    output = output_cls(components=components)
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    dist = output.distribution(output.get_args_proj(8)(hidden))
    assert dist.batch_shape == (2, 4)
    assert dist.sample((5,)).shape == (5, 2, 4)
    loss = -dist.log_prob(torch.zeros(2, 4)).mean()
    loss.backward()
    assert torch.isfinite(hidden.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_squareplus_extremes_and_gradients(dtype):
    x = torch.tensor([-1e30, -1e4, 0.0, 1e4, 1e30], dtype=dtype, requires_grad=True)
    mapped = squareplus(x)
    assert (mapped > 0).all() and torch.isfinite(mapped).all()
    mapped.log().sum().backward()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(
        squareplus(torch.tensor([0.0], dtype=dtype)), torch.ones(1, dtype=dtype)
    )


@pytest.mark.parametrize(
    "output_cls",
    [
        PoissonOutput,
        NegativeBinomialOutput,
        ZeroInflatedPoissonOutput,
        ZeroInflatedNegativeBinomialOutput,
    ],
)
def test_count_scaling_does_not_mutate_args(output_cls):
    output = output_cls()
    args = output.get_args_proj(3)(torch.randn(2, 3, requires_grad=True))
    snapshot = [arg.detach().clone() for arg in args]
    original_mean = output.distribution(args).mean
    scaled = output.distribution(args, scale=torch.tensor(2.0))
    torch.testing.assert_close(scaled.mean, original_mean * 2)
    for arg, old in zip(args, snapshot, strict=True):
        torch.testing.assert_close(arg, old)
    with pytest.raises(ValueError, match="centering"):
        output.distribution(args, loc=torch.tensor(1.0))


def test_zero_inflation_boundaries_and_underflow():
    gates = torch.tensor([0.0, 1.0, 0.5], requires_grad=True)
    rates = torch.tensor([1000.0, 2.0, 1000.0], requires_grad=True)
    dist = ZeroInflatedPoisson(gates, rates)
    result = dist.log_prob(torch.zeros(3))
    torch.testing.assert_close(result, torch.tensor([-1000.0, 0.0, -0.69314718]))
    (-result.sum()).backward()
    assert torch.isfinite(gates.grad).all() and torch.isfinite(rates.grad).all()
    assert torch.all(dist.variance >= 0)
    expanded = dist.expand((2, 3))
    assert expanded.sample((4,)).shape == (4, 2, 3)


@pytest.mark.parametrize("output_cls", [MultivariateNormalOutput, LowRankMultivariateNormalOutput])
def test_multivariate_covariance_and_affine_scaling(output_cls):
    output = (
        output_cls(3, rank=2) if output_cls is LowRankMultivariateNormalOutput else output_cls(3)
    )
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    dist = output.distribution(
        output.get_args_proj(8)(hidden), loc=torch.ones(2, 1, 3), scale=torch.ones(2, 1, 3) * 2
    )
    assert dist.batch_shape == (2, 4) and dist.event_shape == (3,)
    assert dist.sample((2,)).shape == (2, 2, 4, 3)
    (-dist.log_prob(torch.zeros(2, 4, 3)).mean()).backward()
    assert torch.isfinite(hidden.grad).all()


def test_lowrank_initial_variance_matches_sigma():
    output = LowRankMultivariateNormalOutput(2, rank=1, sigma_init=0.3, sigma_minimum=0.0)
    loc, factor, diag = output.domain_map(torch.zeros(2), torch.zeros(2), torch.zeros(2))
    torch.testing.assert_close(diag, torch.full((2,), 0.09))


def test_zero_inflation_rejects_continuous_density_and_accepts_zero_rate():
    from transtimegrad.distributions import ZeroInflatedDistribution

    with pytest.raises(ValueError, match="discrete"):
        ZeroInflatedDistribution(torch.tensor(0.5), torch.distributions.Normal(0.0, 1.0))
    dist = ZeroInflatedPoisson(0.3, 0.0)
    torch.testing.assert_close(dist.log_prob(torch.tensor(0.0)), torch.tensor(0.0))
    assert dist.sample((10,)).eq(0).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_squareplus_log_gradient_at_representable_boundaries(dtype):
    bound = torch.finfo(dtype).max
    values = torch.tensor([-bound, -1.0, 0.0, 1.0, bound], dtype=dtype, requires_grad=True)
    mapped = squareplus(values)
    mapped.log().sum().backward()
    assert (mapped > 0).all() and torch.isfinite(mapped).all()
    assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize(
    "gate,value,expected",
    [(0.0, 0.0, 1.718281828459), (0.0, 1.0, -1.0), (1.0, 0.0, 0.632120558829)],
)
def test_zero_inflation_gate_endpoint_derivatives(gate, value, expected):
    parameter = torch.tensor(gate, dtype=torch.float64, requires_grad=True)
    dist = ZeroInflatedPoisson(parameter, torch.tensor(1.0, dtype=torch.float64))
    dist.log_prob(torch.tensor(value, dtype=torch.float64)).backward()
    torch.testing.assert_close(parameter.grad, torch.tensor(expected, dtype=torch.float64))


def test_zero_inflated_interior_gradcheck_and_zero_rate_derivative():
    gate = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    rate = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    for value in (0.0, 1.0):
        assert torch.autograd.gradcheck(
            lambda g, r, value=value: ZeroInflatedPoisson(g, r).log_prob(
                torch.tensor(value, dtype=torch.float64)
            ),
            (gate, rate),
        )
    zero_rate = torch.tensor(0.0, requires_grad=True)
    ZeroInflatedPoisson(0.3, zero_rate).log_prob(torch.tensor(0.0)).backward()
    torch.testing.assert_close(zero_rate.grad, torch.tensor(-0.7))


def test_zero_inflated_negative_binomial_zero_count_derivative_is_finite():
    from transtimegrad.distributions import ZeroInflatedNegativeBinomial

    count = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    gate = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    logits = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    value = ZeroInflatedNegativeBinomial(gate, count, logits=logits).log_prob(torch.tensor(0.0))
    value.backward()
    torch.testing.assert_close(value, torch.tensor(0.0, dtype=torch.float64), atol=1e-15, rtol=0)
    torch.testing.assert_close(
        count.grad, torch.tensor(-0.7 * torch.log(torch.tensor(2.0)).item(), dtype=torch.float64)
    )
    assert torch.isfinite(gate.grad) and torch.isfinite(logits.grad)
