"""Network projections and immutable forecast distributions."""

import warnings
from typing import Dict, Optional, Tuple

import torch
from gluonts.core.component import validated
from gluonts.torch.distributions import AffineTransformed, DistributionOutput
from gluonts.torch.distributions.studentT import StudentT
from torch.distributions import (
    Beta,
    Categorical,
    Distribution,
    Independent,
    LowRankMultivariateNormal,
    MixtureSameFamily,
    MultivariateNormal,
    NegativeBinomial,
    Normal,
    Poisson,
    constraints,
)

from ..distributions import ZeroInflatedNegativeBinomial, ZeroInflatedPoisson


def squareplus(x):
    """Positive map without cancellation at large negative inputs or x**2 overflow."""
    radius = torch.hypot(x, torch.full_like(x, 2))
    denominator = (radius / 2 - x / 2).clamp_min(torch.finfo(x.dtype).tiny)
    return torch.where(x >= 0, x / 2 + radius / 2, denominator.reciprocal()).clamp_min(
        torch.finfo(x.dtype).tiny
    )


def _count_scale(loc, scale):
    # Counts must remain integer valued; use the established mean/rate scaling.
    if loc is not None and torch.any(torch.as_tensor(loc) != 0):
        raise ValueError("count outputs do not support centering; use mean/no scaling")
    if scale is not None:
        value = torch.as_tensor(scale)
        if not torch.isfinite(value).all() or (value <= 0).any():
            raise ValueError("count scale must be finite and positive")
    return scale


class IndependentDistributionOutput(DistributionOutput):
    @validated()
    def __init__(self, dim: Optional[int] = None) -> None:
        if dim is not None and dim < 1:
            raise ValueError("dim must be positive")
        self.dim = dim
        if dim is not None:
            self.args_dim = {key: dim for key in self.args_dim}

    @property
    def event_shape(self) -> Tuple:
        return () if self.dim is None else (self.dim,)

    def _event_arg(self, value):
        return value.squeeze(-1) if self.dim is None else value

    def independent(self, distr: Distribution) -> Distribution:
        return distr if self.dim is None else Independent(distr, 1)

    def distribution(self, distr_args, loc=None, scale=None) -> Distribution:
        distr = self.independent(self.distr_cls(*distr_args))
        return AffineTransformed(distr, loc=loc, scale=scale)

    squareplus = staticmethod(squareplus)


class NormalOutput(IndependentDistributionOutput):
    args_dim: Dict[str, int] = {"loc": 1, "scale": 1}
    distr_cls = Normal

    def domain_map(self, loc, scale):
        return self._event_arg(loc), self._event_arg(squareplus(scale))


class IndependentNormalOutput(NormalOutput):
    @validated()
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        warnings.warn(
            "IndependentNormalOutput is deprecated. Use NormalOutput instead.",
            DeprecationWarning,
            stacklevel=2,
        )


class BetaOutput(IndependentDistributionOutput):
    args_dim = {"concentration1": 1, "concentration0": 1}
    distr_cls = Beta

    def domain_map(self, concentration1, concentration0):
        return self._event_arg(squareplus(concentration1)), self._event_arg(
            squareplus(concentration0)
        )

    @property
    def value_in_support(self):
        return 0.5


class PoissonOutput(IndependentDistributionOutput):
    args_dim = {"rate": 1}
    distr_cls = Poisson

    def domain_map(self, rate):
        return (self._event_arg(squareplus(rate)),)

    def distribution(self, distr_args, loc=None, scale=None):
        scale = _count_scale(loc, scale)
        (rate,) = distr_args
        return self.independent(Poisson(rate if scale is None else rate * scale))


class ZeroInflatedPoissonOutput(IndependentDistributionOutput):
    args_dim = {"gate": 1, "rate": 1}
    distr_cls = ZeroInflatedPoisson

    def domain_map(self, gate, rate):
        return self._event_arg(gate.sigmoid()), self._event_arg(squareplus(rate))

    def distribution(self, distr_args, loc=None, scale=None):
        scale = _count_scale(loc, scale)
        gate, rate = distr_args
        return self.independent(ZeroInflatedPoisson(gate, rate if scale is None else rate * scale))


class NegativeBinomialOutput(IndependentDistributionOutput):
    args_dim = {"total_count": 1, "logits": 1}
    distr_cls = NegativeBinomial

    def domain_map(self, total_count, logits):
        return self._event_arg(squareplus(total_count)), self._event_arg(logits)

    def distribution(self, distr_args, loc=None, scale=None):
        scale = _count_scale(loc, scale)
        total_count, logits = distr_args
        if scale is not None:
            logits = logits + torch.as_tensor(scale, device=logits.device, dtype=logits.dtype).log()
        return self.independent(NegativeBinomial(total_count=total_count, logits=logits))


class ZeroInflatedNegativeBinomialOutput(IndependentDistributionOutput):
    args_dim = {"gate": 1, "total_count": 1, "logits": 1}
    distr_cls = ZeroInflatedNegativeBinomial

    def domain_map(self, gate, total_count, logits):
        return (
            self._event_arg(gate.sigmoid()),
            self._event_arg(squareplus(total_count)),
            self._event_arg(logits),
        )

    def distribution(self, distr_args, loc=None, scale=None):
        scale = _count_scale(loc, scale)
        gate, total_count, logits = distr_args
        if scale is not None:
            logits = logits + torch.as_tensor(scale, device=logits.device, dtype=logits.dtype).log()
        return self.independent(ZeroInflatedNegativeBinomial(gate, total_count, logits=logits))


class StudentTOutput(IndependentDistributionOutput):
    args_dim = {"df": 1, "loc": 1, "scale": 1}
    distr_cls = StudentT

    def domain_map(self, df, loc, scale):
        return (
            self._event_arg(2 + squareplus(df)),
            self._event_arg(loc),
            self._event_arg(squareplus(scale)),
        )


class NormalMixtureOutput(DistributionOutput):
    @validated()
    def __init__(self, components: int = 1) -> None:
        if components < 1:
            raise ValueError("components must be positive")
        self.components = components
        self.args_dim = {"mix_logits": components, "loc": components, "scale": components}

    squareplus = staticmethod(squareplus)

    @classmethod
    def domain_map(cls, mix_logits, loc, scale):
        # The component axis is required even for a single component.
        return mix_logits, loc, squareplus(scale)

    def distribution(self, distr_args, loc=None, scale=None):
        mix_logits, dist_loc, dist_scale = distr_args
        distr = MixtureSameFamily(Categorical(logits=mix_logits), Normal(dist_loc, dist_scale))
        return AffineTransformed(distr, loc=loc, scale=scale)

    @property
    def event_shape(self):
        return ()


class StudentTMixtureOutput(NormalMixtureOutput):
    @validated()
    def __init__(self, components: int = 1) -> None:
        super().__init__(components)
        self.args_dim = {
            "mix_logits": components,
            "df": components,
            "loc": components,
            "scale": components,
        }

    @classmethod
    def domain_map(cls, mix_logits, df, loc, scale):
        return mix_logits, 2 + squareplus(df), loc, squareplus(scale)

    def distribution(self, distr_args, loc=None, scale=None):
        mix_logits, df, dist_loc, dist_scale = distr_args
        distr = MixtureSameFamily(
            Categorical(logits=mix_logits), StudentT(df, dist_loc, dist_scale)
        )
        return AffineTransformed(distr, loc=loc, scale=scale)


class LowRankMultivariateNormalOutput(DistributionOutput):
    @validated()
    def __init__(
        self, dim: int, rank: int, sigma_init: float = 1.0, sigma_minimum: float = 1e-3
    ) -> None:
        if dim < 1 or rank < 1 or rank > dim or sigma_init <= 0 or sigma_minimum < 0:
            raise ValueError("require dim >= rank >= 1, sigma_init > 0 and sigma_minimum >= 0")
        self.distr_cls = LowRankMultivariateNormal
        self.dim, self.rank = dim, rank
        self.sigma_init, self.sigma_minimum = sigma_init, sigma_minimum
        self.args_dim = {"loc": dim, "cov_factor": dim * rank, "cov_diag": dim}

    def domain_map(self, loc, cov_factor, cov_diag):
        variance = self.sigma_init**2
        diag_bias = variance - 1 / variance  # inverse of squareplus
        factor = cov_factor.reshape(cov_factor.shape[:-1] + (self.dim, self.rank))
        return loc, factor, squareplus(cov_diag + diag_bias) + self.sigma_minimum**2

    squareplus = staticmethod(squareplus)

    @staticmethod
    def inv_softplus(y):
        # Retained for callers of the old helper; the covariance map uses squareplus.
        import math

        return y + math.log(-math.expm1(-y))

    @property
    def event_shape(self):
        return (self.dim,)


class MultivariateNormalOutput(DistributionOutput):
    @validated()
    def __init__(self, dim: int) -> None:
        if dim < 1:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.args_dim = {"loc": dim, "scale_tril": dim * dim}

    squareplus = staticmethod(squareplus)

    def domain_map(self, loc, scale):
        matrix = scale.reshape(scale.shape[:-1] + (self.dim, self.dim))
        diagonal = squareplus(matrix.diagonal(dim1=-2, dim2=-1))
        return loc, matrix.tril(-1) + torch.diag_embed(diagonal)

    def distribution(self, distr_args, loc=None, scale=None):
        dist_loc, scale_tril = distr_args
        return AffineTransformed(
            MultivariateNormal(dist_loc, scale_tril=scale_tril), loc=loc, scale=scale
        )

    @property
    def event_shape(self):
        return (self.dim,)


def _bound_parameters(cond, loc, scale, dim):
    cond = cond.clone()
    loc = torch.as_tensor(0 if loc is None else loc, device=cond.device, dtype=cond.dtype).clone()
    scale = torch.as_tensor(
        1 if scale is None else scale, device=cond.device, dtype=cond.dtype
    ).clone()
    if not torch.isfinite(loc).all() or not torch.isfinite(scale).all() or (scale == 0).any():
        raise ValueError("loc must be finite and scale finite and nonzero")
    shape = torch.broadcast_shapes(cond.shape[:-1] + (dim,), loc.shape, scale.shape)
    if shape[-1] != dim:
        raise ValueError("loc/scale do not match the event dimension")
    return cond, loc, scale, shape[:-1]


class _ConditionalFlowDistribution(Distribution):
    """A flow density with independent, captured conditioning and affine parameters."""

    arg_constraints = {}
    support = constraints.real_vector

    def __init__(self, flow, cond, loc, scale, dim):
        self.flow = flow
        self.cond, self.loc, self.scale, batch = _bound_parameters(cond, loc, scale, dim)
        super().__init__(batch_shape=batch, event_shape=(dim,))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = torch.broadcast_tensors(
            value, self.loc, self.scale, self.cond.new_zeros(self.batch_shape + self.event_shape)
        )[0]
        normalized = (value - self.loc) / self.scale
        cond = self.cond.expand(value.shape[:-1] + (self.cond.shape[-1],))
        latent, jacobian = self.flow.net(normalized, cond)
        return (self.flow.base_dist.log_prob(latent) + jacobian - self.scale.abs().log()).sum(-1)

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        shape = torch.Size(sample_shape) + self.batch_shape
        latent = self.flow.base_dist.sample(shape)
        cond = self.cond.expand(shape + (self.cond.shape[-1],))
        value, _ = self.flow.net.inverse(latent, cond)
        return value * self.scale + self.loc


class FlowOutput(DistributionOutput, torch.nn.Module):
    @validated()
    def __init__(self, flow, input_size, cond_size):
        torch.nn.Module.__init__(self)
        if input_size < 1 or cond_size < 1:
            raise ValueError("input_size and cond_size must be positive")
        if getattr(flow, "input_size", input_size) != input_size:
            raise ValueError("input_size must match the flow/diffusion event dimension")
        self.args_dim = {"cond": cond_size}
        self.flow, self.dim = flow, input_size

    @classmethod
    def domain_map(cls, cond):
        return (cond,)

    def distribution(self, distr_args, loc=None, scale=None):
        return _ConditionalFlowDistribution(self.flow, distr_args[0], loc, scale, self.dim)

    @property
    def event_shape(self):
        return (self.dim,)


class _ConditionalDiffusionForecast:
    """Bound diffusion sampler and denoising loss, deliberately not a probability density."""

    def __init__(self, diffusion, cond, loc, scale, dim):
        self.diffusion = diffusion
        self.cond, self.loc, self.scale, self.batch_shape = _bound_parameters(cond, loc, scale, dim)
        self.event_shape = torch.Size((dim,))

    def sample(self, sample_shape=torch.Size()):
        cond = self.cond.expand(self.batch_shape + (self.cond.shape[-1],))
        return self.diffusion.sample(sample_shape, cond=cond, loc=self.loc, scale=self.scale)

    def loss(self, value, **kwargs):
        return self.diffusion.denoising_loss(
            value, cond=self.cond, loc=self.loc, scale=self.scale, **kwargs
        )

    def log_prob(self, value):
        raise NotImplementedError(
            "diffusion has no exact log_prob; use .loss(value) for denoising training"
        )


class DiffusionOutput(FlowOutput):
    """Legacy output adapter. Its forecast offers .sample() and .loss(), not exact log_prob."""

    @validated()
    def __init__(self, diffusion, input_size, cond_size):
        super().__init__(diffusion, input_size, cond_size)

    @property
    def diffusion(self):
        return self.flow

    def distribution(self, distr_args, loc=None, scale=None):
        return _ConditionalDiffusionForecast(self.diffusion, distr_args[0], loc, scale, self.dim)
