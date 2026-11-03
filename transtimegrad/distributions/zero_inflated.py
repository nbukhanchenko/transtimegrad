# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import Distribution, NegativeBinomial, Poisson, constraints
from torch.distributions.utils import broadcast_all, lazy_property

from .utils import broadcast_shape


class _ZeroInflatedLogProb(torch.autograd.Function):
    """Log-space mixture with analytic one-sided gate derivatives.

    For impossible events or derivatives exceeding the dtype range, gate
    derivatives use a finite saturated limit. Those singular endpoints do not
    have a finite exact gradient; ordinary/interior derivatives remain exact.
    """

    @staticmethod
    def forward(ctx, gate, base_log_prob, is_zero):
        ordinary = (-gate).log1p() + base_log_prob
        zero = torch.logaddexp(gate.log(), ordinary)
        result = torch.where(is_zero, zero, ordinary)
        ctx.save_for_backward(gate, base_log_prob, is_zero, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        gate, base_log_prob, is_zero, result = ctx.saved_tensors
        tiny = torch.finfo(gate.dtype).tiny
        # Keep coefficients finite so zero upstream gradients cannot cause 0*inf.
        log_limit = gate.new_tensor(1 / tiny).log()
        log_zero_derivative = (-base_log_prob.expm1()).log() - result
        zero_derivative = log_zero_derivative.clamp(max=log_limit).exp()
        ordinary_derivative = -(1 - gate).clamp_min(tiny).reciprocal()
        gate_derivative = torch.where(is_zero, zero_derivative, ordinary_derivative)
        log_base_weight = (-gate).log1p() + base_log_prob - result
        base_weight = torch.nan_to_num(log_base_weight.exp(), nan=0.0).clamp(max=1)
        base_derivative = torch.where(is_zero, base_weight, torch.ones_like(base_weight))
        return grad_output * gate_derivative, grad_output * base_derivative, None


class ZeroInflatedDistribution(Distribution):
    """
    Generic Zero Inflated distribution.

    This can be used directly or can be used as a base class as e.g. for
    :class:`ZeroInflatedPoisson` and :class:`ZeroInflatedNegativeBinomial`.

    :param torch.Tensor gate: probability of extra zeros given via a Bernoulli distribution.
    :param TorchDistribution base_dist: the base distribution.
    """

    arg_constraints = {"gate": constraints.unit_interval}

    def __init__(self, gate, base_dist, validate_args=None):
        if not base_dist.support.is_discrete:
            raise ValueError("zero inflation requires a discrete base distribution")
        if base_dist.event_shape:
            raise ValueError(
                "ZeroInflatedDistribution expected empty base_dist.event_shape but got {}".format(
                    base_dist.event_shape
                )
            )
        gate = torch.as_tensor(gate, device=base_dist.mean.device, dtype=base_dist.mean.dtype)
        batch_shape = broadcast_shape(gate.shape, base_dist.batch_shape)
        self.gate = gate.expand(batch_shape)
        self.base_dist = base_dist.expand(batch_shape)
        event_shape = torch.Size()

        super().__init__(batch_shape, event_shape, validate_args)

    @property
    def support(self):
        return self.base_dist.support

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        gate, value = broadcast_all(self.gate, value)
        if isinstance(self.base_dist, Poisson):
            # Poisson's xlogy backward is undefined at rate=0 even for value=0.
            rate = self.base_dist.rate
            safe_rate = rate.clamp_min(torch.finfo(rate.dtype).tiny)
            base_log_prob = value * safe_rate.log() - rate - (value + 1).lgamma()
            base_log_prob = torch.where((rate == 0) & (value > 0), -torch.inf, base_log_prob)
        elif isinstance(self.base_dist, NegativeBinomial):
            count, logits = self.base_dist.total_count, self.base_dist.logits
            # Gamma-function derivatives at total_count=0 otherwise produce
            # inf-inf even for the valid zero-count event.
            safe_count = count.clamp_min(torch.finfo(count.dtype).tiny)
            ordinary = NegativeBinomial(safe_count, logits=logits).log_prob(value)
            zero = -count * torch.nn.functional.softplus(logits)
            base_log_prob = torch.where(value == 0, zero, ordinary)
            base_log_prob = torch.where((count == 0) & (value > 0), -torch.inf, base_log_prob)
        else:
            base_log_prob = self.base_dist.log_prob(value)
        return _ZeroInflatedLogProb.apply(gate, base_log_prob, value == 0)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            mask = torch.bernoulli(self.gate.expand(shape)).bool()
            samples = self.base_dist.expand(shape).sample()
            samples = torch.where(mask, samples.new_zeros(()), samples)
        return samples

    @lazy_property
    def mean(self):
        return (1 - self.gate) * self.base_dist.mean

    @lazy_property
    def variance(self):
        return (1 - self.gate) * self.base_dist.variance + self.gate * (
            1 - self.gate
        ) * self.base_dist.mean.square()

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(type(self), _instance)
        batch_shape = torch.Size(batch_shape)
        gate = self.gate.expand(batch_shape)
        base_dist = self.base_dist.expand(batch_shape)
        ZeroInflatedDistribution.__init__(new, gate, base_dist, validate_args=False)
        new._validate_args = self._validate_args
        return new


class ZeroInflatedPoisson(ZeroInflatedDistribution):
    """
    A Zero Inflated Poisson distribution.

    :param torch.Tensor gate: probability of extra zeros.
    :param torch.Tensor rate: rate of poisson distribution.
    """

    arg_constraints = {"gate": constraints.unit_interval, "rate": constraints.nonnegative}
    support = constraints.nonnegative_integer

    def __init__(self, gate, rate, validate_args=None):
        base_dist = Poisson(rate=rate, validate_args=False)
        base_dist._validate_args = validate_args

        super().__init__(gate, base_dist, validate_args=validate_args)

    @property
    def rate(self):
        return self.base_dist.rate


class ZeroInflatedNegativeBinomial(ZeroInflatedDistribution):
    """
    A Zero Inflated Negative Binomial distribution.

    :param torch.Tensor gate: probability of extra zeros.
    :param total_count: non-negative number of negative Bernoulli trials.
    :type total_count: float or torch.Tensor
    :param torch.Tensor probs: Event probabilities of success in the half open interval [0, 1).
    :param torch.Tensor logits: Event log-odds for probabilities of success.
    """

    arg_constraints = {
        "gate": constraints.unit_interval,
        "total_count": constraints.greater_than_eq(0),
        "probs": constraints.half_open_interval(0.0, 1.0),
        "logits": constraints.real,
    }
    support = constraints.nonnegative_integer

    def __init__(self, gate, total_count, probs=None, logits=None, validate_args=None):
        base_dist = NegativeBinomial(
            total_count=total_count,
            probs=probs,
            logits=logits,
            validate_args=False,
        )
        base_dist._validate_args = validate_args

        super().__init__(gate, base_dist, validate_args=validate_args)

    @property
    def total_count(self):
        return self.base_dist.total_count

    @property
    def probs(self):
        return self.base_dist.probs

    @property
    def logits(self):
        return self.base_dist.logits
