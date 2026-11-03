"""Distribution losses independent of GluonTS's removed loss module."""

import torch
from gluonts.core.component import validated
from torch import nn
from torch.distributions import Independent, TransformedDistribution


class DistributionLoss(nn.Module):
    def forward(self, distribution, target):
        raise NotImplementedError


class NegativeLogLikelihood(DistributionLoss):
    @validated()
    def __init__(self):
        super().__init__()

    def forward(self, distribution, target):
        return -distribution.log_prob(target)


def masked_loss(distribution, target, observed, loss):
    """Mask scalar events or marginal independent coordinates without data loss."""
    try:
        filler = distribution.mean
    except NotImplementedError:
        # Flow outputs have real support but generally no analytic mean.
        filler = torch.zeros_like(target)
    target = torch.where(observed.bool(), target, filler)
    if observed.ndim < 3:
        return loss(distribution, target) * observed
    full = observed.bool().all(-1)
    partial = observed.bool().any(-1) & ~full
    if not partial.any():
        return loss(distribution, target) * full
    if not isinstance(loss, NegativeLogLikelihood):
        raise ValueError("Partial vector observations require NegativeLogLikelihood")
    if isinstance(distribution, Independent):
        elementwise = distribution.base_dist
    elif isinstance(distribution, TransformedDistribution) and isinstance(
        distribution.base_dist, Independent
    ):
        elementwise = TransformedDistribution(
            distribution.base_dist.base_dist, distribution.transforms
        )
    else:
        raise ValueError("Partial vector observations require an independent distribution output")
    values = -elementwise.log_prob(target)
    return torch.where(observed.bool(), values, 0.0).sum(-1)
