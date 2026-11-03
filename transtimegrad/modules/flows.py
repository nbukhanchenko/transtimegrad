import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from gluonts.core.component import validated
from torch.distributions import Normal


def create_masks(input_size, hidden_size, n_hidden, input_order="sequential", input_degrees=None):
    """Build masks whose output coordinate depends only on lower input ranks."""
    if input_size < 1 or hidden_size < 1 or n_hidden < 0:
        raise ValueError("input_size/hidden_size must be positive and n_hidden nonnegative")
    if input_order not in ("sequential", "random"):
        raise ValueError("input_order must be 'sequential' or 'random'")
    if input_degrees is None:
        input_degrees = (
            torch.arange(input_size) if input_order == "sequential" else torch.randperm(input_size)
        )
    input_degrees = torch.as_tensor(input_degrees, dtype=torch.long)
    if input_degrees.shape != (input_size,) or not torch.equal(
        input_degrees.sort().values, torch.arange(input_size)
    ):
        raise ValueError("input_degrees must be a permutation of 0..input_size-1")
    degrees = [input_degrees]
    for _ in range(n_hidden + 1):
        if input_order == "sequential":
            hidden_degrees = torch.arange(hidden_size) % input_size - 1
        else:
            hidden_degrees = torch.randint(-1, input_size - 1, (hidden_size,))
            hidden_degrees[0] = -1
        degrees.append(hidden_degrees)
    # The -1 hidden rank carries conditioning without depending on any input.
    degrees.append(input_degrees - 1)
    masks = [
        (out.unsqueeze(-1) >= inp.unsqueeze(0)).float()
        for inp, out in zip(degrees[:-1], degrees[1:], strict=True)
    ]
    return masks, input_degrees


class FlowSequential(nn.Sequential):
    """Container for layers of a normalizing flow"""

    def forward(self, x, y):
        sum_log_abs_det_jacobians = torch.zeros_like(x)
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, y):
        sum_log_abs_det_jacobians = torch.zeros_like(u)
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians


class BatchNorm(nn.Module):
    """Affine flow normalization with running statistics.

    Training transforms use the previous frozen running statistics. The current
    batch updates only the state used by subsequent forwards. A paired inverse
    uses the exact affine state captured by the latest training forward.
    """

    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        if input_size < 1 or not 0 <= momentum <= 1 or eps <= 0:
            raise ValueError("require input_size > 0, momentum in [0,1] and eps > 0")
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer("running_mean", torch.zeros(input_size))
        self.register_buffer("running_var", torch.ones(input_size))
        self.register_buffer("batch_mean", None, persistent=False)
        self.register_buffer("batch_var", None, persistent=False)

    def forward(self, x, cond_y=None):
        if self.training:
            self.batch_mean = self.running_mean.detach().clone()
            self.batch_var = self.running_var.detach().clone()
            flat = x.reshape(-1, x.shape[-1])
            if flat.shape[0] > 1:
                with torch.no_grad():
                    self.running_mean.lerp_(flat.mean(0).detach(), 1 - self.momentum)
                    self.running_var.lerp_(flat.var(0, unbiased=False).detach(), 1 - self.momentum)
            mean, var = self.batch_mean, self.batch_var
        else:
            mean, var = self.running_mean, self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        # compute log_abs_det_jacobian (cf RealNVP paper)
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
        return y, log_abs_det_jacobian.expand_as(x)

    def inverse(self, y, cond_y=None):
        if self.training and self.batch_mean is not None:
            mean, var = self.batch_mean, self.batch_var
        else:
            mean, var = self.running_mean, self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, log_abs_det_jacobian.expand_as(x)


class LinearMaskedCoupling(nn.Module):
    """Modified RealNVP Coupling Layers per the MAF paper"""

    def __init__(self, input_size, hidden_size, n_hidden, mask, cond_label_size=None):
        super().__init__()

        self.register_buffer("mask", mask)

        # scale function
        s_net = [
            nn.Linear(
                input_size + (cond_label_size if cond_label_size is not None else 0),
                hidden_size,
            )
        ]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)

        # translation function
        self.t_net = copy.deepcopy(self.s_net)
        # replace Tanh with ReLU's per MAF paper
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear):
                self.t_net[i] = nn.ReLU()

    def forward(self, x, y=None):
        # apply mask
        mx = x * self.mask

        # run through model
        s = self.s_net(mx if y is None else torch.cat([y, mx], dim=-1))
        t = self.t_net(mx if y is None else torch.cat([y, mx], dim=-1)) * (1 - self.mask)

        # cf RealNVP eq 8 where u corresponds to x (here we're modeling u)
        log_s = torch.tanh(s) * (1 - self.mask)
        u = x * torch.exp(log_s) + t

        # log det du/dx; cf RealNVP 8 and 6; note, sum over input_size done at model log_prob
        log_abs_det_jacobian = log_s

        return u, log_abs_det_jacobian

    def inverse(self, u, y=None):
        # apply mask
        mu = u * self.mask

        # run through model
        s = self.s_net(mu if y is None else torch.cat([y, mu], dim=-1))
        t = self.t_net(mu if y is None else torch.cat([y, mu], dim=-1)) * (1 - self.mask)

        log_s = torch.tanh(s) * (1 - self.mask)
        x = (u - t) * torch.exp(-log_s)

        log_abs_det_jacobian = -log_s

        return x, log_abs_det_jacobian


class MaskedLinear(nn.Linear):
    """MADE building block layer"""

    def __init__(self, input_size, n_outputs, mask, cond_label_size=None):
        super().__init__(input_size, n_outputs)

        self.register_buffer("mask", mask)

        self.cond_label_size = cond_label_size
        if cond_label_size is not None:
            self.cond_weight = nn.Parameter(
                torch.rand(n_outputs, cond_label_size) / math.sqrt(cond_label_size)
            )

    def forward(self, x, y=None):
        out = F.linear(x, self.weight * self.mask, self.bias)
        if y is not None:
            out = out + F.linear(y, self.cond_weight)
        return out


class MADE(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_hidden,
        cond_label_size=None,
        activation="ReLU",
        input_order="sequential",
        input_degrees=None,
    ):
        """
        Args:
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the autoregressive masks (sequential|random)
                            or the order flipped from the previous layer in a stack of MADEs
            conditional -- bool; whether model is conditional
        """
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer("base_dist_mean", torch.zeros(input_size))
        self.register_buffer("base_dist_var", torch.ones(input_size))

        # create masks
        masks, input_degrees = create_masks(
            input_size, hidden_size, n_hidden, input_order, input_degrees
        )

        self.register_buffer("input_degrees", input_degrees)

        # setup activation
        if activation == "ReLU":
            activation_fn = nn.ReLU()
        elif activation == "Tanh":
            activation_fn = nn.Tanh()
        else:
            raise ValueError("Check activation function.")

        # construct model
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [
            activation_fn,
            MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2, 1)),
        ]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        # MAF eq 4 -- return mean and log std
        m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=-1)
        u = (x - m) * torch.exp(-loga)
        # MAF eq 5
        log_abs_det_jacobian = -loga
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None, sum_log_abs_det_jacobians=None):
        # MAF eq 3
        x = torch.zeros_like(u)
        # run through reverse model
        for i in self.input_degrees.argsort():
            m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=-1)
            candidate = u * torch.exp(loga) + m
            x = torch.where(torch.arange(x.shape[-1], device=x.device) == i, candidate, x)
        log_abs_det_jacobian = loga
        return x, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=-1)


class Flow(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        if input_size < 1:
            raise ValueError("input_size must be positive")
        self.input_size = input_size
        self.register_buffer("_scale", None, persistent=False)
        self.register_buffer("_loc", None, persistent=False)
        self.net = None

        # base distribution for calculation of log prob under the model
        self.register_buffer("base_dist_mean", torch.zeros(input_size))
        self.register_buffer("base_dist_var", torch.ones(input_size))

    @property
    def base_dist(self):
        return Normal(self.base_dist_mean, self.base_dist_var)

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        if scale is not None:
            value = torch.as_tensor(
                scale, device=self.base_dist_mean.device, dtype=self.base_dist_mean.dtype
            )
            if not torch.isfinite(value).all() or (value == 0).any():
                raise ValueError("flow scale must be finite and nonzero")
        self._scale = None if scale is None else value

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, loc):
        value = (
            None
            if loc is None
            else torch.as_tensor(
                loc, device=self.base_dist_mean.device, dtype=self.base_dist_mean.dtype
            )
        )
        if value is not None and not torch.isfinite(value).all():
            raise ValueError("flow loc must be finite")
        self._loc = value

    def forward(self, x, cond=None):
        if self.loc is not None:
            x = x - self.loc
        if self.scale is not None:
            x = x / self.scale
        u, log_abs_det_jacobian = self.net(x, cond)
        if self.scale is not None:
            log_abs_det_jacobian = log_abs_det_jacobian - torch.log(torch.abs(self.scale))
        return u, log_abs_det_jacobian

    def inverse(self, u, cond=None):
        x, log_abs_det_jacobian = self.net.inverse(u, cond)
        if self.scale is not None:
            x = x * self.scale
            log_abs_det_jacobian = log_abs_det_jacobian + torch.log(torch.abs(self.scale))
        if self.loc is not None:
            x = x + self.loc
        return x, log_abs_det_jacobian

    def log_prob(self, x, cond=None):
        u, sum_log_abs_det_jacobians = self.forward(x, cond)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=-1)

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size(), cond=None):
        shape = torch.Size(sample_shape)
        if cond is not None:
            shape = shape + cond.shape[:-1]
            cond = cond.expand(shape + (cond.shape[-1],))

        u = self.base_dist.sample(shape)
        sample, _ = self.inverse(u, cond)
        return sample


class RealNVP(Flow):
    @validated()
    def __init__(
        self,
        n_blocks,
        input_size,
        hidden_size,
        n_hidden,
        cond_label_size=None,
        batch_norm=True,
    ):
        super().__init__(input_size)
        if (
            n_blocks < 1
            or hidden_size < 1
            or n_hidden < 0
            or (cond_label_size is not None and cond_label_size < 1)
        ):
            raise ValueError("require positive block/hidden/conditioning sizes and n_hidden >= 0")

        # construct model
        modules = []
        mask = torch.arange(input_size).float() % 2
        for _ in range(n_blocks):
            modules += [
                LinearMaskedCoupling(input_size, hidden_size, n_hidden, mask, cond_label_size)
            ]
            mask = 1 - mask
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)


class MAF(Flow):
    @validated()
    def __init__(
        self,
        n_blocks,
        input_size,
        hidden_size,
        n_hidden,
        cond_label_size=None,
        activation="ReLU",
        input_order="sequential",
        batch_norm=True,
    ):
        super().__init__(input_size)
        if (
            n_blocks < 1
            or hidden_size < 1
            or n_hidden < 0
            or (cond_label_size is not None and cond_label_size < 1)
        ):
            raise ValueError("require positive block/hidden/conditioning sizes and n_hidden >= 0")

        # construct model
        modules = []
        self.input_degrees = None
        for _ in range(n_blocks):
            modules += [
                MADE(
                    input_size,
                    hidden_size,
                    n_hidden,
                    cond_label_size,
                    activation,
                    input_order,
                    self.input_degrees,
                )
            ]
            self.input_degrees = modules[-1].input_degrees.flip(0)
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)
