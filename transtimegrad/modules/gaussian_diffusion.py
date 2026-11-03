from functools import partial
from inspect import isfunction

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False, dtype=None):
    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device, dtype=dtype).repeat(
            shape[0], *((1,) * (len(shape) - 1))
        )

    def noise():
        return torch.randn(shape, device=device, dtype=dtype)

    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        input_size,
        beta_end=0.1,
        diff_steps=100,
        loss_type="l2",
        betas=None,
        beta_schedule="linear",
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.input_size = input_size
        self.register_buffer("_scale", None, persistent=False)
        self.register_buffer("_loc", None, persistent=False)
        if input_size < 1 or diff_steps < 1:
            raise ValueError("input_size and diff_steps must be positive")
        if loss_type not in ("l1", "l2", "huber"):
            raise ValueError("loss_type must be l1, l2 or huber")

        if betas is not None:
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            if beta_schedule == "linear":
                betas = np.linspace(1e-4, beta_end, diff_steps)
            elif beta_schedule == "quad":
                betas = np.linspace(1e-4**0.5, beta_end**0.5, diff_steps) ** 2
            elif beta_schedule == "const":
                betas = beta_end * np.ones(diff_steps)
            elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
                betas = np.minimum(1.0 / np.linspace(diff_steps, 1, diff_steps), 0.999)
            elif beta_schedule == "sigmoid":
                betas = np.linspace(-6, 6, diff_steps)
                betas = (beta_end - 1e-4) / (np.exp(-betas) + 1) + 1e-4
            elif beta_schedule == "cosine":
                betas = cosine_beta_schedule(diff_steps)
            else:
                raise NotImplementedError(beta_schedule)

        betas = np.asarray(betas, dtype=np.float64)
        if (
            betas.ndim != 1
            or betas.size == 0
            or not np.isfinite(betas).all()
            or np.any((betas <= 0) | (betas >= 1))
        ):
            raise ValueError(
                "betas must be a nonempty vector of finite values strictly between 0 and 1"
            )
        alphas = 1.0 - betas
        with np.errstate(under="ignore"):
            alphas_cumprod = np.cumprod(alphas, axis=0)
        if np.any(alphas_cumprod < 1 / np.finfo(np.float32).max) or np.any(alphas_cumprod >= 1):
            raise ValueError("noise schedule is not representable with finite diffusion buffers")
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)),
        )

        if any(not torch.isfinite(buffer).all() for buffer in self.buffers(recurse=False)):
            raise ValueError("noise schedule is not representable with finite diffusion buffers")

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        if scale is None:
            self._scale = None
            return
        scale = torch.as_tensor(scale, device=self.betas.device, dtype=self.betas.dtype)
        if not torch.isfinite(scale).all() or (scale == 0).any():
            raise ValueError("scale must be finite and nonzero")
        self._scale = scale

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, loc):
        self._loc = (
            None
            if loc is None
            else torch.as_tensor(loc, device=self.betas.device, dtype=self.betas.dtype)
        )
        if self._loc is not None and not torch.isfinite(self._loc).all():
            raise ValueError("loc must be finite")

    def _affine_parameters(self, loc, scale):
        loc = self.loc if loc is None else loc
        scale = self.scale if scale is None else scale
        loc = torch.as_tensor(
            0 if loc is None else loc, device=self.betas.device, dtype=self.betas.dtype
        )
        scale = torch.as_tensor(
            1 if scale is None else scale, device=self.betas.device, dtype=self.betas.dtype
        )
        if not torch.isfinite(loc).all() or not torch.isfinite(scale).all() or (scale == 0).any():
            raise ValueError("loc must be finite and scale finite and nonzero")
        return loc, scale

    def _flatten_condition(self, cond, batch_shape):
        if cond is None:
            return None
        return cond.expand(batch_shape + (cond.shape[-1],)).reshape(-1, 1, cond.shape[-1])

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t, cond=cond))

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, clip_denoised=False, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, cond=cond, t=t, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise, dtype=x.dtype)
        # no noise when t == 0
        nonzero_mask = (t != 0).to(x.dtype).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device, dtype=self.betas.dtype)

        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, cond, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size(), cond=None, loc=None, scale=None):
        """Sample with shape sample_shape + cond.batch_shape + (input_size,)."""
        batch_shape = torch.Size(sample_shape) + (() if cond is None else cond.shape[:-1])
        count = int(np.prod(batch_shape)) if batch_shape else 1
        condition = self._flatten_condition(cond, batch_shape)
        samples = self.p_sample_loop((count, 1, self.input_size), condition)
        samples = samples.reshape(batch_shape + (self.input_size,))
        loc, scale = self._affine_parameters(loc, scale)
        return samples * scale + loc

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5, cond=None):
        """Interpolate equal-shaped targets in normalized diffusion space."""
        if x1.shape != x2.shape or x1.shape[-1] != self.input_size:
            raise ValueError("x1/x2 must have equal shape ending in input_size")
        t = self.num_timesteps - 1 if t is None else t
        if not isinstance(t, int) or not 0 <= t < self.num_timesteps or not 0 <= lam <= 1:
            raise ValueError("t must be a valid integer diffusion step and lam in [0,1]")
        loc, scale = self._affine_parameters(None, None)
        batch_shape = x1.shape[:-1]
        first = ((x1 - loc) / scale).reshape(-1, 1, self.input_size)
        second = ((x2 - loc) / scale).reshape_as(first)
        condition = self._flatten_condition(cond, batch_shape)
        times = torch.full((first.shape[0],), t, dtype=torch.long, device=first.device)
        img = (1 - lam) * self.q_sample(first, times) + lam * self.q_sample(second, times)
        for step in reversed(range(t + 1)):
            img = self.p_sample(img, condition, torch.full_like(times, step))
        return img.reshape(x1.shape) * scale + loc

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, cond, t, noise=None, reduction="mean"):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, cond=cond)

        if self.loss_type == "l1":
            loss = F.l1_loss(x_recon, noise, reduction=reduction)
        elif self.loss_type == "l2":
            loss = F.mse_loss(x_recon, noise, reduction=reduction)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(x_recon, noise, reduction=reduction)
        else:
            raise NotImplementedError()

        return loss

    def denoising_loss(self, x, cond=None, *, loc=None, scale=None, timesteps=None, noise=None):
        """Return a per-event denoising objective, not an exact negative log density."""
        if x.shape[-1] != self.input_size:
            raise ValueError("target must end in input_size")
        loc, scale = self._affine_parameters(loc, scale)
        normalized = (x - loc) / scale
        if not torch.isfinite(normalized).all():
            raise ValueError("target must be finite after normalization")
        batch_shape = normalized.shape[:-1]
        flat = normalized.reshape(-1, 1, self.input_size)
        condition = self._flatten_condition(cond, batch_shape)
        if timesteps is None:
            timesteps = torch.randint(self.num_timesteps, (flat.shape[0],), device=x.device)
        else:
            timesteps = (
                torch.as_tensor(timesteps, dtype=torch.long, device=x.device)
                .expand(batch_shape)
                .reshape(-1)
            )
        if noise is not None:
            noise = noise.reshape_as(flat)
        values = self.p_losses(flat, condition, timesteps, noise=noise, reduction="none")
        return values.mean(dim=(-1, -2)).reshape(batch_shape)

    def log_prob(self, x, cond=None, *args, **kwargs):
        raise NotImplementedError("GaussianDiffusion has no exact log_prob; use denoising_loss()")
