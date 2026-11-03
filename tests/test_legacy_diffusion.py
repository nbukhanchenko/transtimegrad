import pytest
import torch
from torch import nn

from transtimegrad.model.epsilon_theta import EpsilonTheta
from transtimegrad.modules.distribution_output import DiffusionOutput
from transtimegrad.modules.gaussian_diffusion import GaussianDiffusion


class ZeroDenoiser(nn.Module):
    def forward(self, inputs, time, cond=None):
        return torch.zeros_like(inputs)


@pytest.mark.parametrize("shape", [(2, 3), (2, 1), ()])
def test_sampling_and_per_event_denoising_loss(shape):
    diffusion = GaussianDiffusion(ZeroDenoiser(), input_size=2, diff_steps=3)
    cond = torch.randn(shape + (4,))
    forecast = DiffusionOutput(diffusion, input_size=2, cond_size=4).distribution((cond,))
    samples = forecast.sample((5,))
    assert samples.shape == (5,) + shape + (2,)
    values = torch.randn(shape + (2,), requires_grad=True)
    original = values.detach().clone()
    # For a zero denoiser and unit supplied noise, squared denoising loss is exactly 1.
    loss = forecast.loss(values, noise=torch.ones_like(values), timesteps=1)
    assert loss.shape == shape
    torch.testing.assert_close(loss, torch.ones(shape))
    torch.testing.assert_close(values, original)
    with pytest.raises(NotImplementedError, match="log_prob"):
        forecast.log_prob(values)


def test_real_denoiser_multitime_training_and_inference():
    diffusion = GaussianDiffusion(EpsilonTheta(2, 4), input_size=2, diff_steps=3)
    cond = torch.randn(2, 3, 4)
    values = torch.randn(2, 3, 2, requires_grad=True)
    loss = diffusion.denoising_loss(values, cond).mean()
    loss.backward()
    assert torch.isfinite(values.grad).all()
    assert diffusion.sample(cond=cond).shape == values.shape


def test_diffusion_forecasts_do_not_mutate_shared_affine_parameters():
    diffusion = GaussianDiffusion(ZeroDenoiser(), input_size=2, diff_steps=3)
    output = DiffusionOutput(diffusion, input_size=2, cond_size=4)
    cond = torch.zeros(2, 3, 4)
    first = output.distribution((cond,), loc=10.0, scale=2.0)
    torch.manual_seed(7)
    before = first.sample()
    output.distribution((cond + 1,), loc=-10.0, scale=3.0)
    torch.manual_seed(7)
    torch.testing.assert_close(first.sample(), before)
    assert diffusion.scale is None and diffusion.loc is None


def test_interpolate_uses_every_reverse_step_and_preserves_input():
    diffusion = GaussianDiffusion(ZeroDenoiser(), input_size=2, diff_steps=3)
    first, second = torch.randn(2, 3, 2), torch.randn(2, 3, 2)
    snapshot = first.clone()
    steps = []
    original = diffusion.p_sample

    def record(x, cond, time, **kwargs):
        steps.append(time[0].item())
        return original(x, cond, time, **kwargs)

    diffusion.p_sample = record
    result = diffusion.interpolate(first, second, t=2, cond=torch.zeros(2, 3, 4))
    assert result.shape == first.shape and torch.isfinite(result).all()
    assert steps == [2, 1, 0]
    torch.testing.assert_close(first, snapshot)


@pytest.mark.parametrize("betas", [[0.0], [1.0], [-0.1], [float("nan")], []])
def test_invalid_noise_schedules_fail_early(betas):
    with pytest.raises(ValueError, match="betas"):
        GaussianDiffusion(ZeroDenoiser(), input_size=2, betas=betas)


@pytest.mark.parametrize("schedule", ["linear", "quad", "const", "jsd", "sigmoid", "cosine"])
def test_builtin_schedules_are_finite(schedule):
    diffusion = GaussianDiffusion(
        ZeroDenoiser(), input_size=2, diff_steps=4, beta_schedule=schedule
    )
    assert all(torch.isfinite(buffer).all() for buffer in diffusion.buffers())
    assert torch.isfinite(diffusion.sample((2,))).all()


def test_extreme_schedule_rejected_when_precision_cannot_represent_buffers():
    with pytest.raises(ValueError, match="representable"):
        GaussianDiffusion(ZeroDenoiser(), input_size=2, betas=[0.999] * 1000)


@pytest.mark.parametrize("scale", [0.0, float("nan"), float("inf")])
def test_direct_diffusion_affine_validation_cannot_be_bypassed(scale):
    diffusion = GaussianDiffusion(ZeroDenoiser(), input_size=2, diff_steps=3)
    with pytest.raises(ValueError, match="scale"):
        diffusion.sample((2,), scale=scale)
    with pytest.raises(ValueError, match="scale"):
        diffusion.denoising_loss(torch.ones(2, 2), scale=scale)


def test_diffusion_rejects_nonfinite_targets_even_when_denoiser_ignores_input():
    diffusion = GaussianDiffusion(ZeroDenoiser(), input_size=2, diff_steps=3)
    with pytest.raises(ValueError, match="target"):
        diffusion.denoising_loss(torch.tensor([[float("nan"), 0.0]]))
