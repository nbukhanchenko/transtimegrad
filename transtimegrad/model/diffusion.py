"""Shared DDPM training and sampling operations for the TimeGrad backbones."""

from copy import deepcopy

import torch
import torch.nn.functional as F

from ..serialization import SCHEDULER_TYPES

SUPPORTED_SCHEDULERS = frozenset(SCHEDULER_TYPES)


def validate_scheduler(scheduler, num_inference_steps):
    if SCHEDULER_TYPES.get(type(scheduler).__name__) is not type(scheduler):
        raise ValueError(
            f"Unsupported scheduler {type(scheduler).__name__}; use an alpha-based DDPM scheduler"
        )
    if type(scheduler).__name__ == "PNDMScheduler" and scheduler.config.prediction_type == "sample":
        raise ValueError("PNDMScheduler supports epsilon or v_prediction, not sample prediction")
    if getattr(scheduler.config, "variance_type", None) in {"learned", "learned_range"}:
        raise ValueError("Learned variance requires an additional variance prediction head")
    if len(scheduler.alphas_cumprod) != scheduler.config.num_train_timesteps:
        raise ValueError("trained_betas length must match num_train_timesteps")
    if scheduler.config.prediction_type not in {"epsilon", "sample", "v_prediction"}:
        raise ValueError("prediction_type must be epsilon, sample, or v_prediction")
    if not 1 <= num_inference_steps <= scheduler.config.num_train_timesteps:
        raise ValueError("num_inference_steps must lie between 1 and num_train_timesteps")
    result = deepcopy(scheduler)
    try:
        result.set_timesteps(num_inference_steps)
    except (ValueError, IndexError) as error:
        raise ValueError("Inference steps are incompatible with this scheduler") from error
    return result


def validate_model_config(
    context_length, prediction_length, input_size, lags_seq, scaling, num_parallel_samples
):
    for name, value in (
        ("context_length", context_length),
        ("prediction_length", prediction_length),
        ("input_size", input_size),
        ("num_parallel_samples", num_parallel_samples),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if not lags_seq or any(not isinstance(lag, int) or lag < 1 for lag in lags_seq):
        raise ValueError("lags_seq must contain positive integer lags")
    if scaling not in {None, "none", "mean", "std"}:
        raise ValueError("scaling must be mean, std, none, or None")


def diffusion_loss(denoiser, scheduler, context, loc, scale, target, observed_values):
    """Return one masked loss per batch/time, retaining observed coordinates."""
    batch, length = context.shape[:2]
    target = target.to(context)
    loc, scale = loc.to(context), scale.to(context)
    observed = observed_values.bool()
    if observed.ndim == 2 and target.ndim == 3:
        observed = observed.unsqueeze(-1).expand_as(target)
    # Never allow NaNs at missing coordinates to enter the joint denoiser.
    target = torch.where(observed, target, loc)
    clean = ((target - loc) / scale).reshape(batch * length, 1, -1)
    noise = torch.randn_like(clean)
    timesteps = torch.randint(
        scheduler.config.num_train_timesteps, (batch * length,), device=context.device
    )
    noisy = scheduler.add_noise(clean, noise, timesteps)
    prediction = denoiser(noisy, timesteps, context.reshape(batch * length, 1, -1))
    prediction_type = scheduler.config.prediction_type
    if prediction_type == "epsilon":
        expected = noise
    elif prediction_type == "sample":
        expected = clean
    else:
        alpha = scheduler.alphas_cumprod.to(clean)[timesteps].reshape(-1, 1, 1)
        expected = alpha.sqrt() * noise - (1 - alpha).sqrt() * clean
    errors = F.smooth_l1_loss(prediction, expected, reduction="none").reshape(batch, length, -1)
    weights = observed.reshape(batch, length, -1).to(errors)
    return (errors * weights).sum(-1) / weights.sum(-1).clamp_min(1)


@torch.no_grad()
def diffusion_sample(denoiser, scheduler, context, loc, scale, input_size, num_inference_steps):
    batch, length = context.shape[:2]
    # Multistep schedulers maintain mutable solver history: each rollout owns it.
    scheduler = deepcopy(scheduler)
    scheduler.set_timesteps(num_inference_steps, device=context.device)
    sample = torch.randn(batch * length, 1, input_size, device=context.device, dtype=context.dtype)
    sample = sample * scheduler.init_noise_sigma
    conditioning = context.reshape(batch * length, 1, -1)
    for timestep in scheduler.timesteps:
        model_input = scheduler.scale_model_input(sample, timestep)
        prediction = denoiser(model_input, timestep.to(context.device), conditioning)
        sample = scheduler.step(prediction, timestep, sample).prev_sample
    shape = (batch, length) if input_size == 1 else (batch, length, input_size)
    return sample.reshape(shape) * scale + loc


def auxiliary_context_losses(
    model, feat_static_cat, feat_static_real, past_time_feat, past_target, past_observed_values
):
    """Score each context target from its strict prefix, including prefix scaling.

    Left padding follows the instance splitter convention. This optional mode
    costs one additional first-step forecast per auxiliary context target.
    """
    values = []
    for offset in range(1, model.context_length):
        index = past_target.shape[1] - model.context_length + offset
        padding = past_target.shape[1] - index

        def history(tensor, padding=padding, index=index):
            zeros = tensor.new_zeros((tensor.shape[0], padding) + tensor.shape[2:])
            return torch.cat((zeros, tensor[:, :index]), dim=1)

        loc, scale, context = model._one_step_conditioning(
            feat_static_cat,
            feat_static_real,
            history(past_time_feat),
            history(past_target),
            history(past_observed_values),
            past_time_feat[:, index : index + 1],
        )
        values.append(
            diffusion_loss(
                model.unet,
                model.scheduler,
                context,
                loc,
                scale,
                past_target[:, index : index + 1],
                past_observed_values[:, index : index + 1],
            )
        )
    return values


def validate_features(dynamic_real, static_real, static_cat, cardinality, embedding_dimension):
    """Validate feature metadata before allocating embeddings or fitting a model."""
    if min(dynamic_real, static_real, static_cat) < 1:
        raise ValueError(
            "model feature counts must be positive (estimators insert absent-feature placeholders)"
        )
    if len(cardinality) != static_cat or any(size < 1 for size in cardinality):
        raise ValueError("cardinality must contain one positive size per categorical feature")
    if embedding_dimension is not None and (
        len(embedding_dimension) != static_cat or any(size < 1 for size in embedding_dimension)
    ):
        raise ValueError(
            "embedding_dimension must contain one positive size per categorical feature"
        )
