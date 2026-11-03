from copy import deepcopy

import pytest
import torch
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    EulerDiscreteScheduler,
)
from test_model_logic import make_batch, make_model, prediction_batch

from transtimegrad.model.diffusion import validate_scheduler


@pytest.mark.parametrize("kind", ["rnn", "trans"])
@pytest.mark.parametrize("scheduler_type", [DDPMScheduler, DDIMScheduler, DEISMultistepScheduler])
@pytest.mark.parametrize("prediction_type", ["epsilon", "sample", "v_prediction"])
def test_all_supported_prediction_targets_train_and_sample(kind, scheduler_type, prediction_type):
    torch.manual_seed(19)
    model = make_model(kind)
    model.scheduler = scheduler_type(num_train_timesteps=10, prediction_type=prediction_type)
    batch = make_batch(model)
    loss = model.loss(**batch).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model(**prediction_batch(batch), num_parallel_samples=1)).all()


@pytest.mark.parametrize("kind", ["rnn", "trans", "transformer"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_explicit_dtype_loss_and_sampling(kind, dtype):
    model = make_model(kind).to(dtype)
    batch = make_batch(model)
    loss = model.loss(**batch).mean()
    assert loss.dtype == dtype and torch.isfinite(loss)
    loss.backward()
    samples = model(**prediction_batch(batch), num_parallel_samples=1)
    assert samples.dtype == dtype and torch.isfinite(samples).all()


@pytest.mark.parametrize("kind", ["rnn", "trans", "transformer"])
def test_partial_observations_retain_training_signal_and_ignore_missing_targets(kind):
    model = make_model(kind)
    batch = make_batch(model)
    batch["future_observed_values"][..., 0] = 0
    batch["future_target"][..., 0] = float("nan")
    batch["past_observed_values"][:, 0] = 0
    batch["past_target"][:, 0] = float("nan")
    loss = model.loss(**batch).mean()
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in gradients)
    assert torch.isfinite(model(**prediction_batch(batch), num_parallel_samples=1)).all()


@pytest.mark.parametrize("kind", ["rnn", "trans"])
def test_missing_coordinates_cannot_change_loss(kind):
    model = make_model(kind)
    batch = make_batch(model)
    batch["future_observed_values"][..., 0] = 0
    torch.manual_seed(13)
    original = model.loss(**batch)
    batch["future_target"][..., 0] = 1e20
    torch.manual_seed(13)
    changed = model.loss(**batch)
    torch.testing.assert_close(original, changed)


def test_unsupported_sigma_scheduler_is_rejected_before_training():
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        validate_scheduler(EulerDiscreteScheduler(), 2)


@pytest.mark.parametrize("kind", ["rnn", "trans", "transformer"])
def test_invalid_sample_count_fails_clearly(kind):
    model = make_model(kind)
    with pytest.raises(ValueError, match="num_parallel_samples"):
        model(**prediction_batch(make_batch(model)), num_parallel_samples=0)


def test_multistep_sampling_does_not_mutate_scheduler_history():
    model = make_model("trans")
    batch = make_batch(model)
    state = deepcopy(model.scheduler.__dict__)
    model(**prediction_batch(batch), num_parallel_samples=1)
    for key in ("model_outputs", "lower_order_nums", "_step_index"):
        assert repr(model.scheduler.__dict__.get(key)) == repr(state.get(key))


@pytest.mark.parametrize("kind", ["rnn", "trans"])
def test_auxiliary_loss_ignores_missing_active_lags(kind):
    model = make_model(kind, context=4)
    batch = make_batch(model)
    batch["past_target"][:, -2] = float("nan")
    batch["past_observed_values"][:, -2] = 0
    value = model.loss(**batch, future_only=False).mean()
    assert torch.isfinite(value)
    value.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("variance_type", ["learned", "learned_range"])
def test_learned_variance_requires_explicit_variance_head(variance_type):
    with pytest.raises(ValueError, match="variance"):
        validate_scheduler(DDPMScheduler(variance_type=variance_type), 2)


def test_masked_beta_loss_uses_valid_support_for_missing_values():
    from torch.distributions import Beta, Independent

    from transtimegrad.model.loss import NegativeLogLikelihood, masked_loss

    distribution = Independent(Beta(torch.ones(2, 3, 2) * 2, torch.ones(2, 3, 2) * 3), 1)
    target = torch.ones(2, 3, 2) * 0.5
    target[..., 0] = float("nan")
    observed = torch.ones_like(target)
    observed[..., 0] = 0
    result = masked_loss(distribution, target, observed, NegativeLogLikelihood())
    assert torch.isfinite(result).all()
    expected = -distribution.base_dist.log_prob(torch.full_like(target, 0.5))[..., 1]
    torch.testing.assert_close(result, expected)
    assert (
        masked_loss(distribution, target, torch.zeros_like(target), NegativeLogLikelihood()) == 0
    ).all()


def test_default_transformer_output_matches_vector_target():
    from transtimegrad.model.transformer import TransformerEstimator, TransformerModel

    model = TransformerModel(
        freq="D", context_length=3, prediction_length=2, input_size=2, lags_seq=[1]
    )
    assert model.target_shape == (2,)
    assert torch.isfinite(model.loss(**make_batch(model))).all()
    estimator = TransformerEstimator(freq="D", prediction_length=2, input_size=2)
    assert estimator.distr_output.event_shape == (2,)


@pytest.mark.parametrize("kind", ["rnn", "trans", "transformer"])
def test_fixed_batch_objective_can_be_optimized(kind):
    torch.manual_seed(43)
    model = make_model(kind)
    batch = make_batch(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    values = []
    for _ in range(25):
        optimizer.zero_grad()
        torch.manual_seed(19)
        loss = model.loss(**batch).mean()
        values.append(float(loss.detach()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    assert values[-1] < values[0] * 0.9


def test_pndm_rejects_unsupported_clean_sample_target():
    from diffusers import PNDMScheduler

    with pytest.raises(ValueError, match="PNDMScheduler"):
        validate_scheduler(PNDMScheduler(prediction_type="sample", skip_prk_steps=True), 2)


def test_scheduler_training_step_count_matches_supplied_betas():
    with pytest.raises(ValueError, match="trained_betas"):
        validate_scheduler(DDPMScheduler(num_train_timesteps=10, trained_betas=[0.1, 0.2]), 2)


@pytest.mark.parametrize("kind", ["rnn", "trans", "transformer"])
def test_invalid_embedding_metadata_is_rejected(kind):
    from transtimegrad.model.rnn_timegrad import TimeGradModel
    from transtimegrad.model.trans_timegrad import TransTimeGradModel
    from transtimegrad.model.transformer import TransformerModel

    kwargs = {"freq": "D", "context_length": 3, "prediction_length": 2, "cardinality": [0]}
    cls = {"rnn": TimeGradModel, "trans": TransTimeGradModel, "transformer": TransformerModel}[kind]
    if kind != "transformer":
        kwargs["scheduler"] = DEISMultistepScheduler()
    with pytest.raises(ValueError, match="cardinality"):
        cls(**kwargs)
