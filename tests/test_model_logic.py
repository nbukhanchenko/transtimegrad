from types import SimpleNamespace

import pytest
import torch
from diffusers import DEISMultistepScheduler

from transtimegrad.model.rnn_timegrad import TimeGradModel
from transtimegrad.model.trans_timegrad import TransTimeGradModel
from transtimegrad.model.transformer import TransformerModel
from transtimegrad.modules import StudentTOutput


def make_model(kind, dim=2, context=3, horizon=3, timesteps=10):
    kwargs = dict(
        freq="D", context_length=context, prediction_length=horizon,
        input_size=dim, lags_seq=[1, 2], num_parallel_samples=3,
    )
    if kind == "transformer":
        model = TransformerModel(
            **kwargs, d_model=8, nhead=2, num_encoder_layers=2,
            num_decoder_layers=2, dim_feedforward=16, dropout=0,
            distr_output=StudentTOutput(dim=dim) if dim > 1 else StudentTOutput(),
        )
    else:
        cls = TransTimeGradModel if kind == "trans" else TimeGradModel
        model = cls(
            **kwargs, scheduler=DEISMultistepScheduler(num_train_timesteps=timesteps),
            hidden_size=8, num_layers=2, dropout_rate=0, num_inference_steps=2,
        )
    return model.eval()


def make_batch(model, batch_size=2):
    shape = (batch_size, model._past_length)
    future_shape = (batch_size, model.prediction_length)
    if model.input_size > 1:
        shape += (model.input_size,)
        future_shape += (model.input_size,)
    return dict(
        feat_static_cat=torch.zeros(batch_size, 1, dtype=torch.long),
        feat_static_real=torch.zeros(batch_size, 1),
        past_time_feat=torch.randn(batch_size, model._past_length, 1),
        past_target=torch.randn(shape), past_observed_values=torch.ones(shape),
        future_time_feat=torch.randn(batch_size, model.prediction_length, 1),
        future_target=torch.randn(future_shape),
        future_observed_values=torch.ones(future_shape),
    )


def prediction_batch(batch):
    return {k: v for k, v in batch.items() if k not in ("future_target", "future_observed_values")}


@pytest.mark.parametrize("kind", ["trans", "transformer"])
def test_teacher_forcing_matches_autoregressive_conditioning(kind, monkeypatch):
    torch.manual_seed(17)
    model = make_model(kind)
    batch = make_batch(model)
    backbone = model.encoder_decoder if kind == "trans" else model
    outputs = []
    handle = backbone.transformer.decoder.register_forward_hook(
        lambda module, args, output: outputs.append(output.detach().clone())
    )
    with torch.no_grad():
        model.loss(**batch)
        teacher_output = outputs.pop()
        step = 0

        def known_sample(*args, **kwargs):
            nonlocal step
            result = batch["future_target"][:, step:step + 1]
            step += 1
            return result

        if kind == "trans":
            monkeypatch.setattr(model, "sample", known_sample)
        else:
            monkeypatch.setattr(
                model, "output_distribution", lambda *a, **kw: SimpleNamespace(sample=known_sample)
            )
        model(**prediction_batch(batch), num_parallel_samples=1)
    handle.remove()
    autoregressive_output = torch.cat([out[:, -1:] for out in outputs], dim=1)
    torch.testing.assert_close(autoregressive_output, teacher_output, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("kind", ["rnn", "trans", "transformer"])
@pytest.mark.parametrize("dim", [1, 2])
def test_loss_backward_and_forecast_shape(kind, dim):
    torch.manual_seed(5)
    model = make_model(kind, dim=dim)
    batch = make_batch(model)
    loss = model.loss(**batch).mean()
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    with torch.no_grad():
        samples = model(**prediction_batch(batch), num_parallel_samples=2)
    expected_shape = (2, 2, model.prediction_length) + ((dim,) if dim > 1 else ())
    assert samples.shape == expected_shape
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize("kind", ["rnn", "trans"])
def test_scheduler_with_1000_steps_supports_last_timestep(kind):
    model = make_model(kind, timesteps=1000)
    dtype = next(model.unet.parameters()).dtype
    noisy = torch.randn(2, 1, 2, dtype=dtype)
    context = torch.randn(2, 1, 8, dtype=dtype)
    output = model.unet(noisy, torch.tensor([500, 999]), context)
    assert output.shape == noisy.shape
    with torch.no_grad():
        samples = model(**prediction_batch(make_batch(model)), num_parallel_samples=1)
    assert torch.isfinite(samples).all()


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("context", [1, 3])
def test_full_context_loss_is_finite_and_causal(dim, context, monkeypatch):
    torch.manual_seed(6)
    model = make_model("trans", dim=dim, context=context)
    batch = make_batch(model)
    captured = []
    original = model.get_loss_values

    def record(model_output, loc, scale, target, observed_values):
        captured.append(model_output.detach().clone())
        assert model_output.shape[1] == target.shape[1] == context - 1 + model.prediction_length
        return original(model_output, loc, scale, target, observed_values)

    monkeypatch.setattr(model, "get_loss_values", record)
    loss = model.loss(**batch, future_only=False).mean()
    assert torch.isfinite(loss)
    loss.backward()
    # Changing y_1 cannot change conditioning for y_0 or y_1.
    changed = {key: value.clone() for key, value in batch.items()}
    changed["future_target"][:, 1] += 10
    with torch.no_grad():
        model.loss(**changed, future_only=False)
    torch.testing.assert_close(captured[0][:, :context + 1], captured[1][:, :context + 1])


@pytest.mark.parametrize("dim", [1, 2])
def test_transformer_log_prob_matches_negative_summed_loss(dim):
    model = make_model("transformer", dim=dim)
    batch = make_batch(model)
    expected = -model.loss(**batch, aggregate_by=torch.sum)
    actual = model.log_prob(**prediction_batch(batch), future_target=batch["future_target"])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("kind", ["rnn", "trans"])
def test_scalar_conditioner_retains_context_dependence(kind):
    torch.manual_seed(11)
    model = make_model(kind, dim=1)
    upsampler = model.unet.cond_upsampler
    dtype = next(upsampler.parameters()).dtype
    context = torch.randn(2, 1, 8, dtype=dtype, requires_grad=True)
    upsampler(context).sum().backward()
    assert context.grad is not None and context.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["rnn", "trans"])
@pytest.mark.parametrize("dim", [1, 2])
def test_single_context_step_has_no_extra_history_targets(kind, dim):
    model = make_model(kind, dim=dim, context=1)
    batch = make_batch(model)
    torch.nn.init.normal_(model.unet.output_projection.weight, std=0.01)
    torch.manual_seed(2)
    future_loss = model.loss(**batch)
    torch.manual_seed(2)
    full_loss = model.loss(**batch, future_only=False)
    assert torch.isfinite(full_loss).all()
    # With no auxiliary context targets the two modes are equivalent.
    torch.testing.assert_close(full_loss, future_loss)


def test_auxiliary_context_conditioning_does_not_see_later_targets(monkeypatch):
    from gluonts.torch.scaler import NOPScaler

    model = make_model("trans", context=4)
    # Keep scale fixed to isolate the attention dependency from normalization.
    model.encoder_decoder.scaler = NOPScaler(dim=1, keepdim=True)
    batch = make_batch(model)
    outputs = []

    def record(model_output, **kwargs):
        outputs.append(model_output.detach().clone())
        return model_output.sum(-1)

    monkeypatch.setattr(model, "get_loss_values", record)
    with torch.no_grad():
        model.loss(**batch, future_only=False)
        batch["past_target"][:, -2] += 10
        model.loss(**batch, future_only=False)
    # The first two auxiliary targets precede or equal the changed target.
    torch.testing.assert_close(outputs[0][:, :2], outputs[1][:, :2])
