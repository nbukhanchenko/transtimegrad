"""Exercise realistic tensor dimensions without downloading benchmark data."""

import pytest
import torch
from diffusers import DEISMultistepScheduler
from test_model_logic import make_batch, prediction_batch

from transtimegrad.model.rnn_timegrad import TimeGradModel
from transtimegrad.model.trans_timegrad import TransTimeGradModel
from transtimegrad.model.transformer import TransformerModel


@pytest.mark.parametrize("kind", ["rnn", "trans", "transformer"])
def test_wide_series_long_lags_and_day_horizon(kind):
    kwargs = {
        "freq": "h",
        "input_size": 137,
        "context_length": 24,
        "prediction_length": 24,
        "lags_seq": [1, 7, 24],
        "num_parallel_samples": 2,
    }
    if kind == "transformer":
        model = TransformerModel(**kwargs, d_model=32, dropout=0)
    else:
        cls = TransTimeGradModel if kind == "trans" else TimeGradModel
        model = cls(
            **kwargs,
            hidden_size=32,
            dropout_rate=0,
            num_inference_steps=2,
            scheduler=DEISMultistepScheduler(num_train_timesteps=100),
        )
    model.eval()
    batch = make_batch(model)
    value = model.loss(**batch).mean()
    assert torch.isfinite(value)
    value.backward()
    forecast = model(**prediction_batch(batch))
    assert forecast.shape == (2, 2, 24, 137)
    assert torch.isfinite(forecast).all()
