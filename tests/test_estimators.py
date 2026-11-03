import numpy as np
import pytest
from diffusers import DEISMultistepScheduler
from gluonts.dataset.common import ListDataset

from transtimegrad.model.rnn_timegrad import TimeGradEstimator
from transtimegrad.model.trans_timegrad import TransTimeGradEstimator
from transtimegrad.model.transformer import TransformerEstimator
from transtimegrad.modules import StudentTOutput


@pytest.mark.parametrize("with_validation", [False, True])
@pytest.mark.parametrize("kind", ["rnn", "trans", "transformer"])
@pytest.mark.parametrize("dim", [1, 2])
def test_estimator_train_validate_and_predict(kind, dim, with_validation, tmp_path):
    values = np.sin(np.arange(40, dtype=np.float32) / 3) + 2
    target = values if dim == 1 else np.stack([values, 2 * values])
    data = ListDataset(
        [{"start": "2024-01-01", "target": target}], freq="D", one_dim_target=dim == 1
    )
    kwargs = dict(
        freq="D",
        prediction_length=3,
        context_length=3,
        input_size=dim,
        lags_seq=[1, 2],
        num_parallel_samples=2,
        batch_size=2,
        num_batches_per_epoch=1,
        trainer_kwargs=dict(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            num_sanity_val_steps=0,
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            default_root_dir=str(tmp_path),
        ),
    )
    if kind == "transformer":
        estimator = TransformerEstimator(
            **kwargs,
            d_model=8,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            distr_output=StudentTOutput(dim=dim) if dim > 1 else StudentTOutput(),
        )
    else:
        cls = TimeGradEstimator if kind == "rnn" else TransTimeGradEstimator
        estimator = cls(
            **kwargs,
            hidden_size=8,
            num_layers=2,
            num_inference_steps=2,
            scheduler=DEISMultistepScheduler(num_train_timesteps=10),
        )
    predictor = estimator.train(data, validation_data=data if with_validation else None)
    forecasts = list(predictor.predict(data, num_samples=2))
    assert len(forecasts) == 1
    expected_shape = (2, 3) + ((dim,) if dim > 1 else ())
    assert forecasts[0].samples.shape == expected_shape
    assert np.isfinite(forecasts[0].samples).all()
