# TransTimeGrad

Probabilistic time-series forecasting with PyTorch and GluonTS. The library contains
an LSTM TimeGrad model, a Transformer-conditioned TimeGrad model, and a Transformer
baseline with distribution outputs.

## Install

Python 3.11 or newer is required. Use a virtual environment:

```bash
python -m pip install -e '.[test]'
python -m pytest tests
```

Install `.[examples]` for notebook plotting, or `.[audit]` for `pip-audit`.
The runtime versions in `setup.py` are tested together. Version 1.1 updates the
original dependency stack and predictor format; see [compatibility](docs/validation.md).

## Train and forecast

This small CPU example uses a two-dimensional series. For a scalar series, use
`input_size=1`, a one-dimensional target, and `one_dim_target=True`.

```python
import numpy as np
from diffusers import DEISMultistepScheduler
from gluonts.dataset.common import ListDataset
from transtimegrad.model.trans_timegrad import TransTimeGradEstimator

values = np.sin(np.arange(96, dtype=np.float32) / 6) + 2
training = ListDataset(
    [{"start": "2024-01-01", "target": np.stack([values, values * 2])}],
    freq="D", one_dim_target=False,
)
estimator = TransTimeGradEstimator(
    freq="D", input_size=2, prediction_length=6, context_length=12,
    lags_seq=[1, 7], hidden_size=32, num_layers=2,
    scheduler=DEISMultistepScheduler(num_train_timesteps=100),
    num_inference_steps=10, num_parallel_samples=20,
    batch_size=4, num_batches_per_epoch=2,
    trainer_kwargs={"max_epochs": 1, "accelerator": "cpu", "devices": 1},
)
predictor = estimator.train(training)
forecast = next(predictor.predict(training, num_samples=20))
assert forecast.samples.shape == (20, 6, 2)
```

`TimeGradEstimator` from `transtimegrad.model.rnn_timegrad` uses an LSTM instead.
`TransformerEstimator` from `transtimegrad.model.transformer` fits a distribution
likelihood without diffusion. The [experiment notebook](examples/experiments.ipynb)
shows dataset grouping, evaluation, plots and missing-data experiments; running its
training cells requires the selected datasets and compute. Outputs are cleared so
historical results are not presented as measurements of the current code.

## Save and load

```python
from pathlib import Path
from transtimegrad.serialization import TransTimeGradPredictor

predictor.serialize(Path("saved-model"))
restored = TransTimeGradPredictor.deserialize(Path("saved-model"), trusted=True)
```

Weights use safetensors. The GluonTS constructor configuration can execute Python
code, so use this concrete loader and pass `trusted=True` **only for artifacts you
trust**. The default refuses loading. Legacy pickle/checkpoint artifacts are not a
portable predictor format and are not imported by this loader.

## Model contracts and validation

See [validation and compatibility](docs/validation.md) for scheduler support,
missing observations, optional context loss, count/flow outputs, artifact boundaries,
regression coverage and unverified hardware/quality boundaries. CI is not configured.

## Acknowledgments

Based on [TimeGrad](https://arxiv.org/abs/2101.12072),
[PyTorchTS](https://github.com/zalandoresearch/pytorch-ts), and
[GluonTS](https://github.com/awslabs/gluonts). Existing source notices are preserved.

## License

[MIT](LICENSE); adapted GluonTS files also retain their [Apache 2.0 notices](LICENSE.gluon-ts).
