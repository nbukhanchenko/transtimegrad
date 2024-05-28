# transtimegrad
TransTimeGrad is a PyTorch probabilistic time series model which is a transformer based version of the TimeGrad model. It utilizes GluonTS backend API.

## Quick Start
### Installation
To install the TransTimeGrad library with pip, you can simply proceed with

```bash
pip install transtimegrad
```

Also you may want to download the library directly from the GitHub repo as follows

```bash
pip install git+https://github.com/nbukhanchenko/transtimegrad
```

### Usage
To build a trained model you may follow the pipepline from `examples/experiments.ipynb`. Otherwise, proceed with following steps:
1. Prepare your datasets:
* if you want to use ready-to-go datasets, exploit `get_dataset` from GluonTS, for instance, `get_dataset("solar_nips")` in order to get the Solar dataset
* split your data into train and test with `MultivariateGrouper` from GluonTS
* an example of your code might look something like this:

```python
def prepare_dataset(dataset_name):
    # you don't need this line if you already have a dataset
    dataset = get_dataset(dataset_name, regenerate=False)

    train_grouper = MultivariateGrouper(
        max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality)
    )
    test_grouper = MultivariateGrouper(
        num_test_dates=int(len(dataset.test) / len(dataset.train)),
        max_target_dim=int(dataset.metadata.feat_static_cat[0].cardinality),
    )

    return {
        "train": train_grouper(dataset.train),
        "test": test_grouper(dataset.test),
        "metadata": dataset.metadata
    }
```

2. Prepare your predictor:
* choose the model to work with: `rnn_timegrad` (standard TimeGrad model) or `trans_timegrad` (our newly proposed TransTimeGrad model)
* choose scheduler from diffusers library, for instance `DEISMultistepScheduler`
* specify parameters of scheduler and estimator
* an example of your code might look something like this:

```python
def prepare_predictor(dataset, mode, max_epochs, accelerator,
                      num_train_timesteps=100, beta_start=1e-4, beta_end=0.1, beta_schedule="linear",
                      context_length_coef=3, num_layers=2, hidden_size=64, lr=1e-3, weight_decay=1e-8, dropout_rate=0.1,
                      lags_seq=[1, 7, 14, 24, 168], num_inference_steps=99, batch_size=64, num_batches_per_epoch=100):
    model = {
        "rnn_timegrad": TimeGradEstimator,
        "trans_timegrad": TransTimeGradEstimator,
    }
    
    scheduler = DEISMultistepScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
    )
    
    estimator = model[mode](
        freq=dataset["metadata"].freq,
        prediction_length=dataset["metadata"].prediction_length,
        input_size=int(dataset["metadata"].feat_static_cat[0].cardinality),
        scheduler=scheduler,
        context_length=context_length_coef*dataset["metadata"].prediction_length,
        num_layers=num_layers,
        hidden_size=hidden_size,
        lr=lr,
        weight_decay=weight_decay,
        dropout_rate=dropout_rate,
        scaling="mean",
        lags_seq=lags_seq,
        num_inference_steps=num_inference_steps,
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
        trainer_kwargs=dict(max_epochs=max_epochs, accelerator=accelerator, devices="1"),
    )

    return estimator.train(
        training_data=dataset["train"],
        cache_data=True,
        shuffle_buffer_length=1024,
    )
```

3. Run the model:
* prepare metrics for `num_samples` runs at the test set, using `MultivariateEvaluator`, `make_evaluation_predictions` and `evaluator` from GluonTS:

```python
def prepare_metrics(dataset, predictor, num_samples=100):
    evaluator = MultivariateEvaluator(
        quantiles=(np.arange(20) / 20.0)[1:], target_agg_funcs={"sum": np.sum}
    )

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset["test"], predictor=predictor, num_samples=num_samples
    )
    forecasts, targets = list(forecast_it), list(ts_it)
    agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset["test"]))

    return forecasts, targets, agg_metric
```

* finally, prepare statistics and visualize predictions, for example, like this (you can find definition of the `plot` function at `examples/experiments.ipynb`):

```python
def prepare_statistics(dataset, forecasts, targets, agg_metric, frame=0):
    metrics = {
        "CRPS": "mean_wQuantileLoss",
        "ND": "ND",
        "NRMSE": "NRMSE",
        "MSE": "MSE"
    }

    for name in metrics:
        print(f"{name}: {round(agg_metric[metrics[name]], 4)}")
        print(f"{name + '-Sum'}: {round(agg_metric['m_sum_' + metrics[name]], 4)}")
        print("-" * 64)

    plot(
        target=targets[frame],
        forecast=forecasts[frame],
        prediction_length=dataset["metadata"].prediction_length,
    )
    plt.show()
```

The full training-inference pipeline for a given dataset then looks like this:

```python
predictor = prepare_predictor(
    dataset=dataset, mode="rnn_timegrad", max_epochs=40, accelerator="gpu",
    num_train_timesteps=100, beta_start=1e-4, beta_end=0.1, beta_schedule="linear",
    context_length_coef=1, num_layers=3, hidden_size=40, lr=1e-3, weight_decay=1e-8, dropout_rate=0.1,
    lags_seq=[1, 7, 14, 24, 168], num_inference_steps=99, batch_size=64, num_batches_per_epoch=100)
forecasts, targets, agg_metric = prepare_metrics(dataset, predictor)
prepare_statistics(
    dataset["dataset"], dataset["forecasts"],
    dataset["targets"], dataset["agg_metric"]
)
```

## Acknowledgments
We thank the following repositories, papers and their authors and do not claim any authorship on their achievments.

* [Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting](https://arxiv.org/abs/2101.12072), [GitHub repo](https://github.com/zalandoresearch/pytorch-ts)

## License
[MIT]() License