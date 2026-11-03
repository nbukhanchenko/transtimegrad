# Model contracts and validation

## Compatibility

Version 1.1 requires Python >=3.11 and the patched runtime stack in `setup.py`.
Old serialized predictors/checkpoints containing Python objects are not an accepted
portable format. Rebuild predictors with the current constructors and weights from
a trusted source; do not disable PyTorch's safe loading to import unknown files.
The original PyTorch 2.2.2 pin is affected by published checkpoint-loading advisories
[GHSA-53q9-r3pm-6pq6](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6)
and [GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p).
The updated pin is beyond their patched-version thresholds. Predictor artifacts contain safetensors weights and a GluonTS JSON constructor
configuration. JSON constructor decoding remains a trusted-code boundary, and the
concrete `TransTimeGradPredictor.deserialize` requires `trusted=True` before decoding.
Lightning checkpoint hyperparameters use a closed plain-data constructor format;
arbitrary constructors are rejected rather than allowlisted into pickle loading.

## Training and inference

- Scalar targets have shape `(batch, time)`; vector targets `(batch, time, dimension)`.
  Distribution event shapes must match. The Transformer default output is scalar
  Student-t or independent vector Student-t according to `input_size`.
- Lags are positive integers in public model constructors. The RNN's internal zero
  lag represents the last known observation, never the prediction target.
- All model/device conversions use PyTorch's normal `.to(...)` mechanism. The default
  is float32. Sampling follows conditioning dtype/device.
- The two diffusion models support alpha-based DDPM, DDIM, PNDM, DEIS, DPM-Solver
  single/multistep and UniPC schedulers with epsilon, sample or v-prediction targets
  (PNDM supports epsilon and v-prediction only).
  Sigma schedulers and learned-variance configurations are rejected because they
  require different denoiser parameterizations. Inference-step counts must be valid
  for the selected scheduler. Each rollout owns its multistep solver state.
- Missing coordinates retain observed-coordinate diffusion loss. Missing NaNs are
  imputed before entering lag features and the denoiser. Fully unobserved steps have
  zero loss. Independent likelihood outputs support coordinate masking. Partial
  vectors with joint likelihoods or custom losses are rejected explicitly rather
  than silently dropping all observed values; supply fully observed events or an
  appropriate marginal likelihood.
- `future_only=False` adds context targets to the two diffusion objectives. Each
  auxiliary target is scored from a strictly earlier, left-padded prefix with its
  own normalization. The primary forecast path is unchanged. This optional mode
  costs `context_length - 1` additional one-step forecasts per batch.

## Distribution and flow APIs

Continuous outputs preserve scalar/vector/component axes and support affine
normalization. Count outputs preserve integer support by scaling rates/means;
nonzero centering is rejected. Flow adapters capture conditioning and affine
parameters per returned distribution, register trainable flow parameters, and expose
normalized `log_prob` and sampling. Training BatchNorm uses prior running statistics
for an exact current affine transform, then updates statistics for subsequent calls.
Random MADE ranks are persisted with model weights.

A Gaussian diffusion denoising objective is not an exact density. The legacy
`GaussianDiffusion` and `DiffusionOutput` expose sampling and explicit denoising
loss, while exact `log_prob` raises `NotImplementedError`. Use the TimeGrad estimators
for diffusion training, not the Transformer's likelihood-loss path.

## Data and filesystem boundaries

M5 generation accepts local Kaggle CSVs; it does not download data or accept terms.
Splits must have consistent identities and aligned daily histories. Leading-zero
trimming uses training history only, and missing prices are filled causally. Generated
metadata and entries are read back before replacing an existing dataset.

Dataset and predictor writes stage complete artifacts before replacing a recognized
output. Unrelated files and symlinks are refused. If installation and rollback both
fail, the old artifact is retained and its recovery path is included in the error.
This guards ordinary local failures, not concurrent adversarial filesystem changes;
use a directory you control without concurrent writers.

## Checks

```bash
python -m pytest tests -W error::RuntimeWarning
ruff check transtimegrad tests
ruff format --check transtimegrad tests
python -m pip_audit
```

The test suite covers train/validation/predict routes for all three estimators with
and without validation; scalar/vector and single-context cases; teacher-forcing vs
autoregressive conditioning; strict auxiliary causality; missing observations;
scheduler prediction targets; optimization of a fixed synthetic batch; dtype
conversions; predictor/checkpoint roundtrips and trust rejection; numerical/Jacobian
properties of distributions/flows; M5 synthetic save/readback and transactional
failures; and missing-data helpers extracted from the notebook.

CPU tests and small local MPS loss/backward/forecast smoke checks have been performed.
CUDA, distributed training, full Kaggle M5 volume, and complete benchmark training
have not been exercised. Synthetic loss reduction checks implementation trainability,
not forecast quality, calibration or superiority over another architecture.
Dependency auditing reports known advisories in the installed environment at the
check time; it cannot prove the absence of undiscovered vulnerabilities. Notebook
training/evaluation must be rerun before citing any current-model benchmark result.

## Acceptance snapshot

The hardening pass completed 258 regression/integration checks on Python 3.11
from the working tree and Python 3.13 from a built wheel in a separate environment.
RuntimeWarnings were treated as errors. Ruff lint/format and dependency consistency
checks passed. `pip-audit` reported no known vulnerabilities in auditable installed dependencies
in the reviewed modern environment. The unpublished project itself is covered by
source review and tests, not the PyPI advisory database. Small MPS loss/backward/forecast smoke checks passed for all three models.
The README train/predict/save/load example was executed successfully.

Independent adversarial reviews covered model/scheduler/normalization consistency,
numerical gradients and flow Jacobians, captured conditioning/state, safe checkpoint
constructors, predictor persistence, and data-write failure recovery. Counterexamples
found during review were repaired and retained as regressions. No confirmed actionable
findings remained in those reviewed paths at acceptance. This is bounded engineering
evidence, not a proof that every possible bug or undiscovered vulnerability is absent.
