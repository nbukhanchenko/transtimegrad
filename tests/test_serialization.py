import numpy as np
import pytest
import torch
from diffusers import DEISMultistepScheduler
from gluonts.core.serde import dump_json, load_json
from gluonts.dataset.common import ListDataset
from gluonts.model.predictor import Predictor

from transtimegrad.model.rnn_timegrad import TimeGradEstimator
from transtimegrad.model.trans_timegrad import TransTimeGradEstimator
from transtimegrad.model.transformer import TransformerEstimator
from transtimegrad.modules import StudentTOutput
from transtimegrad.serialization import TransTimeGradPredictor


@pytest.mark.parametrize("kind", ["rnn", "trans", "transformer"])
@pytest.mark.parametrize("dim", [1, 2])
def test_predictor_forecasts_roundtrip_through_tensor_only_artifact(kind, dim, tmp_path):
    kwargs = dict(
        freq="D",
        prediction_length=3,
        context_length=4,
        input_size=dim,
        lags_seq=[1, 2],
        batch_size=2,
        num_parallel_samples=2,
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
        estimator_type = TimeGradEstimator if kind == "rnn" else TransTimeGradEstimator
        estimator = estimator_type(
            **kwargs,
            scheduler=DEISMultistepScheduler(num_train_timesteps=20),
            hidden_size=8,
            num_layers=2,
            num_inference_steps=3,
        )
    predictor = estimator.create_predictor(
        estimator.create_transformation(), estimator.create_lightning_module()
    )
    values = np.linspace(1, 4, 20, dtype=np.float32)
    target = values if dim == 1 else np.stack([values, values * 2])
    dataset = ListDataset(
        [{"start": "2024-01-01", "target": target}], freq="D", one_dim_target=dim == 1
    )
    torch.manual_seed(71)
    expected = list(predictor.predict(dataset, num_samples=2))[0].samples
    predictor.serialize(tmp_path)
    assert (tmp_path / "prediction-net.safetensors").is_file()
    assert not list(tmp_path.glob("*.pt"))
    restored = Predictor.deserialize(tmp_path, device="cpu", trusted=True)
    assert isinstance(restored, TransTimeGradPredictor)
    torch.manual_seed(71)
    actual = list(restored.predict(dataset, num_samples=2))[0].samples
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert next(restored.prediction_net.parameters()).device.type == "cpu"


def test_scheduler_config_roundtrip_excludes_runtime_sampling_state():
    scheduler = DEISMultistepScheduler(num_train_timesteps=37, beta_start=0.001, beta_end=0.02)
    scheduler.set_timesteps(5)
    restored = load_json(dump_json(scheduler))
    assert type(restored) is type(scheduler)
    assert dict(restored.config) == dict(scheduler.config)
    assert restored.num_inference_steps is None


def test_predictor_artifact_preserves_model_parameter_dtype(tmp_path):
    estimator = TransformerEstimator(
        freq="D",
        prediction_length=2,
        context_length=3,
        input_size=1,
        lags_seq=[1],
        num_parallel_samples=2,
    )
    module = estimator.create_lightning_module().double()
    predictor = estimator.create_predictor(estimator.create_transformation(), module)
    predictor.serialize(tmp_path)
    restored = TransTimeGradPredictor.deserialize(tmp_path, trusted=True)
    assert next(restored.prediction_net.parameters()).dtype == torch.float64


def _small_predictor():
    estimator = TransformerEstimator(
        freq="D",
        prediction_length=2,
        context_length=3,
        input_size=1,
        lags_seq=[1],
        num_parallel_samples=2,
    )
    return estimator.create_predictor(
        estimator.create_transformation(), estimator.create_lightning_module()
    )


def _artifact_bytes(path):
    return {item.name: item.read_bytes() for item in path.iterdir()}


@pytest.mark.parametrize("failing_file", ["weights", "config"])
def test_serialization_write_failure_preserves_existing_artifact(
    tmp_path, monkeypatch, failing_file
):
    from pathlib import Path

    import transtimegrad.serialization as serialization

    destination = tmp_path / "artifact"
    predictor = _small_predictor()
    predictor.serialize(destination)
    before = _artifact_bytes(destination)
    if failing_file == "weights":

        def fail_weights(*args, **kwargs):
            raise OSError("injected weight write failure")

        monkeypatch.setattr(serialization, "save_file", fail_weights)
    else:
        original = Path.write_text

        def fail_config(path, *args, **kwargs):
            if path.name == "predictor.json":
                raise OSError("injected config write failure")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail_config)
    with pytest.raises(OSError, match="injected"):
        predictor.serialize(destination)
    assert _artifact_bytes(destination) == before
    assert not list(tmp_path.glob(".artifact-stage-*"))


def test_serialization_refuses_unrelated_files_and_symlink_targets(tmp_path):
    predictor = _small_predictor()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "notes.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="unrelated"):
        predictor.serialize(unrelated)
    assert (unrelated / "notes.txt").read_text() == "keep"
    link = tmp_path / "link"
    link.symlink_to(unrelated, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        predictor.serialize(link)
    assert link.is_symlink()


def test_serialization_refuses_symlink_files_inside_an_artifact(tmp_path):
    predictor = _small_predictor()
    predictor.serialize(tmp_path / "artifact")
    metadata = tmp_path / "artifact" / "predictor.json"
    stored = metadata.read_text()
    metadata.unlink()
    outside = tmp_path / "outside.json"
    outside.write_text(stored)
    metadata.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        predictor.serialize(tmp_path / "artifact")
    assert outside.read_text() == stored
    assert metadata.is_symlink()


def test_publication_failure_restores_previous_artifact(tmp_path, monkeypatch):
    from pathlib import Path

    predictor = _small_predictor()
    destination = tmp_path / "artifact"
    predictor.serialize(destination)
    before = _artifact_bytes(destination)
    original = Path.rename

    def fail_install(path, target):
        if path.name.startswith(".artifact-stage-"):
            raise OSError("injected publication failure")
        return original(path, target)

    monkeypatch.setattr(Path, "rename", fail_install)
    with pytest.raises(OSError, match="publication"):
        predictor.serialize(destination)
    assert _artifact_bytes(destination) == before
    assert not list(tmp_path.glob(".artifact-backup-*"))


def test_failed_rollback_retains_previous_artifact_and_reports_backup(tmp_path, monkeypatch):
    from pathlib import Path

    predictor = _small_predictor()
    destination = tmp_path / "artifact"
    predictor.serialize(destination)
    before = _artifact_bytes(destination)
    original = Path.rename

    def fail_install_and_restore(path, target):
        if path.name.startswith((".artifact-stage-", ".artifact-backup-")):
            raise OSError("injected rename failure")
        return original(path, target)

    monkeypatch.setattr(Path, "rename", fail_install_and_restore)
    with pytest.raises(RuntimeError, match="previous artifact retained at") as error:
        predictor.serialize(destination)
    backups = list(tmp_path.glob(".artifact-backup-*"))
    assert len(backups) == 1
    assert str(backups[0]) in str(error.value)
    assert _artifact_bytes(backups[0]) == before


def test_failure_to_move_old_artifact_leaves_no_empty_backups(tmp_path, monkeypatch):
    from pathlib import Path

    predictor = _small_predictor()
    destination = tmp_path / "artifact"
    predictor.serialize(destination)
    before = _artifact_bytes(destination)
    original = Path.rename

    def fail_old_move(path, target):
        if path == destination:
            raise OSError("injected old artifact rename failure")
        return original(path, target)

    monkeypatch.setattr(Path, "rename", fail_old_move)
    with pytest.raises(OSError, match="old artifact"):
        predictor.serialize(destination)
    assert _artifact_bytes(destination) == before
    assert not list(tmp_path.glob(".artifact-backup-*"))
    assert not list(tmp_path.glob(".artifact-stage-*"))


@pytest.mark.parametrize("flow_type", ["realnvp", "maf"])
def test_transformer_flow_output_train_checkpoint_and_forecast_roundtrip(flow_type, tmp_path):
    from transtimegrad.modules.distribution_output import FlowOutput
    from transtimegrad.modules.flows import MAF, RealNVP

    cls = RealNVP if flow_type == "realnvp" else MAF
    options = {"input_order": "random"} if flow_type == "maf" else {}
    flow = cls(
        n_blocks=2,
        input_size=2,
        hidden_size=8,
        n_hidden=1,
        cond_label_size=4,
        batch_norm=False,
        **options,
    )
    output = FlowOutput(flow, input_size=2, cond_size=4)
    estimator = TransformerEstimator(
        freq="D",
        prediction_length=2,
        context_length=3,
        input_size=2,
        lags_seq=[1, 2],
        num_parallel_samples=2,
        batch_size=2,
        num_batches_per_epoch=1,
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        distr_output=output,
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
            default_root_dir=str(tmp_path / "train"),
        ),
    )
    target = np.stack([np.linspace(1, 2, 20), np.linspace(2, 4, 20)]).astype(np.float32)
    dataset = ListDataset(
        [{"start": "2024-01-01", "target": target}], freq="D", one_dim_target=False
    )
    predictor = estimator.train(dataset, validation_data=dataset)
    torch.manual_seed(13)
    expected = list(predictor.predict(dataset))[0].samples
    predictor.serialize(tmp_path / "artifact")
    restored = TransTimeGradPredictor.deserialize(tmp_path / "artifact", trusted=True)
    torch.manual_seed(13)
    actual = list(restored.predict(dataset))[0].samples
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_scheduler_checkpoint_config_supports_explicit_numpy_beta_array():
    from diffusers import DDPMScheduler

    from transtimegrad.serialization import checkpoint_decode, checkpoint_encode

    scheduler = DDPMScheduler(num_train_timesteps=3, trained_betas=np.array([0.01, 0.02, 0.03]))
    restored = checkpoint_decode(checkpoint_encode(scheduler))
    torch.testing.assert_close(restored.alphas_cumprod, scheduler.alphas_cumprod)
