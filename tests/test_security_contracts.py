import json

import pytest
from diffusers import DDIMScheduler
from gluonts.core.serde import dump_json

import transtimegrad.serialization as serialization
from transtimegrad.serialization import TransTimeGradPredictor, scheduler_from_config


@pytest.mark.parametrize("trusted", [False, None, 0, 1, "yes"])
def test_untrusted_deserialization_is_rejected_before_reading_or_decoding(
    tmp_path, monkeypatch, trusted
):
    def unexpected_decode(*args, **kwargs):
        raise AssertionError("An untrusted artifact must not be decoded")

    monkeypatch.setattr(serialization, "load_json", unexpected_decode)
    # No files exist: the trust guard must run before even opening configuration.
    with pytest.raises(ValueError, match="trusted=True"):
        TransTimeGradPredictor.deserialize(tmp_path, trusted=trusted)


def test_unsupported_artifact_version_rejected_before_constructor_decode(tmp_path, monkeypatch):
    for name in ("prediction-net.safetensors", "predictor.json", "gluonts-config.json"):
        (tmp_path / name).write_bytes(b"unused")
    (tmp_path / "transtimegrad-format.json").write_text(
        json.dumps({"format": "transtimegrad.safetensors", "version": 999}), encoding="utf-8"
    )
    monkeypatch.setattr(
        serialization, "load_json", lambda *args: pytest.fail("Decoded invalid artifact")
    )
    with pytest.raises(ValueError, match="format or version"):
        TransTimeGradPredictor.deserialize(tmp_path, trusted=True)


def test_scheduler_restoration_does_not_accept_arbitrary_import_paths():
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        scheduler_from_config("os.system", {"command": "unused"})


def test_unregistered_scheduler_subclass_is_not_silently_reconstructed_as_base():
    class CustomScheduler(DDIMScheduler):
        pass

    with pytest.raises(ValueError, match="Unsupported scheduler"):
        dump_json(CustomScheduler())


def test_checkpoint_parameters_never_execute_unlisted_constructors():
    payload = {
        "transtimegrad_checkpoint_config": 1,
        "value": {"transtimegrad_kind": "instance", "class": "builtins.eval", "args": ["1 + 1"]},
    }
    with pytest.raises(ValueError, match="Unsupported checkpoint constructor"):
        serialization.checkpoint_decode(payload)


@pytest.mark.parametrize("kind", ["type", "stateful", "unknown"])
def test_checkpoint_parameters_reject_other_serde_object_kinds(kind):
    payload = {
        "transtimegrad_checkpoint_config": 1,
        "value": {
            "transtimegrad_kind": kind,
            "class": "transtimegrad.model.loss.NegativeLogLikelihood",
        },
    }
    with pytest.raises(ValueError, match="whitelisted instances"):
        serialization.checkpoint_decode(payload)


def test_checkpoint_configuration_is_plain_json_and_weights_only_readable(tmp_path):
    import torch
    from diffusers import DEISMultistepScheduler

    from transtimegrad.modules import StudentTOutput

    config = serialization.checkpoint_encode(
        {
            "scheduler": DEISMultistepScheduler(num_train_timesteps=20),
            "distr_output": StudentTOutput(dim=2),
        }
    )
    assert json.loads(json.dumps(config)) == config
    checkpoint = tmp_path / "parameters.pt"
    torch.save(config, checkpoint)
    restored = serialization.checkpoint_decode(torch.load(checkpoint, weights_only=True))
    assert restored["scheduler"].config.num_train_timesteps == 20
    assert restored["distr_output"].event_shape == (2,)


def test_trusted_loader_refuses_symlinked_artifact_before_decoding(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    link = tmp_path / "link"
    link.symlink_to(artifact, target_is_directory=True)
    monkeypatch.setattr(
        serialization, "load_json", lambda *args: pytest.fail("Decoded symlinked artifact")
    )
    with pytest.raises(ValueError, match="symlink"):
        TransTimeGradPredictor.deserialize(link, trusted=True)
