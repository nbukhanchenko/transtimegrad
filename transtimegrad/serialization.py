"""Predictor persistence with tensor-only weights and an explicit config trust boundary.

GluonTS represents Python constructors in predictor.json. That configuration is
executable, even though it is JSON: load only artifacts from a trusted source.
Weights use safetensors; this module never reads pickle or torch.load files.
"""

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    PNDMScheduler,
    SchedulerMixin,
    UniPCMultistepScheduler,
)
from gluonts.core.serde import Kind, encode, load_json
from gluonts.model.predictor import Predictor
from gluonts.torch.model.predictor import PyTorchPredictor
from safetensors.torch import load_file, save_file

from .storage import atomic_directory_write

SCHEDULER_TYPES = {
    cls.__name__: cls
    for cls in (
        DDIMScheduler,
        DDPMScheduler,
        DEISMultistepScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler,
        UniPCMultistepScheduler,
        PNDMScheduler,
    )
}
_FORMAT = {"format": "transtimegrad.safetensors", "version": 1}


def scheduler_from_config(class_name: str, config: dict) -> SchedulerMixin:
    """Reconstruct a supported scheduler without importing arbitrary class paths."""
    if not isinstance(class_name, str) or class_name not in SCHEDULER_TYPES:
        raise ValueError(f"Unsupported scheduler class: {class_name}")
    if not isinstance(config, dict):
        raise ValueError("Scheduler config must be a dictionary")
    return SCHEDULER_TYPES[class_name].from_config(config)


def _scheduler_json_value(value):
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _scheduler_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scheduler_json_value(item) for item in value]
    return value


@encode.register(SchedulerMixin)
def _encode_scheduler(scheduler: SchedulerMixin):
    name = type(scheduler).__name__
    if SCHEDULER_TYPES.get(name) is not type(scheduler):
        raise ValueError(f"Unsupported scheduler class: {name}")
    return {
        "__kind__": Kind.Instance,
        "class": "transtimegrad.serialization.scheduler_from_config",
        "kwargs": encode(
            {"class_name": name, "config": _scheduler_json_value(dict(scheduler.config))}
        ),
    }


class TransTimeGradPredictor(PyTorchPredictor):
    """GluonTS predictor saved with safetensors and explicitly trusted config loading."""

    @staticmethod
    def _validate_artifact_directory(path: Path) -> None:
        if path.is_symlink() or not path.is_dir():
            raise ValueError("Predictor artifact must be a regular directory without symlinks")
        entries = list(path.iterdir())
        if not entries:
            return
        expected = {
            "prediction-net.safetensors",
            "predictor.json",
            "transtimegrad-format.json",
            "gluonts-config.json",
        }
        if {entry.name for entry in entries} != expected:
            raise ValueError("Artifact directory contains unrelated files or is incomplete")
        if any(entry.is_symlink() or not entry.is_file() for entry in entries):
            raise ValueError("Predictor artifact files must be regular files without symlinks")
        metadata = json.loads((path / "transtimegrad-format.json").read_text(encoding="utf-8"))
        if metadata != _FORMAT:
            raise ValueError("Unsupported predictor artifact format or version")

    def serialize(self, path: Path) -> None:
        representation = encode(self)
        # Build on CPU before restoring weights or choosing a device, so artifacts
        # created on CUDA remain portable to CPU-only machines.
        representation["kwargs"]["device"] = "cpu"
        weights = {
            key: value.detach().cpu().contiguous().clone()
            for key, value in self.prediction_net.state_dict().items()
        }

        def write(stage: Path) -> None:
            save_file(weights, str(stage / "prediction-net.safetensors"))
            (stage / "predictor.json").write_text(json.dumps(representation), encoding="utf-8")
            (stage / "transtimegrad-format.json").write_text(json.dumps(_FORMAT), encoding="utf-8")
            Predictor.serialize(self, stage)

        atomic_directory_write(Path(path), write, self._validate_artifact_directory)

    @classmethod
    def deserialize(
        cls,
        path: Path,
        device: Optional[Union[str, torch.device]] = None,
        *,
        trusted: bool = False,
    ) -> "TransTimeGradPredictor":
        """Load an artifact only after the caller explicitly trusts its config.

        ``trusted=True`` permits GluonTS constructor reconstruction, which can
        execute Python code. Safetensors protects the weight file, not that JSON
        constructor format. Do not opt in for downloaded, untrusted artifacts.
        """
        if trusted is not True:
            raise ValueError(
                "Predictor config can execute Python code. Load only a trusted artifact "
                "and explicitly pass trusted=True."
            )
        path = Path(path)
        cls._validate_artifact_directory(path)
        metadata = json.loads((path / "transtimegrad-format.json").read_text(encoding="utf-8"))
        if metadata != _FORMAT:
            raise ValueError("Unsupported predictor artifact format or version")
        predictor = load_json((path / "predictor.json").read_text(encoding="utf-8"))
        if type(predictor) is not cls:
            raise ValueError("Artifact does not contain the expected predictor class")
        weights = load_file(str(path / "prediction-net.safetensors"), device="cpu")
        predictor.prediction_net.load_state_dict(weights, strict=True, assign=True)
        predictor.to("cpu" if device is None else device)
        return predictor


def _checkpoint_constructors():
    """Closed constructor set for weights_only Lightning checkpoint parameters."""
    from gluonts.torch import distributions as gluonts_distributions

    from .model.loss import NegativeLogLikelihood
    from .modules import distribution_output as outputs
    from .modules.flows import MAF, RealNVP

    constructors = {
        "transtimegrad.serialization.scheduler_from_config": scheduler_from_config,
        "transtimegrad.model.loss.NegativeLogLikelihood": NegativeLogLikelihood,
        "transtimegrad.modules.flows.MAF": MAF,
        "transtimegrad.modules.flows.RealNVP": RealNVP,
    }
    output_names = (
        "NormalOutput",
        "BetaOutput",
        "PoissonOutput",
        "NegativeBinomialOutput",
        "ZeroInflatedPoissonOutput",
        "ZeroInflatedNegativeBinomialOutput",
        "StudentTOutput",
        "StudentTMixtureOutput",
        "NormalMixtureOutput",
        "LowRankMultivariateNormalOutput",
        "MultivariateNormalOutput",
        "FlowOutput",
    )
    for name in output_names:
        cls = getattr(outputs, name)
        constructors[f"{cls.__module__}.{cls.__name__}"] = cls
    for name in (
        "NormalOutput",
        "StudentTOutput",
        "BetaOutput",
        "GammaOutput",
        "PoissonOutput",
        "NegativeBinomialOutput",
    ):
        cls = getattr(gluonts_distributions, name)
        constructors[f"{cls.__module__}.{cls.__name__}"] = cls
    return constructors


def _decode_checkpoint_value(value):
    if isinstance(value, list):
        return [_decode_checkpoint_value(item) for item in value]
    if not isinstance(value, dict):
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise ValueError("Checkpoint configuration must contain only JSON values")
    if not all(isinstance(key, str) for key in value):
        raise ValueError("Checkpoint configuration keys must be strings")
    if "transtimegrad_kind" not in value:
        return {key: _decode_checkpoint_value(item) for key, item in value.items()}
    if value.get("transtimegrad_kind") != "instance":
        raise ValueError("Checkpoint configuration only supports whitelisted instances")
    class_name = value.get("class")
    if not isinstance(class_name, str):
        raise ValueError("Checkpoint constructor name must be a string")
    constructor = _checkpoint_constructors().get(class_name)
    if constructor is None:
        raise ValueError(f"Unsupported checkpoint constructor: {value.get('class')}")
    if set(value) - {"transtimegrad_kind", "class", "args", "kwargs"}:
        raise ValueError("Unexpected checkpoint constructor fields")
    args = _decode_checkpoint_value(value.get("args", []))
    kwargs = _decode_checkpoint_value(value.get("kwargs", {}))
    if not isinstance(args, list) or not isinstance(kwargs, dict):
        raise ValueError("Invalid checkpoint constructor arguments")
    return constructor(*args, **kwargs)


def _checkpoint_tags(value):
    # GluonTS must not eagerly decode the nested safe checkpoint representation
    # while rebuilding a predictor. Only our closed decoder interprets these tags.
    if isinstance(value, list):
        return [_checkpoint_tags(item) for item in value]
    if isinstance(value, dict):
        return {
            ("transtimegrad_kind" if key == "__kind__" else key): _checkpoint_tags(item)
            for key, item in value.items()
        }
    return value


def checkpoint_encode(value):
    """Store constructor parameters as plain data accepted by weights_only=True.

    This intentionally supports a closed set of schedulers, distributions and
    losses, rather than permitting arbitrary Python constructors in checkpoints.
    """
    encoded = _checkpoint_tags(json.loads(json.dumps(encode(value), allow_nan=False)))
    _decode_checkpoint_value(encoded)  # Fail before training for unsupported classes.
    return {"transtimegrad_checkpoint_config": 1, "value": encoded}


def checkpoint_decode(value):
    """Restore the closed constructor format, or accept ordinary live parameters."""
    if isinstance(value, dict) and "transtimegrad_checkpoint_config" in value:
        valid_keys = set(value) == {"transtimegrad_checkpoint_config", "value"}
        version = value["transtimegrad_checkpoint_config"]
        if not valid_keys or type(version) is not int or version != 1:
            raise ValueError("Unsupported checkpoint configuration version")
        return _decode_checkpoint_value(value["value"])
    return value
