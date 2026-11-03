"""Reproducible missing-observation helpers used by the experiment notebook."""

from copy import deepcopy

import numpy as np


def _target_matrix(target):
    target = np.asarray(target)
    if target.ndim not in (1, 2):
        raise ValueError("target must be a vector or a (series, time) matrix")
    return target[None, :] if target.ndim == 1 else target


def get_max_shape(dataset):
    """Return independent maxima of series count and history length."""
    shapes = [_target_matrix(entry["target"]).shape for entry in dataset]
    return tuple(max((shape[axis] for shape in shapes), default=0) for axis in (0, 1))


def prepare_dataset_with_missing_values(
    dataset, missing_rate, missing_type, missing_value=np.nan, rng=None
):
    """Copy a dataset and mask test observations; never mutate the caller's data.

    The same mask is reused across aligned test windows. Pass a NumPy generator
    to reproduce corruption independently of the process-global random state.
    """
    if not np.isfinite(missing_rate) or not 0 <= missing_rate <= 1:
        raise ValueError("missing_rate must be between zero and one")
    if missing_type not in {"random", "time-level", "series-level"}:
        raise ValueError("unknown missing_type")
    if not np.isscalar(missing_value) or (
        not np.isnan(missing_value) and not np.isfinite(missing_value)
    ):
        raise ValueError("missing_value must be finite or NaN")
    result = deepcopy(dataset)
    height, width = get_max_shape(result["test"])
    random = np.random if rng is None else rng
    mask_shape = {"random": (height, width), "time-level": (1, width), "series-level": (height, 1)}[
        missing_type
    ]
    mask = np.broadcast_to(random.random(mask_shape) < missing_rate, (height, width))
    for entry in result["test"]:
        original = np.asarray(entry["target"])
        target = _target_matrix(original).astype(float, copy=True)
        rows, columns = target.shape
        target[mask[:rows, :columns]] = missing_value
        entry["target"] = target[0] if original.ndim == 1 else target
    return result


def prepare_dataset_without_missing_values(dataset_with_missing_values, missing_value=np.nan):
    """Forward fill missing observations using only earlier observations.

    Missing values before the first observation, including entirely missing
    series, are zero-filled. Existing observations (including the first one)
    and the original dataset are preserved.
    """
    result = deepcopy(dataset_with_missing_values)
    for entry in result["test"]:
        original = np.asarray(entry["target"])
        target = _target_matrix(original).astype(float, copy=True)
        mask = (
            np.isnan(target)
            if np.isnan(missing_value)
            else np.isnan(target) | (target == missing_value)
        )
        if np.isinf(target).any():
            raise ValueError("target contains infinite observations")
        if target.shape[1]:
            positions = np.where(~mask, np.arange(target.shape[1]), -1)
            np.maximum.accumulate(positions, axis=1, out=positions)
            target = np.where(
                positions >= 0, np.take_along_axis(target, positions.clip(min=0), axis=1), 0.0
            )
        entry["target"] = target[0] if original.ndim == 1 else target
    return result
