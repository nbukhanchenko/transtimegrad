import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from transtimegrad.dataset.missing import (
    get_max_shape,
    prepare_dataset_with_missing_values,
    prepare_dataset_without_missing_values,
)


def data(targets):
    return {
        "train": [{"target": np.array([4.0, 5.0])}],
        "test": [{"target": np.array(target)} for target in targets],
        "metadata": {"freq": "D"},
    }


@pytest.mark.parametrize("count", [0, 1, 4])
def test_fill_every_test_entry_without_changing_observed_values_or_input(count):
    dataset = data([[[1.0, np.nan, 3.0], [np.nan, 2.0, np.nan], [np.nan, np.nan, np.nan]]] * count)
    original = deepcopy(dataset)
    result = prepare_dataset_without_missing_values(dataset, np.nan)
    for entry in result["test"]:
        np.testing.assert_array_equal(
            entry["target"], [[1.0, 1.0, 3.0], [0.0, 2.0, 2.0], [0.0, 0.0, 0.0]]
        )
    for entry, old in zip(dataset["test"], original["test"], strict=True):
        np.testing.assert_array_equal(entry["target"], old["target"])
    np.testing.assert_array_equal(result["train"][0]["target"], dataset["train"][0]["target"])


def test_fill_finite_sentinel_empty_history_and_scalar_series():
    dataset = data([[-99, 3, -99, 4], np.empty((2, 0))])
    result = prepare_dataset_without_missing_values(dataset, -99)
    np.testing.assert_array_equal(result["test"][0]["target"], [0.0, 3.0, 3.0, 4.0])
    assert result["test"][1]["target"].shape == (2, 0)


def test_corruption_ragged_axes_rates_reproducibility_and_no_mutation():
    dataset = data([np.ones((3, 2)), np.ones((2, 5))])
    assert get_max_shape(dataset["test"]) == (3, 5)
    for missing_type in ("random", "time-level", "series-level"):
        out = prepare_dataset_with_missing_values(
            dataset, 0.5, missing_type, np.nan, np.random.default_rng(2)
        )
        repeated = prepare_dataset_with_missing_values(
            dataset, 0.5, missing_type, np.nan, np.random.default_rng(2)
        )
        for before, after, repeat in zip(
            dataset["test"], out["test"], repeated["test"], strict=True
        ):
            assert np.isfinite(before["target"]).all()
            np.testing.assert_array_equal(after["target"], repeat["target"])
        if missing_type == "time-level":
            assert np.all(
                np.isnan(out["test"][0]["target"]) == np.isnan(out["test"][0]["target"])[0]
            )
        if missing_type == "series-level":
            assert np.all(
                np.isnan(out["test"][0]["target"]) == np.isnan(out["test"][0]["target"])[:, :1]
            )
    assert np.isnan(
        prepare_dataset_with_missing_values(dataset, 1, "random")["test"][0]["target"]
    ).all()
    assert np.isfinite(
        prepare_dataset_with_missing_values(dataset, 0, "random")["test"][0]["target"]
    ).all()
    assert prepare_dataset_with_missing_values(data([]), 0.5, "random")["test"] == []


@pytest.mark.parametrize(
    "rate,kind", [(-0.1, "random"), (1.1, "random"), (np.nan, "random"), (0.2, "invalid")]
)
def test_corruption_rejects_invalid_arguments(rate, kind):
    with pytest.raises(ValueError):
        prepare_dataset_with_missing_values(data([[1.0, 2.0]]), rate, kind)


def test_notebook_uses_tested_helpers_and_does_not_present_stale_missing_data_results():
    notebook = json.loads((Path(__file__).parents[1] / "examples/experiments.ipynb").read_text())
    assert "from transtimegrad.dataset.missing import" in "".join(notebook["cells"][2]["source"])
    assert all(not cell.get("outputs") for cell in notebook["cells"][14:])
