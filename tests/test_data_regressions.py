from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from gluonts.dataset.common import load_datasets
from gluonts.dataset.jsonl import JsonLinesWriter
from gluonts.exceptions import GluonTSDataError
from pandas.tseries.holiday import Holiday

from transtimegrad.dataset.repository._m5 import generate_pts_m5_dataset
from transtimegrad.feature import CustomDateFeatureSet, CustomHolidayFeatureSet
from transtimegrad.util import get_lags_for_frequency, lagged_sequence_values, weighted_average


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("M", "ME"),
        ("MS", "ME"),
        ("Q", "QE"),
        ("QS", "QE"),
        ("Y", "YE"),
        ("YS", "YE"),
        ("T", "min"),
        ("2Q", "2QE"),
        ("BME", "ME"),
        ("BQE", "QE"),
        ("BYE", "YE"),
    ],
)
def test_frequency_aliases_preserve_seasonal_lags(alias, canonical):
    assert get_lags_for_frequency(alias) == get_lags_for_frequency(canonical)
    assert all(lag > 0 for lag in get_lags_for_frequency(alias))


def test_lag_upper_bound_and_default_count_are_honored():
    assert get_lags_for_frequency("D", lag_ub=3) == [1, 2, 3]
    lags = get_lags_for_frequency("D", num_default_lags=2, lag_ub=10)
    assert 6 in lags and 7 in lags
    assert get_lags_for_frequency("D", num_lags=0) == []
    for kwargs in ({"lag_ub": 0}, {"num_default_lags": -1}, {"num_lags": -1}):
        with pytest.raises(ValueError):
            get_lags_for_frequency("D", **kwargs)
    with pytest.raises(ValueError):
        get_lags_for_frequency("-1D")


@pytest.mark.parametrize("dim", [0, -2, 1, None])
def test_weighted_average_axes_fractional_weights_and_gradients(dim):
    x = torch.tensor([[1.0, 10.0], [3.0, 30.0]], requires_grad=True)
    weights = torch.tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)
    result = weighted_average(x, weights, dim=dim)
    expected = (x * weights).sum(dim=dim) / weights.sum(dim=dim)
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(weights.grad).all()


def test_weighted_average_masks_missing_values_and_supports_integer_inputs():
    x = torch.tensor([float("nan"), 2.0], requires_grad=True)
    weights = torch.tensor([0.0, 0.2], requires_grad=True)
    result = weighted_average(x, weights)
    assert result.item() == 2
    result.backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(weights.grad).all()
    assert weighted_average(torch.tensor([1, 3]), torch.tensor([1, 1])).item() == 2
    assert weighted_average(x, torch.zeros(2)).item() == 0
    torch.testing.assert_close(
        weighted_average(torch.ones(2, 3), torch.ones(2, 1)), torch.tensor(1.0)
    )


def test_lagged_sequence_empty_and_zero_lag_boundary():
    prior = torch.tensor([[1.0, 2.0]])
    sequence = torch.tensor([[3.0, 4.0]])
    torch.testing.assert_close(
        lagged_sequence_values([0, 2], prior, sequence), torch.tensor([[[3.0, 1.0], [4.0, 2.0]]])
    )
    assert lagged_sequence_values([0, 2], prior, sequence[:, :0]).shape == (1, 0, 2)
    torch.testing.assert_close(
        lagged_sequence_values([0], prior, sequence, dim=-1), sequence.unsqueeze(-1)
    )
    for indices in ([], [-1], [3], [1.5]):
        with pytest.raises(ValueError):
            lagged_sequence_values(indices, prior, sequence)


def test_empty_holiday_axes_and_nonempty_values():
    dates = pd.date_range("2024-01-01", periods=3)
    np.testing.assert_array_equal(CustomDateFeatureSet([])(dates), np.zeros((1, 3)))
    assert CustomDateFeatureSet([dates[0]])(dates[:0]).shape == (1, 0)
    assert CustomHolidayFeatureSet([])(dates).shape == (0, 3)
    assert CustomHolidayFeatureSet([Holiday("New Year", month=1, day=1)])(dates[:0]).shape == (1, 0)
    np.testing.assert_array_equal(CustomDateFeatureSet([dates[1]])(dates), [[0, 1, 0]])


@pytest.fixture
def m5_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    days = 8
    calendar = pd.DataFrame(
        {
            "d": [f"d_{day + 1}" for day in range(days)],
            "date": pd.date_range("2024-01-01", periods=days),
            "wm_yr_wk": [1] * 7 + [2],
            "event_name_1": [None] * days,
            "event_name_2": [None] * days,
            "snap_CA": [1, 0, 0, 0, 0, 0, 0, 0],
            "snap_TX": [0] * days,
            "snap_WI": [0] * days,
        }
    )
    calendar.iloc[::-1].to_csv(source / "calendar.csv", index=False)
    targets = [[0] * 6 + [2, 3], [0, 0, 1, 2, 0, 3, 4, 5]]
    for split, length in (("validation", 6), ("evaluation", 8)):
        rows = []
        for item, target in enumerate(targets):
            row = {
                "id": f"item{item}_CA_1_{split}",
                "item_id": f"item{item}",
                "dept_id": "dept",
                "cat_id": "cat",
                "store_id": "CA_1",
                "state_id": "CA",
            }
            row.update({f"d_{day + 1}": value for day, value in enumerate(target[:length])})
            rows.append(row)
        pd.DataFrame(rows).to_csv(source / f"sales_train_{split}.csv", index=False)
    pd.DataFrame(
        [
            {"item_id": "item1", "store_id": "CA_1", "wm_yr_wk": 1, "sell_price": 4.0},
            {"item_id": "item1", "store_id": "CA_1", "wm_yr_wk": 2, "sell_price": 6.0},
        ]
    ).to_csv(source / "sell_prices.csv", index=False)
    return source


def generate(source, destination, writer=None):
    generate_pts_m5_dataset(
        destination, source, writer or JsonLinesWriter(use_gzip=False), prediction_length=2
    )


def test_m5_recipe_roundtrip_preserves_alignment_categories_and_all_zero_history(
    m5_source, tmp_path
):
    destination = tmp_path / "dataset"
    generate(m5_source, destination)
    dataset = load_datasets(destination, destination / "train", destination / "test")
    training, testing = list(dataset.train), list(dataset.test)
    assert len(training) == len(testing) == 2
    assert [feature.name for feature in dataset.metadata.feat_dynamic_real] == [
        "sell_price",
        "event_1",
        "event_2",
        "snap",
    ]
    for train, test in zip(training, testing, strict=True):
        assert train["item_id"] == test["item_id"]
        assert train["start"] == test["start"]
        assert len(test["target"]) == len(train["target"]) + 2
        np.testing.assert_array_equal(train["target"], test["target"][:-2])
        assert np.isfinite(train["feat_dynamic_real"]).all()
        np.testing.assert_array_equal(train["feat_static_cat"], test["feat_static_cat"])
    np.testing.assert_array_equal(training[0]["target"], np.zeros(6))
    np.testing.assert_array_equal(training[0]["feat_dynamic_real"][0], np.zeros(6))
    assert training[1]["start"] == pd.Period("2024-01-03", freq="D")
    assert testing[1]["feat_dynamic_real"][0, -1] == 6
    # Re-generation is supported for a recognizable generated dataset.
    generate(m5_source, destination)
    assert (destination / "metadata.json").exists()


def test_m5_protects_source_ancestors_and_unrelated_files(m5_source, tmp_path):
    for destination in (m5_source, tmp_path):
        with pytest.raises(ValueError, match="source CSV"):
            generate(m5_source, destination)
    destination = tmp_path / "unrelated"
    destination.mkdir()
    sentinel = destination / "important.txt"
    sentinel.write_text("preserve")
    with pytest.raises(ValueError, match="unrelated"):
        generate(m5_source, destination)
    assert sentinel.read_text() == "preserve"


def test_m5_failed_write_preserves_previous_dataset(m5_source, tmp_path):
    destination = tmp_path / "dataset"
    generate(m5_source, destination)
    original = (destination / "train/data.json").read_bytes()

    class FailingWriter(JsonLinesWriter):
        def write_to_folder(self, *args, **kwargs):
            raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        generate(m5_source, destination, FailingWriter())
    assert (destination / "train/data.json").read_bytes() == original


@pytest.mark.parametrize(
    "filename",
    ["calendar.csv", "sales_train_validation.csv", "sales_train_evaluation.csv", "sell_prices.csv"],
)
def test_m5_validates_all_required_csvs_before_writing(m5_source, tmp_path, filename):
    (m5_source / filename).unlink()
    with pytest.raises(RuntimeError, match=filename):
        generate(m5_source, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


def test_m5_keeps_a_training_window_for_late_first_sales(m5_source, tmp_path):
    for split in ("validation", "evaluation"):
        filename = m5_source / f"sales_train_{split}.csv"
        frame = pd.read_csv(filename)
        frame.loc[1, [f"d_{day}" for day in range(1, 7)]] = [0, 0, 0, 0, 0, 1]
        frame.to_csv(filename, index=False)
    destination = tmp_path / "dataset"
    generate(m5_source, destination)
    dataset = load_datasets(destination, destination / "train", destination / "test")
    assert len(list(dataset.train)[1]["target"]) == 2


def test_millisecond_frequency_is_not_silently_treated_as_month_start():
    with pytest.raises(ValueError, match="invalid frequency"):
        get_lags_for_frequency("ms")


def test_m5_allows_a_shorter_custom_forecast_horizon(m5_source, tmp_path):
    destination = tmp_path / "dataset"
    generate_pts_m5_dataset(
        destination, m5_source, JsonLinesWriter(use_gzip=False), prediction_length=1
    )
    dataset = load_datasets(destination, destination / "train", destination / "test")
    assert dataset.metadata.prediction_length == 1


def test_weighted_average_preserves_small_nonzero_weights():
    torch.testing.assert_close(
        weighted_average(torch.tensor([1.0, 3.0]), torch.tensor([1e-40, 1e-40])), torch.tensor(2.0)
    )


@pytest.mark.parametrize("location", ["train/important.txt", "test/unrelated/note.txt"])
def test_m5_preserves_unrelated_nested_files(m5_source, tmp_path, location):
    destination = tmp_path / "dataset"
    generate(m5_source, destination)
    sentinel = destination / location
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("preserve")
    with pytest.raises(ValueError, match="unrelated"):
        generate(m5_source, destination)
    assert sentinel.read_text() == "preserve"


def test_m5_rejects_symlinked_split_and_preserves_its_target(m5_source, tmp_path):
    destination = tmp_path / "dataset"
    generate(m5_source, destination)
    (destination / "train").rename(tmp_path / "original_train")
    (destination / "train").symlink_to(tmp_path / "original_train", target_is_directory=True)
    with pytest.raises(ValueError, match="regular directories"):
        generate(m5_source, destination)
    assert (tmp_path / "original_train/data.json").is_file()


def test_m5_staged_readback_catches_corrupt_writer_without_replacing_old_data(m5_source, tmp_path):
    destination = tmp_path / "dataset"
    generate(m5_source, destination)
    original = (destination / "train/data.json").read_bytes()

    class CorruptWriter(JsonLinesWriter):
        def write_to_folder(self, dataset, folder, name=None):
            (folder / "data.json").write_text("not valid json\n")

    with pytest.raises(GluonTSDataError):
        generate(m5_source, destination, CorruptWriter())
    assert (destination / "train/data.json").read_bytes() == original


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_m5_failed_install_preserves_previous_even_when_rollback_fails(
    m5_source, tmp_path, monkeypatch, rollback_fails
):
    destination = tmp_path / "dataset"
    generate(m5_source, destination)
    original = (destination / "train/data.json").read_bytes()
    rename = Path.rename

    def injected_failure(path, target):
        if "-stage-" in path.name or (rollback_fails and "-backup-" in path.name):
            raise OSError("injected rename failure")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", injected_failure)
    with pytest.raises((RuntimeError, OSError)) as caught:
        generate(m5_source, destination)
    if rollback_fails:
        recovery = list(tmp_path.glob(".dataset-backup-*"))
        assert len(recovery) == 1
        assert (recovery[0] / "train/data.json").read_bytes() == original
        assert str(recovery[0]) in str(caught.value)
    else:
        assert (destination / "train/data.json").read_bytes() == original
