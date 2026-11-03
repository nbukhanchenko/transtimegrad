"""Build the M5 validation/evaluation split from the user's local Kaggle CSVs."""

from pathlib import Path

import numpy as np
import pandas as pd
from gluonts.dataset import DatasetWriter
from gluonts.dataset.common import (
    BasicFeatureInfo,
    CategoricalFeatureInfo,
    MetaData,
    TrainDatasets,
    load_datasets,
)
from gluonts.time_feature.holiday import squared_exponential_kernel

from ...feature import CustomDateFeatureSet
from ...storage import atomic_directory_write

_ID_FIELDS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
_STATIC_FIELDS = ["state_id", "store_id", "cat_id", "dept_id", "item_id"]
_DYNAMIC_FIELDS = ["sell_price", "event_1", "event_2", "snap"]


def _day_columns(sales):
    columns = [column for column in sales if column.startswith("d_") and column[2:].isdigit()]
    columns.sort(key=lambda column: int(column[2:]))
    if not columns or [int(column[2:]) for column in columns] != list(range(1, len(columns) + 1)):
        raise ValueError("M5 sales must contain consecutive day columns starting at d_1")
    return columns


def _validate_saved_dataset(path, expected_metadata=None, expected_count=None):
    """Verify the manifest and stream every stored record before replacement."""
    if path.is_symlink() or not path.is_dir():
        raise ValueError("dataset destination must be a regular directory")
    if {entry.name for entry in path.iterdir()} != {"metadata.json", "train", "test"}:
        raise ValueError("refusing to overwrite a directory containing unrelated files")
    metadata = path / "metadata.json"
    if metadata.is_symlink() or not metadata.is_file():
        raise ValueError("dataset metadata must be a regular file")
    for name in ("train", "test"):
        directory = path / name
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("dataset splits must be regular directories")
        files = list(directory.iterdir())
        if (
            len(files) != 1
            or files[0].name
            not in {"data.json", "data.json.gz", "data.arrow", "data.feather", "data.parquet"}
            or files[0].is_symlink()
            or not files[0].is_file()
        ):
            raise ValueError("dataset split contains unrelated files or symlinks")
    loaded = load_datasets(path, path / "train", path / "test")
    if expected_metadata is not None and loaded.metadata != expected_metadata:
        raise ValueError("saved metadata differs from the generated dataset")
    identifiers = []
    for split in (loaded.train, loaded.test):
        seen = set()
        for entry in split:
            identity = entry.get("item_id")
            target = entry["target"]
            dynamic = entry.get("feat_dynamic_real")
            static = entry.get("feat_static_cat")
            if identity is None or identity in seen:
                raise ValueError("saved dataset has missing or duplicate series identifiers")
            if target.ndim != 1 or not target.size or not np.isfinite(target).all():
                raise ValueError("saved dataset contains invalid target observations")
            if (
                dynamic is None
                or dynamic.shape != (4, len(target))
                or not np.isfinite(dynamic).all()
            ):
                raise ValueError("saved dataset has invalid dynamic features")
            if (
                static is None
                or len(static) != len(loaded.metadata.feat_static_cat)
                or any(
                    value < 0 or value >= int(info.cardinality)
                    for value, info in zip(static, loaded.metadata.feat_static_cat, strict=True)
                )
            ):
                raise ValueError("saved dataset has invalid categorical features")
            seen.add(identity)
        if not seen or (expected_count is not None and len(seen) != expected_count):
            raise ValueError("saved dataset contains an unexpected number of series")
        identifiers.append(seen)
    if identifiers[0] != identifiers[1]:
        raise ValueError("saved training/test series identities differ")


def _save_dataset(dataset, destination, writer, expected_count):
    """Publish a fully read-back dataset using the shared recoverable transaction."""

    def write(stage):
        # GluonTS creates its output root itself; stage is our new empty directory.
        stage.rmdir()
        dataset.save(str(stage), writer=writer, overwrite=False)

    atomic_directory_write(
        destination,
        write,
        _validate_saved_dataset,
        lambda path: _validate_saved_dataset(path, dataset.metadata, expected_count),
    )


def generate_pts_m5_dataset(
    dataset_path: Path,
    m5_file_path: Path,
    dataset_writer: DatasetWriter,
    pandas_freq: str = "D",
    prediction_length: int = 28,
    alpha: float = 0.5,
):
    """Generate aligned train/test series without deleting source or unrelated data.

    Leading zero sales are trimmed using the training split only; all-zero series
    retain their complete history. Missing historical prices are forward filled,
    with zero before the first known price, so future prices do not leak backwards.
    """
    source = Path(m5_file_path).resolve()
    destination = Path(dataset_path).absolute()
    if source == destination.resolve() or destination.resolve() in source.parents:
        raise ValueError("dataset destination must not contain the source CSV directory")
    filenames = [
        "calendar.csv",
        "sales_train_validation.csv",
        "sales_train_evaluation.csv",
        "sell_prices.csv",
    ]
    missing = [name for name in filenames if not (source / name).is_file()]
    if missing:
        raise RuntimeError(
            "M5 data must be downloaded from Kaggle after accepting its terms; "
            f"missing local files at {source}: {', '.join(missing)}"
        )
    if prediction_length < 1 or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("prediction_length must be positive and alpha must be nonnegative")
    if pd.tseries.frequencies.to_offset(pandas_freq) != pd.tseries.offsets.Day():
        raise ValueError("M5 observations have daily frequency")

    calendar = pd.read_csv(source / filenames[0])
    calendar["date"] = pd.to_datetime(calendar["date"])
    if calendar["d"].duplicated().any():
        raise ValueError("calendar has duplicate day identifiers")
    calendar = calendar.set_index("d")
    train = pd.read_csv(source / filenames[1]).set_index(_ID_FIELDS).sort_index()
    test = pd.read_csv(source / filenames[2]).set_index(_ID_FIELDS).sort_index()
    if train.empty or train.index.to_frame(index=False).isna().any().any():
        raise ValueError("M5 training data must contain nonempty, identified series")
    if (
        train.index.has_duplicates
        or test.index.has_duplicates
        or not train.index.equals(test.index)
    ):
        raise ValueError("M5 training and evaluation series identities must match uniquely")
    train_days, test_days = _day_columns(train), _day_columns(test)
    if len(train_days) < prediction_length:
        raise ValueError("training data is shorter than prediction_length")
    if (
        test_days[: len(train_days)] != train_days
        or len(test_days) - len(train_days) < prediction_length
    ):
        raise ValueError(
            "evaluation data must extend training data by at least prediction_length days"
        )
    if not set(test_days) <= set(calendar.index):
        raise ValueError("calendar does not cover the sales observations")
    calendar = calendar.loc[test_days]
    if (
        not calendar["date"]
        .reset_index(drop=True)
        .equals(pd.Series(pd.date_range(calendar["date"].iloc[0], periods=len(calendar), freq="D")))
    ):
        raise ValueError("calendar sales dates must be consecutive daily observations")
    if not np.array_equal(train[train_days].to_numpy(), test[train_days].to_numpy()):
        raise ValueError("evaluation history differs from training history")
    for sales, days in ((train, train_days), (test, test_days)):
        values = sales[days].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("M5 sales must be finite and nonnegative")

    prices = pd.read_csv(source / filenames[3])
    if prices.duplicated(["item_id", "store_id", "wm_yr_wk"]).any():
        raise ValueError("sell_prices has duplicate item/store/week entries")
    if not np.isfinite(prices["sell_price"]).all() or (prices["sell_price"] < 0).any():
        raise ValueError("sell prices must be finite and nonnegative")
    price_groups = {
        key: group.set_index("wm_yr_wk")["sell_price"]
        for key, group in prices.groupby(["item_id", "store_id"])
    }
    kernel = squared_exponential_kernel(alpha=alpha)
    event_features = [
        CustomDateFeatureSet(calendar.loc[calendar[name].notna(), "date"], kernel)(calendar["date"])
        for name in ("event_name_1", "event_name_2")
    ]
    snap_features = {
        state: CustomDateFeatureSet(calendar.loc[calendar[f"snap_{state}"] == 1, "date"], kernel)(
            calendar["date"]
        )
        for state in ("CA", "TX", "WI")
    }
    categories = {
        name: {
            value: code
            for code, value in enumerate(sorted(train.index.get_level_values(name).unique()))
        }
        for name in _STATIC_FIELDS
    }
    meta = MetaData(
        freq=pandas_freq,
        prediction_length=prediction_length,
        feat_static_cat=[
            CategoricalFeatureInfo(name=name, cardinality=str(len(categories[name])))
            for name in _STATIC_FIELDS
        ],
        feat_dynamic_real=[BasicFeatureInfo(name=name) for name in _DYNAMIC_FIELDS],
    )

    def entries(sales, days):
        # Stream entries: materializing both full M5 splits as Python lists costs
        # many gigabytes beyond the input tables themselves.
        for identity, row in sales.iterrows():
            fields = dict(zip(_ID_FIELDS, identity, strict=True))
            training_values = train.loc[identity, train_days].to_numpy(dtype=np.float32)
            nonzero = np.flatnonzero(training_values)
            start = (
                min(int(nonzero[0]), max(0, len(train_days) - prediction_length))
                if len(nonzero)
                else 0
            )
            state = fields["state_id"]
            if state not in snap_features:
                raise ValueError(f"unknown M5 state: {state}")
            price = price_groups.get(
                (fields["item_id"], fields["store_id"]), pd.Series(dtype=float)
            )
            price = calendar["wm_yr_wk"].map(price).ffill().fillna(0).to_numpy(dtype=np.float32)
            dynamic = np.concatenate((price[None], *event_features, snap_features[state]), axis=0)
            static = [categories[name][fields[name]] for name in _STATIC_FIELDS]
            yield {
                "start": str(calendar["date"].iloc[start]),
                "item_id": f"{fields['item_id']}_{fields['store_id']}",
                "feat_static_cat": static,
                "target": row[days].to_numpy(dtype=np.float32)[start:].tolist(),
                "feat_dynamic_real": dynamic[:, start : len(days)].astype(np.float32).tolist(),
            }

    dataset = TrainDatasets(
        metadata=meta, train=entries(train, train_days), test=entries(test, test_days)
    )
    _save_dataset(dataset, destination, dataset_writer, expected_count=len(train))
