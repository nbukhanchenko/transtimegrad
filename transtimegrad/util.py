# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import inspect
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from pandas.tseries.frequencies import to_offset


def _make_lags(middle: int, delta: int) -> np.ndarray:
    """
    Create a set of lags around a middle point including +/- delta.
    """
    return np.arange(middle - delta, middle + delta + 1).tolist()


# adapation of gluonts documentation
def get_lags_for_frequency(
    freq_str: str,
    lag_ub: int = 1200,
    num_lags: Optional[int] = None,
    num_default_lags: int = 7,
) -> List[int]:
    """
    Generates a list of lags that that are appropriate for the given frequency
    string.

    By default all frequencies have the following lags: [1, 2, 3, 4, 5, 6, 7].
    Remaining lags correspond to the same `season` (+/- `delta`) in previous
    `k` cycles. Here `delta` and `k` are chosen according to the existing code.

    Parameters
    ----------

    freq_str
        Frequency string of the form [multiple][granularity] such as "12H",
        "5min", "1D" etc.

    lag_ub
        The maximum value for a lag.

    num_lags
        Maximum number of lags; by default all generated lags are returned.

    num_default_lags
        The number of default lags; by default it is 7.
    """

    # Lags are target values at the same `season` (+/- delta) but in the
    # previous cycle.
    def _make_lags_for_second(multiple, num_cycles=3):
        # We use previous ``num_cycles`` hours to generate lags
        return [_make_lags(k * 60 // multiple, 2) for k in range(1, num_cycles + 1)]

    def _make_lags_for_minute(multiple, num_cycles=3):
        # We use previous ``num_cycles`` hours to generate lags
        return [_make_lags(k * 60 // multiple, 2) for k in range(1, num_cycles + 1)]

    def _make_lags_for_hour(multiple, num_cycles=7):
        # We use previous ``num_cycles`` days to generate lags
        return [_make_lags(k * 24 // multiple, 1) for k in range(1, num_cycles + 1)]

    def _make_lags_for_day(multiple, num_cycles=4, days_in_week=7, days_in_month=30):
        # We use previous ``num_cycles`` weeks to generate lags
        # We use the last month (in addition to 4 weeks) to generate lag.
        return [_make_lags(k * days_in_week // multiple, 1) for k in range(1, num_cycles + 1)] + [
            _make_lags(days_in_month // multiple, 1)
        ]

    def _make_lags_for_week(multiple, num_cycles=3):
        # We use previous ``num_cycles`` years to generate lags
        # Additionally, we use previous 4, 8, 12 weeks
        return [_make_lags(k * 52 // multiple, 1) for k in range(1, num_cycles + 1)] + [
            [4 // multiple, 8 // multiple, 12 // multiple]
        ]

    def _make_lags_for_month(multiple, num_cycles=3):
        # We use previous ``num_cycles`` years to generate lags
        return [_make_lags(k * 12 // multiple, 1) for k in range(1, num_cycles + 1)]

    if lag_ub < 1 or num_default_lags < 0 or (num_lags is not None and num_lags < 0):
        raise ValueError("lag_ub must be positive and lag counts must be nonnegative")
    offset = to_offset(freq_str)
    if offset.n <= 0:
        raise ValueError("frequency must have a positive multiple")
    # pandas 2.2 renamed end offsets; start/end variants have the same season.
    offset_name = offset.name.split("-")[0]
    offset_name = {
        "M": "m",
        "ME": "m",
        "MS": "m",
        "BM": "m",
        "BME": "m",
        "BMS": "m",
        "Q": "q",
        "QE": "q",
        "QS": "q",
        "BQ": "q",
        "BQE": "q",
        "BQS": "q",
        "A": "a",
        "Y": "a",
        "YE": "a",
        "YS": "a",
        "AS": "a",
        "BA": "a",
        "BYE": "a",
        "BYS": "a",
        "min": "t",
        "T": "t",
    }.get(offset_name, offset_name.lower())

    if offset_name == "a":
        lags = []
    elif offset_name == "q":
        lags = _make_lags_for_month(offset.n * 3.0)
    elif offset_name == "m":
        lags = _make_lags_for_month(offset.n)
    elif offset_name == "w":
        lags = _make_lags_for_week(offset.n)
    elif offset_name == "d":
        lags = _make_lags_for_day(offset.n) + _make_lags_for_week(offset.n / 7.0)
    elif offset_name == "b":
        lags = _make_lags_for_day(offset.n, days_in_week=5, days_in_month=22) + _make_lags_for_week(
            offset.n / 5.0
        )
    elif offset_name == "h":
        lags = (
            _make_lags_for_hour(offset.n)
            + _make_lags_for_day(offset.n / 24)
            + _make_lags_for_week(offset.n / (24 * 7))
        )
    # minutes
    elif offset_name == "t":
        lags = (
            _make_lags_for_minute(offset.n)
            + _make_lags_for_hour(offset.n / 60)
            + _make_lags_for_day(offset.n / (60 * 24))
            + _make_lags_for_week(offset.n / (60 * 24 * 7))
        )
    # second
    elif offset_name == "s":
        lags = (
            _make_lags_for_second(offset.n)
            + _make_lags_for_minute(offset.n / 60)
            + _make_lags_for_hour(offset.n / (60 * 60))
        )
    else:
        raise ValueError(
            f"invalid frequency | `freq_str={freq_str}` -> `offset_name={offset_name}`"
        )

    # flatten lags list and filter
    seasonal = [int(lag) for group in lags for lag in group]
    lags = sorted(
        {lag for lag in list(range(1, num_default_lags + 1)) + seasonal if 0 < lag <= lag_ub}
    )
    return lags[:num_lags]


def lagged_sequence_values(
    indices: List[int],
    prior_sequence: torch.Tensor,
    sequence: torch.Tensor,
    dim: int = 1,
    keepdim: bool = False,
) -> torch.Tensor:
    """
    Constructs an array of lagged values from a given sequence.

    Parameters
    ----------
    indices
        Indices of the lagged observations. For example, ``[0]`` indicates
        that, at any time ``t``, the model will have only the observation from
        time ``t`` itself; instead, ``[0, 24]`` indicates that the output
        will have observations from times ``t`` and ``t-24``.
    prior_sequence
        Tensor containing the input sequence prior to the time range for
        which the output is required.
    sequence
        Tensor containing the input sequence in the time range where the
        output is required.
    dim
        Time dimension.
    keepdim
        Whether to keep the last dimension of the output tensor.

    Returns
    -------
    Tensor
        A tensor of shape (*sequence.shape, len(indices)).
    """
    if not indices or any(
        isinstance(index, bool) or not isinstance(index, (int, np.integer)) or index < 0
        for index in indices
    ):
        raise ValueError("indices must be a nonempty list of nonnegative integers")
    if prior_sequence.ndim != sequence.ndim or sequence.ndim < 2:
        raise ValueError("prior_sequence and sequence must have matching ranks of at least two")
    if not -sequence.ndim <= dim < sequence.ndim:
        raise ValueError("dim is outside the sequence rank")
    dim = dim % sequence.ndim
    if max(indices) > prior_sequence.shape[dim]:
        raise ValueError("lags cannot extend beyond the prior sequence")
    if sequence.ndim == 2:
        prior_sequence = prior_sequence.unsqueeze(-1)
        sequence = sequence.unsqueeze(-1)

    full_sequence = torch.cat((prior_sequence, sequence), dim=dim)
    lags_values = []
    for lag_index in indices:
        end_index = full_sequence.shape[dim] - lag_index
        begin_index = end_index - sequence.shape[dim]
        selection = [slice(None)] * full_sequence.ndim
        selection[dim] = slice(begin_index, end_index)
        lags_values.append(full_sequence[tuple(selection)].unsqueeze(-1))

    lags_values = torch.cat(lags_values, dim=-1)

    if not keepdim:
        # merge the last two dimensions
        lags_values = lags_values.reshape(
            *lags_values.shape[:-2], lags_values.shape[-2] * lags_values.shape[-1]
        )

    return lags_values


def weighted_average(
    x: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None
) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given dim, masking
    values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Parameters
    ----------
    x
        Input tensor, of which the average must be computed.
    weights
        Weights tensor, of the same shape as `x`.
    dim
        The dim along which to average `x`

    Returns
    -------
    Tensor:
        The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        weights = torch.broadcast_to(weights, x.shape)
        # Mask before multiplying so missing values cannot poison gradients.
        values = torch.where(weights != 0, x, torch.zeros_like(x))
        weighted_tensor = values * weights
        if not weighted_tensor.is_floating_point():
            weighted_tensor = weighted_tensor.to(torch.get_default_dtype())
        sum_weights = weights.sum(dim=dim)
        denominator = torch.where(sum_weights != 0, sum_weights, torch.ones_like(sum_weights))
        return weighted_tensor.sum(dim=dim) / denominator
    else:
        return x.mean(dim=dim)


def get_module_forward_input_names(module: nn.Module):
    params = inspect.signature(module.forward).parameters
    param_names = [k for k, v in params.items() if not str(v).startswith("*")]
    return param_names
