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

from gluonts.time_feature import norm_freq_str
from gluonts.torch.util import slice_along_dim
import numpy as np
from pandas.tseries.frequencies import to_offset
import torch
import torch.nn as nn


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
        return [
            _make_lags(k * 60 // multiple, 2) for k in range(1, num_cycles + 1)
        ]

    def _make_lags_for_minute(multiple, num_cycles=3):
        # We use previous ``num_cycles`` hours to generate lags
        return [
            _make_lags(k * 60 // multiple, 2) for k in range(1, num_cycles + 1)
        ]

    def _make_lags_for_hour(multiple, num_cycles=7):
        # We use previous ``num_cycles`` days to generate lags
        return [
            _make_lags(k * 24 // multiple, 1) for k in range(1, num_cycles + 1)
        ]

    def _make_lags_for_day(
        multiple, num_cycles=4, days_in_week=7, days_in_month=30
    ):
        # We use previous ``num_cycles`` weeks to generate lags
        # We use the last month (in addition to 4 weeks) to generate lag.
        return [
            _make_lags(k * days_in_week // multiple, 1)
            for k in range(1, num_cycles + 1)
        ] + [_make_lags(days_in_month // multiple, 1)]

    def _make_lags_for_week(multiple, num_cycles=3):
        # We use previous ``num_cycles`` years to generate lags
        # Additionally, we use previous 4, 8, 12 weeks
        return [
            _make_lags(k * 52 // multiple, 1) for k in range(1, num_cycles + 1)
        ] + [[4 // multiple, 8 // multiple, 12 // multiple]]

    def _make_lags_for_month(multiple, num_cycles=3):
        # We use previous ``num_cycles`` years to generate lags
        return [
            _make_lags(k * 12 // multiple, 1) for k in range(1, num_cycles + 1)
        ]

    # multiple, granularity = get_granularity(freq_str)
    offset = to_offset(freq_str)
    # normalize offset name, so that both `W` and `W-SUN` refer to `W`
    offset_name = norm_freq_str(offset.name).lower()

    if offset_name == "a":
        lags = []
    elif offset_name == "q":
        assert (
            offset.n == 1
        ), "Only multiple 1 is supported for quarterly. Use x month instead."
        lags = _make_lags_for_month(offset.n * 3.0)
    elif offset_name == "m":
        lags = _make_lags_for_month(offset.n)
    elif offset_name == "w":
        lags = _make_lags_for_week(offset.n)
    elif offset_name == "d":
        lags = _make_lags_for_day(offset.n) + _make_lags_for_week(
            offset.n / 7.0
        )
    elif offset_name == "b":
        lags = _make_lags_for_day(
            offset.n, days_in_week=5, days_in_month=22
        ) + _make_lags_for_week(offset.n / 5.0)
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
        raise ValueError(f"invalid frequency | `freq_str={freq_str}` -> `offset_name={offset_name}`")

    # flatten lags list and filter
    lags = [
        int(lag) for sub_list in lags for lag in sub_list if 7 < lag <= lag_ub
    ]
    lags = list(range(1, num_default_lags + 1)) + sorted(list(set(lags)))

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
    assert max(indices) <= prior_sequence.shape[dim], (
        f"lags cannot go further than prior sequence length, found lag"
        f" {max(indices)} while prior sequence is only"
        f"{prior_sequence.shape[dim]}-long"
    )

    # if prior_sequence is a 2-tensor add an extra dimension
    if len(prior_sequence.shape) == 2:
        prior_sequence = prior_sequence.unsqueeze(-1)
    if len(sequence.shape) == 2:
        sequence = sequence.unsqueeze(-1)

    full_sequence = torch.cat((prior_sequence, sequence), dim=dim)

    lags_values = []
    for lag_index in indices:
        begin_index = -lag_index - sequence.shape[dim]
        end_index = -lag_index if lag_index > 0 else None
        lags_values.append(
            slice_along_dim(
                full_sequence, dim=dim, slice_=slice(begin_index, end_index)
            ).unsqueeze(-1)
        )

    lags_values = torch.cat(lags_values, dim=-1)

    if not keepdim:
        # merge the last two dimensions
        lags_values = lags_values.reshape(*lags_values.shape[:-2], -1)

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
        weighted_tensor = torch.where(weights != 0, x * weights, torch.zeros_like(x))
        sum_weights = torch.clamp(
            weights.sum(dim=dim) if dim else weights.sum(), min=1.0
        )
        return (
            weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()
        ) / sum_weights
    else:
        return x.mean(dim=dim)


def get_module_forward_input_names(module: nn.Module):
    params = inspect.signature(module.forward).parameters
    param_names = [k for k, v in params.items() if not str(v).startswith("*")]
    return param_names
