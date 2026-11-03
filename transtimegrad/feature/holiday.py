from typing import Callable, List

import numpy as np
import pandas as pd
from gluonts.time_feature.holiday import distance_to_holiday, indicator
from pandas.tseries.holiday import Holiday


class CustomDateFeatureSet:
    """
    Implements calculation of date features. The CustomDateFeatureSet is
    applied on a pandas Series with Datetimeindex and returns a 1D array of
    the shape (1, len(date_indices)).

    Note that for lower than daily granularity the distance to the holiday is
    still computed on a per-day basis.

    Example use:

        >>> import pandas as pd
        >>> cfs = CustomDateFeatureSet([pd.to_datetime('20191129', format='%Y%m%d'),
        ...                             pd.to_datetime('20200101', format='%Y%m%d')])
        >>> date_indices = pd.date_range(
        ...     start="2019-11-24",
        ...     end="2019-12-31",
        ...     freq='D'
        ... )
        >>> cfs(date_indices)
        array([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0.]])

    Example use for using a squared exponential kernel:

        >>> kernel = squared_exponential_kernel(alpha=0.5)
        >>> cfs = CustomDateFeatureSet([pd.to_datetime('20191129', format='%Y%m%d'),
        ...                             pd.to_datetime('20200101', format='%Y%m%d')], kernel)
        >>> cfs(date_indices)
        array([[3.72665317e-06, 3.35462628e-04, 1.11089965e-02, 1.35335283e-01,
            6.06530660e-01, 1.00000000e+00, 6.06530660e-01, 1.35335283e-01,
            1.11089965e-02, 3.35462628e-04, 3.72665317e-06, 1.52299797e-08,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            1.52299797e-08, 3.72665317e-06, 3.35462628e-04, 1.11089965e-02,
            1.35335283e-01, 6.06530660e-01]])
    """

    def __init__(
        self,
        reference_dates: List[pd.Timestamp],
        kernel_function: Callable[[int], int] = indicator,
    ):
        """
        Parameters
        ----------
        reference_dates
            list of panda timestamps for which features should be created.
        kernel_function
            kernel function to pass the feature value based
            on distance in days. Can be indicator function (default),
            exponential_kernel, squared_exponential_kernel or user defined.
        """
        self.reference_dates = reference_dates
        self.kernel_function = kernel_function

    def __call__(self, dates):
        """
        Transform a pandas series with timestamps to holiday features.

        Parameters
        ----------
        dates
            Pandas series with Datetimeindex timestamps.
        """
        values = np.zeros((1, len(dates)), dtype=float)
        for ref_date in self.reference_dates:
            values[0] += [self.kernel_function((index - ref_date).days) for index in dates]
        return values


class CustomHolidayFeatureSet:
    """
    Implements calculation of holiday features. The CustomHolidayFeatureSet is
    applied on a pandas Series with Datetimeindex and returns a 2D array of
    the shape (len(dates), num_features), where num_features are the number
    of holidays.

    Note that for lower than daily granularity the distance to the holiday is
    still computed on a per-day basis.

    Example use:

        >>> from pts.features import (
        ...    squared_exponential_kernel,
        ...    SpecialDateFeatureSet,
        ...    CHRISTMAS_DAY,
        ...    CHRISTMAS_EVE
        ... )
        >>> import pandas as pd
        >>> from pandas.tseries.holiday import Holiday
        >>> cfs = CustomHolidayFeatureSet([
            Holiday("New Years Day", month=1, day=1), Holiday("Christmas Day", month=12, day=25)
        ])
        >>> date_indices = pd.date_range(
        ...     start="2016-12-24",
        ...     end="2016-12-31",
        ...     freq='D'
        ... )
        >>> cfs(date_indices)
        array([[1., 0., 0., 0., 0., 0., 0., 0.],
               [0., 1., 0., 0., 0., 0., 0., 0.]])

    Example use for using a squared exponential kernel:

        >>> kernel = squared_exponential_kernel(alpha=1.0)
        >>> sfs = SpecialDateFeatureSet([
            Holiday("New Years Day", month=1, day=1), Holiday("Christmas Day", month=12, day=25)
        ], kernel)
        >>> sfs(date_indices)
        array([[1.00000000e+00, 3.67879441e-01, 1.83156389e-02, 1.23409804e-04,
                1.12535175e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
               [3.67879441e-01, 1.00000000e+00, 3.67879441e-01, 1.83156389e-02,
                1.23409804e-04, 1.12535175e-07, 0.00000000e+00, 0.00000000e+00]])

    """

    def __init__(
        self,
        custom_holidays: List[Holiday],
        kernel_function: Callable[[int], int] = indicator,
    ):
        """
        Parameters
        ----------
        feature_names
            list of strings with holiday names for which features should be created.
        kernel_function
            kernel function to pass the feature value based
            on distance in days. Can be indicator function (default),
            exponential_kernel, squared_exponential_kernel or user defined.
        """
        self.custom_holidays = custom_holidays
        self.kernel_function = kernel_function

    def __call__(self, dates):
        """
        Transform a pandas series with timestamps to holiday features.

        Parameters
        ----------
        dates
            Pandas series with Datetimeindex timestamps.
        """
        values = np.zeros((len(self.custom_holidays), len(dates)), dtype=float)
        for row, holiday in enumerate(self.custom_holidays):
            distance = distance_to_holiday(holiday)
            values[row] = [self.kernel_function(distance(index)) for index in dates]
        return values
