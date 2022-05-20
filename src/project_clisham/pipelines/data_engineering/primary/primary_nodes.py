# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organisation
# and QuantumBlack, and any unauthorised use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organisation with the prior written
# permission of QuantumBlack.
"""
Grid creation nodes.
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _create_grid_from_series(
    time_series: pd.Series,
    freq: str,
    offset_start: str = None,
    offset_end: str = None,
    timezone: str = None,
) -> pd.DatetimeIndex:
    """
    Create a time grid from the first to the last timestamp in a time series
    with optional offsets at both ends. See
    https://pandas.pydata.org/pandas-docs/version/0.25.0/
    user_guide/timeseries.html#timeseries-offset-aliases
    for valid frequency strings.
    Resulting time stamps are rounded in order to get merge-able grids
    for different data sources.
    Args:
        time_series: series of timestamps
        freq: pandas frequency string
        offset_start: time to add to first timestamp
        offset_end: time to subtract from last timestamp
        timezone: timezone information
    Returns:
        time grid
    """
    time_series = pd.to_datetime(time_series)
    lhs, rhs = time_series.agg(["min", "max"])
    if offset_start:
        lhs += pd.to_timedelta(offset_start)
    if offset_end:
        rhs -= pd.to_timedelta(offset_end)

    return pd.date_range(lhs.ceil(freq), rhs.floor(freq), freq=freq, tz=timezone)


def create_time_grid(params: dict, *dfs: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Creates a time grid from a data frame.
    Args:
        params: dictionary of parameters
        dfs: any number of dataframes with time series column
    """
    freq = params["grid"]["frequency"]
    offset_start = params["grid"].get("offset_start")
    offset_end = params["grid"].get("offset_end")
    timezone = params["pipeline_timezone"]
    datetime_col = params["datetime_col"]

    all_timestamps = pd.concat([df[datetime_col] for df in dfs], ignore_index=True)

    return _create_grid_from_series(
        all_timestamps,
        freq,
        offset_start,
        offset_end,
        timezone,
    )


def merge_to_grid(
    params: dict, grid: pd.DatetimeIndex, *to_merge: pd.DataFrame
) -> pd.DataFrame:
    """
    Left-merges any number of dataframes to the grid.
    Ensures that the size of the resulting df remains unchanged.
    Args:
        grid: time points at which to aggregate
        to_merge: any number of dataframes to merge in
    Returns:
        merged df
    """
    n_rows = len(grid)
    datetime_col = params["datetime_col"]

    merged = pd.DataFrame({datetime_col: grid})

    for i, df in enumerate(to_merge):
        merged = pd.merge(
            merged, df, how="left", left_on=datetime_col, right_on=datetime_col
        )
        if len(merged) != n_rows:
            raise RuntimeError(
                (
                    "Merging dataframe {} led to a change in the number of rows "
                    "from {} to {}. Please check for duplicate timestamps."
                ).format(i + 1, n_rows, len(merged))
            )

    return merged
