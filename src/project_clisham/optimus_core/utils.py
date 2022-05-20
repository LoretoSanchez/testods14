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

"""This module provides a set of helper functions being used across different components
of optimus package.
"""
import datetime as dt
import importlib
import logging
import uuid
from functools import partial, reduce, update_wrapper
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.
    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path.
    Returns:
        Extracted object.
    Raises:
        AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            "Object `{}` cannot be loaded from `{}`.".format(obj_name, obj_path)
        )
    return getattr(module_obj, obj_name)


def partial_wrapper(func: Callable, *args, **kwargs) -> Callable:
    """Enables user to pass in arguments that are not datasets when function is called
    in a Kedro pipeline e.g. a string or int value.
    Args:
        func: Callable node function
     Returns:
        Callable
    """
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def generate_uuid(data: pd.DataFrame, col_name: str = "uuid") -> pd.DataFrame:
    """Extract the parameters saved in conf
    Args:
        data: original DataFrame
        col_name: name for column for UUID
    Returns:
        DataFrame with UUID added
    Raises:
        AttributeError: When the param does not exist
    """
    columns = data.columns
    data[col_name] = [str(uuid.uuid4()) for _ in range(len(data.index))]
    data = data[[col_name, *columns]]
    return data


def norm_columns_name(
    df: pd.DataFrame, symbols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Normalize the name of the columns.
    The main purpose is to lowercase and replace any "problematic" char with an underscore.
    It can accept a specific list of symbols,  by default it will replace [' ',':','-','.']

    Args:
        df: Pandas dataframe
        symbols: a List containing the symbols to be replaced.
    Returns:
        pd.DataFrame with the corrected column name
    """

    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.strip()
    if symbols is None:
        symbol_list = [" ", ":", "-", ".", ",", "(", ")"]
    else:
        symbol_list = symbols
    # Replace the symbols
    for sym in symbol_list:
        df.columns = df.columns.str.replace(sym, "_")

    return df


def norm_columns_rows_name(
    df: pd.DataFrame, symbols: Optional[List[str]] = None, col_name: str = None
) -> pd.DataFrame:
    """Normalize the name of the columns.
    The main purpose is to lowercase and replace any "problematic" char with an underscore.
    It can accept a specific list of symbols,  by default it will replace [' ',':','-','.']

    Args:
        df: Pandas dataframe
        symbols: a List containing the symbols to be replaced.
    Returns:
        pd.DataFrame with the corrected column name
    """
    df[col_name] = df[col_name].str.lower()
    df[col_name] = df[col_name].str.strip()
    if symbols is None:
        symbol_list = [" ", ":", "-", ".", ",", "(", ")"]
    else:
        symbol_list = symbols
    # Replace the symbols
    for sym in symbol_list:
        df[col_name] = df[col_name].str.replace(sym, "_")
    df = norm_columns_name(df)

    return df


def cut_values_from_dict(data: pd.DataFrame, cut_dict: dict) -> pd.DataFrame:
    """Replace values outside a given range with np.nan.

    It takes a dictionary of variables and their allowed ranges and replaces any values outside
    that range with np.nan.

    Args:
        data: DataFrame for which outliers will be removed.
        cut_dict: Dictionary of columns and ranges allowed for each variable (e.g. {'var1': [0, 1], 'var2': [-1, 1]}).

    Returns:
        df: DataFrame with values outside range for the specified columns converted to np.nan.

    """

    for col in cut_dict:
        data.loc[
            (data[col] < cut_dict[col][0]) | (data[col] > cut_dict[col][1]), col
        ] = np.nan

    return data


def cut_values_from_list(
    data: pd.DataFrame, tag_list: List[str], interval: List[float]
) -> pd.DataFrame:
    """Replace values outside a given range with np.nan.

    It takes a list of variables and a unique allowed range, replacing any values outside
    that range with np.nan, for all variables.

    Args:
        data: DataFrame for which outliers will be removed.
        tag_list: List of tags.
        interval: Interval (closed) of allowed values (e.g. [0, 1]).

    Returns:
        df: DataFrame with values outside range for the specified columns converted to np.nan.

    """

    data[tag_list] = data[tag_list].where(
        (data[tag_list] >= interval[0]) & (data[tag_list] <= interval[1])
    )

    return data


def merge_tables_on_timestamp(
    parameters: dict, df_list: List[pd.DataFrame]
) -> pd.DataFrame:
    """Left-merge all DataFrames in the list, based on 1-1 indices for each.

    Args:
        parameters: Dictionary of parameters.
        df_list: List of DataFrames to be merged.

    Returns:
        merged: DataFrame of merged sources.

    """

    merged = reduce(
        partial(_left_merge_on, on=parameters["timestamp_col_name"]), df_list
    )

    return merged


def _left_merge_on(df_1: pd.DataFrame, df_2: pd.DataFrame, on: str) -> pd.DataFrame:
    """Left-merge two DataFrames based on specified column.

    Args:
        df_1: DataFrame 1 to be merged.
        df_2: DataFrame 2 to be merged.
        on: Column to be merged on.

    Returns:
        df: Merged DataFrame.

    """
    # Merge on indices making sure there is 1-1 correspondence
    df = pd.merge(
        df_1, df_2, how="left", left_on=on, right_on=on, validate="one_to_one"
    )

    return df


def filter_date_range(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Filter data based on a time range.
    Args:
        df: DataFrame containing the data to be filtered, must include a column with the
        name explicited on the Parameters
        params: Parameters arguments with the values of start and end date, along with the column
    Returns:
        DataFrame with the applied filter

    """

    logger = logging.getLogger(__name__)
    try:
        months = params["months"]
        col_name = params["col_name"]
    except KeyError:
        logger.error("Empty start date or col_name in params")
        raise
    # end date is an optional
    end_date = None
    if "end_date" in params:
        end_date = params["end_date"]
    start_date = dt.datetime.now() - dt.timedelta(days=months * 30)

    logger.info(f"Filtering the range from {start_date} to {end_date}")
    # Filter the data
    logger.info(f"Dataframe shape pre-filter: {df.shape}")
    df = df.query(f'{col_name} >= "{start_date}" ').copy()
    # Filter data if end_date is present
    if end_date is not None:
        df = df.query(f'{col_name} <= "{end_date}"').copy()
    logger.info(f"Dataframe shape after filter: {df.shape}")
    return df
