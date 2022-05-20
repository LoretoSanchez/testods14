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
Nodes of the data splitting pipeline.
"""
import datetime
import logging
import numbers
from typing import Dict

import pandas as pd


logger = logging.getLogger(__name__)


def split_data_by_date(params: dict, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Args:
        params: dictionary of parameters
        data: input data
    Returns:
        dict(train=train_data, test=opt_data)
    """
    data = data.copy()
    split_datetime_val = params.get("datetime_val")
    split_datetime_col = params.get("datetime_col")
    if split_datetime_col not in data.columns:
        raise KeyError(
            f"Provided split_datetime_col: '{split_datetime_col}' does not exist in "
            f"data"
        )
    try:
        data[split_datetime_col] = pd.to_datetime(
            data[split_datetime_col], infer_datetime_format=True
        )
    except Exception as e:
        logger.warning("Type casting failed for column %s", split_datetime_col)
        raise e
    if not isinstance(split_datetime_val, datetime.datetime):
        raise ValueError(
            f"Provided datetime_val: '{split_datetime_val}' is not a Datetime "
            f"object "
        )

    # it is not possible to compare offset-naive and offset-aware datetimes
    # if we have a timezone aware column, we assume that the time set in
    # the parameters comes from the same timezone
    if isinstance(data[split_datetime_col].dtype, pd.DatetimeTZDtype):
        split_tz = data[split_datetime_col].dt.tz
        split_datetime_val = split_datetime_val.replace(tzinfo=split_tz)

    logger.info("Splitting by datetime: %s", split_datetime_val)
    min_date = data[split_datetime_col].min()
    max_date = data[split_datetime_col].max()

    if not min_date <= split_datetime_val <= max_date:
        raise ValueError(
            f"Provided datetime_val: '{split_datetime_val}' lies outside of the "
            f"range of the dataset [{min_date},{max_date}] "
        )

    train_data = data[data[split_datetime_col] <= split_datetime_val]
    opt_data = data[data[split_datetime_col] > split_datetime_val]
    return dict(train=train_data, test=opt_data)


def split_data_by_frac(params: dict, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Args:
        params: dictionary of parameters
        data: input data
    Returns:
        dict(train=train_data, test=opt_data)
    """
    data = data.copy()
    split_datetime_col = params.get("datetime_col", "timestamp")
    train_fraction = params.get("train_split_fract")
    if not train_fraction:
        raise KeyError("No train_split_ratio provided.")
    if not isinstance(train_fraction, numbers.Number):
        raise ValueError(f"Non-numeric train_fract: '{train_fraction}' provided.")
    if not 0 < train_fraction < 1.0:
        raise ValueError(f"train_fract: '{train_fraction}' is out of range (0,1).")
    if split_datetime_col:
        if split_datetime_col in data.columns:
            logger.info("Sorting df by sort_df_col: '%s'.", split_datetime_col)
            data.sort_values(by=[split_datetime_col], inplace=True)
        else:
            logger.warning(
                "Provided sort_df_col: '%s' does not exist in data. Proceeding with "
                "split without sort. ",
                split_datetime_col,
            )

    logger.info(
        "Splitting data with train fraction of %2.2f ...",
        train_fraction,
    )
    n_obs, _ = data.shape
    split_n = int(n_obs * train_fraction)

    train_data = data.iloc[:split_n, :]
    opt_data = data.iloc[split_n:, :]

    return dict(train=train_data, test=opt_data)


supported_splits = {"date": split_data_by_date, "frac": split_data_by_frac}


def split_data(params: dict, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    split_type = params["type"]
    if split_type not in supported_splits:
        raise ValueError(
            f"Split of type: {split_type} is not supported. Supported splits are"
            f" {', '.join(supported_splits.keys())}"
        )
    return supported_splits[split_type](params, data)
