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

"""This module provides a set of helper functions being used for interact with
Great Expectations in the notebook
"""
import numpy as np
import pandas as pd
from optimus_pkg.core.tag_management import TagDict
from adtk.data import validate_series
from adtk.detector import MinClusterDetector
from sklearn.cluster import KMeans
from adtk.detector import OutlierDetector
from sklearn.ensemble import IsolationForest


def create_sensor_exist_expectation(batch: pd.DataFrame, params: dict) -> None:
    """
    Validate if tags are part of the dataframe
    Args:
        batch: a Great Expectations DataAsset with expectation_suite_name attached
        params: kedro parameters from parameters.yml
    Returns:
        None
    """
    column_list = params["dataset_1"]["column_list"]
    for column in column_list:
        batch.expect_column_to_exist(column)


def create_data_length_expectation(batch: pd.DataFrame, params: dict) -> None:
    """
    c
    Args:
        batch: a Great Expectations DataAsset with expectation_suite_name attached
        params: kedro parameters from parameters.yml
    Returns:
        None
    """
    batch.expect_table_row_count_to_be_between(
        params["dataset_1"]["data_length"]["min_value"],
        params["dataset_1"]["data_length"]["max_value"],
    )


def create_not_null_expectations_from_tagdict(batch: pd.DataFrame) -> None:
    """
    Validate a dataset has no null values in column
    Args:
        batch: a Great Expectations DataAsset with expectation_suite_name attached
    Returns:
        None
    """
    for col in batch.columns:
        batch.expect_column_values_to_not_be_null(col)


def create_data_schema_expectation(batch: pd.DataFrame, params: dict) -> None:
    """
    Validate the schema of a dataframe  with predefined key-pairs
    Args:
        batch: a Great Expectations DataAsset with expectation_suite_name attached
        params: kedro parameters from parameters.yml
    Returns:
        None
    """
    val_schema = params["dataset_1"]["schema"]
    batch.expect_table_schema(schema=val_schema)


def create_time_format_expectation(batch: pd.DataFrame, params: dict) -> None:
    """
    Validate the timestamp column of the dataframe and ensure it conforms to
    the format provided
    Args:
        batch: a Great Expectations DataAsset with expectation_suite_name attached
        params: kedro parameters from parameters.yml
    Returns:
        None
    """
    time_format = params["dataset_1"]["time"]["format"]
    timestamp_column = params["dataset_1"]["time"]["column"]
    batch.expect_column_values_to_match_strftime_format(timestamp_column, time_format)


def create_range_expectations_from_tagdict(batch: pd.DataFrame, td: TagDict) -> None:
    """
    Validate the value range of a dataset based on expected values defined
    in the TagDict
    Args:
        batch: a Great Expectations DataAsset with expectation_suite_name attached
        td: TagDict
    Returns:
        None
    """
    for col in batch.columns:
        if col in td:
            # skip tag that has no range min and max
            if np.isnan(td[col]["range_min"]) and np.isnan(td[col]["range_max"]):
                continue

            batch.expect_column_values_to_be_between(
                col, td[col]["range_min"], td[col]["range_max"]
            )


def create_sensor_pair_equals_expectation(batch: pd.DataFrame, params: dict) -> None:
    """
    Validate the sensor pairs to ensure if they have the same values
    Args:
        batch: a Great Expectations DataAsset with expectation_suite_name attached
        params: kedro parameters from parameters.yml
    Returns:
        None
    """
    first_sensor, second_sensor = (
        params["dataset_1"]["sensor_pair_1"]["first_sensor"],
        params["dataset_1"]["sensor_pair_1"]["second_sensor"],
    )
    batch.expect_column_pair_values_to_be_equal(first_sensor, second_sensor)


# Custom expectations
# ADTK Integrated Expectations
def create_flatline_expectation(batch: pd.DataFrame, params: dict) -> None:
    """
    Validate sensor values are not violating flatline rules i.e. no data change
    with in a process period
    Args:
        batch: a Great Expectations DataAsset with expectation_suite_name attached
        params: kedro parameters from parameters.yml
    Returns:
        None
    """
    process_window = params["dataset_1"]["process_window"]
    batch.index = pd.to_datetime(batch["status_time"])
    validation_columns = batch.select_dtypes(include=["float"]).columns
    for col in validation_columns:
        batch.validate_column_flatline_anomaly(col, process_window)


# ADTK Integrated Expectations
def create_level_shift_expectation(batch: pd.DataFrame, params: dict) -> None:
    """
    Validate sensor values are not violating level shift anomaly
    detection
    Args:
        batch: a Great Expectations DataAsset with expectation_suite_name attached
        params: kedro parameters from parameters.yml
    Returns:
        None
    """

    # set adtk params for level shift outlier detection
    window = params["dataset_1"]["levelshift_anomaly"]["level_window"]
    factor = params["dataset_1"]["levelshift_anomaly"]["level_factor"]
    side = params["dataset_1"]["levelshift_anomaly"]["shift_side"]
    min_period = params["dataset_1"]["levelshift_anomaly"]["min_periods"]

    batch.index = pd.to_datetime(batch["status_time"])
    validation_columns = batch.select_dtypes(include=["float"]).columns
    for col in validation_columns:
        batch.validate_column_levelshift_anomaly(col, window, factor, side, min_period)


def validate_column_quantile_anomaly(batch: pd.DataFrame, params: dict) -> None:
    """
    Validate sensor values are not violating quantile anomaly detection
    Args:
        batch: a Great Expectations DataAsset with expectation_suite_name attached
        params: kedro parameters from parameters.yml
    Returns:
        None
    """
    # set adtk params for quantile outlier detection
    low = params["dataset_1"]["quantile_anomaly"]["low"]
    high = params["dataset_1"]["quantile_anomaly"]["high"]

    batch.index = pd.to_datetime(batch["status_time"])
    validation_columns = batch.select_dtypes(include=["float"]).columns
    for col in validation_columns:
        batch.validate_column_quantile_anomaly(col, low, high)


def validate_column_persist_anomaly(batch: pd.DataFrame, params: dict) -> None:
    """
    Validate sensor values are not violating persist i.e. gaps in median
    or mean within a window anomaly detection
    Args:
        batch: a Great Expectations DataAsset with expectation_suite_name attached
        params: kedro parameters from parameters.yml
    Returns:
        None
    """
    # set adtk params for persist outlier detection
    window = params["dataset_1"]["persist_shift_anomaly"]["persist_window"]
    factor = params["dataset_1"]["persist_shift_anomaly"]["persist_factor"]
    side = params["dataset_1"]["persist_shift_anomaly"]["shift_side"]
    min_period = params["dataset_1"]["persist_shift_anomaly"]["min_periods"]
    aggregation = params["dataset_1"]["persist_shift_anomaly"]["aggregation"]

    batch.index = pd.to_datetime(batch["status_time"])
    validation_columns = batch.select_dtypes(include=["float"]).columns
    for col in validation_columns:
        batch.validate_column_persist_shift_anomaly(
            col, window, factor, side, min_period, aggregation
        )


def validate_multi_dimension_cluster_anomaly(batch: pd.DataFrame, params: dict) -> None:
    """
    Validate a set of sensor values together using Kmeans Clustering algorithm.
    Args:
        batch: a Great Expectations DataAsset with expectation_suite_name attached
        params: kedro parameters from parameters.yml

    Returns:
        None

    """
    batch.index = pd.to_datetime(batch["status_time"])
    clusters = params["dataset_1"]["multi_dim_cluster_anomaly"]["cluster_size"]
    columns = params["dataset_1"]["multi_dim_cluster_anomaly"]["columns_to_evaluate"]
    data = batch[columns]
    data = validate_series(data)
    multi_dim_cluster_ad = MinClusterDetector(KMeans(clusters))
    anomalies = multi_dim_cluster_ad.fit_detect(data)

    for column in columns:
        batch[column] = anomalies
        batch.validate_multi_dimension_cluster_anomaly(column, columns)


def validate_multi_dimension_isolationforest_anomaly(
    batch: pd.DataFrame, params: dict
) -> None:
    """
    Validate a set of sensor values together using IsolationForest technique.
    Args:
        batch: a Great Expectations DataAsset with expectation_suite_name attached
        params: kedro parameters from parameters.yml

    Returns:
        None

    """
    batch.index = pd.to_datetime(batch["status_time"])
    perc_outliers = params["dataset_1"]["multi_dim_lof_anomaly"]["contamination"]
    columns = params["dataset_1"]["multi_dim_cluster_anomaly"]["columns_to_evaluate"]
    data = batch[columns]
    data = validate_series(data)
    outlier_detector = OutlierDetector(IsolationForest(perc_outliers))
    anomalies = outlier_detector.fit_detect(data)

    for column in columns:
        batch[column] = anomalies
        batch.validate_multi_dim_isoforest_anomaly(column, columns)
