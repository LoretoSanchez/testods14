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


from great_expectations.dataset import PandasDataset, MetaPandasDataset
import pandas as pd
from adtk.data import validate_series
from adtk.detector import QuantileAD
from adtk.detector import LevelShiftAD
from adtk.detector import PersistAD

from typing import List


class CustomADTKExpectations(PandasDataset):
    @MetaPandasDataset.column_aggregate_expectation
    def validate_column_flatline_anomaly(self, column: pd.Series, process_window: int):
        """
        Validate sensor values are not violating quantile anomaly detection
        Args:
            column: a Great Expectations DataAsset ccolumn to validate
            process_window: window to look for flat lining
        Returns:
            None
        """
        # set adtk params for process window i.e. time to finish processing material

        data = self[column]
        data = validate_series(data)

        anomalies = data.rolling(process_window).std()
        return {
            "success": anomalies[anomalies == 0].empty,
            "result": {
                "observed_value": {
                    "length": len(anomalies[anomalies == 0]),
                    "time_period": anomalies[anomalies == 0].index.values,
                }
            },
        }

    @MetaPandasDataset.column_aggregate_expectation
    def validate_column_quantile_anomaly(
        self, column: pd.Series, low: float, high: float
    ):
        """
        Validate sensor values are not violating quantile anomaly detection
        Args:
            column: a Great Expectations DataAsset ccolumn to validate
            low: quantile of historical data lower which a value is regarded as anomaly
            high: quantile of historical data above which a value is regarded as anomaly
        Returns:
            None
        """
        data = self[column]
        data = validate_series(data)

        quantile_ad = QuantileAD(low, high)
        anomalies = quantile_ad.fit_detect(data)

        return {
            "success": anomalies[anomalies].empty,
            "result": {
                "observed_value": {
                    "length": len(anomalies[anomalies]),
                    "time_period": anomalies[anomalies].index.values,
                }
            },
        }

    @MetaPandasDataset.column_aggregate_expectation
    def validate_column_levelshift_anomaly(
        self, column: pd.Series, window: int, factor: int, side: str, min_period: int
    ):
        """
        Validate sensor values are not violating level shift anomaly
        detection
        Args:
            column: a Great Expectations DataAsset ccolumn to validate
            window: kedro parameters from parameters.yml
            factor: bound of normal range based on historical interquartile range
            side:  detect anomalous values in both positive, negative changes
            min_period: minimum number of observations required in each window required
        Returns:
            None
        """
        data = self[column]
        data = validate_series(data)

        quantile_range_anomalies = LevelShiftAD(window, factor, side, min_period)
        anomalies = quantile_range_anomalies.fit_detect(data)

        return {
            "success": anomalies[anomalies == 1].empty,
            "result": {
                "observed_value": {
                    "length": len(anomalies[anomalies == 1]),
                    "time_period": anomalies[anomalies == 1].index.values,
                }
            },
        }

    @MetaPandasDataset.column_aggregate_expectation
    def validate_column_persist_shift_anomaly(
        self,
        column: pd.Series,
        window: int,
        factor: int,
        side: str,
        min_period: int,
        aggregation: str,
    ):
        """
        Validate sensor values are not violating persist i.e. gaps in median
        or mean within a window anomaly detection
        Args:
            column: a Great Expectations DataAsset ccolumn to validate
            window: kedro parameters from parameters.yml
            factor: bound of normal range based on historical interquartile range
            side:  detect anomalous values in both positive, negative changes
            min_period: minimum number of observations required in each window required
            aggregation: aggregation operation of the time window
        Returns:
            None
        """
        data = self[column]
        data = validate_series(data)

        persist_ad = PersistAD(window, factor, side, min_period, aggregation)
        anomalies = persist_ad.fit_detect(data)
        return {
            "success": anomalies[anomalies == 1].empty,
            "result": {
                "observed_value": {
                    "length": len(anomalies[anomalies == 1]),
                    "time_period": anomalies[anomalies == 1].index.values,
                }
            },
        }

    @MetaPandasDataset.column_aggregate_expectation
    def validate_multi_dimension_cluster_anomaly(
        self, column: pd.Series, sensor_list: List[str]
    ):
        """
        Validate a set of sensor values together using Kmeans Clustering algorithm.
        Args:
            column: a Great Expectations DataAsset ccolumn to validate
            sensor_list: list of sensors to find anomalies in

        Returns:
            A dictionary indicating if the validation is a success or not,
            and all the anomaly timestamps where it fails.

        """
        data = self[column]

        return {
            "success": data[data].empty,
            "result": {
                "observed_value": {
                    "length": len(data[data]),
                    "time_period": data[data].index.values,
                    "validated_sensors": sensor_list,
                }
            },
        }

    @MetaPandasDataset.column_aggregate_expectation
    def validate_multi_dim_isoforest_anomaly(
        self, column: pd.Series, sensor_list: List[str]
    ):
        """
        Validate a set of sensor values together using Isolation Forest Clustering algorithm.
        Args:
            column: a Great Expectations DataAsset ccolumn to validate
            sensor_list: list of sensors to find anomalies in

        Returns:
            A dictionary indicating if the validation is a success or not,
            and all the anomaly timestamps where it fails.

        """
        data = self[column]

        return {
            "success": data[data].empty,
            "result": {
                "observed_value": {
                    "length": len(data[data]),
                    "time_period": data[data].index.values,
                    "validated_sensors": sensor_list,
                }
            },
        }
