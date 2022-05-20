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
Base class and helpers
"""
from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DIMENSIONALITY_ERROR_PANDAS = (
    "Expected DataFrame, got a Series instead.\n"
    "Convert your data to a DataFrame using pd.DataFrame(series) if "
    "your data has a single feature or series.to_frame().T "
    "if it contains a single sample."
)


class Transformer(TransformerMixin, BaseEstimator, ABC):
    """ Transformer base class """

    @staticmethod
    def check_x(x: Any):
        """
        Checks that the input is a pandas dataframe.
        Throws a helpful, sklearn-like exception if it is a series.
        Args:
            x: input to check
        Raises:
            TypeError: if input is not a dataframe
        """
        if isinstance(x, pd.Series):
            raise TypeError(DIMENSIONALITY_ERROR_PANDAS)
        if not isinstance(x, pd.DataFrame):
            raise TypeError("x should be a dataframe, got {}".format(str(type(x))))

    def fit(
        self, x: pd.DataFrame, y: Union[np.ndarray, pd.Series] = None
    ):  # pylint:disable=unused-argument
        """
        Train the transformer on trainig data.
        Args:
            x: training data
            y: training y (optional)
        Returns:
            self
        """
        return self

    @abstractmethod
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Transform x according to what was learned on the test data.
        Args:
            x: dataframe
        """
        raise NotImplementedError()
