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
Sklearn Transformers wrapper for Pandas compatibility
"""
from typing import Union

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

from .base import Transformer


class SklearnTransform(Transformer):
    """
    Generic Transformer with that accepts an input Sklearn compatible transformer
    with a fit and transform method, and ensures compatibility with preceding and
    proceeding transformers by maintaining Pandas DataFrames.
    """

    def __init__(self, transformer: Union[TransformerMixin, BaseEstimator]):
        """Constructor.

        Args:
            transformer: callable class representing the sklearn Transformer
        """
        self._transformer = transformer

    def fit(
        self, x: pd.DataFrame, y: Union[pd.Series, pd.DataFrame] = None, **kwargs
    ):  # pylint:disable=arguments-differ
        """
        Calls the fit function of Sklearn Transformer
        Args:
            x: training data
            y: training y (no effect)
        Returns:
            self
        """
        self.check_x(x)
        self._transformer.fit(x, y, **kwargs)
        return self

    def transform(self, x: pd.DataFrame):
        """
        Transforms values of x per the transform function of Sklearn transformer
        Args:
            x: training data
        Returns:
            pd.DataFrame
        """
        self.check_x(x)
        transformed = self._transformer.transform(x)
        return pd.DataFrame(transformed, columns=x.columns, index=x.index)
