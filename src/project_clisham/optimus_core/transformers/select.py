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
Transformers for columns selection
"""
import re
from typing import List

from sklearn.exceptions import NotFittedError

from .base import Transformer


class SelectColumns(Transformer):
    """
    Selects columns from the dataset, with most of the
    functionality provided by ``pandas.DataFrame.filter``.

    Args:
        items: a list of columns to select, see
            ``pandas.DataFrame.filter`` for more details
        regex: a regex used to select columns, see
            ``pandas.DataFrame.filter`` for more details
    """

    def __init__(self, items: List[str] = None, regex: str = None):
        # items and regex are mutually exclusive as enforced by pandas
        if items is None and regex is None:
            raise ValueError("Must state either items or regex.")
        if items is not None and regex is not None:
            raise ValueError("Must state either items or regex but not both.")

        if isinstance(items, str):
            items = [items]

        self.regex = re.compile(regex) if regex else None
        self.items = items
        self.selected_columns = None

    def fit(self, x, y=None):  # pylint:disable=unused-argument
        """
        Stores a list of selected columns.
        Args:
            x: training data
            y: training y (no effect)
        Returns:
            self
        """
        self.check_x(x)
        self.selected_columns = x.filter(
            items=self.items, regex=self.regex
        ).columns.tolist()
        return self

    def transform(self, x):
        """
        Reduces x to the columns learned in the .fit step.
        Args:
            x: dataframe
        """
        self.check_x(x)
        if self.selected_columns is None:
            raise NotFittedError(".transform called before .fit.")
        return x.filter(items=self.selected_columns)


class DropColumns(Transformer):
    """
    Drops columns from the dataset, with most of the
    functionality provided by ``pandas.DataFrame.filter``.

    Args:
        items: a list of columns to drop, see
            ``pandas.DataFrame.filter`` for more details
        regex: a regex used to drop columns, see
            ``pandas.DataFrame.filter`` for more details
    """

    def __init__(self, items: List[str] = None, regex: str = None):
        # items and regex are mutually exclusive as enforced by pandas
        if items is None and regex is None:
            raise ValueError("Must state either items or regex.")
        if items is not None and regex is not None:
            raise ValueError("Must state either items or regex but not both.")

        if isinstance(items, str):
            items = [items]

        self.regex = re.compile(regex) if regex else None
        self.items = items
        self.selected_columns_to_drop = None

    def fit(self, x, y=None):  # pylint:disable=unused-argument
        """
        Stores a list of selected columns.
        Args:
            x: training data
            y: training y (no effect)
        Returns:
            self
        """
        self.check_x(x)
        self.selected_columns_to_drop = x.filter(
            items=self.items, regex=self.regex
        ).columns.tolist()
        return self

    def transform(self, x):
        """
        Drops x to the columns learned in the .fit step.
        Args:
            x: dataframe
        """
        self.check_x(x)
        if self.selected_columns_to_drop is None:
            raise NotFittedError(".transform called before .fit.")
        return x.drop(columns=self.selected_columns_to_drop)


class DropAllNull(Transformer):
    """
    Drops columns where everything is ``null``
    """

    def __init__(self):
        self.selected_columns_to_drop = None

    def fit(self, x, y=None):  # pylint:disable=unused-argument
        self.check_x(x)
        self.selected_columns_to_drop = x.columns[x.isnull().all(0)].tolist()
        return self

    def transform(self, x):
        self.check_x(x)
        if self.selected_columns_to_drop is None:
            raise NotFittedError(".transform called before .fit.")
        return x.drop(columns=self.selected_columns_to_drop)
