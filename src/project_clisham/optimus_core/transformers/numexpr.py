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
Transformers using numexpr
"""
import re
import pandas as pd

from .base import Transformer


class NumExprEval(Transformer):
    """
    Dynamically creates a new column with most of the
    functionality provided by ``pandas.DataFrame.eval``.

    Args:
        exprs: the expressions to evaluate, see ``pandas.eval`` for
        more details
    """

    def __init__(self, exprs):
        assignment_regexp = re.compile(r"[^<>=]=[^=]")
        if isinstance(exprs, str):
            exprs = [exprs]
        for expr in exprs:
            if not isinstance(expr, str):
                raise ValueError(
                    f"Must provide a str for eval, type {type(expr)} given"
                )
            if not assignment_regexp.search(expr):
                raise ValueError("Must provide assignment for resulting column name")
        self.exprs = exprs

    def fit(self, x, y=None):  # pylint:disable=unused-argument
        """
        Checks if given input is a pandas DataFrame
        Args:
            x: training data
            y: training y (no effect)
        Returns:
            self
        """
        self.check_x(x)
        return self

    def transform(self, x):
        """
        Reduces x to the columns learned in the .fit step.
        Args:
            x: pandas.DataFrame
        """
        self.check_x(x)
        for expr in self.exprs:
            result_df = x.eval(expr)
            if not isinstance(result_df, pd.DataFrame):
                raise ValueError(
                    f"Please provide a column name assignment for "
                    f"the expression: '{expr}'"
                )
            x = result_df
        return x
