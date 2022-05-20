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

import logging
from typing import Any, Mapping

import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
)

from .metrics import mean_absolute_percentage_error


logger = logging.getLogger(__name__)


def generate_prediction_metrics(
    data: pd.DataFrame, y_true_col: str, y_pred_col: str
) -> pd.Series:
    """
    Calculate various metrics:
     - MAE
     - RMSE
     - MSE
     - MAPE
     - R squared
     - Explained variance

    Args:
        data: Dataframe containing features and predictions
        y_true_col: the actual values column name
        y_pred_col: the predicted values column name

    Returns:
        A pandas series of metric values
    """
    y_true = data[y_true_col]
    y_pred = data[y_pred_col]
    metrics_vals = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "mse": mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "var_score": explained_variance_score(y_true, y_pred),
        "mean_true": y_true.mean(),
        "mean_pred": y_pred.mean(),
    }
    return pd.Series(metrics_vals)


def tree_feature_importance(model_pipeline: Any) -> Mapping:
    """
    Produce feature importances by pulling the feature importances
    vector from the model.

    Args:
        model_pipeline: the model pipeline

    Returns:
        A Pandas Series of Feature Importances
    """
    try:
        model = model_pipeline.named_steps["regressor"]
        importance = model.feature_importances_
    except AttributeError:
        msg = (
            "Model did not look like a tree, it needs "
            "a feature_importances_ attribute. Model "
            "type: {}".format(type(model))
        )
        raise AttributeError(msg)

    return pd.Series(importance, model_pipeline.named_steps["select_columns"].items)
