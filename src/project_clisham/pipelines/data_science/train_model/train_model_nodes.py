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
Nodes of the model training pipeline.
"""
import logging
import datetime
from copy import deepcopy
from typing import Any, Dict
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline as SklearnPipeline
from xgboost import XGBRegressor

from project_clisham.optimus_core import utils
from project_clisham.optimus_core.model_helpers.performance import (
    generate_prediction_metrics,
    tree_feature_importance,
)
from project_clisham.optimus_core.model_helpers.tuning import sklearn_tune, xgb_tune
from project_clisham.optimus_core.reporting_html.nodes import create_html_report

from project_clisham.optimus_core.tag_management import TagDict
from project_clisham.optimus_core.transformers import NumExprEval, SelectColumns

logger = logging.getLogger(__name__)

SUPPORTED_MODEL_FEATURE_IMPORTANCE = {"tree": tree_feature_importance}


def load_regressor(params: dict):
    """
    Loads a regressor object based on given parameters.
    Args:
        params: dictionary of parameters
    Returns:
        sklearn compatible model
    """
    model_class = params["regressor"]["class"]
    model_kwargs = params["regressor"]["kwargs"]
    regressor = utils.load_obj(model_class)(**model_kwargs)
    assert hasattr(regressor, "fit"), "Model object must have a .fit method"
    assert hasattr(regressor, "predict"), "Model object must have a .predict method"
    return regressor


def add_transformers(params: dict, td: TagDict, regressor: Any):
    """
    Creates a sklearn model pipeline based on the regressor and adds
    the desired transformers. This is where things like imputation,
    scaling, feature selection, and dynamic feature generation should plug in.
    Args:
        params: dictionary of parameters
        td: tag dictionary
        regressor: regressor object
    Returns:
        sklearn model pipeline with transformers
    """

    # Transformer which reduces the model input to the
    # relevant features
    model_feature = params["dict_model_feature"]
    feat_cols = td.select(model_feature)
    column_selector = SelectColumns(feat_cols)

    model = SklearnPipeline(
        [
            ("select_columns", column_selector),
            ("regressor", regressor),
        ]
    )

    return model


def train_tree_model(
    params: dict, td: TagDict, data: pd.DataFrame, model: SklearnPipeline
) -> Dict[str, Any]:
    """
    Args:
        params: dictionary of parameters
        td: tag dictionary
        data: input data
        model: sklearn pipeline with regressor and transformers
    Returns:
        Trained sklearn compatible model
        Hyper-parameter Tuning Search Results
        Feature importance
    """
    return _train_model(params, td, data, model, "tree")


def _train_model(
    params: dict,
    td: TagDict,
    data: pd.DataFrame,
    model: SklearnPipeline,
    model_type: str,
) -> Dict[str, Any]:
    """
    Args:
        params: dictionary of parameters
        td: tag dictionary
        data: input data
        model: sklearn pipeline with regressor and transformers
        model_type: string used for determining feature importance
            Supported values: ["tree"]
    Returns:
        Trained sklearn compatible model
        Hyper-parameter Tuning Search Results
        Feature importance
    """
    model_target = params["dict_model_target"]
    model_feature = params["dict_model_feature"]
    target_col = td.select("target", model_target)[0]
    target = data[target_col]

    # strictly speaking, selection features should not be necessary
    # as this is done by the model transformer. However, we do it
    # regardless to reduce our memory footprint and protect against
    # accidental removal of the transformer
    feat_cols = td.select(model_feature)
    feature_df = data[feat_cols]

    regressor = model.named_steps["regressor"]
    if isinstance(regressor, XGBRegressor):
        logger.info("Tuning using `xgb_tune`.")
        tuned_model, cv_results_df = xgb_tune(params, feature_df, target, model)
    else:
        logger.info("Tuning using `sklearn_tune`.")
        tuned_model, cv_results_df = sklearn_tune(params, feature_df, target, model)

    feature_importance = SUPPORTED_MODEL_FEATURE_IMPORTANCE.get(model_type)
    importances = feature_importance(tuned_model)

    # print(importances)
    # exit()

    return dict(
        model=tuned_model, cv_results=cv_results_df, feature_importance=importances
    )


def create_predictions(
    params: dict, td: TagDict, data: pd.DataFrame, model: SklearnPipeline
) -> Dict[str, pd.DataFrame]:
    """
    Creates model predictions for a given data set
    Args:
        params: dictionary of parameters
        td: tag dictionary
        data: input data
        model: sklearn pipeline with regressor and transformers
    Returns:
        predictions, metrics
    """
    model_target = params["dict_model_target"]
    prediction_col = "prediction"
    predictions = model.predict(data)
    target_col = td.select("target", model_target)[0]
    res_df = data.copy()
    if predictions.shape[0] != data.shape[0]:
        missing_rows = data.shape[0] - predictions.shape[0]
        res_df = res_df.iloc[missing_rows:, :]
    res_df[prediction_col] = predictions
    prediction_metrics_df = pd.DataFrame()
    prediction_metrics_df["opt_perf_metrics"] = generate_prediction_metrics(
        res_df, target_col, prediction_col
    )
    print(prediction_metrics_df)
    return dict(predictions=res_df, metrics=prediction_metrics_df)


def retrain_tree_model(
    td: TagDict, model: SklearnPipeline, data: pd.DataFrame
) -> SklearnPipeline:
    """
    Retraining the model object with the new dataset.
    Args:
        td: tag dictionary
        model: sklearn pipeline with regressor and transformers
        data: input data

    Returns:
        retrained SklearnPipeline model

    """
    target_col = td.select("target", model_target)[0]
    target = data[target_col]

    retrain_model = deepcopy(model)
    retrain_model.fit(data, target)
    return retrain_model


def generate_performance_report(
    params: Dict,
    kedro_env: str,
    test_predictions,  # to force node order
):  # pylint:disable=unused-argument
    namespace = params["namespace"]
    template_path = Path(params["report"])
    reporting_dir = Path(params["report_dir"])
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    html_report_filename = f"{namespace}_training_perf_report_{timestamp_str}"
    report_params = {
        "template_path": template_path,
        "output_dir": reporting_dir,
        "report_name": html_report_filename,
        "namespace": namespace,
        "remove_code": True,
    }
    try:
        create_html_report(
            report_params,
            kedro_env=kedro_env,
        )
    except RuntimeError as err:
        # keep going if the report fails
        logger.warning("Could not generate performance report: {}".format(str(err)))
