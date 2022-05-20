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
from typing import Any

import attr
import pandas as pd
from sklearn import base as sk_base
import shap

logger = logging.getLogger(__name__)


@attr.s
class ShapExplanation:
    """
    A simple class to hold the results of the SHAP stuff
    """

    shap_values = attr.ib()
    expectation = attr.ib()
    raw_features = attr.ib()


def calculate_shap_values(
    model: sk_base.RegressorMixin,
    train_dataset: pd.DataFrame,
    shap_dataset: pd.DataFrame,
    shap_explainer: shap.explainers.explainer.Explainer,
    **shap_values_kwargs,
) -> ShapExplanation:
    """
    Calculates a set of shap values. Should be reasonably
    tolerant of different model types.

    Supported model types:
     - Linear models, which extend `sklearn.linear_model.base.LinearModel`.
       Supported through `shap.LinearExplainer`
     - Tree based models, including ones extending `sklearn.ensemble.BaseForest`,
       `XGboost`models`
     - Deep learning models, through `shap.DeepExplainer`

    Args:
        model: The model being evaluated.
        train_dataset: training data. Used in the construction of some SHAP explainers
        shap_dataset: the dataset to explain
        shap_explainer: Sets the SHAP explainer
        **shap_values_kwargs: additional arguments to `explainer.shap_values`

    Returns:
        a `ShapExplanation`

    """
    if not hasattr(train_dataset, "columns") or not hasattr(shap_dataset, "columns"):
        msg = "SHAP calculations require pandas dataframes as inputs, got {} and {}"
        raise TypeError(msg.format(type(train_dataset), type(shap_dataset)))
    if not hasattr(shap_explainer, "shap_values"):
        raise ValueError(
            f"The provided shap explainer is of type f{str(type(shap_explainer))} and "
            f"does not have a shap_values attribute "
        )

    explainer = shap_explainer(model, train_dataset)

    # TODO: remove check_additivity=False once SHAP issue
    # https://github.com/slundberg/shap/issues/950 is resolved

    shap_explanation = explainer.shap_values(
        shap_dataset, check_additivity=False, **shap_values_kwargs
    )

    shap_frame = pd.DataFrame(
        shap_explanation, index=shap_dataset.index, columns=shap_dataset.columns
    )

    # noinspection PyUnresolvedReferences
    return ShapExplanation(shap_frame, explainer.expected_value, shap_dataset)


def tf_calculate_shap_values(
    model: Any,
    train_dataset: pd.DataFrame,
    shap_dataset: pd.DataFrame,
    shap_explainer: shap.explainers.explainer.Explainer,
    **shap_values_kwargs,
) -> ShapExplanation:
    """
    Create a shap object for a tensorflow/nn model.

    Our model is a composite SKlearn pipeline, which has out estimator/model
    as the final stage. The penultimate stage is a feature scaling operation
    which converts selected columns to a numpy array. To fetch the columns of the
    input dataframe which are actual inputs to our model we apply the transform methods
    of the earlier stages.
    Args:
        model: tf.keras model
        train_dataset: training data. Used in the construction of some SHAP explainers
        shap_dataset: the dataset to explain
        shap_explainer: Sets the SHAP explainer (shap.DeepExplainer)
        **shap_values_kwargs:

    Returns: a `ShapExplaination`

    """
    explainer = shap_explainer(model, train_dataset.values)

    shap_explanation = explainer.shap_values(
        shap_dataset.values, check_additivity=False, **shap_values_kwargs
    )
    shap_frame = pd.DataFrame(
        shap_explanation[0], index=shap_dataset.index, columns=shap_dataset.columns
    )

    # noinspection PyUnresolvedReferences
    return ShapExplanation(shap_frame, explainer.expected_value, shap_dataset)


def model_wrapper_for_shap(mdl, x):
    """
    Wraps a model and dataset to make a
    SHAP `KernelExplainer`.
    Args:
        mdl: The model, which has a `predict` function
        x: the dataset to explain

    Returns: a `shap.KernelExplainer`

    """
    try:
        mdl.predict
    except AttributeError:
        msg = "Model passed did not have predict function, model type: {}"
        raise TypeError(msg.format(type(mdl)))

    def inner_model_call(x_inner):
        return mdl.predict(x_inner)

    return shap.KernelExplainer(inner_model_call, x)
