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
Model tuning procedures
"""
from copy import deepcopy
import numbers
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import BaseCrossValidator, train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline

from project_clisham.optimus_core import utils
from project_clisham.optimus_core.model_helpers.metrics import (
    mean_absolute_percentage_error,
)


def get_cv_from_params(params: dict) -> BaseCrossValidator:
    """
    Creates CV Splitting Strategy from Params
    Default is 5-fold cross validation
    Args:
        params: dictionary of parameters
    Returns:
        Cross Validation Iterator
    """
    cv = params["cv"]
    if not isinstance(cv, numbers.Integral):
        cv = utils.load_obj(params["cv"]["class"])(**params["cv"]["kwargs"])
    return cv


def get_hp_from_params(params: dict, model: SklearnPipeline):
    """
    Instantiates Hyper-parameter Tuning strategy from Params.
    If scoring is specified in the params, it also adds MAPE in
    the scoring calculation.
    Args:
        params: dictionary of parameters
        model: sklearn pipeline with regressor and transformers
    Returns:
        Hyperparameter Search Strategy
    """
    cv = get_cv_from_params(params)

    args = params["tuner"]["kwargs"].copy()
    args["estimator"] = model
    args["cv"] = cv

    # if "mape" is selected as a metric, add it programmatically
    # as it is not built into sklearn
    if args["scoring"] and "mape" in args["scoring"]:
        args["scoring"]["mape"] = make_scorer(
            mean_absolute_percentage_error, greater_is_better=False
        )

    hp = utils.load_obj(params["tuner"]["class"])(**args)
    return hp


def sklearn_tune(
    params: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    model: SklearnPipeline,
    **fit_kwargs,
) -> List[Union[Any, Dict]]:
    """
    Generic tuning procedure for sklearn models.

    Args:
        params: dictionary of parameters
        X: training data X
        y: trainig data y
        model: sklearn pipeline with regressor and transformers
        cv: cross validation iterator
        **fit_kwargs: keyword args for `gs_cv.fit`
    Returns:
        Trained sklearn compatible model
        Hyper-parameter Tuning Search Results
    """
    gs_cv = get_hp_from_params(params, model)
    gs_cv.fit(X, y, **fit_kwargs)

    best_estimator = gs_cv.best_estimator_
    cv_results_df = pd.DataFrame(gs_cv.cv_results_)

    return best_estimator, cv_results_df


def xgb_tune(
    params: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    model: SklearnPipeline,
) -> List[Union[Any, Dict]]:
    """
    Tuning procedure for xgb regressors models. Under the hood,
    we use sklearn_tune but with a dedicated validation set for early stopping.

    Args:
        params: dictionary of parameters
        X: training data X
        y: training data y
        model: sklearn pipeline with regressor and transformers
        cv: cross validation iterator
    Returns:
        Trained sklearn compatible model
        Hyper-parameter Tuning Search Results
    """
    xgb_tune_params = params["xgb_tune"]
    if xgb_tune_params.get('active'):
        X_train, X_val, y_train, y_val = train_test_split(X, 
                            y, 
                            test_size=xgb_tune_params['p_eval'], 
                            random_state=xgb_tune_params['random_state'])
        eval_set = [(X_train,y_train), (X_val,y_val)]

        fit_kwargs = dict(
            regressor__eval_set = eval_set,
            regressor__early_stopping_rounds = xgb_tune_params['early_stopping_rounds'],
            regressor__eval_metric = xgb_tune_params['eval_metric'],
            regressor__verbose=False,
        )

        return sklearn_tune(params, X_train, y_train, model, **fit_kwargs)
    
    else:
        fit_kwargs = dict(
            regressor__verbose=False,
        )

        return sklearn_tune(params, X, y, model, **fit_kwargs)
