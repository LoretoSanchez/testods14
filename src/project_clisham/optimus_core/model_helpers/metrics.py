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
import numpy as np

from sklearn.metrics._regression import _check_reg_targets, check_consistent_length


def mean_absolute_percentage_error(y_true, y_pred, sample_weight=None):
    """
    Mean absolute percentage error regression loss
    Args:
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
        sample_weight : array-like of shape (n_samples,), optional
        Sample weights.
    Returns:
        loss : float the weighted average of all output errors is returned.
    """
    _, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, None)
    check_consistent_length(y_true, y_pred, sample_weight)
    mask = y_true != 0
    result = (
        np.average(
            (np.abs(y_true - y_pred) / y_true)[mask], weights=sample_weight, axis=0
        )
        * 100
    )
    return np.average(result)
