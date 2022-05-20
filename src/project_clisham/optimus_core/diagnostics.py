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
Problem set-up diagnostics functions
"""

from copy import copy
from typing import List, Tuple

import numpy as np
import pandas as pd

from optimizer.constraint import InequalityConstraint
from optimizer.problem.problem import OptimizationProblem


def evaluate_on_historic_data(
    data: pd.DataFrame, problem: OptimizationProblem, bounds: List[Tuple]
) -> pd.DataFrame:
    """
    Evaluates the problem statement on historical data. This is
    helpful to sanity check the constraints and problem set-up.
    Args:
        data: input data
        problem: Optimization Problem
        bounds: List of bounds for each variables in the input data. It is expected
        that the bounds are ordered tuples (i.e. (min, max) and is provided for
        each column in the input data.)

    Returns:
        A dataframe with input data along with the evaluated penalties,
        slack, bounds check and objective values. If the inequality constraint
        is violated, the slack is NaN.

    Raises:
        ValueError: if number of columns in data does not match
        the number of bounds

    """
    if len(data.columns) != len(bounds):
        raise ValueError("Bounds list length do not match the number of columns")

    return _evaluate_on_historic_data(data, problem, bounds)


def _evaluate_on_historic_data(
    data: pd.DataFrame, problem: OptimizationProblem, bounds: List[Tuple]
) -> pd.DataFrame:
    input_data = copy(data)

    # Bounds
    check_bounds = _check_bounds(bounds, input_data)

    # Penalties
    penalty_list = problem.penalties
    penalty_table = get_penalties_table(input_data, penalty_list)

    # Objective Value
    objective_values = _get_objective_value(input_data, problem)

    # Slack Value
    slack_table = get_slack_table(input_data, penalty_list)

    return (
        input_data.join(check_bounds, how="left")
        .join(penalty_table, how="left")
        .join(slack_table, how="left")
        .join(objective_values, how="left")
    )


def _get_objective_value(data: pd.DataFrame, problem_stmt: OptimizationProblem):
    """
    Computes the objective value with and without penalty.
    Here, we pass the inputs to the "objective" function of the problem class, so that
    the states are not replaced to the user provided values.
    Similarly, the "apply_penalty" function is used to account for both maximization
    and minimization use-case.
    """

    objective_value = pd.Series(
        problem_stmt.objective(data), name="objective_value", index=data.index
    )

    # Note that we are adding penalty separately to the objective value rather
    # than calling problem_stmt(data), so that the states are historical values
    # from the data and not those provided by user
    objective_value_with_penalty = pd.Series(
        objective_value + problem_stmt.apply_penalties(data),
        name="objective_value_with_penalty",
    )

    return pd.concat([objective_value, objective_value_with_penalty], axis=1)


def _check_bounds(bounds: List[Tuple], data: pd.DataFrame):
    """
    Checks if each column is within the specified bounds. It is expected
    that the bounds are ordered tuples (i.e. (min, max) and provided for each column
    in the dataset and in the same order as the dataset.)
    Returns a dataframe with boolean flags.
    """
    bounds_stack = pd.DataFrame(
        np.stack(bounds), columns=["min", "max"], index=data.columns
    )
    bounds_table = pd.DataFrame(
        np.stack(
            [
                (bounds_stack.loc[col, "min"] <= data[col])
                & (data[col] <= bounds_stack.loc[col, "max"])
                for col in data.columns
            ],
            axis=1,
        ),
        columns=["within_bounds_" + col_name for col_name in data.columns],
        index=data.index,
    )
    return bounds_table


def get_penalties_table(solutions: pd.DataFrame, penalties: List) -> pd.DataFrame:
    """
    Computes absolute values for all penalties on the given data.
    Returns a dataframe with calculated penalties.

    Args:
        solutions: Dataframe with state and controls
        penalties: List of penalty constraints

    Returns:
        Dataframe with penalties for each constraint

    """
    if not penalties:
        return pd.DataFrame()
    penalty_matrix = np.stack([p(solutions) for p in penalties], axis=1)
    penalty_check = pd.DataFrame(
        penalty_matrix, columns=[p.name for p in penalties], index=solutions.index
    )
    return penalty_check


def get_slack_table(solutions: pd.DataFrame, penalties: List) -> pd.DataFrame:
    """
    Compute slack (distance between function and boundary) for the
    InequalityConstraints in the penalties list.
    The value denotes the amount of slack left before the constraint is violated.
    Returns a dataframe with slack values

    Args:
        solutions: Dataframe with state and controls
        penalties: List of penalty constraints

    Returns:
        Dataframe with slack for each inequality penalty constraint

    """
    if not penalties:
        return pd.DataFrame()

    inequality_penalties = [
        p for p in penalties if isinstance(p.constraint, InequalityConstraint)
    ]

    if not inequality_penalties:
        return pd.DataFrame()

    slack_matrix = np.stack(
        [p.constraint(solutions) for p in inequality_penalties], axis=1
    )
    slack_matrix = np.where(slack_matrix > 0, np.NaN, slack_matrix * -1)
    slack_table = pd.DataFrame(
        slack_matrix,
        columns=[p.name.replace("_penalty", "_slack") for p in inequality_penalties],
        index=solutions.index,
    )
    return slack_table
