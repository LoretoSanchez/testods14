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
Tests that the single-row optimization function works as expected.
"""
from datetime import datetime

import pandas as pd
import numpy as np
import pytest
from numpy.testing import assert_allclose

from optimizer import penalty
from optimizer.problem import StatefulOptimizationProblem
from optimizer.solvers import DifferentialEvolutionSolver
from optimizer.testing.problems import Rastrigin
from project_clisham.pipelines.optimization.recommendation.recommendation_nodes import (
    optimize,
    get_solver_bounds,
)
from project_clisham.optimus_core.tag_management import TagDict


class RastriginModel:
    """
    Model formulation of the rastrigin function.
    Enforces given dimension.
    """

    def __init__(self, dim=5):
        self.dim = dim
        self.rastrigin = Rastrigin()

    def predict(self, x):
        pred_cols = [
            c for c in x.columns if c.startswith("state") or c.startswith("ctrl")
        ]
        assert len(pred_cols) == self.dim
        return self.rastrigin(x[pred_cols].values)


@pytest.fixture
def data():
    """ fixture for everything df -related """
    row = pd.DataFrame(
        index=[0],
        data={
            "run_id": "test-run-id",
            "timestamp": datetime(2000, 1, 1),
            "state1": 0,
            "state2": 0,
            "ctrl1": 500,
            "ctrl2": 500,
            "ctrl3": 510,
            "ctrl3": 510,
        },
    )
    return dict(
        timestamp=row["timestamp"][0],
        row=row,
        ui_states=["state1"],  # don't show state 2
        controls=[c for c in list(row.columns) if c.startswith("ctrl")],
        target="output1",
        on_controls=[c for c in list(row.columns) if c.startswith("ctrl")],
    )


@pytest.fixture
def td():
    """ A complete dataframe which can be converted into a TagDict """
    df = pd.DataFrame(
        {
            "tag": ["state1", "state2", "ctrl1", "ctrl2", "ctrl3", "output1"],
            "tag_type": ["input", "state", "control", "control", "control", "output"],
            "op_min": [0, -100, 500, 475, 475, 70],
            "op_max": [0, 100, 501, 525, 525, 75],
            "max_delta": [None, None, None, 10, 10, None],
            "model_feature": [True, None, True, True, True, None],
            "model_target": [None, None, None, None, None, True],
        }
    )

    return TagDict(df, validate=False)


def penalty_list():
    """ penalty fixture """
    penalty_list = [
        penalty(lambda x: np.sum(x, axis=1), "<=", 10, name="row_sum<=10"),
        penalty(lambda x: np.sum(x, axis=1), ">=", 1, name="row_sum>=1"),
    ]
    return penalty_list


@pytest.fixture
def problem_with_penalty(data):
    """ problem fixture """
    model = RastriginModel(dim=5)
    problem = StatefulOptimizationProblem(
        model,
        state=data["row"],
        optimizable_columns=data["controls"],
        penalties=penalty_list(),
    )
    return problem


@pytest.fixture
def problem(data):
    """ problem fixture """
    model = RastriginModel(dim=5)
    problem = StatefulOptimizationProblem(
        model,
        state=data["row"],
        optimizable_columns=data["controls"],
    )
    return problem


@pytest.fixture
def solver(data):
    """ solver fixture """
    return DifferentialEvolutionSolver(
        bounds=Rastrigin.bounds(n=len(data["controls"])), seed=0
    )


def test_optimize(data, problem, solver):
    """
    Test expected function of optimize routine
    """
    row = data["row"]

    result, _, _, _ = optimize(
        problem=problem, solver=solver, stopper=None, model_dict=None, **data
    )

    assert result["fecha"] == str(data["timestamp"])
    assert list(result["state"].keys()) == ["state1"]
    assert_allclose([result["state"]["state1"]], row["state1"])

    ctrl_cols = data["controls"]
    expected_current = row[ctrl_cols].iloc[0]
    got_current = [result["controls"][c]["current"] for c in ctrl_cols]
    assert_allclose(got_current, expected_current)

    expected_opt = [0] * len(ctrl_cols)
    got_opt = [result["controls"][c]["suggested"] for c in ctrl_cols]
    assert_allclose(expected_opt, got_opt, atol=1e-07)

    exp_pred = problem.objective(row)
    got_pred = result["outputs"]["target_pred_current"]
    assert_allclose(exp_pred, got_pred)

    exp_opt = 0
    got_opt = result["outputs"]["target_pred_optimized"]
    assert_allclose(exp_opt, got_opt, atol=1e-07)


def test_constraints_info(data, problem_with_penalty, solver):
    """
    Test expected function of optimize routine for penalty and slack
    """
    result, _, _, _ = optimize(
        problem=problem_with_penalty,
        solver=solver,
        stopper=None,
        model_dict=None,
        **data
    )

    exp_penalty = 0
    got_penalty = result["penalties"]["row_sum<=10_penalty"]["suggested"]
    assert_allclose(exp_penalty, got_penalty)

    exp_slack = np.NaN
    got_slack = result["slack"]["row_sum>=1_slack"]["suggested"]
    assert_allclose(exp_slack, got_slack)


def test_bounded_solver(data, td):
    row = data["row"]
    result = get_solver_bounds(row, td, data["controls"])
    assert result == [(500, 501), (490.0, 510.0), (500.0, 520.0)]
