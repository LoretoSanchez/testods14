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
Reporting of optimization results.
"""

import logging
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib

import pandas as pd
import numpy as np

from optimizer.problem import StatefulContextualOptimizationProblem
from project_clisham.pipelines.optimization.recommendation.recommendation_nodes import (
    _get_features,
)

from project_clisham.optimus_core.tag_management import TagDict

SMALL_SIZE = 12
matplotlib.rc("font", size=SMALL_SIZE)
matplotlib.rc("axes", titlesize=SMALL_SIZE)
matplotlib.rc("xtick", labelsize=SMALL_SIZE)
matplotlib.rc("ytick", labelsize=SMALL_SIZE)
matplotlib.rc("legend", fontsize=SMALL_SIZE)

logger = logging.getLogger(__name__)


def create_sensitivity_plot_data(
    params: Dict,
    td: TagDict,
    model,
    opt_df: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Generates the data to display sensitivity charts within the UI
    Args:
        params: Dict w/2 keys, a list of the unique ids defining each run (e.g.
        run_id and timestamp), and n_points, how many control values to
        plot objective values at.
        td: A tagdict object.
        model: A predictive model defining the objective.
        opt_df: Dataframe holding our optimization set.
        recommendations: List of dictionaries holding recommendations.

    Returns: Dictionary keyed by `sensitivity_plot_df`,
    holding the long-form sensitivity plot data.

    """

    recommendations = recommendations.to_dict(orient="records")
    opt_target = params["opt_target"]
    params = params["recommend_sensitivity"]
    dict_controls = td.select("tag_type", "control")

    return_list = [pd.DataFrame()] * len(recommendations)
    # Copy to avoid modifying original reference
    opt_df = opt_df.copy().set_index(params["unique_ids"])
    for rec_idx, rec in enumerate(recommendations):
        control_range_dict = {
            control: determine_control_range(td, control, params["n_points"])
            for control in rec["controls"].keys()
            if control in dict_controls
        }
        rec_return_list = []
        # .loc returns a Series, .to_frame().T converts to 1-row DF.
        opt_data_row = (
            opt_df.loc[tuple([rec[uid] for uid in params["unique_ids"]])].to_frame().T
        )

        # Set up the objective (with context) for evaluation.
        # Don't worry about repairs/penalties
        model.current_vector = opt_data_row
        problem = _create_opt_problem(opt_df, rec, model, params)
        if len(rec["controls"]) > 0:
            for control in rec["controls"].keys():
                if control in dict_controls:
                    # Create dataframe holding all features *except* control constant.
                    return_df = pd.DataFrame(
                        np.repeat(
                            opt_data_row.values,
                            control_range_dict[control].shape[0],
                            axis=0,
                        ),
                        columns=opt_data_row.columns,
                    )
                    return_df[control] = control_range_dict[control]
                    return_df["target_value"] = problem.objective(return_df)

                    return_df = (
                        return_df[["target_value", control]]
                        .rename(columns={control: "control_value"})
                        .copy()
                    )
                    return_df["control_tag_id"] = control
                    return_df["target_tag_id"] = td.select("target", opt_target)[0]
                    for uid in params["unique_ids"]:
                        return_df[uid] = rec[uid]
                    rec_return_list.append(return_df)
                    # Store the intermediate result
        if len(rec_return_list) > 0:
            return_list[rec_idx] = pd.concat(rec_return_list, axis=0)
        else:
            return_list[rec_idx] = pd.DataFrame(
                columns=[
                    "target_value",
                    "control_value",
                    "control_tag_id",
                    "target_tag_id",
                    "run_id",
                    "fecha",
                ]
            )
    return {"sensitivity_plot_df": pd.concat(return_list, axis=0, ignore_index=True)}


def _create_opt_problem(opt_df, rec, model, params):
    try:
        window_size = (
            model.steps[-1][-1]
            .params.get("data_reformatter", {})
            .get("kwargs", {})
            .get("length", 1)
        )
    except AttributeError:
        window_size = 1

    idx = int(
        np.argwhere(opt_df.reset_index().run_id.values == rec["run_id"]).flatten()
    )

    problem = StatefulContextualOptimizationProblem(
        model,
        state=opt_df.iloc[[idx], :],
        context_data=opt_df.iloc[max(idx - window_size + 1, 0) : idx, :],
        optimizable_columns=rec["controls"].keys(),
        sense="maximize",
        objective_kwargs=params["objective_kwargs"],
    )
    return problem


def determine_control_range(
    td: TagDict, control: str, n_points: int = 50
) -> np.ndarray:
    """
    Determine the range over which we should generate the
    sensitivity plot for a given recommendation.
    Heavily depends upon the ranges and constraint
    sets present in the TagDict.

    Args:
        td: TagDictionary dataframe
        control: String identifying a tag.
        n_points: Number of points to plot

    Returns: 1-dimensional np.ndarray.

    """
    tag_list = td.select("tag", lambda x: True)
    # load params from config
    if control not in tag_list:
        raise ValueError(f"{control} is not in the tag_dict")
    td_row = td.to_frame()[td.to_frame().tag == control].copy()
    # If constraint_range is specified, we should use these

    if any(td_row["constraint_set"].notnull()):
        control_range = np.asarray(pd.eval(td_row["constraint_set"]))[0].astype("int")
    # else look at op_min
    elif all(td_row["op_min"].notnull()) and all(td_row["op_max"].notnull()):
        control_range = np.arange(
            td_row["op_min"].values,
            td_row["op_max"].values,
            (td_row["op_max"] - td_row["op_min"]).values / n_points,
        )
    # Otherwise look at range_min/range_max
    elif all(td_row["range_min"].notnull()) and all(td_row["range_max"].notnull()):
        control_range = np.arange(
            td_row["range_min"].values,
            td_row["range_max"].values,
            (td_row["range_max"] - td_row["range_min"]).values / n_points,
        )
    else:
        raise ValueError(f"{control} has no defined range")

    return control_range


def generate_model_sensitivity_plots(params, td, data, model):
    """
    Generates plots to evaluate sensitivity to the controls for current model

    Args:
        dict_optim: dictionary with max and min updated
        x_test: the dataframe used for optimization
        model: the model object
    Returns:
        a dictionary of plots
    """

    col_timestamp = params["datetime_col"]
    points = params["recommend_sensitivity"]["n_points"]
    model_col = params["opt_target"]

    plots_dict = dict()
    temp = []
    current_vars, current_controls = _get_features(td, params)
    data.set_index(col_timestamp, inplace=True)

    for variable in current_vars:
        min_var, max_var = _determine_control_range(td, variable)
        if min_var is None or max_var is None:
            continue
        subtitle = td.name(variable)
        if min_var < max_var:
            cur_x_test = data[current_vars]
            var_samples = np.linspace(min_var, max_var, num=points).repeat(
                len(cur_x_test)
            )
            df_repeat_prod = pd.concat([data] * points)
            df_repeat_prod[variable] = var_samples
            df_res = pd.DataFrame(index=df_repeat_prod.index)
            df_res[variable] = var_samples
            df_res["target"] = model.predict(df_repeat_prod)
            df_res.reset_index(inplace=True)
            pivot_res = df_res.pivot(
                index=variable, columns=col_timestamp, values="target"
            )
            plot = pivot_res.plot().legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
            plt.suptitle(variable)
            plt.title(subtitle)
            plt.xlabel("Range of values")
            plt.ylabel(model_col)
            fig_sensitivity = plot.get_figure()
            plots_dict[variable + "_sensitivity.png"] = fig_sensitivity
            plt.close()
    return plots_dict


def _determine_control_range(td: TagDict, tag_name: str) -> np.ndarray:
    """
    Determine the range over which we should generate the
    sensitivity plot for a given recommendation.

    Args:
        td: TagDictionary dataframe
        control: String identifying a tag.
        n_points: Number of points to plot

    Returns: 2 values

    """
    tag_list = td.select("tag", lambda x: True)
    # load params from config
    if tag_name not in tag_list:
        raise ValueError(f"{tag_name} is not in the tag_dict")
    td_row = td.to_frame()[td.to_frame().tag == tag_name].copy()

    # else look at op_min
    if all(td_row["op_min"].notnull()) and all(td_row["op_max"].notnull()):
        min_val = td_row["op_min"].values
        max_val = td_row["op_max"].values
    # Otherwise look at range_min/range_max
    else:
        min_val = td_row["range_min"].values
        max_val = td_row["range_max"].values

    return min_val, max_val
