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
Core nodes performing the optimization.
"""
import logging
from ast import literal_eval
from copy import deepcopy
from datetime import datetime
from multiprocessing import Pool
from typing import Any, List, Tuple, Dict

import pandas as pd
import numpy as np
from tqdm import tqdm

from optimizer.constraint import Repair, repair
from optimizer.problem import (
    OptimizationProblem,
    StatefulOptimizationProblem,
    StatefulContextualOptimizationProblem,
)
from optimizer.solvers import Solver
from optimizer.stoppers.base import BaseStopper
from optimizer.loggers import BestTrajectoryLogger

from project_clisham.optimus_core.tag_management import TagDict
from project_clisham.optimus_core.utils import load_obj
from project_clisham.optimus_core.diagnostics import (
    get_penalties_table,
    get_slack_table,
)

import math

logger = logging.getLogger(__name__)


def optimize(  # pylint:disable=too-many-locals
    timestamp: datetime,
    row: pd.DataFrame,
    ui_states: List[str],
    controls: List[str],
    on_controls: List[str],
    target: str,
    problem: OptimizationProblem,
    solver: Solver,
    stopper: BaseStopper,
    model_dict: Dict,
    obj_model=None,
) -> Tuple:
    """
    Optimizes a single row and returns the expected json object.
    Args:
        timestamp: timestamp key
        row: row of data to score
        ui_states: list of state columns to include in the results
        controls: list of controllable features
        on_controls: list of controllable features that are on
        target: target to optimize objective function
        problem: optimization problem
        solver: solver
        stopper: stopper
    Returns:
        Tuple of JSON results
    """

    # Parameter to print what happens with control variables:
    print_info_controls = True
    if print_info_controls:
        logger.info(f"\n\n\tGetting inside 'optimize'...\n\n")

    # score with current and with optimal controls
    scores = pd.concat([row] * 2, ignore_index=True)
    scores.index = ["curr", "opt"]
    best_trajectory_logger = BestTrajectoryLogger()

    if solver:
        stop_condition = False
        while not stop_condition:
            parameters = solver.ask()
            obj_vals, parameters = problem(parameters)
            solver.tell(parameters, obj_vals)
            stop_condition = solver.stop()
            best_trajectory_logger.log(solver)
            if stopper:
                stopper.update(solver)
                stop_condition |= stopper.stop()

        best_controls, _ = solver.best()
        scores.loc["opt", on_controls] = best_controls
    # note the subtle difference between re-scoring with the objective
    # which translates to a model.predict vs calling the problem, directly
    # which includes constraint penalties
    # Plots
    traj_plot_info = {}
    if solver != None:
        traj_plot_info = {
            "fecha": str(timestamp),
            "info": {
                "trajectory_logger": best_trajectory_logger,
                "optim_problem": problem,
                "bounds": solver.bounds,
            },
        }

    # Update controls to recommend
    if (obj_model is not None) & (hasattr(obj_model, "tags_recommend")):
        if (model_dict is not None) & (len(obj_model.tags_recommend) > 0):
            off_controls_set = set(controls).difference(set(on_controls))
            tags_recom = list(
                set(obj_model.tags_recommend).difference(off_controls_set)
            )
            tags_control = list(set(controls).intersection(tags_recom))
            tags_model = list(set(row.columns).intersection(tags_recom))
            controls = list(set(tags_control + tags_model))
    # Update model predictions
    if (model_dict is not None) & (obj_model is not None):
        # Gets total tph and rec and updates of reaction variables
        scores, model_dict, final_update = _update_model_dict_predictions(
            scores, model_dict, obj_model
        )

        # updates in scores the reaction variables
        scores = _update_scores(scores, final_update)

        # updates the prediction for single models
        scores = _get_model_predictions(scores, model_dict)

    # the predict of the model object has the option to not update the values
    scores.loc["curr", "_pred"] = obj_model.predict(
        scores.loc[["curr"]], update_values=False
    ).values[0]
    scores.loc["opt", "_pred"] = problem.objective(scores.loc[["opt"]]).values[0]

    # Saving original curr and opt estimations:
    original_opt_pred = scores.loc["opt", "_pred"]
    original_curr_pred = scores.loc["curr", "_pred"]
    original_controls = controls.copy()

    # Validate optimized > predicted, do not recommend
    if scores.loc["opt", "_pred"] < scores.loc["curr", "_pred"]:
        controls = []
        scores.loc["opt", "_pred"] = scores.loc["curr", "_pred"]

        if print_info_controls:
            logger.info(
                f"\n\n\t\tAll controls have been dropped because opt < curr when MAXIMIZING.\n"
                f"\t\t\tscores.loc['opt', '_pred'] < scores.loc['curr', '_pred']?\n"
                f"\t\t\t\t{original_opt_pred} < {original_curr_pred}?\n"
                f"\t\t\t\t\tObjective value has been updated as the current {original_curr_pred}.\n"
                f"\t\t\t\t\t\tcontrols that has been discarded = {original_controls}.\n"
            )

    penalty_slack_table = pd.DataFrame()
    if hasattr(problem, "penalties"):
        penalty_slack_table = _get_penalty_slack(scores, problem.penalties)

    # Validating if the lines are off to set opt = curr and the recommendation will not
    # be taken into account:
    #if (obj_model is not None) & (
    #    hasattr(obj_model, "apply_on_off_to_tph_recommendations")
    #):
    #    scores = obj_model.apply_on_off_to_tph_recommendations(scores)

    # Validating if there's chance to open more some lime valves, in case the pH
    # tag is going to be recommended:
    if (obj_model is not None) & (
        hasattr(obj_model, "recommend_ph_chance_open_lime_valves")
    ):
        scores = obj_model.recommend_ph_chance_open_lime_valves(scores, controls)

    # validating the conditions to recommend TPH (SAG) when the high limit is open
    # tag is going to be recommended:
    if (obj_model is not None) & (hasattr(obj_model, "not_recommend_tph_SAG_in_range")
    ):
        if obj_model.model_tph_name == "ma2":
            scores = obj_model.not_recommend_tph_SAG_in_range(scores)

    # Validating if the difference between pHs in plant A0 is less than a given
    # parameter to recommend in both lines the mean value:
    if (obj_model is not None) & (
        hasattr(obj_model, "correct_pHs_recommended_plant")
    ):
        scores = obj_model.correct_pHs_recommended_plant(scores, controls)
    
    if (obj_model is not None) & (
        hasattr(obj_model, "not_show_tph_recs_if_tolva_conc_low")
    ):
        controls = obj_model.not_show_tph_recs_if_tolva_conc_low(scores, controls)

    # Validating if the recommended value for Profit MV 'Molienda' tags are
    # not recommended:
    # TODO Evaluate some analyses with JulioG to check if this filter is going
    # to be held in some future.
    # if (obj_model is not None) & (hasattr(obj_model, "recommend_profit_mv_out_of_bounds")):
    #    scores = obj_model.recommend_profit_mv_out_of_bounds(scores, controls)

    # Validating that 'obj_model' has 'params' attribute:
    if (obj_model is not None) & (hasattr(obj_model, "params")):
        # Bringing the 'min percentage change' parameter to filter which
        # controls to recommend here:
        mpc_param = obj_model.params["recommend"]["translator_layer"][
            "min_percentage_change"
        ]

    # Getting the tag_dict_master with desired columns:
    tdm = obj_model.td.to_frame()[["tag", "min_delta","round_val"]]
    tdm.replace({np.nan:None}, inplace = True)
    # filter controls with no change
    new_controls = []
    for ctrl in controls:
        r = tdm[tdm["tag"] == ctrl]["round_val"].values[0]
        r = r if r!=None else 3
        scores.loc["opt", ctrl] = round(float(scores.loc["opt", ctrl]), int(r))
        
        cur = round(float(scores.loc["curr", ctrl]),3)
        sug = scores.loc["opt", ctrl]
        cambio = abs(cur - sug)
        perc = cur * mpc_param

        # Getting the min_delta from tdm:
        min_delta_related = tdm[tdm["tag"] == ctrl]["min_delta"].values[0]
        
        # Validating if min_delta is set for this ctrl tag:
        if pd.isna(min_delta_related):
            # Validating change with the default parameter:
            if (cur != sug) and cur == cur and sug == sug and cambio > perc:
                new_controls.append(ctrl)
            else:
                if print_info_controls:
                    logger.info(
                        f"\n\n\t\tControl tag '{ctrl} has been rejected from recommendations because small change.'\n"
                        f"\t\t\tcambio > perc? --> {cambio} > {perc}?\n"
                    )
        else:
            # Threshold to ensure the max value between pmc and the given by dictionary:
            threshold_mn = max(perc, min_delta_related)

            # Validating change with the value coming from dictionary:
            if (cur != sug) and cur == cur and sug == sug and cambio > threshold_mn:
                new_controls.append(ctrl)
            else:
                if print_info_controls:
                    logger.info(
                        f"\n\n\t\tControl tag '{ctrl}' has been rejected from recommendations because small change.\n"
                        f"\t\t\tcambio > threshold_mn? --> {cambio} > {threshold_mn}?\n"
                    )
        
    controls = new_controls

    # if there are no recommendations predicted and optimized are the same
    if len(controls) == 0:
        scores.loc["opt", "_pred"] = scores.loc["curr", "_pred"]
        for model in model_dict:
            scores.loc["opt", model + "_pred"] = scores.loc["curr", model + "_pred"]

    # assemble result
    states_ = {state: float(row[state].values[0]) for state in ui_states}
    tags = row.columns.to_list().copy()
    tags.remove("run_id")
    tags.remove("fecha")
    states_updated_ = {
        state_updated: float(scores.loc["opt", state_updated])
        for state_updated in tags
        if round(float(scores.loc["curr", state_updated]), 3)
        != round(float(scores.loc["opt", state_updated]), 3)
    }
    context_ = {state: float(row[state].values[0]) for state in tags}

    # Checking if Malla is over the max value for skipping tph recommendations
    # We define that we change this after the report
    if (obj_model is not None) & (hasattr(obj_model, "constraint_malla_range_over_tph")
    ):
        _controls_to_recommend = obj_model.constraint_malla_range_over_tph(scores, controls)
    else:
        _controls_to_recommend = controls
    controls_ = {
        ctrl: {
            "current": float(scores.loc["curr", ctrl]),
            "suggested": float(scores.loc["opt", ctrl]),
            "delta": float(scores.loc["opt", ctrl] - scores.loc["curr", ctrl]),
        }
        for ctrl in _controls_to_recommend
    }
    models_ = {}
    if model_dict is not None:
        models_ = {
            model: {
                f"{model}_current": float(row[model]),
                f"{model}_predicted": float(scores.loc["curr", model + "_pred"]),
                f"{model}_optimized": float(scores.loc["opt", model + "_pred"]),
                f"{model}_delta": float(
                    scores.loc["opt", model + "_pred"]
                    - scores.loc["curr", model + "_pred"]
                ),
            }
            for model in model_dict
        }
    outputs_ = {
        f"target_{target}_current": float(row[target])
        if target in row.columns
        else np.nan,
        "target_pred_current": float(scores.loc["curr", "_pred"]),
        "target_pred_optimized": float(scores.loc["opt", "_pred"]),
    }

    penalty_, slack_ = {}, {}
    if len(penalty_slack_table) > 0:
        penalty_, slack_ = _get_penalty_slack_results(penalty_slack_table)

    uplift_report_dict = {
        "run_id": row["run_id"].values[0],
        "fecha": str(timestamp),
        "context": context_,
        "state": states_,
        "state_updated": states_updated_,
        "controls": controls_,
        "outputs": outputs_,
        "models": models_,
        "penalties": penalty_,
        "slack": slack_,
    }

    control_recommendations = []
    for control in on_controls:
        control_recommendations.append(
            {
                "tag_id": control,
                "run_id": row["run_id"].values[0],
                "value": scores.loc["opt", control],
            }
        )

    output_recommendation = {
        "run_id": row["run_id"].values[0],
        "tag_id": target,
        "baseline": float(scores.loc["curr", "_pred"]),
        "optimized": float(scores.loc["opt", "_pred"]),
    }
    return (
        uplift_report_dict,
        control_recommendations,
        output_recommendation,
        traj_plot_info,
    )


def _optimize_dict(kwargs):
    return optimize(**kwargs)


def _get_penalty_slack(solutions: pd.DataFrame, penalties: List) -> pd.DataFrame:
    """
    Get the penalties and slacks for all constraints. Slack is calculated
    only for inequality constraints. Returns a dataframe.
    """
    if not penalties:
        return pd.DataFrame(index=solutions.index)
    penalty_table = get_penalties_table(solutions, penalties)
    slack_table = get_slack_table(solutions, penalties)
    return penalty_table.join(slack_table, how="left")


def _get_penalty_slack_results(penalty_slack_table):
    penalty_ = {
        penalty_column: {
            "current": float(penalty_slack_table.loc["curr", penalty_column]),
            "suggested": float(penalty_slack_table.loc["opt", penalty_column]),
        }
        for penalty_column in penalty_slack_table.columns
        if "_penalty" in penalty_column
    }
    slack_ = {
        slack_column: {
            "current": float(penalty_slack_table.loc["curr", slack_column]),
            "suggested": float(penalty_slack_table.loc["opt", slack_column]),
        }
        for slack_column in penalty_slack_table.columns
        if "_slack" in slack_column
    }
    return penalty_, slack_


def _get_model_predictions(scores: pd.DataFrame, model_dict):
    for key in model_dict:
        if "model_obj" in model_dict[key]:
            cur_model = model_dict[key]["model_obj"]
            range_v = model_dict[key]["range"]
            cond = (scores[key]["curr"] > range_v[0]) & (
                scores[key]["curr"] < range_v[1]
            )
            if cond:
                pred = cur_model.predict(scores)
                scores[key + "_pred"] = pred
                # TODO: HOTFIX- manualmente forzando que la recomendacion de TPH sea mayor
                if float(scores.loc["curr", key]) < pred[1]:
                    scores.loc["opt", key] = pred[1]
                else:
                    scores.loc["opt", key] = float(scores.loc["curr", key])
            else:
                scores[key + "_pred"] = float(scores.loc["curr", key])

    return scores


def _update_model_dict_predictions(scores: pd.DataFrame, model_dict, obj_model):
    final_update = scores
    if obj_model and obj_model.cuf:
        # update the prediction of the optimized vector updating the values
        tph, update_scores, tph_var = obj_model.predict_tph(scores.loc[["opt"]])
        rec, final_update, rec_var = obj_model.predict_rec(update_scores)
        # Update opt values
        scores.loc[["opt"]] = final_update
        scores.loc["opt", tph_var + "_pred"] = tph.values[0]
        scores.loc["opt", rec_var + "_pred"] = rec.values[0]
        model_dict[tph_var] = []
        model_dict[rec_var] = []
        # update the prediction of the current vector without updating vectors
        tph_c, _, tph_var = obj_model.predict_tph(scores.loc[["curr"]], False)
        rec_c, _, rec_var = obj_model.predict_rec(scores.loc[["curr"]], False)
        scores.loc["curr", tph_var + "_pred"] = tph_c.values[0]
        scores.loc["curr", rec_var + "_pred"] = rec_c.values[0]
    return scores, model_dict, final_update


def _update_scores(scores: pd.DataFrame, final_update: pd.DataFrame):
    for col in final_update.columns:
        if scores.loc["opt", col] != final_update.loc["opt", col]:
            scores.loc["opt", col] = final_update.loc["opt", col]
    return scores


def _get_features(td: TagDict, 
    params: dict,
    ) -> Tuple[List[str], List[str]]:
    """ return features and optimizable features """
    feat_columns = params["model_features"]
    features = list(set([tag for col in feat_columns for tag in td.select(col)]))
    all_controls = td.select("tag_type", "control")
    controls = [f for f in features if f in all_controls]

    return features, controls


def _get_target(td: TagDict, params: dict) -> str:
    """ return target feature """
    opt_target = params["opt_target"]
    target = td.select("target", opt_target)[0]
    return target


def _get_on_features(
    current_value: pd.DataFrame, 
    td: TagDict, 
    controls: List[str],
    print_info_controls: bool,
) -> List[str]:
    """
    Return controls that are on by checking current state.
    """
    on_controls = []
    for feature in controls:
        on_flag = all(
            [current_value[d].iloc[0] > 0.5 for d in td.dependencies(feature)]
        )
        if on_flag:
            if np.isnan(current_value[feature].values[0]):
                logger.warning(f"Current Value for feature {feature} is NaN.")
                if print_info_controls:
                    dict_2_print = dict(zip([d for d in td.dependencies(feature)], 
                                            [current_value[d].iloc[0] for d in td.dependencies(feature)]))
                    logger.info(f"\n\n\t\tFeature '{feature}' is considered OFF @_get_on_features because its value is NaN.\n"
                    f"\t\t\tDependencies = '{dict_2_print}.'\n"
                    )
            else:
                on_controls.append(feature)
        else:
            logger.warning(f"Current feature {feature} is OFF.")
            if print_info_controls:
                dict_2_print = dict(zip([d for d in td.dependencies(feature)], 
                                            [current_value[d].iloc[0] for d in td.dependencies(feature)]))
                logger.info(f"\n\n\t\tFeature '{feature}' is considered OFF @_get_on_features.\n"
                f"\t\t\tDependencies = '{dict_2_print}.'\n"
                )
    return on_controls


def _get_in_range_features(
    current_value: pd.DataFrame, td: TagDict, on_controls: List[str],
    print_info_controls: bool,
) -> List[str]:
    """Return controls that are within their specified bounds."""

    in_range_controls = []
    for feature in on_controls:
        control_entry = td[feature]
        op_min = control_entry["op_min"]
        op_max = control_entry["op_max"]
        max_delta = control_entry["max_delta"]
        max_delta_down = control_entry["max_delta_down"]
        if np.isnan(current_value[feature].iloc[0]):
            logger.warning(f"Current Value for feature {feature} is NaN.")
            if print_info_controls:
                logger.info(f"\n\n\t\tFeature '{feature}' will not be control @_get_in_range_features by some NaN Issue.\n"
                )
        else:
            in_range_flag = (
                (op_min - max_delta_down)
                <= current_value[feature].iloc[0]
                <= (op_max + max_delta)
            )
            if in_range_flag:
                in_range_controls.append(feature)
            else:
                logger.warning(f"Current feature {feature} is out of range.")
                if print_info_controls:
                    logger.info(f"\n\n\t\tFeature '{feature}' will not be control @_get_in_range_features because is out-of-range.\n"
                                f"\t\t\t(op_min - max_delta) <= current <= (op_max + max_delta)?\n"
                                f"\t\t\t\t({op_min} - {max_delta}) <= {current_value[feature].iloc[0]} <= ({op_max} + {max_delta})?\n"
                                f"\t\t\t\t\t{op_min - max_delta} <= {current_value[feature].iloc[0]} <= {op_max + max_delta}?\n"
                )

    return in_range_controls


def _get_in_Profit_range_Molienda_Profit_MV_features(
    current_value: pd.DataFrame,
    td: TagDict,
    in_range_controls: List[str],
    dict_tags_limits: dict(),
) -> List[str]:
    """Return a subset of in_range_controls that."""

    # Initializing the function name:
    funct_name = "_get_in_Profit_range_Molienda_Profit_MV_features"

    # Initializing the in_Profit_range_Tags:
    in_Profit_range_Tags = []

    # Initializing the list of tags turned off:
    list_tags_turned_off = []

    # Filling it with all the in_range_controls that are in Profit limits:
    for inRngCntrl in in_range_controls:
        # Starting the boolean value:
        inProfitRange = True

        # Validating if it is in dict_tags_limits:
        if inRngCntrl in dict_tags_limits:
            # Getting the limits and current value of the specific tags
            low_lim = current_value[dict_tags_limits[inRngCntrl][0]].values[0]
            hig_lim = current_value[dict_tags_limits[inRngCntrl][1]].values[0]
            profit_mv_tag_value = current_value[inRngCntrl].values[0]

            # Validating that they are numerical:
            if (
                (np.isnan(low_lim))
                or (np.isnan(hig_lim))
                or (np.isnan(profit_mv_tag_value))
            ):
                # ################################### #
                # CASE null in actual value or limits #
                # ################################### #

                # What to do? Option: Filter that tag:
                inProfitRange = False

                # Saving this information in the list of tags dropped:
                list_tags_turned_off.append(inRngCntrl)

                # Printing information:
                print(f"NAN in Molienda Profit MV or limit TAGs @{funct_name}.")
                print(
                    f"low limit = {low_lim} | actual value = {profit_mv_tag_value} | high limit = {hig_lim} @{funct_name}.\n"
                )

            else:
                # ################################## #
                # CASE Profit MV Tag in Profit Range #
                # ################################## #

                # Validating if the current value lies within that range:
                if low_lim <= profit_mv_tag_value and profit_mv_tag_value <= hig_lim:
                    inProfitRange = True

        # Saving the tag if it is worthy:
        if inProfitRange:
            in_Profit_range_Tags.append(inRngCntrl)

    # Printing the resume of this method:
    if len(list_tags_turned_off) > 0:
        print(
            f"{len(list_tags_turned_off)} Profit MV Molienda TAGs dropped @{funct_name}"
        )
        print(list_tags_turned_off)
        print("\n\n")

    # Returning the filtered list:
    return in_range_controls  # in_Profit_range_Tags


def _calibrated_pHs(
    current_value: pd.DataFrame,
    in_range_controls: List[str],
    validate_calibrated_phs_sensors: dict,
    print_info_controls: bool,
) -> List[str]:
    """Return those in_range_controls filtering pHs if not calibrated.

    The idea here is to check the difference between the pH tags in the
    'current_value' dataframe is lower-equal than a parameter. If the
    difference is small, then the control tag proceeds to be recommended,
    if not, those tags will be taken from 'in_range_controls'.
    """

    # Getting the function name:
    funct_name = "_calibrated_pHs"

    # Getting the list of tags to test on, and the threshold to test:
    list_tags = validate_calibrated_phs_sensors["tags"]
    lt_threshold = validate_calibrated_phs_sensors["lt_threshold_clbrtd"]

    # Initiliazing the output_list as the input, and if the conditions apply,
    # it will be updated:
    output_list = in_range_controls.copy()

    # Validating if, for this plant being optimized, there is information in the
    # 'validate_calibrated_phs_sensors' dictionary:
    if (lt_threshold > 0) and (len(list_tags) == 2):
        # The threshold is no-negative and the list has exactly two entries:
        # There's information to make the test:

        # Getting the first and the second sensor:
        sensor_1 = current_value[list_tags[0]].values[0]
        sensor_2 = current_value[list_tags[1]].values[0]

        # Checking if they are both numerical:
        if (not np.isnan(sensor_1)) and (not np.isnan(sensor_2)):
            # Validating if the sensor is calibrated:
            if abs(sensor_1 - sensor_2) > lt_threshold:
                # ############################ #
                # CASE: Sensors not calibrated #
                # ############################ #

                # Not going to recommend any:
                output_list = list(set(in_range_controls).difference(set(list_tags)))

                if print_info_controls:
                    logger.info(f"\n\n\t\tThe tags ('{list_tags[0]}', '{list_tags[1]}') has been removed from control list @{funct_name}.\n"
                                f"\t\t\tThey are not calibrated (difference beyond {lt_threshold} pH units).\n"
                )
                
            else:
                # ######################## #
                # CASE: Sensors calibrated #
                # ######################## #

                # Giving the chance to optimize them:
                output_list = in_range_controls

        elif (not np.isnan(sensor_1)) and (np.isnan(sensor_2)):
            # ###################################### #
            # CASE: Only the first sensor is working #
            # ###################################### #

            # Only chance to recommend Sensor 1, dropping the second:
            output_list = list(set(in_range_controls).difference(set([list_tags[1]])))

            if print_info_controls:
                    logger.info(f"\n\n\t\tThe tag '{list_tags[1]}' has been removed from control list @{funct_name}, because it is not working.\n"
                                f"\t\t\tTag '{list_tags[1]}' value = {sensor_2}.\n"
                )

        elif (np.isnan(sensor_1)) and (not np.isnan(sensor_2)):
            # ####################################### #
            # CASE: Only the second sensor is working #
            # ####################################### #

            # Only chance to recommend Sensor 2, dropping the first:
            output_list = list(set(in_range_controls).difference(set([list_tags[0]])))

            if print_info_controls:
                    logger.info(f"\n\n\t\tThe tag '{list_tags[0]}' has been removed from control list @{funct_name}, because it is not working.\n"
                                f"\t\t\tTag '{list_tags[0]}' value = {sensor_1}.\n"
                )

        else:
            # ##################################### #
            # CASE: The sensors have no information #
            # ##################################### #

            # Not going to recommend any:
            output_list = list(set(in_range_controls).difference(set(list_tags)))

            if print_info_controls:
                    logger.info(f"\n\n\t\tThe tags ('{list_tags[0]}', '{list_tags[1]}') has been removed from control list @{funct_name}.\n"
                                f"\t\t\tThey are not working. Values for tags ('{list_tags[0]}','{list_tags[1]}') = ({sensor_1}, {sensor_2}).\n"
                )

    # Returning the output:
    return output_list


def make_solver(params: dict, bounds: Tuple) -> Solver:
    """
    Creates an ask-tell solver from the tag dict

    Args:
        params: dict of pipeline parameters
        bounds: Tuple of Bounds of Solver
    Returns:
        optimization solver object
    """
    solver_class = load_obj(params["solver"]["class"])
    solver_kwargs = params["solver"]["kwargs"]

    solver_kwargs.update({"bounds": bounds})

    return solver_class(**solver_kwargs)


def get_solver_bounds(
    current_value: pd.DataFrame,
    td: TagDict,
    controls: List[str],
    ramp_up_sags16_17_velocities: pd.DataFrame,
    revestimiento: pd.DataFrame,
    ph_standard_data: pd.DataFrame,
    params: dict,
    print_info_controls: bool) -> Tuple[List[Tuple], List[str]]:
    """
    Add more appropriate bounds to controls, applying max_deltas
    if available.
    For specific tags, this function will set the original op ranges
    if there is no intersection between them and the max delta variation.
    Args:
        current_value: DataFrame of current value for optimization.
        td: tag dictionary.
        controls: List of strings indicating controls to optimize.
        ramp_up_sags16_17_velocities: DataFrame with velocities rampup info.
        revestimiento: DataFrame with revestimientos rampup info.
        ph_standard_data: DataFrame with ph standard info.
        params: params dict.
        print_info_controls: Boolean to print information about controls.
    Returns:
        solver_bounds: list with updated min and max op ranges.
        final_controls: list with updated controls list.
    """
    # Getting special tags for skip possibles breaking
    # optimization process
    final_controls = controls.copy()
    no_intersection_tags = [] if not params.get('no_intersection_tags') else params.get('no_intersection_tags')
    control_to_context_tags = [] if not params.get('control_to_context_tags') else params.get('control_to_context_tags')
    solver_bounds = []
    for control in controls:
        control_entry = td[control]
        op_min = control_entry["op_min"]
        op_max = control_entry["op_max"]

        # Updating op_min and op_max values according to the
        # velocity of SAG Mills RAMP UP Standard:
        # Hotfix for missing historic tph
        
        op_min, op_max = update_mill_vels_sag16_17_standard(
           current_value, ramp_up_sags16_17_velocities, revestimiento,
           control, op_min, op_max, params, print_info_controls
        )

        # Updating op_min and op_max values according to the
        # presion de los descansos of SAG Mills RAMP UP Standard:
        # Commented for Cumsum TPH with errors 15-07-2021
        # TODO: Deprecated
        #op_min, op_max = update_mill_pdd_sag16_17_standard(
        #   current_value, ramp_up_sags16_17_pdd, revestimiento,
        #   control, op_min, op_max, params
        #)

        # Updating op_min and op_max values looking the last High limit value.
        op_min, op_max = update_mill_pdd_sag16_17_profit(
           current_value,control, op_min, op_max, params, print_info_controls
        )

        # Updating op_min and op_max values according to the pH Standard:
        op_min, op_max = update_pH_ranges_with_standard(
           current_value, ph_standard_data, control, op_min, op_max, params,
           print_info_controls
        )
        # Updating the op_min and op_max values according to BM-19 water constraint:
        op_min, op_max = update_recommendation_for_bm19(
            current_value, control, op_min, op_max, params, print_info_controls
        )
        op_min, op_max = update_classification_conditions_s16(
            current_value, control, op_min, op_max, params
        )
        op_min, op_max = update_classification_conditions_s17(
            current_value, control, op_min, op_max, params
        )
        op_min, op_max = update_classification_conditions_ma1(
            current_value, control, op_min, op_max, params
        )
        op_min, op_max = update_classification_conditions_ma0(
            current_value, control, op_min, op_max, params
        )
        lower_bounds = [op_min]
        upper_bounds = [op_max]

        if (not pd.isna(control_entry["max_delta"])) and (not pd.isna(control_entry["max_delta_down"])):
            current_val = current_value[control].iloc[0]
            if (current_val < op_min) or (current_val > op_max):
                logger.warning(
                    f"Current Value for Control {control} {current_val} "
                    f"is outside of range [{op_min}, {op_max}]"
                )
            lower_bounds.append(current_val - control_entry["max_delta_down"])
            upper_bounds.append(current_val + control_entry["max_delta"])

            if control in no_intersection_tags and max(lower_bounds) >= min(upper_bounds):
                lower_bounds = [op_min]
                upper_bounds = [op_max]
                logger.warning(
                    f"""Control tag {control} was forced to use it's original optimization
                    bounds due it's current value: {current_val} breaks the intersection logic"""
                    )
            
            if max(lower_bounds) >= min(upper_bounds):
                logger.warning(
                        f"""Control {control} had interval problems, setting as context variable.
                        So it will be dropped from final_controls."""
                    )
                final_controls.remove(control)
            else:
                #assert max(lower_bounds) <= min(upper_bounds), f"""There is no intersection for {control} bounds"""
                solver_bounds.append((max(lower_bounds), min(upper_bounds)))
        else:
            final_controls.remove(control)
            logger.warning(f"Control {control} without max delta info. Neither is not a control or max delta needs to be supplied")
    return solver_bounds, final_controls


def update_mill_vels_sag16_17_standard(current_value: pd.DataFrame,
    ramp_up_sags16_17_velocities: pd.DataFrame,
    revestimiento: pd.DataFrame,
    control: str,
    op_min: float,
    op_max: float,
    params: dict,
    print_info_controls: bool) -> (float, float):
    """
        The purpose of this method is to update the op_min and op_max
    values for velocities of SAG16 and SAG17 according to the "Ramp Up"
    standard depicted in the 'ramp_up_sags16_17' dataframe (check catalog).

    Args:
        current_value: Dataframe with the picture of the plants to be optimized.
        ramp_up_sags16_17_velocities: the velocity standard df coming from csv.
        revestimiento: dates of changes in revestimiento, coming from csv.
        control: control variable being analyzed.
        op_min: initial min value.
        op_max: initial max value.
        params: parameters of the process being optimized.
        print_info_controls: Boolean to print information about controls.
    Returns:
        op_min, op_max: if conditions applies to change them.

    """
    # Getting the function name:
    funct_name = "update_mill_vels_sag16_17_standard"

    # Getting initial op_min and op_max, for reporting use:
    init_op_min = op_min
    init_op_max = op_max

    # Validating if the parameters has been set for this plant optimization:
    _params_vel = params.get("sag_mill_rampup_standard")
    if _params_vel:
        if control in _params_vel["control_tags"]["vel"]:
            sag = _get_sag_from_params(_params_vel["tph_model_features"],
                        control,
                        "vel")
            # Getting the current date of optimization:
            opt_date = str(current_value["fecha"].values[0])[:-10].replace("T", " ")
            opt_date = pd.to_datetime(opt_date)
            
            # Getting the processed tph label from parameters:
            tph_prcsd_lbl =_params_vel["processed_tph_label"]

            # Getting the min and max velocity labels from parameters:
            min_mill_velocity_lbl =_params_vel["min_max_labels"]["vel"]["min"]
            max_mill_velocity_lbl =_params_vel["min_max_labels"]["vel"]["max"]

            # Getting the start date campaign label from parameters:
            strt_dt_cmpng_lbl =_params_vel["start_date_campaing_label"]

            # Getting the cumulated tph for this model from parameters:
            model_tph_params = _params_vel["tph_model_features"][sag]
            cum_tph_tag_name = model_tph_params["cum_sum_tph_tag"]
            rev_col_mdl = model_tph_params["revest_file_label"]
            # Getting the label of the "work_with_tph_label" column:
            wwtph_lbl = _params_vel["work_with_tph_label"]

            # Getting the label of the "date_change_label" column:
            dt_chng_lbl = _params_vel["date_change_label"]

            # Subset for associated tag model:
            cond_001 = ramp_up_sags16_17_velocities["model"] == sag
            ramp_up = ramp_up_sags16_17_velocities[cond_001]

            # Getting the lastest change in "revestimiento" file before today
            # (the optimization period):
            cond_002 = pd.to_datetime(revestimiento[rev_col_mdl]) <= opt_date
            if (len(cond_002) > 0) and (sum(cond_002) > 0):
                # Find latest date in ramp_up file until the optimization date:
                latest_date = max(
                    pd.to_datetime(revestimiento[cond_002][rev_col_mdl])
                )
                latest_date = pd.to_datetime(latest_date)
                
                # --------------------------------------------
                #  Determining if operating by days or by tph
                # --------------------------------------------

                # Subsetting the last date lower or equal than today (opt_date): pd.to_datetime(
                cond_005 = pd.to_datetime(ramp_up[dt_chng_lbl]) <= opt_date
                ramp_up2 = ramp_up[cond_005].tail(1)

                # Validating if this period is operated by days:
                working_tph = ramp_up2[wwtph_lbl].values[0]
                if working_tph == 0:
                    # -------------------------
                    #  Case working with dates
                    # -------------------------

                    # Updating op_min and op_max:
                    op_min = ramp_up2[min_mill_velocity_lbl].values[0]
                    op_max = ramp_up2[max_mill_velocity_lbl].values[0]
                    
                else:
                    # ---------------------------------
                    #  Case working with cumulated tph
                    # ---------------------------------

                    # Subsetting for the active campaign:
                    cond_006 = pd.to_datetime(ramp_up[strt_dt_cmpng_lbl]) == latest_date
                    ramp_up = ramp_up[cond_006]
                    ramp_up.reset_index(inplace = True)

                    # Checking the cumulated TPH for the optimization date:
                    cum_tph_today = current_value[cum_tph_tag_name].values[0]

                    # Checking the lower tph historic proccess:
                    cond_003 = ramp_up[tph_prcsd_lbl] <= cum_tph_today
                    
                    # Validating if there's intersection:
                    if (len(cond_003) > 0) and (sum(cond_003) > 0):
                        # Finding the ceiling of the processed TPH:
                        ceil_tph = max(
                            ramp_up[cond_003][tph_prcsd_lbl]
                        )
                    else:
                        # BORDER CASE:
                        # Not a intersection, selecting the first ceil_tph:
                        ceil_tph = min(ramp_up[tph_prcsd_lbl])
                    
                    # Final subsetting of the ramp_up file:
                    cond_004 = ramp_up[tph_prcsd_lbl] == ceil_tph
                    ramp_up = ramp_up[cond_004][
                        [min_mill_velocity_lbl, max_mill_velocity_lbl]
                    ]

                    # Updating op_min and op_max:
                    op_min = ramp_up[min_mill_velocity_lbl].values[0]
                    op_max = ramp_up[max_mill_velocity_lbl].values[0]
                    
                # Validating chances, because of historic values
                # treatment:
                if (op_min != init_op_min) or (op_max != init_op_max):
                    logger.warning(
                                f"op_min and op_max updated for tag {control} " +
                                f"from [{init_op_min}, {init_op_max}] to " +
                                f"[{op_min}, {op_max}] @{funct_name}."
                            )
                    if print_info_controls:
                        logger.info(f"\n\n\t\top_min and op_max updated for tag '{control}' @{funct_name}.\n"
                                    f"\t\t\tInitial range [{init_op_min}, {init_op_max}].\n"
                                    f"\t\t\t\tFinal range [{op_min}, {op_max}].\n"
                    )
    # Returning the op_min and op_max:
    return op_min, op_max

# # # def update_mill_pdd_sag16_17_standard(current_value: pd.DataFrame,
# # #     ramp_up_sags16_17_pdd: pd.DataFrame,
# # #     revestimiento: pd.DataFrame,
# # #     control: str,
# # #     op_min: float,
# # #     op_max: float,
# # #     params: dict) -> (float, float):
# # #     """
# # #         The purpose of this method is to update the op_min and op_max
# # #     values for pdd of SAG16 and SAG17 according to the "Ramp Up"
# # #     standard depicted in the 'ramp_up_sags16_17' dataframe (check catalog).

# # #     Args:
# # #         current_value: Dataframe with the picture of the plants to be optimized.
# # #         ramp_up_sags16_17_pdd: the presion de los descansos standard df coming from csv.
# # #         revestimiento: dates of changes in revestimiento, coming from csv.
# # #         control: control variable being analyzed.
# # #         op_min: initial min value.
# # #         op_max: initial max value.
# # #         params: parameters of the process being optimized.
# # #     Returns:
# # #         op_min, op_max: if conditions applies to change them.

# # #     """
# # #     # Getting initial op_min and op_max, for reporting use:
# # #     init_op_min = op_min
# # #     init_op_max = op_max

# # #     # Validating if the parameters has been set for this plant optimization:
# # #     _params_pdd = params.get("sag_mill_rampup_standard")
# # #     if _params_pdd:
# # #         # Getting specific model params
# # #         if control in _params_pdd["control_tags"]["pdd"]:
# # #             sag = _get_sag_from_params(_params_pdd["tph_model_features"],
# # #                         control,
# # #                         "pdd")
# # #             model_tph_params = _params_pdd["tph_model_features"][sag]
# # #             # Getting the current date of optimization:
# # #             opt_date = pd.to_datetime(str(current_value["fecha"].values[0])[:-10].replace("T", " "))

# # #             # Getting labels from parameters:
# # #             tph_prcsd_lbl =_params_pdd["processed_tph_label"]
# # #             min_pdd_label =_params_pdd["min_max_labels"]["pdd"]["min"]
# # #             max_pdd_label =_params_pdd["min_max_labels"]["pdd"]["max"]
# # #             strt_dt_cmpng_lbl =_params_pdd["start_date_campaing_label"]
# # #             rev_col_mdl = model_tph_params["revest_file_label"]

# # #             # Getting the model associated with ths control from parameters:
# # #             cum_tph_tag_name = model_tph_params["cum_sum_tph_tag"]

# # #             # Subset for associated tag model:
# # #             mask = ramp_up_sags16_17_pdd["model"] == sag
# # #             ramp_up = ramp_up_sags16_17_pdd[mask].copy()
# # #             # Getting the lastest change in "revestimiento" file before today
# # #             # (the optimization period):
# # #             mask = pd.to_datetime(revestimiento[rev_col_mdl]) <= opt_date
# # #             if sum(mask)> 0 and len(ramp_up)> 0:
# # #                 # Find latest date in ramp_up file until the optimization date:
# # #                 latest_date = max(
# # #                     pd.to_datetime(revestimiento[mask][rev_col_mdl])
# # #                 )
# # #                 latest_date = pd.to_datetime(latest_date)

# # #                 # Subsetting for the active campaign:
# # #                 mask = pd.to_datetime(ramp_up[strt_dt_cmpng_lbl]) == latest_date
# # #                 ramp_up = ramp_up[mask]
# # #                 ramp_up.reset_index(inplace = True)
# # #                 # Checking the cumulated TPH for the optimization date:
# # #                 cum_tph_today = current_value[cum_tph_tag_name].values[0]
# # #                 logger.info(f"Acummulated Tph is:{round(cum_tph_today)} for {sag}")
# # #                 # Checking the lower tph historic proccess:
# # #                 mask = ramp_up[tph_prcsd_lbl] >= cum_tph_today
# # #                 # Validating if there's intersection:
# # #                 if sum(mask)>0 and len(ramp_up[mask])> 0:
# # #                     # Finding the ceiling of the processed TPH:
# # #                     ceil_tph = min(
# # #                         ramp_up[mask][tph_prcsd_lbl]
# # #                     )
# # #                 else:
# # #                     # BORDER CASE:
# # #                     # Not a intersection, selecting the first ceil_tph:
# # #                     ceil_tph = min(ramp_up[tph_prcsd_lbl])

# # #                 # Final subsetting of the ramp_up file:
# # #                 mask = ramp_up[tph_prcsd_lbl] == ceil_tph
# # #                 (op_min,op_max) = ramp_up[mask][[min_pdd_label, max_pdd_label]].values[0]

# # #                 # Validating changes, because of historic values
# # #                 # treatment:
# # #                 if (op_min != init_op_min) or (op_max != init_op_max):
# # #                     logger.warning(
# # #                                 f"op_min and op_max updated for tag {control} " +
# # #                                 f"from [{init_op_min}, {init_op_max}] to " +
# # #                                 f"[{op_min}, {op_max}] @update_mill_pdd_sag16_17_standard."
# # #                             )

# # #     # Returning the op_min and op_max:
# # #     return op_min, op_max

def update_mill_pdd_sag16_17_profit(current_value: pd.DataFrame,
    control: str,
    op_min: float,
    op_max: float,
    params: dict,
    print_info_controls: bool) -> (float, float):
    """
        The purpose of this method is to update the op_min and op_max
    values for pdd of SAG16 and SAG17 according to the actual profit high limit.

    Args:
        current_value: Dataframe with the picture of the plants to be optimized.
        control: control variable being analyzed.
        op_min: initial min value.
        op_max: initial max value.
        params: parameters of the process being optimized.
        print_info_controls: Boolean to print information about controls.
    Returns:
        op_min, op_max: if conditions applies to change them.

    """
    # The name of the function or method:
    funct_name = "update_mill_pdd_sag16_17_profit"

    # Getting initial op_min and op_max, for reporting use:
    init_op_min = op_min
    init_op_max = op_max

    # Validating if the parameters has been set for this plant optimization:
    _params_pdd = params.get("sag_mill_rampup_standard")
    if _params_pdd:
        # Getting specific model params
        if control in _params_pdd["control_tags"]["pdd"]:
            sag = _get_sag_from_params(_params_pdd["tph_model_features"],
                        control,
                        "pdd")
            model_tph_params = _params_pdd["tph_model_features"][sag]
            pdd_hl = model_tph_params["pdd_hl"]

            op_max = current_value[pdd_hl].values[0]
            op_min = op_max - _params_pdd["pdd_dif"]
            # Getting the current date of optimization:
            opt_date = pd.to_datetime(str(current_value["fecha"].values[0])[:-10].replace("T", " "))

            if (op_min != init_op_min) or (op_max != init_op_max):
                logger.warning(
                            f"op_min and op_max updated for tag {control} at {opt_date} " +
                            f"from [{init_op_min}, {init_op_max}] to " +
                            f"[{op_min}, {op_max}] @update_mill_pdd_sag16_17_profit."
                        )
                if print_info_controls:
                        logger.info(f"\n\n\t\top_min and op_max updated for tag '{control}' at {opt_date} @{funct_name}.\n"
                                    f"\t\t\tInitial range [{init_op_min}, {init_op_max}].\n"
                                    f"\t\t\t\tFinal range [{op_min}, {op_max}].\n"
                    )

    # Returning the op_min and op_max:
    return op_min, op_max

def _get_sag_from_params(params: dict,
                        control:str,
                        feature:str) -> str:
    """
    Aux function to get the right Sag in the RampUp params
    Args:
        params: Parameters dictionary.
        control: Control variable to check.
        feature: Velocidad or Presion de los descansos (vel or pdd).
    Return:
        sag: Sag.
    """
    sag = [x for x in params.keys() if control == params[x][feature]][0]
    return sag

def update_pH_ranges_with_standard(current_value: pd.DataFrame,
    ph_standard_data: pd.DataFrame,
    control: str,
    op_min: float,
    op_max: float,
    params: dict,
    print_info_controls: bool
) -> (float, float):
    """
        The purpose of this method is to update the op_min and op_max
    values for pH and FoamLevelForBanks declared in the 'ph_standard_data' and
    in the parameters set.

    Args:
        current_value: Dataframe with the picture of the plants to be optimized.
        ph_standard_data: the ph standard df coming from csv.
        control: control variable being analyzed.
        op_min: initial min value.
        op_max: initial max value.
        params: parameters of the process being optimized.
        print_info_controls: Boolean to print information about controls.
    Returns:
        op_min, op_max: if conditions applies to change them.
    
    ph_standard_data contains the following information:
    - last updated time of the standard
    - the tag subjected to that protocol
    - min and max rango of the fe/cu ratio
    - new optmin/optmax ranges
    - left and right senses: whether the left and right senses of the inequation is
    gonna be a less than or less or equal than.
        ge: greater or equal
        le: lower or equal
        gt: greater than
        lt: lower than
    """

    # Getting the function name:
    funct_name = "update_pH_ranges_with_standard"

    # Getting initial op_min and op_max, for reporting use:
    init_op_min = op_min
    init_op_max = op_max

    # Validating if the parameters has been set for this plant optimization
    # and that the actual control is defined in that list:
    cntrl_tgs_prms = params["ph_standard"]["control_tags"]
    if (cntrl_tgs_prms is not None) and (control in cntrl_tgs_prms):
        # Getting the current date of optimization:
        opt_date = str(current_value["fecha"].values[0])[:-10].replace("T", " ")
        opt_date = pd.to_datetime(opt_date)

        # Getting the label for the datetime column:
        strtDt_lbl = params["ph_standard"]["StartDate_label"]

        # Getting the label for the tag to be limited:
        tag_lbl_2b_lmtd = params["ph_standard"]["Limited_tag_label"]

        # Getting the labels for the fe over cu tag:
        minrng_fecu_lbl = params["ph_standard"]["minRange_fe_cu_label"]
        maxrng_fecu_lbl = params["ph_standard"]["maxRange_fe_cu_label"]

        # Getting the labels of the opmin and opmax range for the control tag:
        opmin_tag_lbl = params["ph_standard"]["op_min_tag_label"]
        opmax_tag_lbl = params["ph_standard"]["op_max_tag_label"]

        # Getting the labels of the left and right sense comparator:
        lft_sns_lbl = params["ph_standard"]["left_sense_label"]
        rght_sns_lbl = params["ph_standard"]["right_sense_label"]

        # Getting the tag of fe over cu:
        dependent_tag = params["ph_standard"]["dependent_tag"]

        # Getting the labels for final subsetting:
        stnds_lft_lbl = params["ph_standard"]["stands_left_label"]
        stnds_rght_lbl = params["ph_standard"]["stands_right_label"]
        rw_of_intrst = params["ph_standard"]["row_of_interest_label"]

        # Subsetting the file for the control tag being studied:
        cond_001 = ph_standard_data[tag_lbl_2b_lmtd] == control
        if len(cond_001) > 0:
            ph_stndrd_dt_cntrl = ph_standard_data[cond_001]

            # Getting the lastest change in "ph_standard_data" file before today
            # (the optimization period):
            cond_002 = ph_stndrd_dt_cntrl[strtDt_lbl] <= opt_date
            if len(cond_002) > 0:
                # Find latest date in ramp_up file until the optimization date:
                latest_date = max(
                    ph_stndrd_dt_cntrl[cond_002][strtDt_lbl]
                )
                
                # Subsetting for the active campaign:
                standard = ph_stndrd_dt_cntrl[ph_stndrd_dt_cntrl[strtDt_lbl] == latest_date]
                dsrd_cols = [minrng_fecu_lbl, maxrng_fecu_lbl, opmin_tag_lbl, 
                            opmax_tag_lbl, lft_sns_lbl, rght_sns_lbl]
                standard = standard[dsrd_cols]
                standard.reset_index(inplace = True, drop = True)
                
                # Adding the actual fe over cu value:
                dpndnt_vl = current_value[dependent_tag].values[0]
                standard[dependent_tag] = [dpndnt_vl] * standard.shape[0]
                
                # Final calculations to get the proper row:
                cols_interest = [minrng_fecu_lbl, lft_sns_lbl, dependent_tag, 
                                    rght_sns_lbl, maxrng_fecu_lbl]
                standard[stnds_lft_lbl] =\
                    standard[cols_interest].apply(lambda m: \
                        (m[0] <= m[2]) if m[1] == 'ge' else (m[0] < m[2]), axis = 1)
                standard[stnds_rght_lbl] =\
                    standard[cols_interest].apply(lambda m: \
                        (m[2] <= m[4]) if m[3] == 'le' else (m[2] < m[4]), axis = 1)
                standard[rw_of_intrst] =\
                    standard[[stnds_lft_lbl, stnds_rght_lbl]].apply(lambda r: \
                        r[0] & r[1], axis = 1)

                # Validating there is a unique active case:
                if sum(standard[rw_of_intrst]) == 1:
                    op_min_max_tags = [opmin_tag_lbl, opmax_tag_lbl]
                    standard = standard.loc[standard[rw_of_intrst], op_min_max_tags]
                    
                    # Updating op_min and op_max:
                    op_min = standard[opmin_tag_lbl].values[0]
                    op_max = standard[opmax_tag_lbl].values[0]

                    # Validating chances, because of historic values
                    # treatment:
                    if (op_min != init_op_min) and (op_max != init_op_max):
                        logger.warning(
                                    f"op_min and op_max updated for tag {control} " +
                                    f"from [{init_op_min}, {init_op_max}] to " +
                                    f"[{op_min}, {op_max}] @{funct_name}."
                        )
                        if print_info_controls:
                            logger.info(f"\n\n\t\top_min and op_max updated for tag '{control}' at {opt_date} @{funct_name}.\n"
                                        f"\t\t\tInitial range [{init_op_min}, {init_op_max}].\n"
                                        f"\t\t\t\tFinal range [{op_min}, {op_max}].\n"
                        )
                elif sum(standard[rw_of_intrst]) == 0:
                    logger.warning(
                                f"No condition found with {dependent_tag} " +
                                f"for control tag {control}."
                                f"\nNo changes will be applied to op_min and " +
                                f"op_max values taken from tag_dict_master @{funct_name}."
                    )
                else:
                    logger.warning(
                                f"More than one condition found with {dependent_tag} " +
                                f"for control tag {control}."
                                f"\nNo changes will be applied to op_min and " +
                                f"op_max values taken from tag_dict_master @{funct_name}."
                    )

    # Returning the op_min and op_max:
    return op_min, op_max

def update_recommendation_for_bm19(
        current_value: pd.DataFrame,
        control: str,
        op_min: float,
        op_max: float,
        params: dict,
        print_info_controls: bool
) -> Tuple[float, float]:
    """
    A method to update the minimum and maximum values for the water flow that goes into the 501 pond
    ("pozo 501"), which is the converging node for all flows in the "quinto molino" (BM19) circuit.
    
    The purpose of this method is to build a constraint for the water flow to be recommended, using
    the solid proportion in the pulp pumped to cyclons as a flag, and the pool level as a buffer.
    This way, we only recommend to increase the water flow when solids are over a certain threshold
    value (70%), and to decrease it when solids are below it. However, if for any case the pool
    level is over 90% of water, we only recommend to increase the water flow in order to drain the
    water excess of the aforementioned pool to avoid eventual overpourings.

    Parameters
    ----------
    current_value : DataFrame that represents all instances to be optimized (with all inputs we need).
    control : Tag for the control feature we are constraining with this method.
    op_min : Initial minimum value for the control feature.
    op_max : Initial maximum value for the control feature.
    params : Dictionary of parameters we fetch to give context and initial conditions for this method.
    print_info_controls: Boolean to print information about controls.

    Returns
    -------
    The updated op_min and op_max values (in that order).

    """
    # 1. Getting the function name
    funct_name = "update_recommendation_for_bm19"
    
    # 2. Getting initial op_min and op_max, for reporting use:
    init_op_min = op_min
    init_op_max = op_max
    
    # 3. Checking if the regarding parameters has been set for this process
    _params_bm19 = params.get("bm19_solids_control")
    
    if _params_bm19:
        # 3.1 Getting the model params for this condition
        if control in _params_bm19["control_tag"]:
            flow   = _params_bm19["control_tag"]
            solids = _params_bm19["solids_in_pulp"]
            level  = _params_bm19["pool_water_level"]
            buffer = _params_bm19["level_buffer"]
            solid_threshold = _params_bm19["solids_threshold"]
            
            # 3.1.1 Constraining the direction of the eventual recommendation
            # 3.1.1.1 CASE A: Pool level under 90%
            if current_value[level].values[0] < buffer: # Pool level is not a problem
                if current_value[solids].values[0] <= solid_threshold: # Solids are low (water flow can only be 
                                                             # decreased)
                    op_max = current_value[flow].values[0]                
                if current_value[solids].values[0] > solid_threshold: # Solids are high (water flow can only be increased)
                    op_min = current_value[flow].values[0] 
            
            # 3.1.1.2 CASE B: Pool level equal or greater than 90%
            if current_value[level].values[0] >= buffer: # Pool level is too high, so we break the high solids
                                               # threshold constraint on purpose to avoid any water
                                               # outpourings. This is, we can only recommend decreasing the
                                               # usage of water.
                op_max = current_value[flow].values[0]
                
            # 3.1.2 We eventually update the min and max values for water flow, so we need to
            # convey this information to our log in order to take account any changed we might
            # have made
            if (op_min != init_op_min) or (op_max != init_op_max):
                logger.warning(
                    f"op_min and op_max updated for tag {control} " +
                    f"from [{init_op_min}, {init_op_max}] to " +
                    f"[{op_min}, {op_max}] @{funct_name}."
                )
                if print_info_controls:
                        logger.info(f"\n\n\t\top_min and op_max updated for tag '{control}' @{funct_name}.\n"
                                    f"\t\t\tInitial range [{init_op_min}, {init_op_max}].\n"
                                    f"\t\t\t\tFinal range [{op_min}, {op_max}].\n"
                    )
    # 4. We have made our changes, so we return the eventually updated values for op_min or op_max
    return op_min, op_max

def update_classification_conditions_s16(
        current_value: pd.DataFrame,
        control: str,
        op_min: float,
        op_max: float,
        params: dict
) -> Tuple[float, float]:
    """
    A method to update the feeding pressure for cyclons in the classification circuit of S16, considering
    the existing trade-off between classification efficiency and the ore-size distribution. This method is
    only for S16 since the SAG grinding circuit has two cyclon batteries for each primary mill.

    The purpose of this method is to build a constraint for the feeding pressure value to be recommended,
    using the solid proportion and the current ore-size reference (+65#) as flags, and the pond level as
    a buffer. This logic is chained as follows:

                          High solids prop (>=75%)    Low solids prop (<=62%)
                          High +65# value [28, 35]    Low +65# value: [26, 28]
                          High pond level (>=90%):    Low pond level (<=40%):
                          ^                           |          
                          | pressure                  v pressure

    However, these relations have limits (of course, we cannot increase the pressure beyond the safety 
    limit of the cyclons, and we cannot decrease the solid proportions indefinitely, because we loss 
    throughput -and fine copper- by doing so), all of them defined as bounds for all variables involved. 
    These bounds are the references we use to constrain the feeding pressures recommendations. Moreover, 
    if for any case the pond level is over 90% of water, we only recommend to decrease the feeding 
    pressure in order to drain the water excess of the aforementioned pond to avoid eventual overpourings.
    On the other side, if for any case the pond level is under 40% of water, we only recommend to increase
    the feeding pressure in order to let the pond level grow up to a standard condition.
    
    Parameters
    ----------
    current_value : DataFrame that represents all instances to be optimized (with all inputs we need).
    control : Tag for the control feature we are constraining with this method.
    op_min : Initial minimum value for the control feature.
    op_max : Initial maximum value for the control feature.
    params : Dictionary of parameters we fetch to give context and initial conditions for this method.

    Returns
    -------
    The updated op_min and op_max values (in that order).

    """
    # 1. Getting the function name
    func_name = "update_classification_conditions_s16"
    
    # 2. Getting initial op_min and op_max, for reporting use:
    init_op_min = op_min
    init_op_max = op_max
    
    # 3. Checking if the regarding parameters has been set for this process
    _params_ma2 = params.get("ma2_classification_control")

    if _params_ma2:
        # 3.1 Getting the model params for this condition
        if control in _params_ma2["control_tags"]["s16"]:
            # 3.1.1 First, the tags for every parameter
            pressure_tags = _params_ma2["control_tags"]["s16"]
            solids_tags = _params_ma2["solids_in_pulp"]["s16"]
            level_tags = [_params_ma2["pond_water_level"]["s16"]] * len(solids_tags)
            size_tags = _params_ma2["ore_size_feats"]["s16"]

            # 3.1.2 Getting the boundaries for every flag and buffer parameter
            high_size_bounds = _params_ma2["ore_size_feats"]["high_size_bounds"]
            low_size_bounds = _params_ma2["ore_size_feats"]["low_size_bounds"]
            solids_bounds = _params_ma2["solids_threshold"]
            levels_bounds = _params_ma2["level_buffer"]
            excldd_frm_size_vldtn = _params_ma2["excldd_frm_size_vldtn"]
            
            # 3.1.3 Validating constraints
            for pressure, solids, level, size in zip(pressure_tags, solids_tags, level_tags, size_tags):
                if pressure == control:
                    if not np.isnan(current_value[solids].values[0]) and not np.isnan(current_value[size].values[0]):
                        # 3.1.3.1 Checking the most important condition (solids)
                        if current_value[solids].values[0] >= solids_bounds[1]: # Solids are too high
                            op_min = current_value[pressure].values[0] # Pressure can only be increased
                        elif current_value[solids].values[0] <= solids_bounds[0]: # Solids are too low
                            op_max = current_value[pressure].values[0] # Pressure can only be decreased
                        else:
                            if excldd_frm_size_vldtn is None:
                                # 3.1.3.2 Checking the second most important condition (size)
                                if current_value[size].values[0] >= high_size_bounds[0]: # Product is high in size
                                    op_min = current_value[pressure].values[0] # Pressure can only be increased
                                elif current_value[size].values[0] < high_size_bounds[0]: # Product is low in size
                                    op_max = current_value[pressure].values[0] # Pressure can only be decreased
                            elif (excldd_frm_size_vldtn is not None) and (pressure not in excldd_frm_size_vldtn):
                                # 3.1.3.2 Checking the second most important condition (size)
                                if current_value[size].values[0] >= high_size_bounds[0]: # Product is high in size
                                    op_min = current_value[pressure].values[0] # Pressure can only be increased
                                elif current_value[size].values[0] < high_size_bounds[0]: # Product is low in size
                                    op_max = current_value[pressure].values[0] # Pressure can only be decreased
                    else:
                        # Solid and size variables NAN: control set as context:
                        op_min = current_value[pressure].values[0]
                        op_max = current_value[pressure].values[0]


            # 3.1.4 We eventually update the min and max values for the cyclones' feeding pressures, so we 
            # need to convey this information to our log in order to take account for any change we might
            # have made
            if (op_min != init_op_min) or (op_max != init_op_max):
                logger.warning(
                    f"op_min and op_max updated for tag {control} " +
                    f"from [{init_op_min}, {init_op_max}] to " +
                    f"[{op_min}, {op_max}] @{func_name}."
                )
    # 4. We have made our changes, so we return the eventually updated values for op_min or op_max
    return op_min, op_max

def update_classification_conditions_s17(
        current_value: pd.DataFrame,
        control: str,
        op_min: float,
        op_max: float,
        params: dict
) -> Tuple[float, float]:
    """
    A method to update the feeding pressure for cyclons in the classification circuit of S17, considering
    the existing trade-off between classification efficiency and the ore-size distribution. This method is
    only for S17 since the SAG grinding circuit has two cyclon batteries for each primary mill.

    The purpose of this method is to build a constraint for the feeding pressure value to be recommended,
    using the solid proportion and the current ore-size reference (+65#) as flags, and the pond level as
    a buffer. This logic is chained as follows:

                          High solids prop (>=75%)    Low solids prop (<=62%)
                          High +65# value [28, 35]    Low +65# value: [26, 28]
                          High pond level (>=90%):    Low pond level (<=40%):
                          ^                           |          
                          | pressure                  v pressure

    However, these relations have limits (of course, we cannot increase the pressure beyond the safety 
    limit of the cyclons, and we cannot decrease the solid proportions indefinitely, because we loss 
    throughput -and fine copper- by doing so), all of them defined as bounds for all variables involved. 
    These bounds are the references we use to constrain the feeding pressures recommendations. Moreover, 
    if for any case the pond level is over 90% of water, we only recommend to decrease the feeding 
    pressure in order to drain the water excess of the aforementioned pond to avoid eventual overpourings.
    On the other side, if for any case the pond level is under 40% of water, we only recommend to increase
    the feeding pressure in order to let the pond level grow up to a standard condition.
    
    Parameters
    ----------
    current_value : DataFrame that represents all instances to be optimized (with all inputs we need).
    control : Tag for the control feature we are constraining with this method.
    op_min : Initial minimum value for the control feature.
    op_max : Initial maximum value for the control feature.
    params : Dictionary of parameters we fetch to give context and initial conditions for this method.

    Returns
    -------
    The updated op_min and op_max values (in that order).

    """
    # 1. Getting the function name
    func_name = "update_classification_conditions_s17"
    
    # 2. Getting initial op_min and op_max, for reporting use:
    init_op_min = op_min
    init_op_max = op_max
    
    # 3. Checking if the regarding parameters has been set for this process
    _params_ma2 = params.get("ma2_classification_control")

    if _params_ma2:
        # 3.1 Getting the model params for this condition
        if control in _params_ma2["control_tags"]["s17"]:
            # 3.1.1 First, the tags for every parameter
            pressure_tags = _params_ma2["control_tags"]["s17"]
            solids_tags = _params_ma2["solids_in_pulp"]["s17"]
            level_tags = [_params_ma2["pond_water_level"]["s17"]] * len(solids_tags)
            size_tags = _params_ma2["ore_size_feats"]["s17"]

            # 3.1.2 Getting the boundaries for every flag and buffer parameter
            high_size_bounds = _params_ma2["ore_size_feats"]["high_size_bounds"]
            low_size_bounds = _params_ma2["ore_size_feats"]["low_size_bounds"]
            solids_bounds = _params_ma2["solids_threshold"]
            levels_bounds = _params_ma2["level_buffer"]
            excldd_frm_size_vldtn = _params_ma2["excldd_frm_size_vldtn"]

            # 3.1.3 Validating constraints
            for pressure, solids, level, size in zip(pressure_tags, solids_tags, level_tags, size_tags):
                if control == pressure:
                    if not np.isnan(current_value[solids].values[0]) and not np.isnan(current_value[size].values[0]):
                        # 3.1.3.1 Checking the most important condition (solids)
                        if current_value[solids].values[0] >= solids_bounds[1]: # Solids are too high
                            op_min = current_value[pressure].values[0] # Pressure can only be increased
                        elif current_value[solids].values[0] <= solids_bounds[0]: # Solids are too low
                            op_max = current_value[pressure].values[0] # Pressure can only be decreased
                        else:
                            if excldd_frm_size_vldtn is None:
                                # 3.1.3.2 Checking the second most important condition (size)
                                if current_value[size].values[0] >= high_size_bounds[0]: # Product is high in size
                                    op_min = current_value[pressure].values[0] # Pressure can only be increased
                                elif current_value[size].values[0] < high_size_bounds[0]: # Product is low in size
                                    op_max = current_value[pressure].values[0] # Pressure can only be decreased
                            elif (excldd_frm_size_vldtn is not None) and (pressure not in excldd_frm_size_vldtn):
                                # 3.1.3.2 Checking the second most important condition (size)
                                if current_value[size].values[0] >= high_size_bounds[0]: # Product is high in size
                                    op_min = current_value[pressure].values[0] # Pressure can only be increased
                                elif current_value[size].values[0] < high_size_bounds[0]: # Product is low in size
                                    op_max = current_value[pressure].values[0] # Pressure can only be decreased
                    else:
                        # Solid and size variables NAN: control set as context:
                        op_min = current_value[pressure].values[0]
                        op_max = current_value[pressure].values[0]

            # 3.1.4 We eventually update the min and max values for the cyclones' feeding pressures, so we 
            # need to convey this information to our log in order to take account for any change we might
            # have made
            if (op_min != init_op_min) or (op_max != init_op_max):
                logger.warning(
                    f"op_min and op_max updated for tag {control} " +
                    f"from [{init_op_min}, {init_op_max}] to " +
                    f"[{op_min}, {op_max}] @{func_name}."
                )
    # 4. We have made our changes, so we return the eventually updated values for op_min or op_max
    return op_min, op_max

def update_classification_conditions_ma1(
        current_value: pd.DataFrame,
        control: str,
        op_min: float,
        op_max: float,
        params: dict
) -> Tuple[float, float]:
    """
    A method to update the feeding pressure for cyclons in the classification circuit of MA1, considering
    the existing trade-off between classification efficiency and the ore-size distribution.

    The purpose of this method is to build a constraint for the feeding pressure value to be recommended,
    using the solid proportion and the current ore-size reference (+65#) as flags, and the pond level as
    a buffer. This logic is chained as follows:

                          High solids prop (>=75%)    Low solids prop (<=62%)
                          High +65# value [28, 35]    Low +65# value: [26, 28]
                          High pond level (>=90%):    Low pond level (<=40%):
                          ^                           |          
                          | pressure                  v pressure

    However, these relations have limits (of course, we cannot increase the pressure beyond the safety 
    limit of the cyclons, and we cannot decrease the solid proportions indefinitely, because we loss 
    throughput -and fine copper- by doing so), all of them defined as bounds for all variables involved. 
    These bounds are the references we use to constrain the feeding pressures recommendations. Moreover, 
    if for any case the pond level is over 90% of water, we only recommend to decrease the feeding 
    pressure in order to drain the water excess of the aforementioned pond to avoid eventual overpourings.
    On the other side, if for any case the pond level is under 40% of water, we only recommend to increase
    the feeding pressure in order to let the pond level grow up to a standard condition.
    
    Parameters
    ----------
    current_value : DataFrame that represents all instances to be optimized (with all inputs we need).
    control : Tag for the control feature we are constraining with this method.
    op_min : Initial minimum value for the control feature.
    op_max : Initial maximum value for the control feature.
    params : Dictionary of parameters we fetch to give context and initial conditions for this method.

    Returns
    -------
    The updated op_min and op_max values (in that order).

    """
    # 1. Getting the function name
    func_name = "update_recommendation_for_ma1"
    
    # 2. Getting initial op_min and op_max, for reporting use:
    init_op_min = op_min
    init_op_max = op_max
    
    # 3. Checking if the regarding parameters has been set for this process
    _params_ma1 = params.get("ma1_classification_control")

    if _params_ma1:
        # 3.1 Getting the model params for this condition
        if control in _params_ma1["control_tags"]:
            # 3.1.1 First, the tags for every parameter
            _solids_prefix = _params_ma1["solids_in_pulp"] # Prefix for solids variable
            _sizes  = _params_ma1["ore_size_feats"] # Dict for all size tags
            _levels = _params_ma1["pond_water_level"] # Dict for all level tags

            pressure_tags = _params_ma1["control_tags"]

            sections = [j for j in range(13, 16)] # List with all sections of MA1
            solids_tags = [f"{_solids_prefix}{j}" for j in sections]
            keys = [f"s{j}" for j in sections]

            size_tags, level_tags = [], []
            # Cycling over the former dicts to fetch everything in a list
            for key in keys:
                size_tag  = _sizes[key]
                level_tag = _levels[key]
                size_tags.append(size_tag)
                level_tags.append(level_tag)

            # 3.1.2 Getting the boundaries for every flag and buffer parameter
            high_size_bounds = _params_ma1["ore_size_feats"]["high_size_bounds"]
            low_size_bounds = _params_ma1["ore_size_feats"]["low_size_bounds"]
            solids_bounds = _params_ma1["solids_threshold"]
            levels_bounds = _params_ma1["level_buffer"]
            excldd_frm_size_vldtn = _params_ma1["excldd_frm_size_vldtn"]
            
            # 3.1.3 Validating constraints
            for pressure, solids, size in zip(pressure_tags, solids_tags, size_tags):
                if not np.isnan(current_value[solids].values[0]) and not np.isnan(current_value[size].values[0]):
                    if pressure == control:
                        # 3.1.3.1 Checking the most important condition (solids)
                        if current_value[solids].values[0] >= solids_bounds[1]: # Solids are too high
                            op_min = current_value[pressure].values[0] # Pressure can only be increased
                        elif current_value[solids].values[0] <= solids_bounds[0]: # Solids are too low
                            op_max = current_value[pressure].values[0] # Pressure can only be decreased
                        else:
                            if excldd_frm_size_vldtn is None:
                                # 3.1.3.2 Checking the second most important condition (size)
                                if current_value[size].values[0] >= high_size_bounds[0]: # Product is high in size
                                    op_min = current_value[pressure].values[0] # Pressure can only be increased
                                elif current_value[size].values[0] < high_size_bounds[0]: # Product is low in size
                                    op_max = current_value[pressure].values[0] # Pressure can only be decreased
                            elif (excldd_frm_size_vldtn is not None) and (pressure not in excldd_frm_size_vldtn):
                                # 3.1.3.2 Checking the second most important condition (size)
                                if current_value[size].values[0] >= high_size_bounds[0]: # Product is high in size
                                    op_min = current_value[pressure].values[0] # Pressure can only be increased
                                elif current_value[size].values[0] < high_size_bounds[0]: # Product is low in size
                                    op_max = current_value[pressure].values[0] # Pressure can only be decreased
                                    
                else:
                    # Solid and size variables NAN: control set as context:
                    op_min = current_value[pressure].values[0]
                    op_max = current_value[pressure].values[0]

            # 3.1.4 We eventually update the min and max values for the cyclones' feeding pressures, so we 
            # need to convey this information to our log in order to take account for any change we might
            # have made
            if (op_min != init_op_min) or (op_max != init_op_max):
                logger.warning(
                    f"op_min and op_max updated for tag {control} " +
                    f"from [{init_op_min}, {init_op_max}] to " +
                    f"[{op_min}, {op_max}] @{func_name}."
                )
    # 4. We have made our changes, so we return the eventually updated values for op_min or op_max
    return op_min, op_max

def update_classification_conditions_ma0(
        current_value: pd.DataFrame,
        control: str,
        op_min: float,
        op_max: float,
        params: dict
) -> Tuple[float, float]:
    """
    A method to update the feeding pressure for cyclons in the classification circuit of MA0, considering
    the existing trade-off between classification efficiency and the ore-size distribution. This method is
    applyable for all MA0 sections at once, since the classification circuits are unique for each section.

    The purpose of this method is to build a constraint for the feeding pressure value to be recommended,
    using the solid proportion and the current ore-size reference (+65#) as flags, and the pond level as
    a buffer. This logic is chained as follows:

                          High solids prop (>=75%)    Low solids prop (<=62%)
                          High +65# value [28, 35]    Low +65# value: [26, 28]
                          High pond level (>=90%):    Low pond level (<=40%):
                          ^                           |          
                          | Speed                     v Speed

    However, these relations have limits (of course, we cannot increase the pressure beyond the safety 
    limit of the cyclons, and we cannot decrease the solid proportions indefinitely, because we loss 
    throughput -and fine copper- by doing so), all of them defined as bounds for all variables involved. 
    These bounds are the references we use to constrain the feeding pressures recommendations. Moreover, 
    if for any case the pond level is over 90% of water, we only recommend to decrease the feeding 
    pressure in order to drain the water excess of the aforementioned pond to avoid eventual overpourings.
    On the other side, if for any case the pond level is under 40% of water, we only recommend to increase
    the feeding pressure in order to let the pond level grow up to a standard condition.
    
    Parameters
    ----------
    current_value : DataFrame that represents all instances to be optimized (with all inputs we need).
    control : Tag for the control feature we are constraining with this method.
    op_min : Initial minimum value for the control feature.
    op_max : Initial maximum value for the control feature.
    params : Dictionary of parameters we fetch to give context and initial conditions for this method.

    Returns
    -------
    The updated op_min and op_max values (in that order).

    """
    # 1. Getting the function name
    func_name = "update_classification_conditions_ma0"
    
    # 2. Getting initial op_min and op_max, for reporting use:
    init_op_min = op_min
    init_op_max = op_max
    
    # 3. Checking if the regarding parameters has been set for this process
    _params_ma0 = params.get("ma0_classification_control")
    
    if _params_ma0:
        # 3.1 Getting the model params for this condition
        if control in _params_ma0["control_tags"]:
            # 3.1.1 First, the tags for every parameter
            _solids_prefix = _params_ma0["solids_in_pulp"] # Prefix for solids variable
            _sizes  = _params_ma0["ore_size_feats"] # Dict for all size tags
            _levels = _params_ma0["pond_water_level"] # Dict for all level tags

            speed_tags = _params_ma0["control_tags"]

            sections_ma0 = [j for j in range(0, 13)] # A list which encompasses all sections in MA0
            sections_not_modelled = [0, 1, 3, 6, 12] # A list which encompasses all sections of MA0 
                                                     # that were not modelled
            sections = [j for j in sections_ma0 if j not in sections_not_modelled] # Sections modelled
            solids_tags = [f"{_solids_prefix}{j}" for j in sections] # Tag list for solids in pulp
            keys = [f"s{j}" for j in sections]

            size_tags, level_tags = [], []
            # Cycling over the former dicts to fetch everything in a list
            for key in keys:
                size_tag  = _sizes[key]
                level_tag = _levels[key]
                size_tags.append(size_tag)
                level_tags.append(level_tag)

            # 3.1.2 Getting the boundaries for every flag and buffer parameter
            high_size_bounds = _params_ma0["ore_size_feats"]["high_size_bounds"]
            low_size_bounds = _params_ma0["ore_size_feats"]["low_size_bounds"]
            solids_bounds = _params_ma0["solids_threshold"]
            levels_bounds = _params_ma0["level_buffer"]
            excldd_frm_size_vldtn = _params_ma0["excldd_frm_size_vldtn"]

            # 3.1.3 Validating constraints
            for speed, solids, size in zip(speed_tags, solids_tags, size_tags):
                if control == speed:
                    if not np.isnan(current_value[solids].values[0]) and not np.isnan(current_value[size].values[0]):
                        # 3.1.3.1 Checking the most important condition (solids)
                        if current_value[solids].values[0] >= solids_bounds[1]: # Solids are too high
                            op_min = current_value[speed].values[0] # Speed can only be increased
                        elif current_value[solids].values[0] <= solids_bounds[0]: # Solids are too low
                            op_max = current_value[speed].values[0] # Speed can only be decreased
                        else:
                            if excldd_frm_size_vldtn is not None:
                                # 3.1.3.2 Checking the second most important condition (size)
                                if current_value[size].values[0] >= high_size_bounds[0]: # Product is high in size
                                    op_min = current_value[speed].values[0] # Speed can only be increased
                                elif current_value[size].values[0] < high_size_bounds[0]: # Product is low in size
                                    op_max = current_value[speed].values[0] # Speed can only be decreased
                            elif (excldd_frm_size_vldtn is not None) and (speed not in excldd_frm_size_vldtn):
                                # 3.1.3.2 Checking the second most important condition (size)
                                if current_value[size].values[0] >= high_size_bounds[0]: # Product is high in size
                                    op_min = current_value[speed].values[0] # Speed can only be increased
                                elif current_value[size].values[0] < high_size_bounds[0]: # Product is low in size
                                    op_max = current_value[speed].values[0] # Speed can only be decreased
                    else:
                        # Solid and size variables NAN: control set as context:
                        op_min = current_value[speed].values[0]
                        op_max = current_value[speed].values[0]

            # 3.1.4 We eventually update the min and max values for the cyclones' feeding pressures, so we 
            # need to convey this information to our log in order to take account for any change we might
            # have made
            if (op_min != init_op_min) or (op_max != init_op_max):
                logger.warning(
                    f"op_min and op_max updated for tag {control} " +
                    f"from [{init_op_min}, {init_op_max}] to " +
                    f"[{op_min}, {op_max}] @{func_name}."
                )
    # 4. We have made our changes, so we return the eventually updated values for op_min or op_max
    return op_min, op_max

def make_stopper(params: dict) -> BaseStopper:
    """
    Creates a stopper using configured params

    Args:
        params: dict of pipeline parameters
    Returns:
        optimization stopper object
    """
    if params["stopper"]:
        stopper_class = load_obj(params["stopper"]["class"])
        stopper_kwargs = params["stopper"]["kwargs"]
        return stopper_class(**stopper_kwargs)
    return None


def _make_set_repair(td: TagDict, column: str) -> Repair:
    """ Creates a new constraint set repair for a given column """
    constraint_set = literal_eval(td[column]["constraint_set"])
    return repair(column, "in", constraint_set)


def bulk_optimize(  # pylint:disable=too-many-locals
    params: dict,
    td: TagDict,
    data: pd.DataFrame,
    model: Any,
    model_dict: Dict,
    revestimiento: pd.DataFrame,
    ramp_up_sags16_17_velocities: pd.DataFrame,
    ph_standard_data: pd.DataFrame,
    penalties=None,
) -> Dict:
    """
    Create recommendations for a whole dataframe in row by row.

    Args:
        params: dict of pipeline parameters
        td: tag dictionary
        data: dataframe to process
        model: model object. Needs to have a `.predict` method
    Returns:
        recommendations, recommended_controls, projected_optimization
    """

    # Param to print information:
    print_info_controls = True

    if not hasattr(model, "predict"):
        raise ValueError("`model` must have a `predict` method.")

    # do not use parallel processing in the model
    # since we are parallelizing over rows in the dataframe
    try:
        model.set_params(regressor__n_jobs=1)
    except (ValueError, AttributeError):
        pass

    n_jobs = params["n_jobs"]
    datetime_col = params["datetime_col"]
    stopper = None

    if params["stopper"]:
        stopper = make_stopper(params)

    features, controls = _get_features(td, params)
    target = _get_target(td, params)

    if print_info_controls:
        logger.info(f"\n\n\t@bulk_optimize\n\n")
    if print_info_controls:
        logger.info(f"\n\n\t\tcontrols from _get_features = {controls}\n\n")

    for feature in controls:
        if pd.isna(td[feature]["op_min"]) or pd.isna(td[feature]["op_max"]):
            raise ValueError(f"Operating Ranges for f{feature} must be specified.")

    # for now, we show all model-states in the UI
    ui_states = [f for f in features if f not in controls]

    if print_info_controls:
        logger.info(f"\n\n\t\tTags in ui_states = {ui_states}\n\n")

    controls_with_constraints = td.select("constraint_set", pd.notnull)

    def yield_dicts():
        # we iterate over rows as single-row dataframes
        # instead of pd.Series in order to preserve dtypes
        for idx in data.index:
            row = data.loc[[idx], :]

            # _get_on_features is already dropping NaNs controls
            on_controls = _get_on_features(row, td, controls, print_info_controls)

            # _get_in_range_features was applying the drop NaNs controls
            # and also making something similar that we have in get_solver_bounds func
            # Redundant function -> not useful right now.
            #in_range_controls = _get_in_range_features(row, td, on_controls)

            clbrtd_ph_controls = _calibrated_pHs(
                row, on_controls, params["validate_calibrated_phs_sensors"],
                print_info_controls
            )

            # Pass current scores to the object model
            if (model is not None) & (hasattr(model, "current_vector")):
                model.current_vector = row

            # Getting final controls:
            final_controls = clbrtd_ph_controls

            if print_info_controls:
                logger.info(f"\n\n\t\tFinal control tags after '_get_on_features', '_get_in_range_features' and '_calibrated_pHs'\n"
                            f"\t\t\t{final_controls}\n"
                )   

            if final_controls:
                # the normal case: we have at least one control variable
                # that we want to optimize
                row_solver_bounds, final_controls = get_solver_bounds(
                    row, td, final_controls, ramp_up_sags16_17_velocities,
                    revestimiento, ph_standard_data, params, print_info_controls
                )
                
                if print_info_controls:
                    logger.info(f"\n\n\t\tFinal control tags after 'get_solver_bounds'\n"
                                f"\t\t\t{final_controls}\n"
                    )  

                if final_controls:
                    # Pass controls to the object model
                    if (model is not None) & (hasattr(model, "controls")):
                        model.controls = final_controls

                    row_solver = make_solver(params, row_solver_bounds)

                    repairs = [
                        _make_set_repair(td, col)
                        for col in (set(final_controls) & set(controls_with_constraints))
                    ] or None

                    problem = StatefulOptimizationProblem(
                        model,
                        state=row,
                        optimizable_columns=final_controls,
                        penalties=penalties,
                        repairs=repairs,
                        sense="maximize",
                    )
                else:
                    # if the mix of off machines and controls way off their op ranges,
                    # there is no recommendation to be delivered and we simply create
                    # a dummy problem
                    logger.warning("All machine off or controls are way off their ranges")
                    if (model is not None) & (hasattr(model, "controls")):
                        model.controls = []
                    
                    row_solver = None
                    problem = OptimizationProblem(model, sense="maximize")
            else:
                # if all machines are off, there is no recommendation to be
                # produced and we simply create a dummy problem
                logger.warning("All Machines are Off")
                if (model is not None) & (hasattr(model, "controls")):
                    model.controls = []
                
                row_solver = None
                problem = OptimizationProblem(model, sense="maximize")

            yield dict(
                timestamp=row.at[idx, datetime_col],
                row=row,
                ui_states=ui_states,
                controls=controls,
                on_controls=final_controls,
                target=target,
                problem=deepcopy(problem),
                solver=row_solver,
                stopper=deepcopy(stopper),
                model_dict=deepcopy(model_dict),
                obj_model=deepcopy(model),
            )

    # we use imap (lazy pool.map) here to make tqdm work
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            results = list(
                tqdm(pool.imap(_optimize_dict, yield_dicts()), total=len(data))
            )
            pool.close()
            pool.join()
    else:
        results = tqdm(
            [_optimize_dict(kwargs) for kwargs in yield_dicts()], total=len(data)
        )
    uplift_results, control_results, output_results, traj_plot_info = list(
        zip(*results)
    )
    uplift_results = pd.DataFrame(list(uplift_results))
    control_results = pd.DataFrame(list(list(cr for cr in control_results)))
    output_results = pd.DataFrame(list(output_results))
    traj_plot_info = {
        inf["fecha"]: inf["info"] for inf in traj_plot_info if "fecha" in inf.keys()
    }
    return {
        "recommendations": uplift_results,
        "recommended_controls": control_results,
        "projected_optimization": output_results,
        "traj_plot_info": traj_plot_info,
    }
