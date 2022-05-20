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
import uuid

from functools import partial
from itertools import product
from multiprocessing import Pool
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from project_clisham.optimus_core.tag_management import TagDict
from optimizer.plotting import best_trajectory_problem_plot
from project_clisham.pipelines.optimization.recommendation.recommendation_nodes import (
    _get_features,
)


logger = logging.getLogger(__name__)


def generate_uuid(data: pd.DataFrame, col_name: str = "run_id") -> pd.DataFrame:
    """Extract the parameters saved in conf
    Args:
        data: original DataFrame
        col_name: name for column for UUID
    Returns:
        DataFrame with UUID added
    Raises:
        AttributeError: When the param does not exist
    """
    columns = data.columns
    data[col_name] = [str(uuid.uuid4()) for _ in range(len(data.index))]
    data = data[[col_name, *columns]]
    return data


def filter_timestamp_optimization(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Filters the dataframe to only keep the timestamps needed for optimization

    Args:
        params: dict of pipeline parameters
        data: dataframe to process
    Returns:
        dataframe for optimization
    """
    datetime_col = params["datetime_col"]
    data.set_index(datetime_col, inplace=True)
    begin_date = params["filter_timestamps"]["begin_date"]
    end_date = params["filter_timestamps"]["end_date"]
    data = data[begin_date:]

    selection_type = params["filter_timestamps"]["type"]
    if selection_type not in ["beginning", "end", "date"]:
        raise RuntimeError(
            """
            Valid arguments are "beginning", "end" or "date". Please, fix it.
            """
        )
    values = params["filter_timestamps"][selection_type]
    if (selection_type in ["beginning", "end"]) & (not isinstance(values, int)):
        raise RuntimeError(
            """
            Valid arguments are integers
            """
        )
    if (selection_type in ["date"]) & (not isinstance(values, list)):
        raise RuntimeError(
            """
            Valid arguments is a list of timestamps with format %Y-%m-%d %H:%M:%S
            """
        )
    if selection_type == "beginning":
        data = data[:values]
    elif selection_type == "end":
        data = data[-values:]
    else:
        index_list = data.index.strftime(date_format="%Y-%m-%d %H:%M:%S").tolist()
        # logger.info(index_list)
        if values in index_list:
            data = data.loc[pd.to_datetime(values)]
        else:
            in_there = list(set(values).intersection(set(index_list)))
            if len(in_there) == 0:
                raise RuntimeError(
                    """
                    None of the selected timestamps are in the test set
                    """
                )
            else:
                data = data.loc[pd.to_datetime(in_there)]
        data.index = data.index.set_names([datetime_col])
    data.reset_index(inplace=True)
    return data


def _filter_stable_conditions(data, conditions):

    for condition in conditions:
        min_val = conditions[condition][0]
        max_val = conditions[condition][1]
        if min_val > max_val:
            raise RuntimeError(
                """
            The min value is above the max value
            """
            )
        else:
            cond = (data[condition] > min_val) & (data[condition] < max_val)
            data = data[cond]
    return data


def generate_recommendation_csv(td: TagDict, params: dict, recomm):
    """
    Process the JSON with recommendations to have nice csv

    Args:
        td: Tag dictionary.
        recomm: recommendations
    Returns:
        objective df
        controls df
    """
    context_cols = params["context_variables"]
    ts = params["timestamp_col"]
    tag_dict = td.to_frame().set_index("tag")
    controls_tags = list(
        set([con for x in range(len(recomm)) for con in recomm["controls"][x].keys()])
    )
    all_controls = pd.DataFrame(index=controls_tags + context_cols)
    all_controls = all_controls.merge(
        tag_dict[["description", "area", "tag_type"]],
        how="left",
        right_index=True,
        left_index=True,
    )
    all_objective = pd.DataFrame()
    for row in recomm.index:
        timestamp = recomm[ts][row]
        controls = pd.DataFrame.from_dict(recomm["controls"][row]).T
        context = pd.DataFrame.from_dict(
            recomm["context"][row], columns=[timestamp], orient="index"
        )
        context.columns = ["current"]
        context = context.loc[context_cols, :]
        controls = controls.append(context)
        controls.columns = [
            col + "_" + str(timestamp).replace(" ", "_") for col in controls.columns
        ]
        all_controls = all_controls.merge(
            controls, how="left", right_index=True, left_index=True
        )
        objective = pd.DataFrame.from_dict(
            recomm["outputs"][row], columns=[timestamp], orient="index"
        ).T
        for model in recomm["models"][row]:
            model_info = pd.DataFrame.from_dict(
                recomm["models"][row][model],
                columns=[timestamp],
                orient="index",
            ).T
            objective = objective.merge(
                model_info, how="left", right_index=True, left_index=True
            )
        all_objective = all_objective.append(objective, sort=True)

    all_objective["uplift_pred"] = 100 * (
        (all_objective["target_pred_optimized"] - all_objective["target_pred_current"])
        / all_objective["target_pred_current"]
    )
    all_controls = all_controls.sort_values("description").T

    return dict(all_objective=all_objective, all_controls=all_controls)


def _extract_state_row(row: dict) -> pd.DataFrame:
    """
    extracts state data from optimization result.
    """
    index = pd.Index([row["run_id"]], name="run_id")
    state_columns = row["state"].keys()

    state_row = pd.DataFrame(index=index, columns=state_columns, dtype=float)
    for col in state_columns:
        state_row.loc[row["run_id"], col] = row["state"][col]
    return state_row


def _extract_ctrl_row(row: dict) -> pd.DataFrame:
    """
    extracts controls data from optimization result.
    """
    ctrl_row = pd.DataFrame()
    if len(row["controls"]) > 0:
        index = pd.Index([row["run_id"]], name="run_id")
        ctrl_columns = list(
            product(row["controls"].keys(), ["current", "suggested", "delta"])
        )
        ctrl_row = pd.DataFrame(
            index=index, columns=pd.MultiIndex.from_tuples(ctrl_columns), dtype=float
        )
        for ctrl, status in ctrl_columns:
            ctrl_row.loc[row["run_id"], (ctrl, status)] = row["controls"][ctrl][status]
    return ctrl_row


def _extract_output_row(row: dict, target: str) -> pd.DataFrame:
    """
    extracts output data from optimization result.
    """
    index = pd.Index([row["run_id"]], name="run_id")
    output_columns = list(product([target], row["outputs"].keys()))
    output_row = pd.DataFrame(
        index=index, columns=pd.MultiIndex.from_tuples(output_columns), dtype=float
    )
    for _, outp in output_columns:
        output_row.loc[row["run_id"], (target, outp)] = row["outputs"][outp]
    return output_row


def create_bulk_result_tables(
    params: dict, td: TagDict, recommendations: pd.DataFrame, opt_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Creates a bulk-optimization output table.

    Args:
        params: dict of pipeline parameters
        td: tag dictionary
        recommendations: output of bulk optimization
        opt_df: opt data
    Returns:
        states, controls, outcomes
    """
    n_jobs = params["n_jobs"]
    opt_target = params["opt_target"]
    target = td.select("target", opt_target)[0]

    recommendations = recommendations.to_dict(orient="records")

    with Pool(n_jobs) as pool:
        all_states = list(pool.map(_extract_state_row, recommendations))
        all_ctrls = list(pool.map(_extract_ctrl_row, recommendations))
        all_outputs = list(
            pool.map(partial(_extract_output_row, target=target), recommendations)
        )

    output_df = pd.concat(all_outputs)
    state_df = pd.concat(all_states)
    ctrl_df = pd.concat(all_ctrls)

    # append true opt set outcomes to output
    actual_df = pd.DataFrame(
        data=opt_df[target].values,
        columns=pd.MultiIndex.from_tuples([(target, "actual")]),
        index=pd.Index(name="run_id", data=opt_df["run_id"]),
    )
    output_df = pd.merge(
        output_df, actual_df, how="left", left_index=True, right_index=True
    )

    # calculate deltas
    output_df[(target, "optimized_vs_predicted")] = (
        output_df[(target, "target_pred_optimized")]
        - output_df[(target, "target_pred_current")]
    )
    output_df[(target, "optimized_vs_predicted_pct")] = (
        output_df[(target, "optimized_vs_predicted")]
        / output_df[(target, "target_pred_current")]
        * 100
    )
    output_df[(target, "optimized_vs_actual")] = (
        output_df[(target, "target_pred_optimized")] - output_df[(target, "actual")]
    )
    output_df[(target, "optimized_vs_actual_pct")] = (
        output_df[(target, "optimized_vs_actual")] / output_df[(target, "actual")] * 100
    )

    return {"states": state_df, "controls": ctrl_df, "outcomes": output_df}


def generate_optim_diag_plots(traj_data: Dict) -> Dict:
    """Extract the info from the optimization and generate
        penalty plots
    Args:
        data: trajectory data
    Returns:
        Dictionary with plots

    """
    plots_dict = dict()
    for ts in traj_data:
        shift = traj_data[ts]
        fig, ax = best_trajectory_problem_plot(
            shift["trajectory_logger"],
            shift["optim_problem"],
            shift["bounds"],
            normalize_penalties=False,
        )
        ts_clean = ts.replace("/", "_")
        plots_dict[f"traj_plot_{ts_clean}.png"] = fig
    return plots_dict


def _get_off_features(
    current_value: pd.DataFrame, td: TagDict, controls: List[str]
) -> List[str]:
    """
    This function checks which control tags are turned off by the on_off features,
    so a report of turned-off control tags can be created.

    Inspired by _get_on_features @ recommendation_nodes.py

    The reason: do not print what already has been checked by _get_on_features.

    Args:
        current_value: The turn to be optimized.
        td: Tag dict master object.
        controls: The list of control tags.
    Returns:
        A list (subset of controls) of control tags that have been turned off.
    """
    off_controls = []
    for feature in controls:
        on_flag = all(
            [current_value[d].iloc[0] > 0.5 for d in td.dependencies(feature)]
        )
        if not on_flag:
            off_controls.append(feature)
    return off_controls


def _get_off_range_features(
    current_value: pd.DataFrame, td: TagDict, controls: List[str]
) -> List[str]:
    """
    This function checks which control tags are turned off by the min-max
    criteria, so a general report of turned-off control tags can be created.

    Inspired by _get_in_range_features @ recommendation_nodes.py

    The reason: do not print what already has been checked by _get_in_range_features.

    Args:
        current_value: The turn to be optimized.
        td: Tag dict master object.
        on_controls: The list of control tags.
    Returns:
        A list (subset of controls) of control tags that have been turned off.
    """

    off_range_controls = []
    for feature in controls:
        control_entry = td[feature]
        op_min = control_entry["op_min"]
        op_max = control_entry["op_max"]
        max_delta = control_entry["max_delta"]
        if np.isnan(current_value[feature].iloc[0]):
            off_range_controls.append(feature)
        else:
            in_range_flag = (
                (op_min - max_delta)
                <= current_value[feature].iloc[0]
                <= (op_max + max_delta)
            )
            if not in_range_flag:
                off_range_controls.append(feature)

    return off_range_controls


def generate_dropped_tags_report(
    params: dict, td: TagDict, data: pd.DataFrame
) -> pd.DataFrame:
    """This function constructs a report for the control tags that are
        turned-off along all the turns of the optimization exercise (with
        on_off logic) and min-max criteria.

        This uses an utility function '_get_off_features' created to
        spot all the turned off controls.

    Args:
        td: tag dict master object.
        data: a dataframe that contains all the optimization turns to be
              analyzed.
        params: The parameters object.
    Returns:
        pd.DataFrame: which contains as index all the control tags passed
                      as input, the percentage of all turns that has been
            turned off and the detail of which turn has been turned down.

    """
    # Getting the features (unused) and the control tags:
    features, controls = _get_features(td, params)

    # Initializing the output dataframe with control tags:
    df_output = pd.DataFrame({"tag": controls})
    df_output.set_index("tag", inplace=True)

    # Initializing the list that will help reordering the output dataframe:
    list_output_cols = ["description", "porc_off_all_turns", "porc_off_range_all_turns"]
    list_off_flag_cols = []
    list_off_range_flag_cols = []

    # Cycling over all the optimization turns:
    for idx in data.index:
        # Getting the i-th turn of the optimization:
        row = data.loc[[idx], :]

        # Checking those that are off:
        off_controls = _get_off_features(row, td, controls)

        # Checking those that are off-range:
        off_range_controls = _get_off_range_features(row, td, controls)

        # Getting a formatted time for 'fecha':
        fecha = str(row["fecha"].values[0])[:-16]

        # Getting the column labels that will be shown in the output:
        output_label_off = "off_flag_" + str(fecha)
        output_label_off_range = "off_range_flag_" + str(fecha)

        # Storing them in the output list of columns:
        list_output_cols.append(output_label_off)
        list_output_cols.append(output_label_off_range)
        list_off_flag_cols.append(output_label_off)
        list_off_range_flag_cols.append(output_label_off_range)

        ##############
        ## OFF_FLAG ##
        ##############

        # Creating a dataframe with those controls and off_flag for the timestamp:
        df_ith = pd.DataFrame(
            {"tag": off_controls, output_label_off: [1] * len(off_controls)}
        )
        df_ith.set_index("tag", inplace=True)

        # Merging that iterative dataframe to the output of the method:
        df_output = df_output.merge(df_ith, how="left", right_on="tag", left_on="tag")

        ####################
        ## OFF_RANGE_FLAG ##
        ####################

        # Creating a dataframe with those controls and off_flag for the timestamp:
        df_ith = pd.DataFrame(
            {
                "tag": off_range_controls,
                output_label_off_range: [1] * len(off_range_controls),
            }
        )
        df_ith.set_index("tag", inplace=True)

        # Merging that iterative dataframe to the output of the method:
        df_output = df_output.merge(df_ith, how="left", right_on="tag", left_on="tag")

    # Filling with 0 all NaN values created by successive merges:
    df_output = df_output.fillna(0.0)

    # Creating the percentage optimization turns that the tags have been dropped
    # by OFF FLAG:
    df_output["porc_off_all_turns"] = df_output[list_off_flag_cols].apply(
        lambda z: sum(z) / data.shape[0] * 100.0, axis=1
    )

    # Creating the percentage optimization turns that the tags have been dropped by
    # OFF RANGE FLAG:
    df_output["porc_off_range_all_turns"] = df_output[list_off_range_flag_cols].apply(
        lambda z: sum(z) / data.shape[0] * 100.0, axis=1
    )

    # Sorting the resulting dataframe with the controls that are mostly turned off:
    df_output["order"] = df_output[
        ["porc_off_all_turns", "porc_off_range_all_turns"]
    ].sum(axis=1)
    df_output = df_output.sort_values(by="order", ascending=False)

    # Adding the tag description and the TAG as index:
    df_description = td._data[["tag", "description"]]
    df_description.set_index("tag", inplace=True)
    df_output = df_output.merge(
        df_description, how="left", left_index=True, right_index=True
    )

    # Returning the dataframe:
    return df_output[list_output_cols]


def generate_sub_optimal_report(
    td: TagDict,
    rep_optim_objective: pd.DataFrame
) -> Dict: #pd.DataFrame:
    """This function creates two reports showing when the tags have sub_optimal
    estimations. 
    
        In the first report, the subobtimal is assumed when 'optimized' 
    values are equal to the 'current' ones.

        In the second report, the suboptimal is assumed when 'optimized'
    values are lower than the 'predicted' ones.

    Args:
        rep_optim_objective: The report of recommendations for all the shifts.
    Returns:
        dict: A dictionary with values the two reports stored in the 'df_result'
              and the 'df_result2' dataframes.
              Keys: 'df_suboptimal_targets_report_Curr_Eq_Opt' and
                    'df_suboptimal_targets_report_Opt_LT_Curr'.
              Values: pd.DataFrame: The desired report: the indexes are the tags, the first
                      column is the percentage of all shifts that have a suboptimal value, 
                      and the binary columns per all shifts in the excercise.
    """

    # ---------- #
    #  REPORT 1  #
    # --------------------------------------------------------------------------- #
    #    The actual report creates a table with the amount of times in all shifts #
    # given to the optimizer, the "current" and "optimized" values are the same.  #
    # --------------------------------------------------------------------------- #

    # Preparing the input df:
    df_rep_optim_obj = rep_optim_objective.rename(columns = {'Unnamed: 0': 'Fecha'}).copy()
    df_rep_optim_obj.set_index('Fecha', inplace = True)

    # Initializing the list of tags to be monitorized:
    list_clean_tags = []
    
    # Cycling over all the dataframe:
    for col_j in df_rep_optim_obj.columns:
        # Getting the cleaned target:
        lista = [word_k for word_k in col_j.split('_')]
        list_clean_tags.append('_'.join(lista[:-1]))

    # Filtering the list:
    list_clean_tags = list(set(list_clean_tags))

    # Validating if all of them have '_current' and '_optimized' information:
    list_clean_tags2 = []
    for tag_k in list_clean_tags:
        if ((tag_k + '_current') in df_rep_optim_obj.columns) and\
           ((tag_k + '_optimized') in df_rep_optim_obj.columns) :
            list_clean_tags2.append(tag_k)

    # Getting the description of those tags:
    list_tag_description = []
    for tag_k in list_clean_tags2:
        try:
            desc_k = td._data[td._data['tag'] == tag_k]['description'].values[0]
        except:
            desc_k = tag_k
        list_tag_description.append(desc_k)
    
    # Initializing the list of columns to keep at the end:
    list_df_result_cols = ['Tag Description', '% Shifts with Curr = Opt']

    # Initializing the output report:
    df_result = pd.DataFrame({'Fecha': df_rep_optim_obj.index}).set_index('Fecha')

    # Cycling to check if all these tag have '_current' and '_optimized' versions:
    for tag_k in list_clean_tags2:
        # Getting a copy of the subset required:
        df_subset = df_rep_optim_obj[[tag_k + '_current', tag_k + '_optimized']].copy()
        df_subset[tag_k] = df_subset.apply(lambda j: 1 if (j[1] == j[0]) else 0, axis = 1)
        
        # Merging this information to the df_result:
        df_result = df_result.merge(df_subset[tag_k], how = 'left', left_index = True, right_index = True)
        
    # Transposing the result:
    df_result = df_result.T

    # Adding the sufix to differentiate the report:
    df_result.columns = [(col_j + '_Curr_Equal_Opt_Flag') for col_j in df_result.columns]
    for name_k in df_result.columns:
        list_df_result_cols.append(name_k)
    
    # Estimating the percentage of 'Curr = Opt' cases in total shifts:
    df_result['% Shifts with Curr = Opt'] = df_result.mean(axis = 1)

    # Adding the description of tags:
    df_result['Tag Description'] = list_tag_description

    # Final touches:
    df_result = df_result[list_df_result_cols]
    df_result = df_result.sort_values(by = '% Shifts with Curr = Opt', ascending = False)


    # ---------- #
    #  REPORT 2  #
    # --------------------------------------------------------------------------- #
    #    The actual report creates a table with the amount of times in all shifts #
    # given to the optimizer, the "optimized" < "pred".                           #
    # --------------------------------------------------------------------------- #

    # Validating if all of them have '_optimized' and '_predicted' information:
    list_clean_tags3 = []
    for tag_k in list_clean_tags:
        if ((tag_k + '_optimized') in df_rep_optim_obj.columns) and\
           ((tag_k + '_predicted') in df_rep_optim_obj.columns):
            list_clean_tags3.append(tag_k)

    # Getting the description of those tags:
    list_tag_description2 = []
    for tag_k in list_clean_tags3:
        try:
            desc_k = td._data[td._data['tag'] == tag_k]['description'].values[0]
        except:
            desc_k = tag_k
        list_tag_description2.append(desc_k)

    # Initializing the list of columns to keep at the end:
    list_df_result2_cols = ['Tag Description', '% Shifts with Opt < Pred']

    # Initializing the output report:
    df_result2 = pd.DataFrame({'Fecha': df_rep_optim_obj.index}).set_index('Fecha')

    # Cycling to check if all these tag have '_optimized' and '_predicted' versions:
    for tag_k in list_clean_tags3:
        # Getting a copy of the subset required:
        df_subset2 = df_rep_optim_obj[[tag_k + '_optimized', tag_k + '_predicted']].copy()
        df_subset2[tag_k] = df_subset2.apply(lambda j: 1 if (j[0] < j[1]) else 0, axis = 1)
        
        # Merging this information to the df_result:
        df_result2 = df_result2.merge(df_subset2[tag_k], how = 'left', left_index = True, right_index = True)
        
    # Transposing the result:
    df_result2 = df_result2.T

    # Adding the sufix to differentiate the report:
    df_result2.columns = [(col_j + '_Opt_<_Pred_Flag') for col_j in df_result2.columns]
    for name_k in df_result2.columns:
        list_df_result2_cols.append(name_k)
    
    # Estimating the percentage of 'Curr = Opt' cases in total shifts:
    df_result2['% Shifts with Opt < Pred'] = df_result2.mean(axis = 1)

    # Adding the description of tags:
    df_result2['Tag Description'] = list_tag_description2

    # Final touches:
    df_result2 = df_result2[list_df_result2_cols]
    df_result2 = df_result2.sort_values(by = '% Shifts with Opt < Pred', ascending = False)

    # Returning the two dataframes with the reports:
    return {'df_suboptimal_targets_report_Curr_Eq_Opt': df_result, 
            'df_suboptimal_targets_report_Opt_LT_Pred': df_result2}