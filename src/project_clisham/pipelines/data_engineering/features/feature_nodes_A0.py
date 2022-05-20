import pandas as pd
import numpy as np

from project_clisham.optimus_core.utils import merge_tables_on_timestamp

from .features_nodes import (
    create_target_flot,
    create_mean_by_line,
    create_fe_over_cu,
    create_delta_aire_by_line,
    create_feature_lags,
    create_cuf_feature,
    create_dosif_reactives,
    create_diff_bancos,
)


def add_a0_features_by_hour(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Add calculated features to the hourly master data table.

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing source data.

    Returns:
        data_feat: Master table with additional calculated features.

    """

    # Create features
    list_features = [data]
    target_flot = create_target_flot(parameters, data, model_name="fa0")
    list_features.append(target_flot)
    mean_lines = create_mean_by_line(parameters, data, model_name="fa0")
    list_features.append(mean_lines)
    fe_cu = create_fe_over_cu(parameters, data, model_name="fa0")
    list_features.append(fe_cu)
    #diff_bancos = create_diff_bancos(parameters, data, model_name="fa0")
    #list_features.append(diff_bancos)
    delta_lines = create_delta_aire_by_line(parameters, data, model_name="fa0")
    list_features.append(delta_lines)
    dosif = create_dosif_reactives(parameters, data, model_name="fa0")
    list_features.append(dosif)
    tph_input = _calculate_tph_targets_a0(parameters, data)
    list_features.append(tph_input)
    sol_flot = _calculate_weighted_a0(parameters, data, variable="sol_flota")
    list_features.append(sol_flot)
    psi_malla = _calculate_weighted_a0(parameters, data, variable="psi_malla")
    list_features.append(psi_malla)
    pres_bhc = _calculate_weighted_a0(parameters, data, variable="pres_bhc")
    list_features.append(pres_bhc)
    pot_bolas = _calculate_binary_pot_mol_bolas(parameters, data)
    list_features.append(pot_bolas)

    # Merge data and created features
    data_feat = merge_tables_on_timestamp(parameters=parameters, df_list=list_features)

    return data_feat


def _calculate_tph_targets_a0(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Add tph for lines in a0

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.

    Returns:
        data: df with new variable.

    """

    # Select mill columns
    timestamp_col_name = parameters["timestamp_col_name"]
    lineas = parameters["ma0_lineas"]
    tph_tags_ma0 = [
        tags for line in range(lineas) for tags in parameters[f"s{line}_tph"]
    ]

    # Select features
    df = data[[timestamp_col_name] + tph_tags_ma0].copy()

    # Add total TPH
    tag_name = parameters["ma0_target_name"]
    df[tag_name] = df[tph_tags_ma0].sum(axis=1)

    return df[[timestamp_col_name, tag_name]]


def _calculate_weighted_a0(
    parameters: dict, data: pd.DataFrame, variable: str
) -> pd.DataFrame:
    """Add tph for lines in a0

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.

    Returns:
        data: df with new variable.

    """
    # Select mill columns
    timestamp_col_name = parameters["timestamp_col_name"]
    lineas = parameters["ma0_lineas"]
    tag_name = parameters[f"ma0_{variable}"]
    tph_tags_ma0 = [
        tags for line in range(lineas) for tags in parameters[f"s{line}_tph"]
    ]
    sol_tags_ma0 = [
        tags for line in range(lineas) for tags in parameters[f"s{line}_{variable}"]
    ]

    # Select features
    df = data[[timestamp_col_name] + tph_tags_ma0 + sol_tags_ma0].copy()

    # calculate weighted % solids
    df["total"] = 0
    for line in range(lineas):
        vars_tph = parameters[f"s{line}_tph"]
        var_sol = parameters[f"s{line}_{variable}"][0]
        tot_variable = df[vars_tph].sum(axis=1)
        df["variable"] = tot_variable * df[var_sol]
        df["total"] = df[["variable", "total"]].sum(axis=1)

    # total TPH
    total_tph = df[tph_tags_ma0].sum(axis=1)
    df[tag_name] = df["total"] / total_tph

    return df[[timestamp_col_name, tag_name]]


def _calculate_binary_pot_mol_bolas(
    parameters: dict, data: pd.DataFrame
) -> pd.DataFrame:
    """Add tph for lines in a0

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.

    Returns:
        data: df with new variable.

    """
    # Select mill columns
    timestamp_col_name = parameters["timestamp_col_name"]
    lineas = parameters["ma0_lineas"]
    potencias = parameters["pot_mol_bolas"]
    tag_name = potencias["tag_prefix"]
    threshold = potencias["threshold"]

    pot_tags_ma0 = [tags for line in range(lineas) for tags in potencias[f"s{line}"]]

    # Select features
    df = data[[timestamp_col_name] + pot_tags_ma0].copy()

    # calculate weighted % solids
    new_vars = []
    for line in range(lineas):
        vars_pot = potencias[f"s{line}"]
        new_var = tag_name + str(line)
        df[new_var] = 1
        for var in vars_pot:
            df.loc[df[var] < threshold, new_var] = 0
        new_vars.append(new_var)

    return df[[timestamp_col_name] + new_vars]


def add_a0_features_by_shift(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Add calculated features to the aggregated-by-shift master data table.

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing source data.

    Returns:
        data_feat: Master table with additional calculated features.

    """

    # Create features
    list_features = [data]
    cuf_features = create_cuf_feature(parameters, data, model_name="fa0")
    list_features.append(cuf_features)

    # Merge data and created features
    data_feat = merge_tables_on_timestamp(parameters=parameters, df_list=list_features)

    return data_feat
