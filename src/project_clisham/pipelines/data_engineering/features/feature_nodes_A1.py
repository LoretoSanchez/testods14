import pandas as pd

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


def add_a1_features_by_hour(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Add calculated features to the hourly master data table.

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing source data.

    Returns:
        data_feat: Master table with additional calculated features.

    """

    # Create features
    list_features = [data]
    target_flot = create_target_flot(parameters, data, model_name="fa1")
    list_features.append(target_flot)
    mean_lines = create_mean_by_line(parameters, data, model_name="fa1")
    list_features.append(mean_lines)
    fe_cu = create_fe_over_cu(parameters, data, model_name="fa1")
    list_features.append(fe_cu)
    #diff_bancos = create_diff_bancos(parameters, data, model_name="fa1")
    #list_features.append(diff_bancos)
    delta_lines = create_delta_aire_by_line(parameters, data, model_name="fa1")
    list_features.append(delta_lines)
    tph_input = _calculate_tph_targets_a1(parameters, data)
    list_features.append(tph_input)
    dosif = create_dosif_reactives(parameters, data, model_name="fa1")
    list_features.append(dosif)

    # Merge data and created features
    data_feat = merge_tables_on_timestamp(parameters=parameters, df_list=list_features)

    return data_feat


def add_a1_features_by_shift(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Add calculated features to the aggregated-by-shift master data table.

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing source data.

    Returns:
        data_feat: Master table with additional calculated features.

    """

    # Create features
    list_features = [data]
    cuf_features = create_cuf_feature(parameters, data, model_name="fa1")
    list_features.append(cuf_features)

    # Merge data and created features
    data_feat = merge_tables_on_timestamp(parameters=parameters, df_list=list_features)

    return data_feat


def _calculate_tph_targets_a1(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Add tph for s13, s14 and s15.

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.

    Returns:
        data: df with new variable.

    """

    # Select mill columns
    timestamp_col_name = parameters["timestamp_col_name"]
    tph_tags_ma1 = parameters["fa1_flotation_target"]["tph_tags"]

    # Select features
    df = data[[timestamp_col_name] + tph_tags_ma1].copy()

    # Add total TPH
    tag_name = parameters["ma1_target_name"]
    df[tag_name] = df[tph_tags_ma1].sum(axis=1)

    return df[[timestamp_col_name, tag_name]]
