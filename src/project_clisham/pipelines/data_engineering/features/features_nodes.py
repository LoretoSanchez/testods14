import pandas as pd
import numpy as np

from project_clisham.optimus_core.tag_management import TagDict
from project_clisham.optimus_core.utils import (
    cut_values_from_dict,
    cut_values_from_list,
)
from project_clisham.optimus_core.utils import merge_tables_on_timestamp
from typing import Dict, List, Union

def create_tag_dict(tag_dict_csv) -> TagDict:
    return TagDict(tag_dict_csv)

def map_columns_names(df: pd.DataFrame, 
    params: Dict):
    """
    Map names due new subscription
    Args:
        df: Pandas DataFrame with all data with
        normalized columns
        params: Parameters containing mapping dict.
    Returns:
        Cleaned DataFrame
    """
    df.rename(columns = params["mapping_columns"], inplace = True)
    return df

def add_on_off_features_by_shift(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Add on off features to the shifted master data table.

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing source data.

    Returns:
        data_feat: Master table with additional calculated features.

    """

    # Create features
    list_features = [data]
    on_off = create_on_off(parameters, data)
    list_features.append(on_off)
    of_off_psi = create_on_off_psi(parameters, data)
    list_features.append(of_off_psi)

    # Merge data and created features
    data_feat = merge_tables_on_timestamp(parameters=parameters, df_list=list_features)

    return data_feat


def add_across_features_by_hour(parameters: dict, data: pd.DataFrame,
    data_traza_clean_intermediate: pd.DataFrame, td:TagDict) -> pd.DataFrame:
    """Add calculated features to the hourly master data table.

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing source data.
        data_traza_clean_intermediate: DataFrame with traza intermediate info.
        td: TagDict.
    Returns:
        data_feat: Master table with additional calculated features.

    """

    # Create features
    list_features = [data]
    group_mean = create_group_mean(parameters, data)
    list_features.append(group_mean)
    group_sum = create_group_sum(parameters, data)
    list_features.append(group_sum)
    group_sol = create_group_sol(parameters, data)
    list_features.append(group_sol)
    specific_power = calculate_specific_power(parameters, data, td)
    list_features.append(specific_power)

    # Adding the cleaned 'data_traza_clean_intermediate':
    feats_traza = features_from_trazabilidad(parameters, data_traza_clean_intermediate)
    list_features.append(feats_traza)
    feats_bhc_solids = create_solid_in_cyclons(parameters, data)
    list_features.append(feats_bhc_solids)
    feats_ch23 = calculate_grinding_features(parameters, data)
    list_features.append(feats_ch23)

    # Merge data and created features
    data_feat = merge_tables_on_timestamp(parameters=parameters, df_list=list_features)

    return data_feat


def group_by_shift(parameters, df) -> pd.DataFrame:
    """group by shift

    Args:
        parameters: dictionary with parameters
        df: all data

    Returns:
        pd.DataFrame: df grouped
    """

    # Getting the average per shift:
    df_agg = get_mean_by_shift(parameters, df)
    list_features = [df_agg]

    # Merge data and created features
    data_grouped = merge_tables_on_timestamp(parameters=parameters, df_list=list_features) 
        
    # Returning the aggregated dataframe:
    return data_grouped

def get_mean_by_shift(parameters, df) -> pd.DataFrame:
    """
        The purpose of this method is to get the average of all the
    features in the parameter df with the parameters specified.

    The output of this method is the grouped-meaned dataframe with the
    datetime, not an index.
    """
    # Getting the parameters for grouping:
    freq_group_rule = parameters["group_shift_freq"]
    col_timestamp = parameters["timestamp_col_name"]
    offset = parameters["grouping_offset"]

    # Creating the average of all the tags in df:
    if freq_group_rule!='1H':
        agg_df = df.resample(
                rule=freq_group_rule,
                on=col_timestamp,
                offset=offset,
                closed="right",
                label="right",
        ).mean()
    else:
        agg_df = df.copy()

    ##dropping left overs due shifts
    agg_df = agg_df.reset_index()

    ##adding an extra hour due we right close
    left_overs_cut = (df[col_timestamp].max() + pd.Timedelta(freq_group_rule) 
                    - pd.Timedelta('1H'))
    skip_non_full_shifts = agg_df[col_timestamp]<left_overs_cut
    agg_df = agg_df[skip_non_full_shifts]

    return agg_df

def get_last_value_by_shift(parameters, df) -> pd.DataFrame:
    """
        The purpose of this method is to get the last value of all the
    features in the parameter df with the specified parameters.

    The output of this method is the grouped-last_value dataframe with the
    datetime, not an index.
    """
    # Getting the parameters for grouping:
    freq_group_rule = parameters["group_shift_freq"]
    col_timestamp = parameters["timestamp_col_name"]
    offset = parameters["grouping_offset"]
    tags = parameters["last_val_grouping"]["tags"]
    tag_prefix = parameters["last_val_grouping"]["tag_prefix"]
    tags.append(col_timestamp)

    _df = df[tags].copy().fillna(method="ffill")
    # Taking the last value of all the tags in df:
    agg_df = _df.resample(
            rule=freq_group_rule,
            on=col_timestamp,
            offset=offset,
            closed="right",
            label="right",
    ).last()
    agg_df.drop(columns = col_timestamp , inplace = True)
    agg_df = agg_df.add_prefix(tag_prefix)

    ##dropping left overs due shifts
    agg_df = agg_df.reset_index()

    ##adding an extra hour due we right close
    left_overs_cut = (df[col_timestamp].max() + pd.Timedelta(freq_group_rule) 
                    - pd.Timedelta('1H'))
    skip_non_full_shifts = agg_df[col_timestamp]<left_overs_cut
    agg_df = agg_df[skip_non_full_shifts]

    return agg_df

def add_cum_sum_RampUp_Standard_Sags(parameters, revestimiento, df_hist_tphs_sags16_17) -> pd.DataFrame:
    """
        The purpose of this method is to get the cummulate sum for the
    TPH tags for the SAG Models, and restart the values when a change in
    a 'revestimiento' change is done.

    """
    # Getting the cumsum prefix:
    cumsum_prefix = parameters['cumsum_grouping']['tag_prefix']

    # Getting the grouping parameters:
    freq_group_rule = parameters["group_shift_freq"]
    col_timestamp = parameters["timestamp_col_name"]
    offset = parameters["grouping_offset"]

    # Preparing the 'revestimiento' file:
    revs_df = revestimiento.copy()

    # Sorting and restarting the index of the table 'df_hist_tphs_sags16_17':
    df_hist_tphs_sags16_17.sort_index(inplace = True)
    df_hist_tphs_sags16_17.reset_index(inplace = True)

    # Initializing the list of tags to get the cumsum:
    list_tags_2_cum_sum = []

    # Initializing the list of new tags to filter at the end:
    list_of_new_tags = []

    # Cycling over all the groups:
    for cs_group_j in parameters['cumsum_grouping']['groups']:
        for tag_k in cs_group_j:
            list_tags_2_cum_sum.append(tag_k)

    # Sorting that table before cumulate the sum:
    df_hist_tphs_sags16_17.sort_values(by = col_timestamp, inplace = True)

    # Cycling over all the tags to get_cum_sum:
    for tag_j in list_tags_2_cum_sum:
        new_tagj = cumsum_prefix + tag_j
        df_hist_tphs_sags16_17[new_tagj] = df_hist_tphs_sags16_17[tag_j].fillna(0.0).cumsum()
        list_of_new_tags.append(new_tagj)

    # Initializing a dictionary:
    dict_work = {'s16_ucf': [cumsum_prefix + tagj for tagj in parameters['cumsum_grouping']['groups'][0]],
                 's17_ucf': [cumsum_prefix + tagj for tagj in parameters['cumsum_grouping']['groups'][1]]}
    
    # Cycling over the SAGs to modify the cumulated values:
    for sagk in dict_work:
        # Cycling over all the tags of interest to update:
        for tag_i in dict_work[sagk]:
            # Cycling for all the change in revestimiento for the sag 16:
            for date_k in revs_df[sagk]:
                # Getting the time filter:
                time_filter = df_hist_tphs_sags16_17[col_timestamp] >= date_k

                # This is a validation that the df has information beyond the date_k
                if sum(time_filter) > 0:
                    # Getting the value to restart the cummulated TPH:
                    #min_value_to_deduce =\
                    #    min(df_hist_tphs_sags16_17[df_hist_tphs_sags16_17[col_timestamp] >= date_k][tag_i])
                    min_value_to_deduce = \
                        df_hist_tphs_sags16_17[time_filter][tag_i].values[0]
                    
                    # Deducing the that minimun value from date_k until the end:
                    df_hist_tphs_sags16_17.loc[time_filter, tag_i] =\
                         df_hist_tphs_sags16_17.loc[time_filter, tag_i] -\
                              min_value_to_deduce
    
    df_hist_tphs_sags16_17 = df_hist_tphs_sags16_17.ffill()
    # Creating the average of all the tags in df:
    agg_df = df_hist_tphs_sags16_17[[col_timestamp] + list_of_new_tags].resample(
                        rule=freq_group_rule,
                        on=col_timestamp,
                        offset=offset,
                        closed="right",
                        label="right",
    ).last() #.sum()
    agg_df = agg_df.ffill()
    del agg_df[col_timestamp]

    # Reseting the index:
    agg_df.reset_index(inplace = True)

    # Returning the aggregated dataframe:
    return agg_df[[col_timestamp] + list_of_new_tags]


def create_target_counts(parameters, td, df_raw, df_agg) -> pd.DataFrame:
    """Append counts of target to aggregate dataframe

    Args:
        parameters: dictionary with parameters
        td: Tag dictionary.
        df_raw: all data
        df_agg: grouped dataframe

    Returns:
        pd.DataFrame: df grouped
    """
    freq_group_rule = parameters["group_shift_freq"]
    col_timestamp = parameters["timestamp_col_name"]
    offset = parameters["grouping_offset"]
    targets = td.get_targets()
    # Get raw target name
    raw_tar = [
        target if len(target.split("lag_")) == 1 else target.split("lag_")[1]
        for target in targets
    ]
    dict_agg = {}
    for var in raw_tar:
        dict_agg[var] = "count"
    agg_counts = df_raw.resample(
        rule=freq_group_rule,
        offset=offset,
        on=col_timestamp,
        closed="right",
        label="right",
    ).agg(dict_agg)

    agg_counts.columns = ["calc_count_" + col for col in agg_counts.columns]
    agg_counts.reset_index(inplace=True)

    df_agg = df_agg.merge(
        agg_counts,
        how="left",
        right_on=col_timestamp,
        left_on=col_timestamp,
        validate="1:1",
    )
    return df_agg


def create_target_lags(
    td: TagDict, df_agg: pd.DataFrame, parameters: dict
) -> pd.DataFrame:
    """Append lags of targets to aggregated dataframe.

    Args:
        td: Tag dictionary.
        df_agg: Grouped dataframe.
        parameters: Dictionary of parameters.

    Returns:
        pd.DataFrame: df grouped
    """
    col_timestamp = parameters["timestamp_col_name"]
    freq = parameters["group_shift_freq"]
    targets = td.get_targets()
    # Get raw target name
    raw_tar = [
        target if len(target.split("lag_")) == 1 else target.split("lag_")[1]
        for target in targets
    ]

    df_agg.set_index(col_timestamp, inplace=True)
    shifts = df_agg[raw_tar].shift(freq="-" + freq)
    shifts.columns = ["calc_p1_lag_" + col for col in shifts.columns]
    df_agg = df_agg.merge(
        shifts, how="left", right_index=True, left_index=True, validate="1:1"
    )
    shifts_back = df_agg[raw_tar].shift(freq=freq)
    shifts_back.columns = ["calc_m1_lag_" + col for col in shifts_back.columns]
    df_agg = df_agg.merge(
        shifts_back, how="left", right_index=True, left_index=True, validate="1:1"
    )
    df_agg.reset_index(inplace=True)

    return df_agg

def create_solid_in_cyclons(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """
    A function that calculates the solid proportion in cyclons regarding to pulp density in their inputs.
    
    Args:
        parameters: Dictionary of parameters.
        data: Master data table containing pulp density features for each section.
    
    Returns:
        df: A DataFrame with the solid proportions calculated.
    """
    density_tags = parameters["densidad_bhc"]
    
    # We create a sub-DataFrame which holds only the information referred to pulp's density before the cyclons
    # in conventional grinding (because, in SAG grinding, this same information has poor quality)
    tag_list = [parameters["timestamp_col_name"]]
    for i in range(len(density_tags)):
        tag_list.append(density_tags[i])
    
    density_df = data[tag_list].copy()
    
    tag_list.pop(0) # We remove the timestamp name of the tag list
    
    tag_list_a0 = tag_list[:int(parameters["ma0_lines"])].copy()
    tag_list_a1 = tag_list[int(parameters["ma0_lines"]):].copy()
    
    intact_rock_scale_density = parameters["solid_proportion_in_cyclons"]["ore_scale_density"]
    model_names = ["ma0", "ma1"]
    
    # We generate the number of sections in each conventional grinding plant
    n_sections = parameters["conventional_lines"]
    section_list = ["s" + str(i) for i in range(n_sections)]
    
    ma0_section_list = section_list[:int(parameters["ma0_lines"])].copy()
    ma1_section_list = section_list[int(parameters["ma0_lines"]):].copy()
    
    # We iterate over every section in the plant and calculate the solid proportion in cyclons
    for section, tag in zip(ma0_section_list, tag_list_a0):
        calc_cyclon_solid_prop_name = (
            f"calc_{model_names[0]}_cyclon_solid_prop_bhc_{section}"
        )
        density_df[calc_cyclon_solid_prop_name] = (100*
            (1 - 1/density_df[tag])/(1 - 1/intact_rock_scale_density)
                                          )
    
    for section, tag in zip(ma1_section_list, tag_list_a1):
        calc_cyclon_solid_prop_name = (
            f"calc_{model_names[1]}_cyclon_solid_prop_bhc_{section}"
        )
        density_df[calc_cyclon_solid_prop_name] = (100*
            (1 - 1/density_df[tag])/(1 - 1/intact_rock_scale_density)
                                          )    
        
    df = density_df[sorted(density_df.columns.difference(tag_list_a0 + tag_list_a1))]
    solid_tags = df.columns.to_list()[:int(len(density_tags))]
    
    # Finally, we assure that solid proportions in cyclons are in the correct range (this is, between 0 and 100).
    # That correct range is expected, considering pulp densities such that they are higher than water's.
    lower_solid_bound = parameters["solid_proportion_in_cyclons"]["lower_solid_bound"]
    higher_solid_bound = parameters["solid_proportion_in_cyclons"]["higher_solid_bound"]

    for i in range(len(df)):
        for tag in solid_tags:
            if df[tag][i] >= higher_solid_bound:
                df[tag][i] = np.nan
            if df[tag][i] < lower_solid_bound:
                df[tag][i] = np.nan
    
    return df

def create_diff_celdas(
    parameters: dict, data: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """Add differences between celdas/bancos for flotation variables.

    This is a generic function meant to be used for each flotation model. It gets the plant's configuration from the
    parameters and creates the following features:
    - 'Dif flujo de aire': difference in 'flujo de aire' between consecutive cells for each line
    - 'Dif vel burbujas': difference in 'flujo de aire' between consecutive cells for each line
    - 'Dif nivel espuma': difference in 'nivel de espuma' between consecutive cells (disabled for lack of data)
    - 'Dif nivel pulpa': difference in 'nivel de pulpa' between consecutive cells for each line

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing flotation variables.
        model_name: Name of the model.

    Returns:
        data: Master table with additional calculated flotation features.

    """
    flot_tags = parameters[f"{model_name}_flotation_tags"]

    # Create sub-df containing only flotation features to be aggregated
    tag_list = [parameters["timestamp_col_name"]]
    for group in flot_tags:
        tag_list.extend(flot_tags[group])
    df = data[tag_list].copy()
    tag_list.pop(0)  # Remove timestamp

    # Create number of lines
    plant_params = parameters[f"{model_name}_flotation_config"]
    lines_list = ["l" + str(i + 1) for i in range(plant_params["n_lines"])]
    n_celdas = plant_params["n_celdas"]

    # Iterate over lineas, celdas and bancos
    for lin in lines_list:

        for cel in range(n_celdas - 1):
            # Flujo aire
            calc_diff_flujo_aire_name = (
                f"calc_{model_name}_diff_flujo_aire_{lin}_c{cel+2}c{cel+1}"
            )
            df[calc_diff_flujo_aire_name] = (
                df[flot_tags[f"flujo_aire_{lin}_tags"][cel + 1]]
                - df[flot_tags[f"flujo_aire_{lin}_tags"][cel]]
            )
            # Velocidad burbujas
            calc_diff_vel_burbujas_name = (
                f"calc_{model_name}_diff_vel_burbuja_{lin}_c{cel+2}c{cel+1}"
            )
            df[calc_diff_vel_burbujas_name] = (
                df[flot_tags[f"vel_burbujas_{lin}_tags"][cel + 1]]
                - df[flot_tags[f"vel_burbujas_{lin}_tags"][cel]]
            )
            # # Nivel espuma  # TODO: ML -  disabled bc' data is missing, might need to remove commented text and parameters
            # calc_diff_nivel_espuma_name = (
            #         f"calc_{model_name}_diff_nivel_espuma_{lin}_c{cel+2}c{cel+1}"
            # )
            # df[calc_diff_nivel_espuma_name] = (
            #         df[flot_tags[f"nivel_espuma_{lin}_tags"][cel+1]]
            #         - df[flot_tags[f"nivel_espuma_{lin}_tags"][cel]]
            # )

    # Select and sort features
    return df[sorted(df.columns.difference(tag_list))]


def create_diff_bancos(
    parameters: dict, data: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """Add differences between celdas/bancos for flotation variables.

    This is a generic function meant to be used for each flotation model. It gets the plant's configuration from the
    parameters and creates the following features:
    - 'Dif flujo de aire': difference in 'flujo de aire' between consecutive cells for each line
    - 'Dif vel burbujas': difference in 'flujo de aire' between consecutive cells for each line
    - 'Dif nivel espuma': difference in 'nivel de espuma' between consecutive cells (disabled for lack of data)
    - 'Dif nivel pulpa': difference in 'nivel de pulpa' between consecutive cells for each line

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing flotation variables.
        model_name: Name of the model.

    Returns:
        data: Master table with additional calculated flotation features.

    """
    flot_tags = parameters[f"{model_name}_flotation_tags"]

    # Create sub-df containing only flotation features to be aggregated
    tag_list = [parameters["timestamp_col_name"]]
    for group in flot_tags:
        tag_list.extend(flot_tags[group])
    df = data[tag_list].copy()
    tag_list.pop(0)  # Remove timestamp

    # Create number of lines
    plant_params = parameters[f"{model_name}_flotation_config"]
    lines_list = ["l" + str(i + 1) for i in range(plant_params["n_lines"])]
    n_bancos = plant_params["n_bancos"]

    # Create bancos
    for lin in lines_list:

        for ban in range(n_bancos):
            # Flujo aire
            if flot_tags[f"flujo_aire_{lin}_tags"]:
                calc_flujo_aire_name = f"calc_{model_name}_flujo_aire_{lin}_b{ban+1}"
                df[calc_flujo_aire_name] = (
                    df[flot_tags[f"flujo_aire_{lin}_tags"][2 * ban]]
                    + df[flot_tags[f"flujo_aire_{lin}_tags"][2 * ban + 1]]
                ) / 2
            # Velocidad burbujas
            if flot_tags[f"vel_burbujas_{lin}_tags"]:
                calc_vel_burbujas_name = f"calc_{model_name}_vel_burbujas_{lin}_b{ban+1}"
                df[calc_vel_burbujas_name] = (
                    df[flot_tags[f"vel_burbujas_{lin}_tags"][2 * ban]]
                    + df[flot_tags[f"vel_burbujas_{lin}_tags"][2 * ban + 1]]
                ) / 2
            # Tamano burbujas
            if flot_tags[f"tam_burbujas_{lin}_tags"]:
                calc_tam_burbujas_name = f"calc_{model_name}_tam_burbujas_{lin}_b{ban+1}"
                df[calc_tam_burbujas_name] = (
                    df[flot_tags[f"tam_burbujas_{lin}_tags"][2 * ban]]
                    + df[flot_tags[f"tam_burbujas_{lin}_tags"][2 * ban + 1]]
                ) / 2

    # Create diff between bancos
    for lin in lines_list:

        for ban in range(n_bancos - 1):
            # Flujo aire
            if flot_tags[f"flujo_aire_{lin}_tags"]:
                calc_diff_flujo_aire_name = (
                    f"calc_{model_name}_diff_flujo_aire_{lin}_b{ban+2}b{ban+1}"
                )
                df[calc_diff_flujo_aire_name] = (
                    df[f"calc_{model_name}_flujo_aire_{lin}_b{ban+2}"]
                    - df[f"calc_{model_name}_flujo_aire_{lin}_b{ban+1}"]
                )
            # Velocidad burbujas
            if flot_tags[f"vel_burbujas_{lin}_tags"]:
                calc_diff_vel_burbujas_name = (
                    f"calc_{model_name}_diff_vel_burbujas_{lin}_b{ban+2}b{ban+1}"
                )
                df[calc_diff_vel_burbujas_name] = (
                    df[f"calc_{model_name}_vel_burbujas_{lin}_b{ban+2}"]
                    - df[f"calc_{model_name}_vel_burbujas_{lin}_b{ban+1}"]
                )
            # Nivel pulpa
            #if flot_tags[f"nivel_pulpa_{lin}_tags"]:
            #    calc_diff_nivel_pulpa_name = (
            #        f"calc_{model_name}_diff_nivel_pulpa_{lin}_b{ban+2}b{ban+1}"
            #    )
            #    df[calc_diff_nivel_pulpa_name] = (
            #        df[flot_tags[f"nivel_pulpa_{lin}_tags"][ban + 1]]
            #        - df[flot_tags[f"nivel_pulpa_{lin}_tags"][ban]]
            #    )

    # Select and sort features
    return df[sorted(df.columns.difference(tag_list))]


def create_mean_by_line(
    parameters: dict, data: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """Create mean across celdas for these flotation variables:
    - Flujo aire
    - Velocidad burbujas
    - Nivel pulpa
    - Tamaño burbujas

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing flotation variables.
        model_name: Name of the model.

    Return:
        df: DataFrame with new features.
    """

    flot_tags = parameters[f"{model_name}_flotation_tags"]
    timestamp_col_name = parameters["timestamp_col_name"]

    # Create sub-df containing only flotation features to be aggregated
    tag_list = []
    for group in flot_tags:
        tag_list.extend(flot_tags[group])
    df = data[[timestamp_col_name] + tag_list].copy()

    # Create number of lines
    plant_params = parameters[f"{model_name}_flotation_config"]
    lines_list = ["l" + str(i + 1) for i in range(plant_params["n_lines"])]

    # Create mean by linea
    for lin in lines_list:

        df[f"calc_{model_name}_mean_flujo_aire_{lin}"] = df[
            flot_tags[f"flujo_aire_{lin}_tags"]
        ].mean(axis=1)
        df[f"calc_{model_name}_mean_vel_burbujas_{lin}"] = df[
            flot_tags[f"vel_burbujas_{lin}_tags"]
        ].mean(axis=1)
        #df[f"calc_{model_name}_mean_nivel_pulpa_{lin}"] = df[
        #    flot_tags[f"nivel_pulpa_{lin}_tags"]
        #].mean(axis=1)
        df[f"calc_{model_name}_mean_tam_burbujas_{lin}"] = df[
            flot_tags[f"tam_burbujas_{lin}_tags"]
        ].mean(axis=1)

    # Select and sort features
    return df[sorted(df.columns.difference(tag_list))]


def create_delta_aire_by_line(
    parameters: dict, data: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """Create delta between C1 and C7 for flujo de aire.

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing flotation variables.
        model_name: Name of the model.
    """

    flot_tags = parameters[f"{model_name}_flotation_tags"]

    # Create sub-df containing only flotation features to be aggregated
    timestamp_col_name = parameters["timestamp_col_name"]
    tag_list = []
    for group in flot_tags:
        tag_list.extend(flot_tags[group])
    df = data[[timestamp_col_name] + tag_list].copy()

    # Create number of lines
    plant_params = parameters[f"{model_name}_flotation_config"]
    lines_list = ["l" + str(i + 1) for i in range(plant_params["n_lines"])]
    n_celdas = plant_params["n_celdas"]

    # Create delta C(N-1)-C(1) by linea
    for lin in lines_list:
        calc_delta_flujo_aire_name = (
            f"calc_{model_name}_delta_flujo_aire_{lin}_c{n_celdas-1}c1"
        )
        df[calc_delta_flujo_aire_name] = (
            df[flot_tags[f"flujo_aire_{lin}_tags"][n_celdas - 2]]
            - df[flot_tags[f"flujo_aire_{lin}_tags"][0]]
        )

    # Select and sort features
    return df[sorted(df.columns.difference(tag_list))]


def create_fe_over_cu(
    parameters: dict, data: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """Create Fe/Cu ratio.

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing flotation variables.
        model_name: Name of the model.
    """

    params = parameters[f"{model_name}_fe_over_cu"]
    timestamp_col_name = parameters["timestamp_col_name"]
    ley_fe = params["ley_fe_alim"]
    ley_cu = params["ley_cu_alim"]

    tag_list = [ley_fe, ley_cu]
    df = data[[timestamp_col_name] + tag_list].copy()

    # Create Fe/Cu
    fe_over_cu_tag_name = f"calc_{model_name}_fe_over_cu"
    df[fe_over_cu_tag_name] = df[ley_fe] / df[ley_cu]
    # Remove outliers  # TODO: DM - move this somewhere else
    df = cut_values_from_dict(df, {fe_over_cu_tag_name: [0, 5]})

    return df[df.columns.difference(tag_list)]


def create_target_flot(
    parameters: dict, data: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """Create calculated recovery based on feed, concentrate and tailings grade.

    Assumptions:
    - There are several tailings lines, the final tailings grade is calculated as the average of the lines
    - Data might have time gaps (e.g. every 2/4 h), so back-fill and forward-fill are applied
    - Data is noisy so temporary clips are introduced here

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing flotation variables.
        model_name: Name of the model, used to name the calculated features.

    Returns:
        data: Master table with additional calculated flotation features.

    """
    # Select target columns
    timestamp_col_name = parameters["timestamp_col_name"]
    params_target = parameters[f"{model_name}_flotation_target"]
    
    if model_name != "fa0":
        ley_conc_rx_tag = params_target["ley_conc_rx_tag"] # Leyes RX
        ley_conc_tf_tag = params_target["ley_conc_tf_tag"] # Analizadores ThermoFisher
    else:
        ley_conc_tag = params_target["ley_conc_tag"]  # Individual
    
    ley_alim_tag = params_target["ley_alim_tag"]  # Individual
    ley_cola_rx_tags = params_target["ley_cola_rx_tags"]  # Leyes RX
    ley_cola_tf_tags = params_target["ley_cola_tf_tags"]  # Analizadores ThermoFisher
    
    if model_name != "fa0":
        target_tags = ley_conc_rx_tag + ley_conc_tf_tag + ley_alim_tag + ley_cola_rx_tags + ley_cola_tf_tags
    else:
        target_tags = ley_conc_tag + ley_alim_tag + ley_cola_rx_tags + ley_cola_tf_tags
    
    # Select features
    if timestamp_col_name in data.columns:
        # MN TEMPORARY
        # Using this space for ffill leyes tags, attacking
        # directly to the original df and not only the copy.
        data[[timestamp_col_name] + target_tags] = data[[timestamp_col_name] + target_tags].fillna(method="ffill")
        df = data[[timestamp_col_name] + target_tags].copy()
    else:
        data[target_tags] = data[target_tags].fillna(method="ffill")
        df = data[target_tags].copy()

    # Fill na for all targets # TODO: ML - temporary! This process should be performed somewhere else
    # MN TEMPORARY
    df[target_tags] = (
        df[target_tags].fillna(method="ffill").fillna(method="bfill", limit=2)
    )
    
    # Filter values  # ToDO: DM - temporary! This filter should be somewhere else (for the moment, the hard-
    # coded values have been replaced with fixed parameters fetched from the params dictionary)
    ley_alim_min_bound = params_target["ley_alim_bounds"][0]
    ley_alim_max_bound = params_target["ley_alim_bounds"][1]
    
    params_config = parameters[f"{model_name}_flotation_config"]
    lines = [f"l{line}" for line in range(1, params_config["n_lines"] + 1)]
    
    tailing_bounds_tags = [f"{model_name}{line}" for line in lines]
    ley_cola_min_bounds = []
    ley_cola_max_bounds = []
    for tag_j in tailing_bounds_tags:
        min_bound = parameters[tag_j]["range"][0]
        max_bound = parameters[tag_j]["range"][1]
        ley_cola_min_bounds.append(min_bound)
        ley_cola_max_bounds.append(max_bound)
         
    df = cut_values_from_dict(
        df, {ley_alim_tag[0]: [ley_alim_min_bound, ley_alim_max_bound]
             }
    )  # Alimentacion
    
    for min_bound, max_bound in zip(ley_cola_min_bounds, ley_cola_max_bounds):
        df = cut_values_from_list(
            df, ley_cola_rx_tags + ley_cola_tf_tags, [min_bound, max_bound]
        )  # Colas
    
    if model_name != "fa0":
        # Calculate recup en peso using colas RX  # TODO ML - split into independent functions
        # Commentary: The update of this function will use the RX information, if available. If not,
        # the thermofisher data will be used. This replacement is made by a "fillna" strategy
        new_feat_list = []
        
        # Handling of zeros and NaNs in RX data
        # Filling NaN values in RX data with the Thermofisher data
        for cola_rx, cola_tf in zip(ley_cola_rx_tags, ley_cola_tf_tags):
            df[cola_rx].fillna(df[cola_tf], inplace=True) # Leyes de cola
        df[ley_conc_rx_tag].fillna(df[ley_conc_tf_tag], inplace=True) # Leyes de concentrado
    
        # Replacing zeros in RX data with Thermofisher data, when possible
        for cola_rx, cola_tf in zip(ley_cola_rx_tags, ley_cola_tf_tags):
            df[cola_rx] = np.where(
                df[cola_rx] == 0, df[cola_tf], df[cola_rx]
            ) # Leyes de cola
        df[ley_conc_rx_tag] = np.where(
            df[ley_conc_rx_tag] == 0, df[ley_conc_tf_tag], df[ley_conc_rx_tag]
        ) # Ley de concentrado (RX to Thermofisher)
        
        # Handling of zeros and NaNs in TF data
        # Filling NaN values in Thermofisher data with the RX data
        for cola_rx, cola_tf in zip(ley_cola_rx_tags, ley_cola_tf_tags):
            df[cola_tf].fillna(df[cola_rx], inplace=True) # Leyes de cola
        df[ley_conc_tf_tag].fillna(df[ley_conc_rx_tag], inplace=True) # Leyes de concentrado
        
        # Replacing zeros in Thermofisher data with RX data, when possible
        for cola_rx, cola_tf in zip(ley_cola_rx_tags, ley_cola_tf_tags):
            df[cola_tf] = np.where(
                df[cola_tf] == 0, df[cola_rx], df[cola_tf]
            ) # Leyes de cola
        df[ley_conc_tf_tag] = np.where(
            df[ley_conc_tf_tag] == 0, df[ley_conc_rx_tag], df[ley_conc_tf_tag]
        ) # Ley de concentrado (Thermofisher to RX)
        
        # Calculating Cu mass recovery
        for line, ley_cola in zip(lines, ley_cola_rx_tags):
            recup_peso_name = f"calc_{model_name}_recup_peso_{line}"
            df[recup_peso_name] = (df[target_tags[2]] - df[ley_cola]) / (
                df[target_tags[1]] - df[ley_cola]
            )
            new_feat_list.append(recup_peso_name)
    
        # Calculate recovery by line using cola TF and alim/conc RX
        weights = params_target["weights_colas"]
        ley_cola_pond = 0
        for line, ley_cola, j in zip(lines, ley_cola_tf_tags, range(len(lines))):
            recup_name = f"calc_{model_name}_recup_{line}"
            new_feat_list.append(recup_name)
            # Calculate recup by line
            df[recup_name] = (
                df[target_tags[1]]
                * (df[target_tags[2]] - df[ley_cola])
                / (df[target_tags[2]] * (df[target_tags[1]] - df[ley_cola]))
            )
            temp = df[ley_cola] * weights[j]
            temp.fillna(0, inplace=True)
            # Update weighted ley cola
            df[ley_cola + "_w"] = weights[j]
        w_tags = [ley_cola + "_w" for ley_cola in ley_cola_tf_tags]
        # Replace by NAN weight where measurement not available
        for ley_cola in ley_cola_tf_tags:
            for j in range(len(df)):
                if pd.isnull(df[ley_cola][j]):
                    df[ley_cola + "_w"][j] = np.nan
        # have total weights
        temp_av = df[w_tags].sum(axis=1)
        # if total weight is=myweight, then other line is down
        # my weight should be 1
        for ley_cola in ley_cola_tf_tags:
            if model_name != 'fa2':
                df.loc[df[ley_cola + "_w"] == temp_av, ley_cola + "_w"] = 1
            else:
            # TODO: avoid hardcoding for line 2
                df.loc[(df[ley_cola + "_w"] == 0.5) & (temp_av > 0.5), ley_cola + "_w"] = 0.75
                df.loc[(df[ley_cola + "_w"] == 0.25) & (temp_av == 0.5), ley_cola + "_w"] = 0.5
        for ley_cola in ley_cola_tf_tags:
            df[ley_cola + "_total"] = df[ley_cola] * df[ley_cola + "_w"]
        tot_tags = [ley_cola + "_total" for ley_cola in ley_cola_tf_tags]
        # Sum as DF so NANs are not taken into account
        ley_cola_pond = df[tot_tags].sum(axis=1, skipna=True)
        # Global recovery (weighted ley colas)
        recup_wt_name = params_target["recup_wt_name"]
        df[recup_wt_name] = (
            df[target_tags[1]]
            * (df[target_tags[2]] - ley_cola_pond)
            / (df[target_tags[2]] * (df[target_tags[1]] - ley_cola_pond))
        )
        new_feat_list.append(recup_wt_name)
        
        df['ley_cola_pond_'+ model_name] = ley_cola_pond 
        new_feat_list.append('ley_cola_pond_'+ model_name)
        
        df = cut_values_from_list(
            df, new_feat_list, [0, 1]
        )  # ToDO: DM - temporary! Where do we clip created features?
    
    else:
        # Calculate recup en peso using colas RX  # TODO ML - split into independent functions
        new_feat_list = []
        for line, ley_cola in enumerate(ley_cola_rx_tags):
            recup_peso_name = f"calc_{model_name}_recup_peso_l{line+1}"
            df[recup_peso_name] = (df[target_tags[1]] - df[ley_cola]) / (
                df[target_tags[0]] - df[ley_cola]
            )
            new_feat_list.append(recup_peso_name)
    
        # Calculate recovery by line using cola TF and alim/conc RX
        weights = params_target["weights_colas"]
        ley_cola_pond = 0
        for linea, ley_cola in enumerate(ley_cola_tf_tags):
            linea_name = str(linea + 1)
            recup_name = f"calc_{model_name}_recup_l{linea_name}"
            new_feat_list.append(recup_name)
            # Calculate recup by line
            df[recup_name] = (
                df[target_tags[0]]
                * (df[target_tags[1]] - df[ley_cola])
                / (df[target_tags[1]] * (df[target_tags[0]] - df[ley_cola]))
            )
            #temp = df[ley_cola] * weights[linea]
            #temp.fillna(0, inplace=True)
            # Update weighted ley cola
            df[ley_cola + "_w"] = weights[linea]
        w_tags = [ley_cola + "_w" for ley_cola in ley_cola_tf_tags]
        # replace by NAN weight where measurement not available
        for ley_cola in ley_cola_tf_tags:
            df.loc[df[ley_cola].isna(), ley_cola + "_w"] = np.nan
        # have total weights
        temp_av = df[w_tags].sum(axis=1)
        # if total weight is=myweight, then other line is down
        # my weight should be 1
        for ley_cola in ley_cola_tf_tags:
            if model_name!='fa2':
                df.loc[df[ley_cola + "_w"] == temp_av, ley_cola + "_w"] = 1
            else:
            # TODO: avoid hardcoding for line 2
                df.loc[(df[ley_cola + "_w"] == 0.5) & (temp_av > 0.5), ley_cola + "_w"] = 0.75
                df.loc[(df[ley_cola + "_w"] == 0.25) & (temp_av == 0.5), ley_cola + "_w"] = 0.5
        for ley_cola in ley_cola_tf_tags:
            df[ley_cola + "_total"] = df[ley_cola] * df[ley_cola + "_w"]
        tot_tags = [ley_cola + "_total" for ley_cola in ley_cola_tf_tags]
        # Sum as DF so NANs are not taken into account
        ley_cola_pond = df[tot_tags].sum(axis=1, skipna=True)
        # If the sum of all weights is 0, replace them by NaN
        ley_cola_pond = ley_cola_pond.replace(0, np.nan)
        # Global recovery (weighted ley colas)
        recup_wt_name = params_target["recup_wt_name"]
        df[recup_wt_name] = (
            df[target_tags[0]]
            * (df[target_tags[1]] - ley_cola_pond)
            / (df[target_tags[1]] * (df[target_tags[0]] - ley_cola_pond))
        )
        new_feat_list.append(recup_wt_name)

        df['ley_cola_pond_'+ model_name] = ley_cola_pond 
        new_feat_list.append('ley_cola_pond_'+ model_name)
        
        df = cut_values_from_list(
            df, new_feat_list, [0, 1]
        )  # ToDO: DM - temporary! Where do we clip created features?
    
    if timestamp_col_name in data.columns:
        return df[[timestamp_col_name] + new_feat_list]
    else:
        return df[new_feat_list]


def create_group_mean(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Create the average of group of variables.

    Assumptions:
    - Each group has more than one tag and we want the average of them

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.

    Returns:
        data: df with new variables.

    """
    # Select mill columns
    timestamp_col_name = parameters["timestamp_col_name"]
    feature_prefix = parameters["mean_grouping"]["tag_prefix"]
    groups = parameters["mean_grouping"]["groups"]
    tags = [tag for mill in groups for tag in parameters[mill]]

    # Select features
    df = data[[timestamp_col_name] + tags].copy()

    new_var_names = []
    for mill in groups:
        name = feature_prefix + mill
        new_var_names.append(name)
        tags_mill = parameters[mill]
        df[name] = df[tags_mill].mean(axis=1)

    return df[[timestamp_col_name] + new_var_names]


def create_group_sum(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Create the sum of group of variables.

    Assumptions:
    - Each group has more than one tag and we want the sum of them

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.

    Returns:
        data: df with new variables.

    """
    # Select mill columns
    timestamp_col_name = parameters["timestamp_col_name"]
    feature_prefix = parameters["sum_grouping"]["tag_prefix"]
    groups = parameters["sum_grouping"]["groups"]
    tags = [tag for mill in groups for tag in parameters[mill]]

    # Select features
    df = data[[timestamp_col_name] + tags].copy()

    new_var_names = []
    for mill in groups:
        name = feature_prefix + mill
        new_var_names.append(name)
        tags_mill = parameters[mill]
        df[name] = df[tags_mill].sum(axis=1)

    return df[[timestamp_col_name] + new_var_names]


def create_group_sol(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Creates the % of solid of group of variables

    Assumptions:
    - Each group has more than one tag and we want the % solid of them

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.

    Returns:
        data: df with new variables.

    """
    # Select mill columns
    timestamp_col_name = parameters["timestamp_col_name"]
    feature_prefix = parameters["sol_grouping"]["tag_prefix"]
    groups = parameters["sol_grouping"]["groups"]
    tags = [timestamp_col_name]
    for group in groups:
        tags_group = [
            tag for mill in parameters[group] for tag in parameters[group][mill]
        ]
        tags = tags + tags_group

    # Select features
    df = data[tags].copy()

    new_var_names = []
    for group in groups:
        tag_name = feature_prefix + group
        tags_water = parameters[group]["water"]
        tags_sol = parameters[group]["sol"]
        sum_all_water = df[tags_water].sum(axis=1)
        sum_all_solid = df[tags_sol].sum(axis=1)
        df[tag_name] = sum_all_solid / (sum_all_water + sum_all_solid)
        df[tag_name] = df[tag_name].replace([np.inf, -np.inf], 0)
        new_var_names.append(tag_name)

    return df[[timestamp_col_name] + new_var_names]


def create_on_off(parameters: dict, data: pd.DataFrame, 
                var: list = None, percentages: List[float] = [1.0,1.0]) -> pd.DataFrame:
    """Create binary on off variables

    Assumptions:
    - Each variable represents an equiment/area being on/off above or below a value

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.
        var: List of specific variables to check from
        on_off groups. It apply the on_off logic to only
        those variables.
        percentages: Percentage of the actual value set in
        the original parameters to make the definition
        of the on/off status.

    Returns:
        data: df with new variables.

    """
    # Select filter columns
    timestamp_col_name = parameters["timestamp_col_name"]
    feature_prefix = parameters["on_off"]["tag_prefix"]
    groups = parameters["on_off"]["groups"]

    # Select features
    df = data.copy()

    new_var_names = []
    for group in groups:
        if var:
            filter_cols = [x for x in parameters[group] if x in var]
        else:
            filter_cols = parameters[group]
        for filter_col in filter_cols:
            name = feature_prefix + filter_col
            new_var_names.append(name)
            tags_filter_col = parameters[group][filter_col]
            tag = tags_filter_col["tag"]
            off_when = tags_filter_col["off_when"]
            value = tags_filter_col["value"]
            df[name] = 1
            if off_when == "less_than":
                m_p = min(percentages)
                df.loc[df[tag] < value*m_p, name] = 0
            elif off_when == "greater_than":
                m_p = max(percentages)
                df.loc[df[tag] > value*m_p, name] = 0
            df.loc[df[tag].isna(), name] = 0

    return df[[timestamp_col_name] + new_var_names]


def create_feature_lags(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Create lagged variables grouped in parameters.

    Assumptions:
    - It is assumed that in parameters_features there is entry 'lag_grouping', which contains different groups of
    variables (e.g. ["group of variables", n_shifts]). Each group of variables in the list will be shifted 'n_shifts'
    number of periods.

    Args:
        parameters: Dictionary of parameters.
        data: Dataset grouped by shift containing all tags to be lagged.

    Returns:
        df: Dataframe of all lagged variables.

    """
    timestamp_col_name = parameters["timestamp_col_name"]
    tag_groups = parameters["lag_grouping"]["groups"]

    # Select all tags to be lagged
    tag_list = [timestamp_col_name]
    for group, n_shifts in tag_groups:
        tag_list.extend(parameters[group])

    tag_list = list(set(tag_list))

    df = data[tag_list].set_index(timestamp_col_name)
    # Create new DF to create lags for same group
    df_lags = pd.DataFrame(index=df.index)

    # Create lagged features
    for group, n_shifts in tag_groups:
        tags = parameters[group]
        type_shift = "p" * (n_shifts <= 0) + "m" * (n_shifts > 0) + str(n_shifts)
        prefix = f"calc_{type_shift}_lag_"
        df_lags[tags] = df[tags].shift(periods=n_shifts)
        df_lags.rename(columns={tag: prefix + tag for tag in tags}, inplace=True)

    return df_lags.reset_index()


def create_cuf_feature(
    parameters: dict, data: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """Create CuF features.
        Two alternatives are provided:
        - Using weighted average of individual recovery lines
        - Using global recovery line

    Args:
        parameters: Dictionary of parameters.
        data: Dataset grouped by shift containing all tags to be lagged.
        model_name: Name of the model for which tags will be selected and tagged.

    Returns:
        df: Dataframe of all lagged variables.

    """
    
    # Select features
    timestamp_col_name = parameters["timestamp_col_name"]
    params = parameters[f"{model_name}_flotation_target"]
    ley_alim = params["ley_alim_tag"][0]
    recup_wt_name = params["recup_wt_name"]
    tph_tags = params["tph_tags"]
    cuf_name = params["cuf_obj_name"]
    target_tags = [ley_alim, recup_wt_name] + tph_tags

    if timestamp_col_name in data.columns:
        df = data[[timestamp_col_name] + target_tags].copy()
    else:
        df = data[target_tags].copy()

    df[cuf_name] = (
        df[tph_tags].sum(axis=1) * df[recup_wt_name] * df[ley_alim] / 100
    )  # TODO: ML split into separate fun

    if timestamp_col_name in data.columns:
        return df[[timestamp_col_name, cuf_name]].replace([np.inf, -np.inf], np.nan)
    else:
        return df[[cuf_name]].replace([np.inf, -np.inf], np.nan)

_typing_scale_helper = List[Union[str,float]]
_typing_inner_unit_helper_helper = Dict[str, Union[List,float]]
_typing_unit_helper_helper = Dict[str,_typing_inner_unit_helper_helper]

def _scaling_units(td: TagDict, 
                scaling_features: List[_typing_scale_helper]) -> _typing_unit_helper_helper:
    """
    Helper function that creates a dictionary with features that 
    need to be re-scaled given a list of [unit,scale_factor] (see parameters_features.yml
    specific_power_grouping:scale_to_mw_features).
    Args:
        td: TagDict.
        scaling_features: List of lists containing the unit and the re-scaling
        factor needed to re-scale.
    Returns:
        unit_helper: Dictionary that has a key with the unit, and 
        contains a dictionary with the list of tags that belong to that unit
        and the needed factor: dict('kw':{'tags':['x','y'],
                                        'scale_factor':0.001}).
    """
    unit_helper = {}
    for unit in scaling_features:
        unit_ = unit[0]
        unit_helper[unit_] = {}
        unit_helper[unit_]['tags'] = td.select('unit',unit_)
        unit_helper[unit_]['scale_factor'] = unit[1]
    return unit_helper

def calculate_specific_power(parameters: dict, data: pd.DataFrame, td: TagDict) -> pd.DataFrame:
    """Calculates the specific power.
    2021-04-06: This function was created thinking that all powers are measured 
    in mW, but the reality is that almost all variables are measured 
    in kW, so there are some problems and we need to address those ones.
    For simplicity, we use TagDict as the source of truth in terms of 
    the unit of the tag, and we will keep using the mW as our pivot.
    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.
        td: TagDict.
    Returns:
        data: df with new variables.

    """

    # Select mill columns
    timestamp_col_name = parameters["timestamp_col_name"]
    feature_prefix = parameters["specific_power_grouping"]["tag_prefix"]
    groups = parameters["specific_power_grouping"]["groups"]
    threshold = parameters["specific_power_grouping"]["threshold"]
    tags = []
    for group in groups:
        tags_group = [
            tag for var in parameters[group] for tag in parameters[group][var]
        ]
        tags = tags + tags_group

    # Select features
    df = data[[timestamp_col_name] + tags].copy()

    # Call unit helper
    scaling_features = parameters["specific_power_grouping"]['scale_to_mw_features']
    unit_helper = _scaling_units(td, scaling_features)

    new_var_names = []
    for group in groups:
        tag_name = feature_prefix + group
        new_var_names.append(tag_name)
        tags_target = parameters[group]
        if "tonelaje" in tags_target and "potencia" in tags_target:
            tons = tags_target["tonelaje"]
            pot = tags_target["potencia"][0]
            total_tons = df[tons].sum(axis=1)
            scale_factor = [v.get('scale_factor') for k,v in unit_helper.items() if pot in v.get('tags')]
            if scale_factor:
                df[pot]*=scale_factor[0]
            df[tag_name] = df[pot] / total_tons 
            df.loc[df[tag_name] > threshold, tag_name] = np.nan
        else:
            raise ValueError(
                f"{group} needs to have tag for potencia and tonelaje to calculate the specific power"
            )
    return df[[timestamp_col_name] + new_var_names]


def add_across_features_by_shift(parameters: dict, *data: pd.DataFrame) -> pd.DataFrame:
    """Add calculated features to the master data table by shift.

    Args:
        parameters: Dictionary of parameters.
        data: *args of multiples dataframes.

    Returns:
        data_feat: Master table with additional calculated features.

    """

    # Create features
    list_features = list(data)
    # TODO: This should be in other place, not in this function
    lag_sol_colas = create_feature_lags(parameters, data[0])
    list_features.append(lag_sol_colas)

    # Merge data and created features
    data_feat = merge_tables_on_timestamp(parameters=parameters, df_list=list_features)

    return data_feat


def create_dosif_reactives(
    parameters: dict, data: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """
    Create additional features transforming units from cc/min to gr/ton.
    
    We must take into consideration that the xantato's tag has been updated since
    1/7 for all the flotation plants. While this might sound obvious in terms of
    simply updating this information in TDM, this is not the case because the new
    tag does not have any historical information before 1/7 and, moreover, the
    new tag already conveys the regarding dosification in g/t. So we must append
    everything manually.

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing source data.
        model_name: Name of the model for which tags will be transformed.

    Returns:
        data_feat: Master table with additional calculated features.

    """
    # Get parameters
    timestamp_col_name = parameters["timestamp_col_name"]
    parameters_r = parameters["conversion_reactivos"]
    parameters_c = parameters["correccion_xantato"]
    suffix = parameters_r["suffix"]
    parameters_m = parameters[parameters_r["tags"][model_name]]
    tph_tags = parameters_m["tph"]
    
    # Corrected tag for "xantato"
    xant_new_tag = parameters_c[model_name]
    xant_old_tag = parameters_m["react"]["xantato"]
    switch_stamp = pd.to_datetime(parameters_c["timestamp_switch"])
    
    # We will use the xantato's new tag to refer to all historic data contained in the old one
    df = data.copy()
    df[timestamp_col_name] = pd.to_datetime(df[timestamp_col_name])
    df.set_index(timestamp_col_name, inplace=True)

    df_xant_new = df[[xant_new_tag]].loc[switch_stamp+pd.DateOffset(hours=1):]
    
    # Select all tags
    tags_used = tph_tags + list(parameters_m["react"].values())
    df_t = data[[timestamp_col_name] + tags_used].copy()

    # Calculate total tph
    df_t["tph_total"] = df_t[tph_tags].sum(axis=1)

    # Calculate dosif
    new_vars = []
    for reactive in parameters_m["react"]:
        tag = parameters_m["react"][reactive]
        dosif_name = f"calc_{tag}_{suffix}"
        df_t[dosif_name] = (df_t[tag] * parameters_r["densidad"][reactive] * 60).divide(
            df_t.tph_total, axis=0)
        # Cut values out of range  # TODO: ML this definitely shouldn't be here
        df_t = cut_values_from_dict(df_t, {dosif_name: parameters_r["min_max"][reactive]})
        new_vars.append(dosif_name)
    
    # Appending the new xantato tag to the old one, already expressed in g/t
    xant_calc_tag = f"calc_{xant_old_tag}_{suffix}"
    df_t[timestamp_col_name] = pd.to_datetime(df_t[timestamp_col_name])
    df_t.set_index(timestamp_col_name, inplace=True)
    df_t[xant_calc_tag].loc[switch_stamp+pd.DateOffset(hours=1):] = df_xant_new.values.reshape(len(df_xant_new),)
    df_t.reset_index(inplace=True)
    
    return df_t[[timestamp_col_name] + new_vars]


def create_on_off_psi(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Create binary on off variables por PSI sensors.

    This is, check if the percentual error between real sensor (PSI)
    and the virtual version: abs(Sensor - Virtual) / abs(Sensor) is
    lower than a parametric threshold.

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.

    Returns:
        data: df with new variables.
    """

    # Select the timestamp column:
    timestamp_col_name = parameters["timestamp_col_name"]

    # Selecting the prefix of these kind of features:
    feature_prefix = parameters["on_off"]["tag_prefix"]

    # Initializing the list of tags to keep at the end of this function:
    list_tags_2keep = [timestamp_col_name]

    # Copying the hourly master table:
    df = data.copy()

    # Defining the structures to iterate over all the plants related to this task:
    dict_match_vars = {
        "A0": parameters["on_off_A0_PSIsensors"],
        "A1": parameters["on_off_A1_PSIsensors"],
    }

    # Cycling over all the plants to work on: A0 and A1
    for plant_Ai in dict_match_vars:
        # Getting the threshold to determine that the sensor is working or not:
        param_percErr_th = dict_match_vars[plant_Ai]["perc_error_th"]

        # Cycling over all the sections in the plant_Ai:
        for sec_j in dict_match_vars[plant_Ai]["sections"]:
            # Name of the percentual error -temporary- variable:
            name_perc_error = "err_rel_malla_psi_sect_" + plant_Ai + "_" + sec_j

            # Name of the on_off variable that will remain in the flow:
            name_new_feat = feature_prefix + "malla_psi_" + plant_Ai + "_" + sec_j

            # Extracting the name of the tags to subset the dataframe df:
            tags_2_work_on = [
                dict_match_vars[plant_Ai]["sections"][sec_j][0],
                dict_match_vars[plant_Ai]["sections"][sec_j][1],
            ]

            # Creating the temporary percentual error:
            df[name_perc_error] = df[tags_2_work_on].apply(
                lambda p: abs(p[0] - p[1]) / abs(p[0]), axis=1
            )
            df[name_perc_error] = df[name_perc_error].replace(np.inf, np.nan)

            # Creating the boolean feature:
            df[name_new_feat] = df[name_perc_error].apply(
                lambda z: 1 if z < param_percErr_th else 0
            )

            # Getting the new feature to the list of tags to keep:
            list_tags_2keep.append(name_new_feat)

    # Returning a dataframe only with the timestamp and the on_off variables
    # of the PSI sensors:
    return df[list_tags_2keep]


def features_from_trazabilidad(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """The purpose of this method is to work with the 'trazabilidad'
    data

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.

    Returns:
        data: df with new variables.

    """

    # Select the timestamp column:
    timestamp_col_name = parameters["timestamp_col_name"]

    # Getting the 'trazabilidad' parameters:
    params_traza = parameters['data_trazabilidad']

    # Defining the desired sources (i.e. tolva_a2, tolva_grueso):
    dsrd_sources = params_traza['desired_sources']

    # Defining the proportional ends that will be kept:
    dsrd_porcs_ends = params_traza['porc_desired']

    # Getting all the qualities that will be kept:
    dsrd_qualities = params_traza['available_qualities']

    # Geting the prefix of the output tags:
    prefix = params_traza['prefix']


    # ##################################### #
    # TASK 1: Filtering the desired columns #
    # ##################################### #

    # Initializing the list of tags to keep:
    list_tags_2keep = []

    # Determining the max length of all 'dsrd_sources':
    max_length_sources = max([len(srck) for srck in dsrd_sources])

    # Cycling over all the desired sources:
    for source_k in dsrd_sources:
        # Iterating over all the columns of 'data':
        for tag_j in data.columns:
            # Validating if the source is in 'tag_j':
            # ASUMPTION: The source is located at the begining of the 'tag_j':
            if (len(tag_j) >= max_length_sources - 1) and (tag_j[:len(source_k)] == source_k):
                # Keeping the tag:
                list_tags_2keep.append(tag_j)

    
    # #################################### #
    # TASK 2: Spliting qualities from tphs #
    # #################################### #

    # Initializing the list of dictionarys:
    dict_tphs = {}
    dict_qualities = {}

    # Iterating over the kept tags:
    for tag_j in list_tags_2keep:
        # Cycling over the desired qualities:
        for qual_k in dsrd_qualities:
            # Validating if it is a quality:
            # ASUMPTION: The quality is located at the end of the 'tag_j':
            if (len(tag_j) >= len(qual_k)) and (tag_j[-len(qual_k):] == qual_k):
                # CASE: Quality tag
                
                # Validating that the quality hasn't been added:
                if qual_k not in dict_qualities:
                    # CASE: Adding the quality to the dictionary
                    dict_qualities[qual_k] = set([tag_j])
                else:
                    # CASE: The quality was already there:
                    dict_qualities[qual_k] = dict_qualities[qual_k].union([tag_j])
                    
        # Cycling over the desired sources:
        for source_k in dsrd_sources:
            # Validating if it is a quality:
            # ASUMPTION: The source is located at the beginning of the 'tag_j':
            if (len(tag_j) >= len(source_k)) and (tag_j[:len(source_k)] == source_k):
                # CASE: Quality tag
                
                # Validating that the quality hasn't been added:
                if source_k not in dict_tphs:
                    # CASE: Adding the quality to the dictionary
                    dict_tphs[source_k] = set([tag_j])
                
                else:
                    # CASE: The quality was already there:
                    dict_tphs[source_k] = dict_tphs[source_k].union([tag_j])

    # Cleaning the dict_tphs from any quality:
    # TODO - Handle better this multi-nested loops
    for dsrd_src_k in dict_tphs:
        for tag_kj in dict_tphs[dsrd_src_k]:
            for qual_p in dict_qualities:
                for tag_pq in dict_qualities[qual_p]:
                    dict_tphs[dsrd_src_k] = dict_tphs[dsrd_src_k].difference([tag_pq])


    # ####################################### #
    # TASK 3: Constructing proportion of tphs #
    # ####################################### #

    # Iterating over all TPH:
    for source_k in dict_tphs:
        # Getting the sum according to the source:
        data['sum_' + source_k] = data[list(dict_tphs[source_k])].sum(axis = 1)
        
        # Iterating over all the parts to get the percentages:
        for tag_kj in dict_tphs[source_k]:
            data[prefix + 'perc_' + tag_kj] = data[tag_kj] / data['sum_' + source_k]

    
    # ##################################### #
    # TASK 4: Selecting the output features #
    # ##################################### #

    # List of final variables to output:
    final_vars = []

    # Cycling over all the tph tags:
    # TODO - Handle better this multi-nested loops
    for dsrd_src_k in dict_tphs:
        for tag_kj in dict_tphs[dsrd_src_k]:
            for desired_percs in dsrd_porcs_ends:
                if (len(tag_kj) >= len(desired_percs)) and\
                            (tag_kj[-len(desired_percs):] == desired_percs):
                    final_vars.append(prefix + 'perc_' + tag_kj)
                    
    # Iterating over the desired qualities:
    for dsrd_qual_k in dict_qualities:
        for tag_kj in dict_qualities[dsrd_qual_k]:
            data = data.rename(columns={tag_kj: prefix + tag_kj})
            final_vars.append(prefix + tag_kj)
            
    # Returning the desired dataframe:
    return data[[timestamp_col_name] + final_vars]

def apply_on_off_to_df(df: pd.DataFrame, 
                    td: TagDict,
                    parameters: Dict):
    """
    Set to NaNs all the dependencies if the equipment
    is off. This grabs the logic from create_on_off and
    iterate over a list of equipments that are set in 
    parameters_features.yml. It also provide a more flexible
    way to make the comparisson, using a percentage for increase
    or decrease the "greater than" or "less than" constrain.
    Args:
        df: Pandas DataFrame with MDT-like data.
        td: TagDict.
        parameters: Context parameters.
    Returns:
        df: Pandas DataFrame with NaNs in each tag-row where
        it's parent equipment is down.
    """
    feature_prefix = parameters["on_off"]["tag_prefix"]
    td_df = td.to_frame()

    equipments = parameters['on_off_general_features']['equipments']
    percentages = parameters['on_off_general_features']['percentages']
    df_ = df.copy()
    for equipment in equipments:
        name = feature_prefix+equipment

        tags_to_exclude = td_df[td_df.on_off_dependencies.str.contains(name)].tag.unique()

        # filtering columns due dependencies has some
        # calculated features that use aggregate values
        tags_to_exclude = [x for x in tags_to_exclude if x in df_.columns]
        
        exc_df = create_on_off(parameters, df_, [equipment], percentages)
        mask = pd.Series(exc_df[name] == 0, index=exc_df.index)
        
        df_.loc[:,tags_to_exclude] = df_.loc[:,tags_to_exclude].mask(mask)
    return df_

def calculate_grinding_features(parameters: dict, data: pd.DataFrame) -> pd.DataFrame:
    """
    A function that calculates three features refered to secondary and terciary crushing, to be used in some
    conventional milling models (A0 and A1). 
    These features are (1): The low-size ore mass proportion (in percentage units) that is classified previously to
    secondary crushing and sent straight to conventional milling; (2): The throughput-weighted sum of currents for 
    the secondary crushing process; and (3): The bow-level-weighted sum current of the terciary crushing process.

    Args:
        parameters : Dictionary of parameters.
        data : Data table containing crushing information for all these calculations.

    Returns:
        df: A DataFrame with the grinding features calculated.
    """
    # Parameters
    currents_params   = parameters["crushing_parameters"]["currents"]
    bowlevels_params  = parameters["crushing_parameters"]["bow_levels"]
    troughputs_params = parameters["crushing_parameters"]["throughputs"]
    
    tag_list = [parameters["timestamp_col_name"]] # Timestamp column for the mdt
    
    currents_params_sec_crsh     = currents_params["sec_crushers"]
    currents_params_ter_crsh     = currents_params["ter_crushers"]
    bowlevels_params_ter_crsh    = bowlevels_params["ter_crushers"]
    throughputs_params_134_belts = troughputs_params["134_belts"] # Weight-meter tags for 134 belts (input of sec. crushing)
    throughputs_params_135_belts = troughputs_params["135_belts"] # Weight-meter tags for 135 belts (output of sec. crushing)
    throughputs_params_139_belt  = troughputs_params["139_belt"] # Weight-meter tag for 139 belt (low-size ore straight to milling)
    
    # We gather all tags in a unique list
    tag_ensemble = currents_params_sec_crsh + currents_params_ter_crsh + bowlevels_params_ter_crsh +\
                   throughputs_params_134_belts + throughputs_params_135_belts + [throughputs_params_139_belt]
    tag_list.extend(tag_ensemble)
    
    crushing_df = data[tag_list].copy()
    
    pd.options.mode.chained_assignment = None 
    
    # 1. Fino %
    ore_prop_name = parameters["crushing_parameters"]["feature_names"]["low_size_ore_name"] # Feature name
    
    crushing_df[ore_prop_name] = np.where(
            crushing_df[throughputs_params_134_belts].sum(axis = 1) == 0, np.nan,
                100*crushing_df[throughputs_params_139_belt]/crushing_df[throughputs_params_134_belts].sum(axis = 1)
        )
    
    crushing_df[ore_prop_name] = np.where(
        crushing_df[ore_prop_name] > parameters["crushing_parameters"]["higher_low_size_prop_bound"], np.nan,
        crushing_df[ore_prop_name]
    )
    # 2. Wheighted sum of current per crushing
    sections = parameters["crushing_parameters"]["feature_names"]["currents_groups"]
    prefix   = parameters["crushing_parameters"]["feature_names"]["tag_prefix"]
    
    # 2.1 Sec crushing
    sec_name = f"{prefix}{sections[0]}"
    
    crushing_df[sec_name] = np.where(
        crushing_df[throughputs_params_134_belts].sum(axis = 1) == 0, np.nan,
        np.where(
            crushing_df[throughputs_params_135_belts].sum(axis = 1) == 0, np.nan,
            (1/2)*(
                np.multiply(
                   crushing_df[currents_params_sec_crsh], 
                   crushing_df[throughputs_params_134_belts]
                     ).sum(axis = 1)/crushing_df[throughputs_params_134_belts].sum(axis = 1) +\
                np.multiply(
                   crushing_df[currents_params_sec_crsh], 
                   crushing_df[throughputs_params_135_belts]
                     ).sum(axis = 1)/crushing_df[throughputs_params_135_belts].sum(axis = 1)
                    )
        )
    )
    crushing_df[sec_name] = np.where(
        crushing_df[sec_name] < parameters["crushing_parameters"]["lower_currents_bound"], np.nan,
        crushing_df[sec_name]
    )
    # 2.2 Terciary crushing
    ter_name = f"{prefix}{sections[1]}"
    
    crushing_df[ter_name] = np.where(
        crushing_df[bowlevels_params_ter_crsh].sum(axis = 1) == 0, np.nan,
        np.multiply(
            crushing_df[bowlevels_params_ter_crsh], 
            crushing_df[currents_params_ter_crsh]
              ).sum(axis = 1)/crushing_df[bowlevels_params_ter_crsh].sum(axis = 1)
    )
    crushing_df[ter_name]  = np.where(
        crushing_df[ter_name] < parameters["crushing_parameters"]["lower_currents_bound"], np.nan,
        crushing_df[ter_name]
    )
    crushing_df.index = data[parameters["timestamp_col_name"]]
    df = crushing_df[sorted(crushing_df.columns.difference(tag_list))]
    return df
