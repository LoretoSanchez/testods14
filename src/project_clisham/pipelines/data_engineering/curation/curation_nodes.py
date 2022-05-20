import datetime as dt
from typing import Dict, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def replace_outliers_by_value(data: pd.DataFrame, params: Dict, go_or_nogo: bool):
    """Replaces all outliers for a specific tag by a set value from the parameters file.
    This is a clipping function plus a collection of stats

    Args:
        data: a dataframe with the tags to be replaced
        params: a dictionary with the tag name as key value, and high_value and low_value
        go_or_nogo: True if DQ validation has passed. Used to enforce execution order.

    Returns:
        corrected_data: a dataframe with the replaced values
        stats_outliers_by_value: a dataframe with basics stats
    """
    # This is just to make Kedro Viz look nice
    my_params = params["curation"]["replace_outliers_by_value"]["tag_list"]
    index_tag = params["timestamp_col_name"]

    stats_outliers_by_value = _calculate_stats_outliers_by_value(
        my_data=data, params=my_params, index_tag=index_tag
    )
    corrected_data = _correct_outliers_by_value(
        my_data=data.copy(), params=my_params, index_tag=index_tag
    )

    return corrected_data, stats_outliers_by_value


def replace_outliers_by_nan(
    data: pd.DataFrame, params: Dict, td: pd.DataFrame, go_or_nogo: bool
):
    """Replaces all outliers for nan
    This is a clipping function plus a collection of stats

    Args:
        data: a dataframe with the tags to be replaced
        td: dictionary with min and max
        go_or_nogo: True if DQ validation has passed. Used to enforce execution order.

    Returns:
        corrected_data: a dataframe with the replaced values
        stats_outliers_by_value: a dataframe with basics stats
    """
    stats_outliers_by_nan = _calculate_stats_outliers_by_nan(
        my_data=data, params=params, td=td
    )
    corrected_data = _correct_outliers_by_nan(data=data.copy(), params=params, td=td)

    return corrected_data, stats_outliers_by_nan


def _correct_outliers_by_value(my_data: pd.DataFrame, params: Dict, index_tag: str):
    """Clips predefined values from a dataframe, the columns are defined
    in the parameters along high and low values.

    Args:
        my_data: a dataframe with the tags to be replaced
        params: a dictionary with the tag name as key value, and high_value and low_value
        index_tag: Index of the dataframe (usually a date field.)
    Returns:
        my_data: a dataframe with the replaced values
    """
    check_tags = params

    for tag, limit in check_tags.items():
        my_data[tag] = my_data[tag].apply(
            lambda x: limit["high_value"] if x > limit["high_value"] else x
        )
        my_data[tag] = my_data[tag].apply(
            lambda x: limit["low_value"] if x < limit["low_value"] else x
        )
    return my_data


def _calculate_stats_outliers_by_value(
    my_data: pd.DataFrame, params: Dict, index_tag: str
):
    """Calculate basic statistics about how many values are replaced by the function.
    Counts the total, the last month (30 days), and the last two months (60 days) of data

    Args:
        my_data: a dataframe with the tags to be replaced
        params: a dictionary with the tag name as key value, and high_value and low_value
        index_tag: Index of the dataframe (usually a date field.)

    Returns:
        stats_outliers_by_value: a dataframe with basics stats
    """
    check_tags = params
    data = my_data.copy()
    now = dt.datetime.now().isoformat()
    one_month_ago = data[index_tag].max() - dt.timedelta(30)
    two_month_ago = data[index_tag].max() - dt.timedelta(60)
    category_name = "outliers_by_value"
    data.set_index(index_tag, inplace=True)
    # TODO: DM - this could be improved a lot
    # TODO: DM - this does not contemplate all possible values or use cases
    for tag, limit in check_tags.items():
        check_tags[tag]["check_date"] = now
        check_tags[tag]["category"] = category_name
        high = data.query(f"{tag} > {limit['high_value']} ")[tag].count()
        low = data.query(f"{tag} < {limit['low_value']} ")[tag].count()
        total = data[tag].count()

        check_tags[tag]["count_high"] = high
        check_tags[tag]["count_low"] = low
        check_tags[tag]["total"] = total

        high = (
            data.query(f'index >= "{one_month_ago}" ')
            .query(f"{tag} > {limit['high_value']} ")[tag]
            .count()
        )
        low = (
            data.query(f'index >= "{one_month_ago}" ')
            .query(f"{tag} < {limit['low_value']} ")[tag]
            .count()
        )
        total = data.query(f'index >= "{one_month_ago}" ')[tag].count()

        check_tags[tag]["count_high_one_month"] = high
        check_tags[tag]["count_low_one_month"] = low
        check_tags[tag]["total_one_month"] = total

        high = (
            data.query(f'index >= "{two_month_ago}" ')
            .query(f"{tag} > {limit['high_value']} ")[tag]
            .count()
        )
        low = (
            data.query(f'index >= "{two_month_ago}" ')
            .query(f"{tag} < {limit['low_value']} ")[tag]
            .count()
        )
        total = data.query(f'index >= "{two_month_ago}" ')[tag].count()

        check_tags[tag]["count_high_two_month"] = high
        check_tags[tag]["count_low_two_month"] = low
        check_tags[tag]["total_two_month"] = total

    stats: pd.DataFrame = pd.DataFrame(check_tags).T
    stats.reset_index(inplace=True)
    stats.rename(columns={"index": "tag"}, inplace=True)
    return stats


def _correct_outliers_by_nan(data: pd.DataFrame, params: Dict, td: pd.DataFrame):
    """Clips predefined values from a dataframe, the columns are defined
    in the parameters along high and low values.

    Args:
        my_data: a dataframe with the tags to be replaced
        params: a dictionary with the tag name as key value, and high_value and low_value
        index_tag: Index of the dataframe (usually a date field.)
    Returns:
        my_data: a dataframe with the replaced values
    """
    # TODO: update when using td
    datetime = params["timestamp_col_name"]
    # td.set_index("tag", inplace=True)
    df = data.copy()
    variabs = df.columns.to_list()

    variabs.remove(datetime)
    tags_lim = td.loc[
        ~td["range_min"].isna() | ~td["range_max"].isna(), ["range_min", "range_max"]
    ]
    variables = list(set(tags_lim.index.to_list()).intersection(df.columns))
    cut_dict = tags_lim.loc[variables, ["range_min", "range_max"]].T.to_dict()
    for col in cut_dict:
        df.loc[
            (df[col] < cut_dict[col]["range_min"])
            | (df[col] > cut_dict[col]["range_max"]),
            col,
        ] = np.nan
    return df


def _calculate_stats_outliers_by_nan(  # TODO: join this function for both replace by value or by nan
    my_data: pd.DataFrame, params: Dict, td: pd.DataFrame
):
    """Calculate basic statistics about how many values are replaced by the function.
    Counts the total, the last month (30 days), and the last two months (60 days) of data

    Args:
        my_data: a dataframe with the tags to be replaced
        params: a dictionary with the tag name as key value, and high_value and low_value
        td: dictionary of tags #TODO: use the td object not the csv

    Returns:
        stats_outliers_by_value: a dataframe with basics stats
    """
    td.set_index("tag", inplace=True)
    data = my_data.copy()
    tags_lim = td.loc[
        ~td["range_min"].isna() | ~td["range_max"].isna(), ["range_min", "range_max"]
    ]
    variables = list(set(tags_lim.index.to_list()).intersection(data.columns))
    cut_dict = tags_lim.loc[variables, ["range_min", "range_max"]].T.to_dict()

    now = dt.datetime.now().isoformat()
    one_month_ago = data[params["timestamp_col_name"]].max() - dt.timedelta(30)
    two_month_ago = data[params["timestamp_col_name"]].max() - dt.timedelta(60)
    category_name = "outliers_by_nan"
    datetime = params["timestamp_col_name"]
    data.set_index(datetime, inplace=True)

    for tag, limit in cut_dict.items():
        cut_dict[tag]["check_date"] = now
        cut_dict[tag]["category"] = category_name

        high = data[tag].loc[data[tag] > limit["range_max"]].count()
        low = data[tag].loc[data[tag] < limit["range_min"]].count()
        total = data[tag].count()

        cut_dict[tag]["count_high"] = high
        cut_dict[tag]["count_low"] = low
        cut_dict[tag]["total"] = total

        high = data.loc[one_month_ago:, tag].loc[data[tag] > limit["range_max"]].count()
        low = data.loc[one_month_ago:, tag].loc[data[tag] < limit["range_min"]].count()
        total = data.loc[one_month_ago:, tag].count()

        cut_dict[tag]["count_high_one_month"] = high
        cut_dict[tag]["count_low_one_month"] = low
        cut_dict[tag]["total_one_month"] = total

        high = data.loc[two_month_ago:, tag].loc[data[tag] > limit["range_max"]].count()
        low = data.loc[two_month_ago:, tag].loc[data[tag] < limit["range_min"]].count()
        total = data.loc[two_month_ago:, tag].count()

        cut_dict[tag]["count_high_two_month"] = high
        cut_dict[tag]["count_low_two_month"] = low
        cut_dict[tag]["total_two_month"] = total

    stats: pd.DataFrame = pd.DataFrame(cut_dict).T
    stats.reset_index(inplace=True)
    stats.rename(columns={"index": "tag"}, inplace=True)
    return stats


def impute_missing_ffill(data: pd.DataFrame, params: Dict):
    """Forward fills values for a specific tags on the parameters file.

    Args:
        data: a dataframe with the tags to be replaced
        params: a dictionary with the tags to fill and the limit of filling


    Returns:
        data: a dataframe with the replaced values
    """
    groups = params["curation"]["ffill"]
    for group in groups:
        ref_param = params["curation"]["ffill"][group]["ref_param"]
        limit = params["curation"]["ffill"][group]["limit"]
        if ref_param:
            ref = params["curation"]["ffill"][group]["tag_list"]
            tags = params[ref]
        else:
            tags = params["curation"]["ffill"][group]["tag_list"]
        if len(set(tags).intersection(data.columns)) == len(tags):
            data.loc[:, tags] = data.loc[:, tags].ffill(limit=limit)
        else:
            raise ValueError(f"Some variables to impute are not in the DF")
    return data