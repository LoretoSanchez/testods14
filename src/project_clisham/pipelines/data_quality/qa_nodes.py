import logging
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from pandas.plotting import register_matplotlib_converters
from pandas.tseries.offsets import DateOffset
from scipy.stats import ks_2samp, kurtosis
from tqdm import tqdm
from PIL import Image

from project_clisham.optimus_core.utils import cut_values_from_dict

matplotlib.rcParams.update({"font.size": 4})
register_matplotlib_converters()


def remove_outliers(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Filter values based on the percentil number.

    This function only calculates the cut values for each column on the dataframe.
    it calls internally to cut_values_from_dict to remove the values.

    Args:
        df: a Dataframe containing variables with outliers
        params: a Dict object with the percentile that should be applied at the top and bottom.

    returns:
        a DataFrame with the outliers removed
    """
    logger = logging.getLogger(__name__)

    outlier_dict = dict()
    bottom = params["bottom"]
    top = params["top"]

    for col in df.columns.to_list():
        try:
            p_bottom = df[col].quantile(bottom)
            p_top = df[col].quantile(top)
            outlier_dict[col] = [p_bottom, p_top]
        except TypeError:
            logger.warning(
                f"TypeError: Cannot calculate percentiles for column:\t{col}"
            )
            continue

    df_no_outliers = cut_values_from_dict(df, outlier_dict)
    return df_no_outliers


def apply_cleaning_filters(**df_list) -> pd.DataFrame:
    """Apply filters to input data based on quality criteria for each variable.

    Each feature of the dataset is checked against the following list of rules:
    - Count monthly missing data
    - Check for large gaps of missing data
    - Check for highly concentrated values
    - Check for stability in time using KS test
    Based on these criteria and threshold values, features are kept or removed from the dataset. An additional csv
    output file is exported containing the list of rejected variables, which can be used in production runs.

    Args:
        df_list: Dict of DataFrames containing input data. Last key should be 'parameters' containing the dict of
        parameters.

    Returns:
        df: DataFrame with removed columns based on quality criteria.

    """
    # Get logger
    logger = logging.getLogger(__name__)
    logger.info("Initializing data quality assessment for all sources")

    # Get parameters
    timestamp_name = df_list["parameters"]["timestamp_col_name"]
    params_qa = df_list["parameters"]["quality_assessment"]
    ignore_dates = params_qa["ignore_dates"]
    # Remove 'parameters' from dictionary
    df_list.popitem()

    # Dictionary of features to be removed {"tag": "filter_function"}
    cols_removed_dict = {}
    # Dictionary of removed tags and source {"tag": "source"}
    tags_per_source_dict = {}

    # Iterate over dfs
    for key in df_list:
        logger.info(f"Applying filters to `{key}` source")
        (
            cols_removed_key,
            stat_na,
            concentrated_data,
            null_data,
            ks_test_sum,
        ) = _single_apply_cleaning_filters(
            df_list[key].set_index(timestamp_name), params_qa[key], ignore_dates
        )
        cols_removed_dict.update(cols_removed_key)
        tags_per_source_dict.update({tag: key for tag in cols_removed_key.keys()})

    # return cols_removed_df, stat_na, concentrated_data, null_data, ks_test_sum
    return (
        stat_na,
        concentrated_data,
        null_data,
        ks_test_sum,
    )  # TODO: ML - add outputs to doctstring


def _single_apply_cleaning_filters(
    data: pd.DataFrame, parameters: dict, ignore_dates: List
):
    """Apply filters to input data based on quality criteria for each variable.

    Args:
        data: DataFrame containing data to be checked for missing values.
        parameters: Dictionary of parameters.
        ignore_dates: List of data ranges to be excluded from the analysis.

    Returns:
        df_filter, rm_cols_dict: DataFrame with columns removed, updated dict of all columns removed from original df.

    """
    # set logger
    # logger = logging.getLogger(__name__) #not used
    data.index = pd.to_datetime(data.index, format="%Y-%m-%d %H:%M:%S")
    df = data.copy()
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")
    assert df.index.is_monotonic, "Index not ascending"

    # Remove blackout periods
    if ignore_dates:
        for [t1, t2] in ignore_dates:
            df = df[~((df.index >= t1) & (df.index <= t2))]

    # Get length of df
    df_old_len = len(df)

    # Outlier removal for QA
    cut_dict = parameters["tag_range"]
    df = cut_values_from_dict(df, cut_dict)

    # Get other parameters
    # Get columns to ignore in cleaning process
    ignore_tags = parameters["ignore_tags"]
    # Filter average monthly missing
    filter_na = parameters["filter_na_monthly"]
    monthly_na_thresh = parameters["monthly_na_thresh"]
    # Filter max window missing
    filter_na_window = parameters["filter_na_window"]
    max_window_na_thresh = parameters["max_window_na_thresh"]
    # Filter KS stability
    filter_ks_stability = parameters["filter_ks_stability"]
    ks_stability_thresh = parameters["ks_stability_thresh"]
    # Filter concentrated values
    parameters_concentrated = parameters["filter_concentrated_values"]
    # Kurtosis
    filter_kurtosis = parameters_concentrated["filter_kurtosis"]
    kurtosis_thresh = parameters_concentrated["kurtosis_thresh"]
    # Count unique
    filter_count_unique = parameters_concentrated["filter_count_unique"]
    count_unique_thresh = parameters_concentrated["count_unique_thresh"]
    # Most common frequency
    filter_mode_freq = parameters_concentrated["filter_mode_freq"]
    mode_freq_thresh = parameters_concentrated["mode_freq_thresh"]
    # Append targets  TODO: add target tags in parameters
    # ignore_tags.append(parameters["mdt_prep"]["throughput_var_name"])
    # ignore_tags.append(parameters["mdt_prep"]["grade_tail_var_name"])

    # Cumulative dictionary of features to be removed
    cols_removed_local = {}

    # Check monthly missing data and apply filter
    if filter_na:
        df, cols_removed_local, stat_na = _count_monthly_na(
            df=data,
            ignore_tags=ignore_tags,
            threshold=monthly_na_thresh,
            rm_cols_dict=cols_removed_local,
        )
    # logger.info(cols_removed_local)
    cols_removed_local = {}

    # Check large window of missing values and apply filter
    if filter_na_window:
        df, cols_removed_local, null_data = _max_window_na(
            df=data,
            ignore_tags=ignore_tags,
            threshold=max_window_na_thresh,
            rm_cols_dict=cols_removed_local,
        )
    # logger.info(cols_removed_local)
    cols_removed_local = {}
    # Check concentration of values and apply filter
    if filter_kurtosis or filter_count_unique or filter_mode_freq:
        filter_types = {
            "filter_kurtosis": filter_kurtosis,
            "filter_count_unique": filter_count_unique,
            "filter_mode_freq": filter_mode_freq,
        }
        threshold_values = {
            "kurtosis_thresh": kurtosis_thresh,
            "count_unique_thresh": count_unique_thresh,
            "mode_freq_thresh": mode_freq_thresh,
        }

        df, cols_removed_local, concentrated_data = _concentrated_data(
            df=data,
            ignore_tags=ignore_tags,
            threshold_values=threshold_values,
            filter_types=filter_types,
            rm_cols_dict=cols_removed_local,
        )

    # Check KS stability and apply filter
    # logger.info(cols_removed_local)
    cols_removed_local = {}
    if filter_ks_stability:
        df, cols_removed_local, ks_test_sum = _ks_stability(
            df=data,
            ignore_tags=ignore_tags,
            threshold=ks_stability_thresh,
            rm_cols_dict=cols_removed_local,
        )

    # Make sure no rows of data have been removed
    assert df_old_len == len(
        df
    ), "Error: some records were removed from dataset after applying cleaning filters"

    return cols_removed_local, stat_na, concentrated_data, null_data, ks_test_sum


def _count_monthly_na(
    df: pd.DataFrame,
    ignore_tags: List[str] = None,
    threshold: float = 0.9,
    rm_cols_dict: dict = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove columns for which the monthly average of missing data is above a threshold value.

    Args:
        df: DataFrame containing data to be checked for missing values.
        threshold (%): Above this % of monthly-average missing data, column will be dropped.
        rm_cols_dict (opt): Cumulative dictionary consisting of {removed_col: filter_function}.

    Returns:
        df_filter, rm_cols_dict: DataFrame with columns removed, updated dict of all columns removed from original df.

    """
    # Initialize empty lists and dictionaries
    if ignore_tags is None:
        ignore_tags = []
    if rm_cols_dict is None:
        rm_cols_dict = {}

    # Get logger
    logger = logging.getLogger(__name__)

    # Copy DataFrame
    df_old = df.copy()

    # Define temp columns
    col_date = "fecha"
    col_date_g = "ref_date_groupby"
    df[col_date] = df.index

    # Get rm_cols from the keys of rm_cols_dict
    rm_cols = list(rm_cols_dict.keys())

    # Make an array of columns to apply rules
    cols = [f for f in df.columns if f not in ignore_tags and f != "fecha"]

    # Formatting numeric and dates
    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")

    # Group by months
    df[col_date_g] = df[col_date].dt.strftime("%Y-%m")
    df_g = df.groupby(col_date_g)

    # Calculates missing percentage
    null_perc = pd.DataFrame(data=list(df_g.groups.keys()), columns=["ref_date"])

    for col in cols:
        null_col_count = df_g.agg({col: lambda x: x.isnull().sum()})
        total_col_count = df_g.agg({col: lambda x: len(x)})
        null_col_perc = round((null_col_count / total_col_count) * 100, 2)
        null_perc = null_perc.merge(
            null_col_perc,
            left_on="ref_date",
            right_on=col_date_g,
            right_index=False,
            validate="one_to_one",
        )

    # Create describe table and transpose
    stat_na = null_perc.describe().transpose()
    stat_na.reset_index(inplace=True)
    # Get the columns above threshold for monthly mean of % missing
    cols_above = _filter_thresh(stat_na, "mean", threshold)

    # Now filter DataFrame and remove columns
    df_filter, rm_cols = _filter_data(df, cols_above, rm_cols)

    # Remove temporary column
    df_filter.drop(columns=[col_date, col_date_g], inplace=True)

    # Output how many columns were kept
    logger.info(
        "_count_monthly_na kept {} columns from {}".format(
            len(df_filter.columns), len(df.drop(columns=[col_date, col_date_g]).columns)
        )
    )

    # Make dictionary of removed columns
    for col in rm_cols:
        if col not in rm_cols_dict.keys():
            rm_cols_dict[col] = "_count_monthly_na"

    # Use df_old which has all data
    df_filter = df_old[df_filter.columns]

    return df_filter, rm_cols_dict, stat_na


def _filter_thresh(
    df: pd.DataFrame, col: str, thr: float, greater: bool = True
) -> List[str]:
    """Function to check which variables meet the quality requirement through a defined threshold.

    Args:
        df: Pandas DataFrame with the name of columns as index and statistics generated in the previous functions.
        col: Column name you want to apply the rule for checking threshold.
        greater: Boolean to choose if you want to check if statistics values are greater or lower than threshold.

    Returns:
        col_names: List with column names that attends the rule.

    """
    if greater:
        df_filter = df.loc[df[col] >= thr, :]
    else:
        df_filter = df.loc[df[col] < thr, :]

    # Get the unique column names
    col_names = np.array(df_filter.index.unique())

    return col_names


def _filter_data(
    df: pd.DataFrame, cols: List[str], rm_cols: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Function to filter dataset removing columns selected by cleaning rules applied.

    Args:
        df: Pandas DataFrame with master data.
        cols: List of column names to be removed from dataset.
        rm_cols: Cumulative list of all columns that have been removed from original dataset.

    Returns:
        df_filter, rm_cols: Filtered dataset, updated list with all columns removed up-to-now.

    """
    rm_cols = np.concatenate([rm_cols, cols])
    df_filter = df.loc[:, ~df.columns.isin(rm_cols)].copy()

    return df_filter, rm_cols


def _max_window_na(
    df: pd.DataFrame,
    ignore_tags: List[str] = None,
    threshold: int = 500,
    rm_cols_dict: dict = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove columns for which the largest window of missing values is above a threshold value.

    Args:
        df: DataFrame containing data to be checked for large gaps of missing values.
        threshold: Above this number of consecutive missing values, column will be dropped.
        rm_cols_dict (opt): Cumulative dictionary consisting of {removed_col: filter_function}.

    Returns:
        df_filter, rm_cols_dict: DataFrame with columns removed, updated dict of all columns removed from original df.

    """
    # Initialize empty lists and dictionaries
    if ignore_tags is None:
        ignore_tags = []
    if rm_cols_dict is None:
        rm_cols_dict = {}

    # Get logger
    logger = logging.getLogger(__name__)

    # Copy DataFrame
    df_old = df.copy()

    # Get rm_cols from the keys of rm_cols_dict
    rm_cols = list(rm_cols_dict.keys())

    # Define temp columns
    col_date = "timestamp_temp"
    df[col_date] = df.index

    # Make an array of columns to apply rules
    cols = [f for f in df.columns if f not in ignore_tags and f != "timestamp_temp"]

    # Formatting numeric and dates
    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")

    # Calculates missing windows
    null_data = pd.DataFrame()
    for col in cols:
        null_data_sum = (
            df[col]
            .isnull()
            .astype(int)
            .groupby(df[col].notnull().astype(int).cumsum())
            .sum()
        )
        null_data_max = null_data_sum.max(axis=0)
        line = pd.DataFrame([[col, null_data_max]])
        null_data = pd.concat([null_data, line], axis=0)
    null_data.columns = ["Variable_Name", "Max_Null_Window"]
    null_data.set_index("Variable_Name", inplace=True)

    # Get the columns above threshold for max number of consecutive missing values and filter from df
    cols = _filter_thresh(null_data, "Max_Null_Window", threshold)
    df_filter, rm_cols = _filter_data(df, cols, rm_cols)

    # Remove temporary column
    df_filter.drop(columns=[col_date], inplace=True)

    # Output how many columns were kept
    logger.info(
        "_max_window_na kept {} columns from {}".format(
            len(df_filter.columns), len(df.drop(columns=[col_date]).columns)
        )
    )

    # Make dictionary of removed columns
    for col in rm_cols:
        if col not in rm_cols_dict.keys():
            rm_cols_dict[col] = "_max_window_na"

    # Use df_old which has all data
    df_filter = df_old[df_filter.columns]

    return df_filter, rm_cols_dict, null_data.reset_index()


def _ks_stability(
    df: pd.DataFrame,
    ignore_tags: List[str] = None,
    threshold: int = 50,
    rm_cols_dict: dict = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove columns for which the value distribution changes too much based on KS test.

    For each variable, perform KS test in moving windows across the training set. If the % of tests showing different
    data distributions is above a threshold value, the column is marked as unstable and removed from dataset.

    Args:
        df: DataFrame containing variables that will be individually checked for data stability.
        ignore_tags (opt): List of columns for which test will not be run.
        threshold (%): Above this % of failed KS tests, the variable is deemed as unstable and removed from df.
        rm_cols_dict (opt): Cumulative dictionary consisting of {removed_col: filter_function}.

    Returns:
        df_filter, rm_cols_dict: DataFrame with columns removed, updated dict of all columns removed from original df.

    """
    # Initialize empty lists and dictionaries
    if ignore_tags is None:
        ignore_tags = []
    if rm_cols_dict is None:
        rm_cols_dict = {}

    # Get logger
    logger = logging.getLogger(__name__)

    # Copy DataFrame
    df_old = df.copy()

    # Get rm_cols from the keys of rm_cols_dict
    rm_cols = list(rm_cols_dict.keys())

    # Define temp columns
    col_date = "fecha"
    df[col_date] = df.index

    # Make an array of columns to apply rules
    cols = [f for f in df.columns if f not in ignore_tags and f != "fecha"]

    # Determine time interval to check stability
    start_date = pd.to_datetime(df[col_date].min().date())
    end_date = pd.to_datetime(df[col_date].max().date())

    assert (
        end_date > start_date
    ), "Error: end date < start date in _ks_stability function"

    # Configure period sizes (period 2 will be KS-tested against period 1 in moving windows of the following sizes)
    n_days_1 = 56  # days
    n_days_2 = 14  # days

    # Create rolling windows, each starting every 7 days
    roll_windows = pd.period_range(start=start_date, end=end_date, freq="28D")
    roll_windows = roll_windows.to_timestamp()
    # Make sure starting date of each window is early enough to fit KS-test without going over end_date
    roll_windows = [
        date
        for date in roll_windows
        if date + DateOffset(days=n_days_1) + DateOffset(days=n_days_2) <= end_date
    ]

    # Calculate KS to check stability
    ks_rows = []
    for window in roll_windows:
        # Period 1
        start_date1 = pd.to_datetime(window)
        end_date1 = window + DateOffset(days=n_days_1) - DateOffset(days=1)
        # Period 2
        start_date2 = window + DateOffset(days=n_days_1)
        end_date2 = window + DateOffset(days=n_days_1) + DateOffset(days=n_days_2)
        # Extract data for both periods
        df_ks1 = df.loc[(df[col_date] >= start_date1) & (df[col_date] <= end_date1), :]
        df_ks2 = df.loc[(df[col_date] >= start_date2) & (df[col_date] <= end_date2), :]

        for col in cols:
            # Remove missing values in col: ks_2samp is not designed to handle nan
            df_ks1_col = df_ks1[col].dropna()
            df_ks2_col = df_ks2[col].dropna()
            # KS-test (KS statistic and p-value can be used to reject null hypothesis)
            if len(df_ks1_col) != 0 and len(df_ks2_col) != 0:
                ks = ks_2samp(df_ks1_col, df_ks2_col)
                # Option 1: use only KS statistic
                reject_h0 = 1 if ks[0] >= 0.5 else 0
                # Option 2: use both (more strict)
                # reject_h0 = 1 if ks[1] <= 0.01 or ks[0] >= 0.5 else 0
                ks_rows.append([col, start_date1, end_date2, ks[0], ks[1], reject_h0])

    # Dump results to df and use it to filter columns
    # TODO: save this table somewhere for future analysis
    ks_df = pd.DataFrame(
        ks_rows,
        columns=["tag", "start_date", "end_date", "KS", "p-value", "is_unstable"],
    ).sort_values(by=["tag", "start_date"])

    if not ks_df.empty:
        # Normalize
        ks_test_sum = (
            100
            * pd.DataFrame(ks_df.groupby("tag").is_unstable.sum())
            / len(roll_windows)
        )

        # Get the columns above threshold for % of failed KS test and filter from df
        cols = _filter_thresh(ks_test_sum, "is_unstable", threshold)
        df_filter, rm_cols = _filter_data(df, cols, rm_cols)

        # Remove temporary column
        df_filter.drop(columns=[col_date], inplace=True)

        # Output how many columns were kept
        logger.info(
            "_ks_stability kept {} columns from {}".format(
                len(df_filter.columns), len(df.drop(columns=[col_date]).columns)
            )
        )

        # Make dictionary of removed columns
        for col in rm_cols:
            if col not in rm_cols_dict.keys():
                rm_cols_dict[col] = "_ks_stability"

        # Use df_old which has all data
        df_filter = df_old[df_filter.columns]

    else:
        df_filter = df_old

    return df_filter, rm_cols_dict, ks_test_sum.reset_index()


def _concentrated_data(
    df: pd.DataFrame,
    ignore_tags: List[str] = None,
    threshold_values: dict = None,
    filter_types: dict = None,
    rm_cols_dict: dict = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove columns for which the value distribution is not diverse enough to contribute to predictions.

    For each variable, concentration of values is optionally assessed based on three filters:
        1. Kurtosis value
        2. High concentration of the most frequent values
        3. Not enough different values

    Args:
        df: DataFrame containing variables that will be individually checked for concentrated data.
        ignore_tags (opt): List of columns for which tests will not be run.
        threshold_values: Dictionary containing threshold values for keys 'kurtosis_thresh', 'count_unique_thresh' and
        'mode_freq_thresh'.
        filter_types: Dictionary containing boolean values for keys 'filter_kurtosis', 'filter_count_unique'
        and 'filter_mode_freq'.
        rm_cols_dict (opt): Cumulative dictionary consisting of {removed_col: filter_function}.

    Returns:
        df_filter, rm_cols_dict: DataFrame with columns removed, updated dict of all columns removed from original df.
    """
    # Initialize empty lists and dictionaries
    if ignore_tags is None:
        ignore_tags = []
    if rm_cols_dict is None:
        rm_cols_dict = {}
    if filter_types is None:
        filter_types = {
            "filter_kurtosis": True,
            "filter_count_unique": True,
            "filter_mode_freq": True,
        }
    else:
        if any(
            x not in ("filter_kurtosis", "filter_count_unique", "filter_mode_freq")
            for x in filter_types
        ):
            raise ValueError("Unexpected key in threshold_types dictionary")
    if threshold_values is None:
        threshold_values = {
            "kurtosis_thresh": 200,
            "count_unique_thresh": 5,
            "mode_freq_thresh": 60,
        }
    else:
        if any(
            x not in ("kurtosis_thresh", "count_unique_thresh", "mode_freq_thresh")
            for x in threshold_values
        ):
            raise ValueError("Unexpected key in threshold_values dictionary")

    # Get logger
    logger = logging.getLogger(__name__)

    # Copy DataFrame
    df_old = df.copy()

    # Get rm_cols from the keys of rm_cols_dict
    rm_cols = list(rm_cols_dict.keys())

    # Make an array of columns to apply rules
    temp_cols = ["timestamp_temp", "ref_date_groupby", "fecha"]
    cols = [f for f in df.columns if f not in ignore_tags and f not in temp_cols]

    # Calculate kurtosis and most frequent values
    counter_rows = []
    df_filter = df.loc[:, df.columns.isin(cols)]
    df_filter = round(
        df_filter[cols], 4
    )  # Round to 4 digits to account for values that vary insignificantly
    n_most_freq = 3  # Number of most frequent values to be extracted
    for col in cols:
        # Calculate % covered by the most common values
        counter_n_most_freq = Counter(df[col]).most_common(n_most_freq)
        pct_n_most_common = (
            100 * np.sum([pair[1] for pair in counter_n_most_freq]) / len(df[col])
        )
        # Calculate Kurtosis
        # logger.warning('Check this col:\t'+col)
        try:
            kurt = kurtosis(df[col], nan_policy="omit")
        except Exception as e:
            print(e)
            continue
        # Create and append row with all information
        counter_rows.append([col, kurt, pct_n_most_common])

    # Transform to df
    counter_df = (
        pd.DataFrame(counter_rows, columns=["tag", "kurtosis", "pct_n_most_common"])
        .sort_values(by=["tag"])
        .set_index("tag")
    )

    # Calculate unique values and merge to final df
    unique_data = pd.DataFrame(df_filter.nunique(), columns=["count_unique"])
    final_data = counter_df.merge(
        unique_data, left_index=True, right_index=True, validate="one_to_one"
    )

    # For Kurtosis, filter out columns above threshold Kurtosis value
    if filter_types["filter_kurtosis"]:
        cols_1 = _filter_thresh(
            final_data, "kurtosis", threshold_values["kurtosis_thresh"]
        )
        df_filter, rm_cols = _filter_data(df, cols_1, rm_cols)
        for col in cols_1:
            rm_cols_dict[col] = "_concentrated_data_kurtosis"

    # For unique values, filter out columns below unique-count threshold
    if filter_types["filter_count_unique"]:
        cols_2 = _filter_thresh(
            final_data,
            "count_unique",
            threshold_values["count_unique_thresh"],
            greater=False,
        )
        df_filter, rm_cols = _filter_data(df_filter, cols_2, rm_cols)
        for col in cols_2:
            rm_cols_dict[col] = "_concentrated_data_count_unique"

    # For mode frequency, filter out columns above threshold value
    if filter_types["filter_mode_freq"]:
        cols_3 = _filter_thresh(
            final_data, "pct_n_most_common", threshold_values["mode_freq_thresh"]
        )
        df_filter, rm_cols = _filter_data(df_filter, cols_3, rm_cols)
        for col in cols_3:
            rm_cols_dict[col] = "_concentrated_data_mode_freq"

    # Output how many columns were kept
    logger.info(
        "_concentrated_data kept {} columns from {}".format(
            len(df_filter.columns), len(df.columns)
        )
    )

    # Use df_old which has all data
    df_filter = df_old[df_filter.columns]

    return df_filter, rm_cols_dict, final_data.reset_index()


def plot_bad_tags(**df_list) -> None:
    """Create a pdf file containing time-series plot and histogram for each tag of bad quality.

    Args:
        df_list: Dict of DataFrames containing input data. Last key should be 'tags' containing the dict of tags to be
        plotted.

    """
    # TODO: fix order of plots as it's not the same as the list in csv output file
    # Get logger
    logger = logging.getLogger(__name__)
    logger.info("Creating plots for tags of insufficient quality")
    ts_col_name = df_list["parameters"]["timestamp_col_name"]

    # Get tags to be removed and remove last item in dictionary
    tags_bad_quality = list(df_list["tags"].tag)
    df_list.popitem()
    # Get parameters and remove last item in dictionary
    parameters = df_list["parameters"]["quality_assessment"]
    df_list.popitem()
    for key in df_list.keys():
        logger.info(f"Plotting tags from ´{key}´ source")
        _single_plot_bad_tags(
            df_list[key].set_index(ts_col_name), tags_bad_quality, key, parameters
        )


def _single_plot_bad_tags(
    data: pd.DataFrame, tags_bad_quality: List, source_name: str, parameters: dict
):
    """Create a pdf file containing time-series plot and histogram for each tag of bad quality.

    Args:
        data: DataFrames containing input data.
        tags_bad_quality: List of tags to be removed
        source_name: Human-readable name of the specific area of the plant.

    """
    # Select columns to be plotted
    df = data[data.columns[data.columns.isin(tags_bad_quality)]]
    # Check that some columns remain
    if df.shape[1] == 0:
        return

    # Outlier removal for QA
    cut_dict = parameters[source_name]["tag_range"]
    if len(cut_dict) != 0:
        # Check that tags exist in tags_bad_quality
        rmv_from_cut_dict = [
            tag for tag in cut_dict.keys() if tag not in tags_bad_quality
        ]
        for tag in rmv_from_cut_dict:
            cut_dict.pop(tag, None)
        df = cut_values_from_dict(df, cut_dict)

    df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")

    # Dimensions for any n-rows x m-cols array of subplots for each page in pdf file
    n, m = 4, 2
    filename_pdf = f"data/08_reporting/data_quality/tags_low_quality_{source_name}.pdf"

    # Start pdf creation
    with PdfPages(filename_pdf) as pdf:

        # Before beginning the iteration through all the data, initialize the layout for the plots and create a
        # representation of the subplots that can be easily iterated over for knowing when to create the next page

        f, axarr = plt.subplots(n, m)
        arr_ij = [(x, y) for x, y in np.ndindex(axarr.shape)]
        subplots = [axarr[index] for index in arr_ij]

        splot_index = 0

        # Iterate through each sample in the data
        for sample in tqdm(range(df.shape[1])):

            subplots[splot_index].set_title(df.columns[sample])
            subplots[splot_index].plot(df.iloc[:, sample])
            subplots[splot_index + 1].set_title(df.columns[sample])
            subplots[splot_index + 1].hist(df.iloc[:, sample].dropna(), 30)

            splot_index += 2

            # If end of page
            if splot_index == n * m:
                plt.tight_layout()
                pdf.savefig()
                plt.close(f)

                f, axarr = plt.subplots(n, m)
                arr_ij = [(x, y) for x, y in np.ndindex(axarr.shape)]
                subplots = [axarr[index] for index in arr_ij]
                splot_index = 0

        # Add last page
        plt.tight_layout()
        pdf.savefig()
        plt.close(f)


def create_qa_table(
    imt: pd.DataFrame,
    stat_na: pd.DataFrame,
    concentrated_data: pd.DataFrame,
    null_data: pd.DataFrame,
    ks_test_sum: pd.DataFrame,
    data_dict: pd.DataFrame,
):
    """TODO: Docstring"""
    target_col_name = "tag"
    complete = pd.DataFrame(imt.columns, columns=[target_col_name])
    complete = pd.merge(
        complete,
        stat_na,
        left_on=target_col_name,
        right_on="index",
        how="left",
        copy=False,
    )
    complete = pd.merge(
        complete,
        concentrated_data,
        left_on=target_col_name,
        right_on="index",
        how="left",
        copy=False,
    )
    complete = pd.merge(
        complete,
        null_data,
        left_on=target_col_name,
        right_on="Variable_Name",
        how="left",
        copy=False,
    )
    complete = pd.merge(
        complete,
        ks_test_sum,
        left_on=target_col_name,
        right_on="tag",
        how="left",
        copy=False,
    )
    # TODO: Move to params this values
    mean_threshold = 90
    max_window_na_thresh = 480
    ks_stability_thresh = 60
    kurtosis_thresh = 200
    count_unique_thresh = 5
    mode_freq_thresh = 60
    # Correct to Numeric
    complete["kurtosis"] = pd.to_numeric(complete["kurtosis"], errors="coerce").fillna(
        0
    )
    # Create the result columns
    complete["too_many_na_per_month"] = complete.apply(
        lambda x: 1 if x["mean"] > mean_threshold else 0, axis=1
    )
    complete["too_many_consecutives_na"] = complete.apply(
        lambda x: 1 if x["Max_Null_Window"] > max_window_na_thresh else 0, axis=1
    )
    complete["unstable_over_70_days"] = complete.apply(
        lambda x: 1 if x["is_unstable"] > ks_stability_thresh else 0, axis=1
    )
    complete["kurtosis_thresh"] = complete.apply(
        lambda x: 1 if x["kurtosis"] > kurtosis_thresh else 0, axis=1
    )
    complete["few_unique_values"] = complete.apply(
        lambda x: 1 if x["count_unique"] < count_unique_thresh else 0, axis=1
    )
    complete["mode_freq_thresh"] = complete.apply(
        lambda x: 1 if x["pct_n_most_common"] > mode_freq_thresh else 0, axis=1
    )
    # Selection of columns
    sum_cols = [
        "too_many_na_per_month",
        "too_many_consecutives_na",
        "unstable_over_70_days",
        "kurtosis_thresh",
        "few_unique_values",
        "mode_freq_thresh",
    ]
    # Add all the interesting columns into a single Score column

    complete["score"] = complete[sum_cols].sum(axis=1)

    # Joined table with the result and the appended Dictionary Values.
    interest_columns = [
        target_col_name,
        "score",
        "mean",
        "std",
        "kurtosis",
        "pct_n_most_common",
        "count_unique",
        "Max_Null_Window",
        "is_unstable",
    ]
    result = pd.merge(
        data_dict,
        complete[interest_columns + sum_cols],
        left_on="tag",
        right_on=target_col_name,
        how="left",
        copy=False,
    )

    return result


def go_or_no_go(data: pd.DataFrame, tag_dict: pd.DataFrame, parameters: dict):
    """TODO: Docstring"""
    logger = logging.getLogger(__name__)
    qa = data[["tag", "score"]].copy()
    tag_dict = pd.merge(
        tag_dict, qa, left_on="tag", right_on="tag", how="left", copy=False
    )
    feature_list_from_colname = [
        x for x in tag_dict.columns.values if x.endswith("_feature")
    ]
    go_or_nogo = True
    for feat in feature_list_from_colname:
        my_data = (
            tag_dict.query(f"{feat} == 1.")
            .query("derived == 0.")[["tag", "score"]]
            .copy()
        )
        stats = my_data.groupby("score").size()
        if len(stats) > 0:
            for i, score in enumerate(stats):
                if i in [0, 1]:
                    logger.info(f" - {score} tags have score: {i} for feature {feat}")
                if i >= 2:
                    logger.error(f" - {score} tags have score: {i} for feature {feat}")
                    logger.error(my_data.query("score >= 2."))
                    go_or_nogo = False

        else:
            logger.info(
                f"No tags were selected for feature: {feat}, please check dictionary"
            )
    if go_or_nogo:
        return go_or_nogo
    if parameters["stop_if_no_go"]:
        raise ValueError(
            "There are one or more tags with quality problems, please check the log"
        )
    logger.info(
        'Code stopping criterion has been bypassed. Change this behavior in parameters["curation"]'
    )
    return go_or_nogo


def generate_qa_plots(
    tag_dict_data: pd.DataFrame, qa_data: pd.DataFrame, dci_data: pd.DataFrame
):
    """TODO: Docstring
    Not Implemented yet.
    """

    tag_dict_data = tag_dict_data.query("derived == 0.0")
    qa_data = qa_data.query("derived == 0.0")

    dci_data.set_index("fecha", inplace=True)
    dci_data = dci_data["2020-04-01":"2020-09-09"]
    # Create a dict with the tags per score
    tags_per_score_dict = dict()
    for name, group in qa_data.groupby("score"):
        tags_per_score_dict[f"score_{int(name)}"] = group["tag"].tolist()

    for key, list_of_tags in tags_per_score_dict.items():
        list_of_plots = generate_plot(tag_dict_data, key, list_of_tags, dci_data)
        export_pdf(key, list_of_plots)


def generate_plot(
    tag_dict_data: pd.DataFrame, score: str, list_of_tags: List, dci_data: pd.DataFrame
):
    """TODO: Docstring
    Not Implemented yet.
    """

    list_of_plots = []
    logger = logging.getLogger(__name__)

    # TODO: remove this filter to enable plotting for all tags
    # for i in list_of_tags:
    for i in list_of_tags[:20]:  # FIXME This is just for develop
        logger.info(f"Plotting {i}")
        fig_dims = (16, 6)
        f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=fig_dims)
        ax1 = sns.distplot(
            dci_data[i], label="Valor", kde=False, ax=ax1, hist_kws=dict(alpha=1)
        )
        ax2 = sns.lineplot(x=dci_data.index, y=i, data=dci_data, ax=ax2)

        ax1.set(title=dci_data[i].name)
        ax1.set(xlabel="Valor")
        ax1.set(ylabel="Frecuencia")

        ax2.set(title=dci_data[i].name)
        ax2.set(xlabel="Fecha")
        ax2.set(ylabel="Valor")
        ax2.axhline(
            tag_dict_data.loc[tag_dict_data["tag"] == i]["range_max"].values[0],
            color="red",
            label="range_max",
        )
        ax2.axhline(
            tag_dict_data.loc[tag_dict_data["tag"] == i]["range_min"].values[0],
            color="green",
            label="range_min",
        )
        list_of_plots.append(i + ".png")

        if (
            pd.isnull(
                tag_dict_data.loc[tag_dict_data["tag"] == i]["range_min"].values[0]
            )
            == False
            and pd.isnull(
                tag_dict_data.loc[tag_dict_data["tag"] == i]["range_min"].values[0]
            )
            == False
        ):
            ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=-1)

        # TODO: Esta ruta necesita ser parametrizada
        plt.savefig(f"data/08_reporting/data_quality/plots/{score}/" + i + ".png")
        # plt.show()
    return list_of_plots


def export_pdf(score: str, list_of_plots: List):
    """TODO: Docstring
    Not Implemented yet.
    """
    imagelist = list()
    # TODO: parametrizar los directorios
    for i in list_of_plots:
        image = Image.open(f"data/08_reporting/data_quality/plots/{score}/" + i)

        if i == list_of_plots[0]:
            imFirst = image.convert("RGB")

        elif i == list_of_plots[-1]:
            im = image.convert("RGB")
            imagelist.append(im)
            imFirst.save(
                f"data/08_reporting/data_quality/plots/pdf/dch_minco_plots_{score}.pdf",
                save_all=True,
                append_images=imagelist,
            )

        else:
            im = image.convert("RGB")
            imagelist.append(im)
