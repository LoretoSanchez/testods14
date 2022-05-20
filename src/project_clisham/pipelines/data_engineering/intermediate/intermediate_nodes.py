# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

# from typing import Any, Dict

import pandas as pd
import numpy as np


def add_duracion_revestimiento(
    parameters, df: pd.DataFrame, revestimiento: pd.DataFrame
) -> pd.DataFrame:
    """This method takes the 'revestimiento' file and counts, per
    SAG MILLs -16 & 17-, the amount of days since a 'revestimiento' has
    been replaced in the sag mills.

    This creates a cumulated number of days since a change in 'revestimiento'
    and restart when a new campaign has been taken (detailed in 'revestimiento'
    file)

    Args:
        parameters: parameters of the project.
        df: Raw data.
        revestimiento: details of change in 'revestimiento' in sag mills.
    
    Returns:
        pd.DataFrame: data with standardized names
    """

    equip = [
        "s16",
        "s17",
    ]
    col_timestamp = parameters["timestamp_col_name"]
    feature_prefix = parameters["rev_cal"]["tag_prefix"]
    df = df.set_index(col_timestamp)
    for eq in equip:
        df[feature_prefix + eq + "_ucf_dias"] = None
        df[feature_prefix + eq + "_ucf_dias"] = df[
            feature_prefix + eq + "_ucf_dias"
        ].astype(float)
        dates_rev_change = np.array(
            sorted([pd.to_datetime(d) for d in revestimiento[eq + "_ucf"].values])
        )
        for dias in df.index:  # TODO: JA - inefficient loop, try to improve
            try:
                last_change = sorted(
                    dates_rev_change[dates_rev_change <= dias], key=lambda x: dias - x
                )[0]
                delta = dias - last_change
                df.loc[dias, feature_prefix + eq + "_ucf_dias"] = delta.days
            except IndexError:
                pass  # TODO: JA - consideracion DM a JA de log

    return df.reset_index()

def update_hist_tphs_table(parameters,
        actual_mdt: pd.DataFrame,
        hist_tphs_table: pd.DataFrame) -> pd.DataFrame:

    """
        The idea of this method is to get the actual MDT and append the values of the
    TPH variables for SAG16 and SAG17. This will help ensure the correct calculation
    of processed TPH for each SAG Mill, and restart the values with changes in inner
    walls.

    Args:
     - parameters: Parameters of the project.
     - actual_mdt: the data fed to train or generate recommendations (last few hours).
     - hist_tphs_table: the historic TPH data processed by sag mills.

    Returns:
     - [pd.DataFrame]: augmented historic TPH data processed by each sag mill
     with the new information seen in 'actual_mdt'.
    """

    # Getting the 'time' column name:
    time_col_name = parameters['timestamp_col_name']

    # Initializing the list of tags to get the cumsum:
    list_tags_2_cum_sum = []

    # Cycling over all the groups:
    for cs_group_j in parameters['cumsum_grouping']['groups']:
        for tag_k in cs_group_j:
            list_tags_2_cum_sum.append(tag_k)

    # Getting the latest data of the hist_tphs_table:
    max_date = max(hist_tphs_table.index)

    # Defining the tags of interest:
    tags_of_interest = [tag_j.lower().replace(':','_') for tag_j in list_tags_2_cum_sum]

    # Getting a smaller copy of the 'actual_mdt':
    actual_mdt_2 = actual_mdt[[time_col_name] + tags_of_interest]
    actual_mdt_2.set_index(time_col_name, inplace = True)

    # New historic mdt for tphs:
    new_hist_tphs_table = pd.concat([hist_tphs_table, actual_mdt_2.loc[max_date:]])
    new_hist_tphs_table = new_hist_tphs_table.drop_duplicates()
    new_hist_tphs_table.sort_index(inplace = True)

    # Returning this new historic mdt for tphs (augmented):
    return new_hist_tphs_table


def update_traza_data(parameters,
        data_traza_clean: pd.DataFrame) -> pd.DataFrame:
    """
        The purpose of the method is to get the data in 'data_traza_clean'
    and sort it, drop duplicates, so there is a good merge with the MDT in
    a later node.

    Args:
    - data_traza_clean: Data that comes from 'trazabilidad' that needs to be
    sorted and cleaned of duplicates (if any).

    Returns:
    - [pd.DataFrame] The 'trazabilidad' data ready to be merged with the MDT
      of the process.

    """

    # Getting the 'time' column name:
    time_col_name = parameters['timestamp_col_name']

    # Initializing the output dataframe:
    output_df = data_traza_clean.copy()

    # Parsing the 'time' to date_time:
    output_df[time_col_name] = pd.to_datetime(output_df[time_col_name])

    # Sorting values by time:
    output_df.sort_values(by = time_col_name, inplace = True)

    # Deleting any NA: 
    output_df = output_df.dropna()

    # Dropping duplicates:
    output_df = output_df.drop_duplicates()

    # Getting rid of the duplicated dates, but with differences in other columns:
    # TODO - Check a smarter way to deal with this
    output_df = output_df.groupby([time_col_name]).last()
    output_df.reset_index(inplace = True)

    # Validating uniqueness in this timestamp column:
    if len(output_df[time_col_name].unique()) < output_df.shape[0]:
        raise ValueError(
                    f"Not unique timestamp values in {time_col_name} column @update_traza_data after cleaning the data."
                )

    # Returning this cleaned version of the 'trazabilidad' data:
    return output_df