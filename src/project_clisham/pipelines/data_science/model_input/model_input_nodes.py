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

from datetime import datetime as dt

import pandas as pd

from project_clisham.optimus_core.tag_management import TagDict


def filter_target(params: dict, td: TagDict, data: pd.DataFrame) -> pd.DataFrame:
    """Filter table based on three criteria for each model:
    1. Min/max values
    2. Count of valid values for each shift
    3. Min/max dates to be considered

    Args:
        params: dictionary of parameters
        td: tag dictionary
        data: input data

    Returns:
        Dataframe filtered based on target limits, counts and dates.
    """
    timestamp_col = params["datetime_col"]
    train_start = params["datetime_start"]
    test_end = params["datetime_end"]
    auto_training = params["auto_training"]
    model_target = params["dict_model_target"]
    target_col = td.select("target", model_target)[0]
    if "lag_" in target_col:
        target_current = target_col.split("lag_")[1]
    else:
        target_current = target_col

    target_lag = "calc_p1_lag_" + target_current
    target_count = "calc_count_" + target_current

    lim_current = params["filter"]["current"]
    lim_lag_p1 = params["filter"]["lag_p1"]
    lim_count = params["filter"]["count"]

    # Filter by min/max ranges

    if lim_current is not None:
        cond = (data[target_current] > lim_current[0]) & (
            data[target_current] < lim_current[1]
        )
        data = data[cond]

    if lim_lag_p1 is not None:
        cond = (data[target_lag] > lim_lag_p1[0]) & (data[target_lag] < lim_lag_p1[1])
        data = data[cond]

    # Filter by counts/shift

    if lim_count is not None:
        cond = data[target_count] > lim_count
        data = data[cond]

    if not auto_training:
        # Filter by start/end dates

        if (train_start is not None) & (test_end is not None):
            d1 = pd.to_datetime(train_start, format="%Y-%m-%d")
            d2 = pd.to_datetime(test_end, format="%Y-%m-%d")

            cond = (data[timestamp_col] >= d1) & (data[timestamp_col] <= d2)
            data = data[cond]
    else:
        all_training_hours = params["all_training_hours"]
        data = data.tail(all_training_hours)


    return data


def remove_shut_downs(params: dict, data: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        params: dictionary of parameters
        data: input data

    Returns:
        Dataframe filtered based on shut down dates
    """
    timestamp_col = params["datetime_col"]
    shutdown_period = params["shut_down_dates"]
    if shutdown_period is None:
        filtered_df = data
    else:
        for period in shutdown_period:
            d1 = dt.strptime(period[0], "%Y-%m-%d")
            d2 = dt.strptime(period[1], "%Y-%m-%d")
            if d2 <= d1:
                raise RuntimeError(
                    """
                        Each element in the list is a pair of dates (d1, d2) with format %Y-%m-%d
                        and d1 < d2
                        """
                )
            else:
                cond = (data[timestamp_col] < d1) | (data[timestamp_col] > d2)
                data = data[cond]
        filtered_df = data

    return filtered_df
