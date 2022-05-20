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

import logging
from typing import Any, Dict

import pandas as pd
import numpy as np
import project_clisham.pipelines.data_engineering.features as features
from project_clisham.pipelines.data_engineering.features.features_nodes import (
    create_target_flot,
    create_cuf_feature,
    create_diff_bancos

)
from project_clisham.optimus_core.tag_management import TagDict
from project_clisham.pipelines.data_engineering.features.feature_nodes_A0 import (
    _calculate_weighted_a0,
)

logger = logging.getLogger(__name__)


def renormalize_ganancia_feeder(parameters: dict, data: pd.DataFrame, td: TagDict) -> pd.DataFrame:
    """Re Normalize the ganancia feeder so they add up to one

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.
        td: TagDict.

    Returns:
        data: df with new variables.

    """

    # Select mill columns
    feeder_parameters = parameters["ma2_feeders_norm"].copy()

    tags = [tag for feeder in feeder_parameters for tag in feeder_parameters[feeder]]

    # Select features
    df = data[tags].copy()

    for feeder in feeder_parameters:
        tags_feeder = feeder_parameters[feeder]
        sum_all = df[tags_feeder].sum(axis=1)
        for tag in tags_feeder:
            df[tag] = (df[tag] / sum_all) * 100
            df[tag].replace([np.inf, -np.inf], 0, inplace=True)

    return df[tags]


def update_pres_bhc(parameters: dict, data: pd.DataFrame, td: TagDict) -> pd.DataFrame:
    """Recalculate the pbhc pressure
    Uses the original function for the feature generation

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.
        td: TagDict.

    Returns:
        data: df with new variables.

    """
    timestamp_col_name = parameters["timestamp_col_name"]
    if timestamp_col_name not in data.columns:
        data[timestamp_col_name] = np.nan

    return _calculate_weighted_a0(parameters, data, variable="pres_bhc")


def translate_air(parameters: dict, data: pd.DataFrame, td: TagDict) -> pd.DataFrame:
    """Transform deltas into air flows per celda.
    
    Considerations: We only set as controls the mean airflow
    value per line and the delta between 0 and -2 airflow cells
    (python list indexes).
    We pivot using the -2 cell and scale up/down
    using the current and suggested mean airflow. Then we add the 
    suggested delta in order to get the 0 airflow cell.

    Args:
        parameters: Dictionary of parameters.
        data: Master table containing variables.
        td: TagDict.

    Returns:
        df: df with new variables.

    """
    param_air = parameters["translate_air"]
    df = data.copy()
    td = td.to_frame()
    for line in param_air:
        line_vars = param_air[line]
        delta = line_vars["delta"][0]
        promedio = line_vars["promedio"][0]
        celdas = line_vars["celdas"]
        c1 = celdas[0]
        c7 = celdas[-2]
        max_delta_c1 = td[td.tag==c1][["max_delta"]].values[0][0]
        max_delta_c7 = td[td.tag==c7][["max_delta"]].values[0][0]

        df["current_c1"] = df[[c1]]
        df["current_c7"] = df[[c7]]
        # If celdas are out of control tags, we can recover the original
        # mean value making the calculus by hand
        df["current_prom"] = df[celdas].mean(axis=1)

        df["new_c1"] = df["current_c7"] * df[promedio] / df["current_prom"] - df[delta]
        df["new_c7"] = df["new_c1"] + df[delta]

        # Is this really necessary? If we set only negative
        # delta ranges this shouldn't be needed. Still I
        # will vectorize it.
        df[c1] = np.where(
            df[delta] < 0, df["new_c1"], df["current_c1"]
        )
        df[c7] = np.where(
            df[delta] < 0, df["new_c7"], df["current_c7"]
        )

        # Check if single cell delta restriction is not broken
        # for both cells simultaneously. If not, return to their original
        # values.
        c1_cond = abs(df["new_c1"] - df["current_c1"]) <= max_delta_c1
        c7_cond = abs(df["new_c7"] - df["current_c7"]) <= max_delta_c7
        
        df[c1] = np.where(
            ~c1_cond | ~c7_cond, df["current_c1"], df["new_c1"]
        )
        df[c7] = np.where(
            ~c1_cond | ~c7_cond, df["current_c7"], df["new_c7"]
        )

    # We need to update the diff bancos variables due
    # we update the air flows. This is only needed for A2
    line_bancos = create_diff_bancos(parameters, df, f"fa2")
    line_bancos.drop(columns = parameters["timestamp_col_name"] , inplace = True)
    for col in [x for x in line_bancos.columns if x in df.columns]:
        df[col] = line_bancos[col]
    return df
