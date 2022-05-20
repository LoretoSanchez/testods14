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


from kedro.pipeline import Pipeline, node

from .features_nodes import (
    add_across_features_by_hour,
    add_across_features_by_shift,
    create_tag_dict,
    group_by_shift,
    create_target_counts,
    create_target_lags,
    add_on_off_features_by_shift,
    apply_on_off_to_df,
    map_columns_names,
    get_last_value_by_shift
)

from .feature_nodes_A1 import add_a1_features_by_hour, add_a1_features_by_shift
from .feature_nodes_A2 import add_a2_features_by_hour, add_a2_features_by_shift
from .feature_nodes_A0 import add_a0_features_by_hour, add_a0_features_by_shift


def create_pipeline():
    return Pipeline(
        [
            node(
                func=create_tag_dict,
                inputs="tag_dict_master",
                outputs="td",
                tags=["dict"],
            ),
            node(
                func=map_columns_names,
                inputs=["data_primary","parameters"],
                outputs="data_primary_mapped",
            ),
            node(
                func=add_a0_features_by_hour,
                inputs=["parameters", "data_primary_mapped"],
                outputs="data_a0_features",
            ),
            node(
                func=add_a1_features_by_hour,
                inputs=["parameters", "data_a0_features"],
                outputs="data_a1_features",
            ),
            node(
                func=add_a2_features_by_hour,
                inputs=["parameters", "data_a1_features"],
                outputs="data_a2_features_by_hour",
            ),
            node(
                func=add_across_features_by_hour,
                inputs=["parameters", "data_a2_features_by_hour", "data_traza_clean_intermediate", "td"],
                outputs="data_general_features",
            ),
            node(
                func=apply_on_off_to_df,
                inputs=["data_general_features", "td", "parameters"],
                outputs="data_general_features_on_off",
            ),
            node(
                func=group_by_shift,
                inputs=["parameters", "data_general_features_on_off"],
                outputs="data_aggregated",
            ),
            node(
                func=add_a0_features_by_shift,
                inputs=["parameters", "data_aggregated"],
                outputs="data_a0_features_by_shift",
            ),
            node(
                func=add_a1_features_by_shift,
                inputs=["parameters", "data_a0_features_by_shift"],
                outputs="data_a1_features_by_shift",
            ),
            node(
                func=add_a2_features_by_shift,
                inputs=["parameters", "data_a1_features_by_shift"],
                outputs="data_a2_features_by_shift",
            ),
            node(
                func=get_last_value_by_shift,
                inputs=["parameters", "data_primary"],
                outputs="data_last_values_features",
            ),
            node(
                func=add_across_features_by_shift,
                inputs=["parameters", "data_a2_features_by_shift","data_last_values_features"],
                outputs="data_features_by_shift",
            ),
            node(
                func=create_target_counts,  # TODO: ML move to features_by_shift
                inputs=[
                    "parameters",
                    "td",
                    "data_general_features",
                    "data_features_by_shift",
                ],
                outputs="data_aggregated_counts",
            ),
            node(
                func=create_target_lags,  # TODO: ML move to features_by_shift
                inputs=["td", "data_aggregated_counts", "parameters"],
                outputs="data_aggregated_lags",
            ),
            node(
                func=add_on_off_features_by_shift,
                inputs=["parameters", "data_aggregated_lags"],
                outputs="data_all_features",
            ),
        ],
        tags=["features", "de"],
    )
