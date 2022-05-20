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

from .train_model_nodes import (
    add_transformers,
    create_predictions,
    load_regressor,
    # retrain_tree_model,
    train_tree_model,
    generate_performance_report,
)


def create_pipeline():
    return Pipeline(
        [
            node(
                load_regressor,
                dict(params="params:train_model"),
                "regressor",
                name="load_regressor",
            ),
            node(
                add_transformers,
                dict(params="params:train_model", td="td", regressor="regressor"),
                "regressor_pipeline",
                name="add_transformers",
            ),
            node(
                train_tree_model,
                dict(
                    params="params:train_model",
                    td="td",
                    data="train_set",
                    model="regressor_pipeline",
                ),
                dict(
                    model="train_model",
                    cv_results="train_set_cv_results",
                    feature_importance="train_set_feature_importance",
                ),
                name="train_tree_model",
            ),
            node(
                create_predictions,
                dict(
                    params="params:train_model",
                    td="td",
                    data="train_set",
                    model="train_model",
                ),
                dict(
                    predictions="train_set_predictions",
                    metrics="train_set_metrics",
                ),
                name="create_predictions_train",
            ),
            node(
                create_predictions,
                dict(
                    params="params:train_model",
                    td="td",
                    data="test_set",
                    model="train_model",
                ),
                dict(
                    predictions="test_set_predictions",
                    metrics="test_set_metrics",
                ),
                name="create_predictions",
            ),
            node(
                generate_performance_report,
                dict(
                    params="params:train_model",
                    kedro_env="params:KEDRO_ENV",
                    test_predictions="test_set_predictions",
                ),
                None,
                name="generate_performance_report",
                tags=["report"],
            ),
            # node(
            #     retrain_tree_model,
            #     dict(td="td", model="train_model", data="input"),
            #     "model",
            #     name="retrain_tree_model",
            # ),
        ],
        tags="train_model",
    )
