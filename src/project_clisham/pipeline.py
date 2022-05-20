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

"""Construction of the master pipeline.
"""

from typing import Dict

from kedro.pipeline import Pipeline

from project_clisham.pipelines import data_engineering as de
from project_clisham.pipelines import data_science as ds
from project_clisham.pipelines import data_quality as dq
from project_clisham.pipelines import optimization as opt
from project_clisham.pipelines.data_engineering import raw, intermediate, primary
from project_clisham.pipelines import model_reporting as mr

# from project_clisham.pipelines import data_curation as dc


def create_pipelines() -> Dict[str, Pipeline]:
    """Create the project's pipeline.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """
    raw_pipeline = raw.create_pipeline()
    intermediate_pipeline = intermediate.create_pipeline()
    data_quality_pipe = dq.create_pipeline()

    data_engineering_pipeline = de.create_pipeline()
    data_science_pipeline = ds.create_pipeline()
    optimization_pipeline = opt.create_pipeline()
    data_quality_pipeline = raw_pipeline + intermediate_pipeline + data_quality_pipe

    combined_pipeline = (
        data_engineering_pipeline + data_science_pipeline + optimization_pipeline
    )

    model_reporting_pipeline = mr.create_pipeline()

    return {
        "de": data_engineering_pipeline,
        "ds": data_science_pipeline,
        "ds_cloud": data_engineering_pipeline + data_science_pipeline - data_science_pipeline.only_nodes_with_tags("report"),
        "dq": data_quality_pipeline,
        "opt": optimization_pipeline,
        "__default__": combined_pipeline,
        # Pipelines no reports
        "cuf_a2_recommend": data_engineering_pipeline
        + optimization_pipeline.only_nodes_with_tags("cuf_a2_optim")
        - optimization_pipeline.only_nodes_with_tags("report"),
        "cuf_a1_recommend": data_engineering_pipeline
        + optimization_pipeline.only_nodes_with_tags("cuf_a1_optim")
        - optimization_pipeline.only_nodes_with_tags("report"),
        "cuf_a0_recommend": data_engineering_pipeline
        + optimization_pipeline.only_nodes_with_tags("cuf_a0_optim")
        - optimization_pipeline.only_nodes_with_tags("report"),
        "cuf_recommend": data_engineering_pipeline
        + optimization_pipeline.only_nodes_with_tags("cuf_a0_optim")
        + optimization_pipeline.only_nodes_with_tags("cuf_a1_optim")
        + optimization_pipeline.only_nodes_with_tags("cuf_a2_optim")
        - optimization_pipeline.only_nodes_with_tags("report"),
        "ma2_train": data_engineering_pipeline
        + data_science_pipeline.only_nodes_with_tags("ma2")
        - data_science_pipeline.only_nodes_with_tags("report"),
        "ma1_train": data_engineering_pipeline
        + data_science_pipeline.only_nodes_with_tags("ma1")
        - data_science_pipeline.only_nodes_with_tags("report"),
        "ma0_train": data_engineering_pipeline
        + data_science_pipeline.only_nodes_with_tags("ma0")
        - data_science_pipeline.only_nodes_with_tags("report"),
        "fa2_train": data_engineering_pipeline
        + data_science_pipeline.only_nodes_with_tags("fa2")
        - data_science_pipeline.only_nodes_with_tags("report"),
        "fa1_train": data_engineering_pipeline
        + data_science_pipeline.only_nodes_with_tags("fa1")
        - data_science_pipeline.only_nodes_with_tags("report"),
        "fa0_train": data_engineering_pipeline
        + data_science_pipeline.only_nodes_with_tags("fa0")
        - data_science_pipeline.only_nodes_with_tags("report"),
        # Pipelines reports no DE
        "cuf_a2_rec": optimization_pipeline.only_nodes_with_tags("cuf_a2_optim"),
        "cuf_a1_rec": optimization_pipeline.only_nodes_with_tags("cuf_a1_optim"),
        "cuf_a0_rec": optimization_pipeline.only_nodes_with_tags("cuf_a0_optim"),
        "cuf_rec": optimization_pipeline.only_nodes_with_tags("cuf_a0_optim")
        + optimization_pipeline.only_nodes_with_tags("cuf_a1_optim")
        + optimization_pipeline.only_nodes_with_tags("cuf_a2_optim"),
        "ma2_tr": data_science_pipeline.only_nodes_with_tags("ma2"),
        "ma1_tr": data_science_pipeline.only_nodes_with_tags("ma1"),
        "ma0_tr": data_science_pipeline.only_nodes_with_tags("ma0"),
        "fa2_tr": data_science_pipeline.only_nodes_with_tags("fa2"),
        "fa1_tr": data_science_pipeline.only_nodes_with_tags("fa1"),
        "fa0_tr": data_science_pipeline.only_nodes_with_tags("fa0"),
        "ds_report": data_science_pipeline.only_nodes_with_tags("report"),
        # Model's health report
        "model_reporting": model_reporting_pipeline
    }
