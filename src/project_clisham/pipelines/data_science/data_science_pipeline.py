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
# NON-INFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
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

from .model_input import model_input_pipeline as mip
from .train_model import train_model_pipeline as tmp
from kedro.pipeline import pipeline


def create_pipeline():

    # Area A2
    tags_m2 = ["ma2", "a2"]
    tags_f2 = ["fa2", "a2"]
    s16_model = create_model_pipeline("s16", tags_m2)
    s17_model = create_model_pipeline("s17", tags_m2)
    fa2l1_model = create_model_pipeline("fa2l1", tags_f2)
    fa2l2_model = create_model_pipeline("fa2l2", tags_f2)
    fa2l3_model = create_model_pipeline("fa2l3", tags_f2)
    # fa2lg_model = create_model_pipeline("fa2lg", tags_f2)

    # Area A1
    tags_m1 = ["ma1", "a1"]
    tags_f1 = ["fa1", "a1"]
    s13_model = create_model_pipeline("s13", tags_m1)
    s14_model = create_model_pipeline("s14", tags_m1)
    s15_model = create_model_pipeline("s15", tags_m1)
    fa1l1_model = create_model_pipeline("fa1l1", tags_f1)
    fa1l2_model = create_model_pipeline("fa1l2", tags_f1)

    # Area A0
    tags_m0 = ["ma0", "a0"]
    tags_f0 = ["fa0", "a0"]
    # s1_model = create_model_pipeline("s1", tags_m0)
    s2_model = create_model_pipeline("s2", tags_m0)
    # s3_model = create_model_pipeline("s3", tags_m0)
    s4_model = create_model_pipeline("s4", tags_m0)
    s5_model = create_model_pipeline("s5", tags_m0)
    # s6_model = create_model_pipeline("s6", tags_m0)
    s7_model = create_model_pipeline("s7", tags_m0)
    s8_model = create_model_pipeline("s8", tags_m0)
    s9_model = create_model_pipeline("s9", tags_m0)
    s10_model = create_model_pipeline("s10", tags_m0)
    s11_model = create_model_pipeline("s11", tags_m0)
    fa0l1_model = create_model_pipeline("fa0l1", tags_f0)
    fa0l2_model = create_model_pipeline("fa0l2", tags_f0)

    all_pipeline = (
        s16_model
        + s17_model
        + fa2l1_model
        + fa2l2_model
        + fa2l3_model
        # + fa2lg_model
        + s13_model
        + s14_model
        + s15_model
        + fa1l1_model
        + fa1l2_model
        # + s1_model
        + s2_model
        # + s3_model
        + s4_model
        + s5_model
        # + s6_model
        + s7_model
        + s8_model
        + s9_model
        + s10_model
        + s11_model
        + fa0l1_model
        + fa0l2_model
    )

    return all_pipeline


def create_model_pipeline(model_name: str, tags: list):

    model_input_pipeline = mip.create_pipeline()
    train_model_pipeline = tmp.create_pipeline()

    model = pipeline(
        model_input_pipeline,
        inputs={"data_all_features": "data_all_features", "td": "td"},
        parameters={"params:model_input": f"params:{model_name}.model_input"},
        namespace=model_name,
    ) + pipeline(
        train_model_pipeline,
        inputs={"td": "td"},
        parameters={"params:train_model": f"params:{model_name}.train_model"},
        namespace=model_name,
    )
    model = model.tag([model_name] + tags)
    return model
