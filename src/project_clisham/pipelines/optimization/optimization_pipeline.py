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
from kedro.pipeline import pipeline, Pipeline, node
from .transform_models import transform_models_pipeline as tmp
from .recommendation import recommendation_pipeline as rp
from .sensitivity import sensitivity_pipeline as sp

from .transform_models.transform_models_nodes import create_models_dictionary


def create_pipeline():

    tph_models_pipeline = tmp.create_tph_pipeline()
    cuf_a2_models_pipeline = tmp.create_cuf_pipeline_a2()
    cuf_a1_models_pipeline = tmp.create_cuf_pipeline_a1()
    cuf_a0_models_pipeline = tmp.create_cuf_pipeline_a0()
    recommendation_pipeline = rp.create_pipeline()
    sensitivity_pipeline = sp.create_pipeline()

    # Area A2 - molienda
    ma2_tph_recommendations = (
        Pipeline(
            [
                node(
                    func=create_models_dictionary,
                    inputs=dict(
                        params="parameters",
                        td="td",
                        s16="s16.train_model",
                        s17="s17.train_model",
                    ),
                    outputs="ma2.tph_dict",
                )
            ]
        )
        + pipeline(
            tph_models_pipeline,
            inputs={},
            parameters={},
            namespace="ma2",
        )
        + pipeline(
            recommendation_pipeline,
            inputs={
                "data_all_features": "data_all_features",
                "td": "td",
                "models_dict": "ma2.tph_dict",  # actualizar con el output de create models_dictionary
                "ramp_up_sags16_17_velocities": "ramp_up_sags16_17_velocities",
                "revestimiento": "revestimiento",
                "ph_standard_data": "ph_standard_data",
            },
            parameters={
                "params:recommend": "params:ma2.recommend",
                "params:uplift_report": "params:ma2.recommend",
            },
            namespace="ma2",
        )
        + pipeline(
            sensitivity_pipeline,
            inputs={"td": "td"},
            parameters={"params:recommend_sensitivity": "params:ma2.recommend"},
            namespace="ma2",
        )
    )
    ma2_tph_recommendations = ma2_tph_recommendations.tag(["ma2", "a2", "ma2_optim"])

    # Area A2 - CuF por linea
    ca2xl_tph_recommendations = (
        Pipeline(
            [
                node(
                    func=create_models_dictionary,
                    inputs=dict(
                        params="parameters",
                        td="td",
                        s16="s16.train_model",
                        s17="s17.train_model",
                        fa2l1="fa2l1.train_model",
                        fa2l2="fa2l2.train_model",
                        fa2l3="fa2l3.train_model",
                    ),
                    outputs="ca2xl.cuf_dict",
                )
            ]
        )
        + pipeline(
            cuf_a2_models_pipeline,
            inputs={"profit_matrices_clean": "profit_matrices_clean"},
            parameters={"params": "parameters", "td": "td"},
            namespace="ca2xl",
        )
        + pipeline(
            recommendation_pipeline,
            inputs={
                "data_all_features": "data_all_features",
                "td": "td",
                "models_dict": "ca2xl.cuf_dict",  # actualizar con el output de create models_dictionary
                "ramp_up_sags16_17_velocities": "ramp_up_sags16_17_velocities",
                "revestimiento": "revestimiento",
                "ph_standard_data": "ph_standard_data",
            },
            parameters={
                "params:recommend": "params:ca2xl.recommend",
                "params:uplift_report": "params:ca2xl.recommend",
            },
            namespace="ca2xl",
        )
        + pipeline(
            sensitivity_pipeline,
            inputs={"td": "td"},
            parameters={"params:recommend_sensitivity": "params:ca2xl.recommend"},
            namespace="ca2xl",
        )
    )
    ca2xl_tph_recommendations = ca2xl_tph_recommendations.tag(
        ["ca2xl", "a2", "cuf_a2_optim"]
    )

    # Area A1 - CuF por linea
    ca1xl_tph_recommendations = (
        Pipeline(
            [
                node(
                    func=create_models_dictionary,
                    inputs=dict(
                        params="parameters",
                        td="td",
                        s13="s13.train_model",
                        s14="s14.train_model",
                        s15="s15.train_model",
                        fa1l1="fa1l1.train_model",
                        fa1l2="fa1l2.train_model",
                    ),
                    outputs="ca1xl.cuf_dict",
                )
            ]
        )
        + pipeline(
            cuf_a1_models_pipeline,
            inputs={"profit_matrices_clean": "profit_matrices_clean"},
            parameters={"params": "parameters", "td": "td"},
            namespace="ca1xl",
        )
        + pipeline(
            recommendation_pipeline,
            inputs={
                "data_all_features": "data_all_features",
                "td": "td",
                "models_dict": "ca1xl.cuf_dict",  # actualizar con el output de create models_dictionary
                "ramp_up_sags16_17_velocities": "ramp_up_sags16_17_velocities",
                "revestimiento": "revestimiento",
                "ph_standard_data": "ph_standard_data",
            },
            parameters={
                "params:recommend": "params:ca1xl.recommend",
                "params:uplift_report": "params:ca1xl.recommend",
            },
            namespace="ca1xl",
        )
        + pipeline(
            sensitivity_pipeline,
            inputs={"td": "td"},
            parameters={"params:recommend_sensitivity": "params:ca1xl.recommend"},
            namespace="ca1xl",
        )
    )
    ca1xl_tph_recommendations = ca1xl_tph_recommendations.tag(
        ["ca1xl", "a1", "cuf_a1_optim"]
    )

    # Area A0 - CuF por linea
    ca0xl_tph_recommendations = (
        Pipeline(
            [
                node(
                    func=create_models_dictionary,
                    inputs=dict(
                        params="parameters",
                        td="td",
                        # s1="s1.train_model",
                        s2="s2.train_model",
                        # s3="s3.train_model",
                        s4="s4.train_model",
                        s5="s5.train_model",
                        # s6="s6.train_model",
                        s7="s7.train_model",
                        s8="s8.train_model",
                        s9="s9.train_model",
                        s10="s10.train_model",
                        s11="s11.train_model",
                        fa0l1="fa0l1.train_model",
                        fa0l2="fa0l2.train_model",
                    ),
                    outputs="ca0xl.cuf_dict",
                )
            ]
        )
        + pipeline(
            cuf_a0_models_pipeline,
            inputs={"profit_matrices_clean": "profit_matrices_clean"},
            parameters={"params": "parameters", "td": "td"},
            namespace="ca0xl",
        )
        + pipeline(
            recommendation_pipeline,
            inputs={
                "data_all_features": "data_all_features",
                "td": "td",
                "models_dict": "ca0xl.cuf_dict",  # actualizar con el output de create models_dictionary
                "ramp_up_sags16_17_velocities": "ramp_up_sags16_17_velocities",
                "revestimiento": "revestimiento",
                "ph_standard_data": "ph_standard_data",
            },
            parameters={
                "params:recommend": "params:ca0xl.recommend",
                "params:uplift_report": "params:ca0xl.recommend",
            },
            namespace="ca0xl",
        )
        + pipeline(
            sensitivity_pipeline,
            inputs={"td": "td"},
            parameters={"params:recommend_sensitivity": "params:ca0xl.recommend"},
            namespace="ca0xl",
        )
    )
    ca0xl_tph_recommendations = ca0xl_tph_recommendations.tag(
        ["ca0xl", "a0", "cuf_a0_optim"]
    )

    all_pipeline = (
        ma2_tph_recommendations
        + ca2xl_tph_recommendations
        + ca1xl_tph_recommendations
        + ca0xl_tph_recommendations
    )

    return all_pipeline
