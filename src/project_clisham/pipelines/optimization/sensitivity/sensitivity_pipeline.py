# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organisation
# and QuantumBlack, and any unauthorised use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organisation with the prior written
# permission of QuantumBlack.

from kedro.pipeline import Pipeline, node, pipeline

from .sensitivity_nodes import (
    create_sensitivity_plot_data,
    generate_model_sensitivity_plots,
)


def create_pipeline():  # pylint:disable=unused-argument
    return pipeline(
        pipe=Pipeline(
            [
                node(
                    create_sensitivity_plot_data,
                    dict(
                        params="params:recommend_sensitivity",
                        td="td",
                        model="optimization_function",
                        opt_df="data_input_optim_uuid",
                        recommendations="recommendations",
                    ),
                    dict(sensitivity_plot_df="sensitivity_plot_df"),
                    name="generate_sensitivity_data",
                ),
                # node(
                #     generate_model_sensitivity_plots,
                #     dict(
                #         params=f"params:recommend_sensitivity",
                #         td="td",
                #         data="data_input_optim_uuid",
                #         model="optimization_function",
                #     ),
                #     "model_sensitivity",
                #     name="generate_sensitivity_plots",
                # ),
            ]
        ),
    )
