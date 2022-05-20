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
"""
Data Cleaning Pipeline
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_html_report


def create_pipeline(**kwargs):

    return pipeline(
        pipe=Pipeline(
            [
                node(
                    create_html_report,
                    [
                        "params:reporting_html",
                        "params:KEDRO_ENV",
                        *kwargs.get("wait_on", []),
                    ],
                    None,
                    name="create_html_report",
                ),
            ]
        ),
        namespace="reporting_html",
    )
