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
"""Interactive Sensitivity Chart for Model Explainability and Understanding"""
from copy import deepcopy
from pathlib import Path

from typing import Dict, Iterable, List, Union
import pandas as pd
import numpy as np
import altair as alt

from kedro.framework.context import KedroContext
from kedro.config import TemplatedConfigLoader
from project_clisham.pipelines.optimization.recommendation.recommendation_nodes import (
    _get_features,
    _get_target,
)
from project_clisham.optimus_core.tag_management import TagDict

import streamlit as st
import sys
import sensitivity_nodes as nodes


@st.cache(allow_output_mutation=True)
def load_data(model: str):
    """
    Loads and caches data and catalog needed for later running the app.
    Users updating the app for their client engagement will need to update
    the parameters file which maps features, model, recs and sensitivity
    data to their corresponding catalog entries.
    Returns: tuple of features, sensitivity data, recommendations,
    model object, tag dict and catalog.

    """

    context = get_kedro_context()
    catalog = context.catalog

    td = catalog.load("td")
    params_general = catalog.load("parameters")[f"{model}.recommend"]
    params = params_general["recommend_sensitivity"]["sensitivity_app_data_mapping"]
    features = catalog.load(params["features"])
    model = catalog.load(params["model"])
    recs = catalog.load(params["recs"])

    sensitivity_data = catalog.load(params["sensitivity_data"])
    sensitivity_data[params["timestamp_col"]] = pd.to_datetime(
        sensitivity_data[params["timestamp_col"]]
    )

    return (
        features,
        sensitivity_data,
        recs,
        model,
        td,
        catalog,
        params_general,
        params["timestamp_col"],
    )


def shift_recs_input(control_numeric_input, noncontrol_sliders, df):
    return_df = df.copy()
    controls_dict_raw = deepcopy(return_df.loc[0, "controls"])
    for control in controls_dict_raw.keys():
        controls_dict_raw[control]["suggested"] = control_numeric_input[control]
    return_df.loc[0, "controls"] = [controls_dict_raw]
    return_df.loc[0, "state"] = [noncontrol_sliders]
    return return_df


def find_starting_values(rec_df: pd.DataFrame) -> Dict:
    """
    Utility for finding slider starting points for a given shift recommendation
    Args:
        rec_df: Dataframe of recommendations.

    Returns:

    """
    # Find state values
    state_dict = rec_df.reset_index().loc[0, "state"]
    # Find suggested rec values
    raw_rec_dict = rec_df.reset_index().loc[0, "controls"]
    rec_dict = {
        control_name: raw_rec_dict[control_name]["suggested"]
        for control_name in raw_rec_dict.keys()
    }
    return {**state_dict, **rec_dict}


def create_altair_sensitivity_chart(
    plot_data: pd.DataFrame, plot_control: str, plot_target: str
):
    """
    Given a control variable, a target variable (e.g. from a model),
    Args:
        plot_data: datafrom of sensitivity data
        plot_control: string of the user-selected control variable (to be plotted
        on x-axis).
        plot_target: User-selected target variable (to be plotted on y-axis).

    Returns: Atlair chart.

    """
    c1 = (
        alt.Chart(
            plot_data.rename(
                columns={"control_value": plot_control, "target_value": plot_target}
            )
        )
        .mark_point(clip=True)
        .encode(
            x=alt.X(plot_control, scale=alt.Scale(zero=False)),
            y=plot_target,
            color=alt.value("red"),
            tooltip=[plot_control, plot_target],
        )
        .interactive()
    )
    c2 = (
        alt.Chart(
            plot_data.rename(
                columns={"control_value": plot_control, "target_value": plot_target}
            )
        )
        .mark_line(clip=True)
        .encode(
            x=alt.X(plot_control, scale=alt.Scale(zero=False)),
            y=plot_target,
            color=alt.value("red"),
        )
    )
    return c1 + c2


def get_widget_dict(
    streamlit_widget: Union[st.sidebar.number_input, st.sidebar.slider],
    features_list: List[str],
    feature_ranges_dict: Dict[str, np.ndarray],
    feature_starting_values: Dict,
    td: TagDict,
):
    """
    Utility for capturing user-input into a dictionary of values, keyed by a user-input
    feature (corresponding to a feature in our data). This way, we can loop over the
    creation of controls that the user sees. These lists will come from the tag
    dictionary, so that if the user changes or updates the TagDict, the app here will
    change accordingly.

    Args:
        streamlit_widget: Either sidebar.slider or sidebar.number_input. The main
        callable capturing user input.
        features_list: List of strings that form the keys of our dictionary (and
        which will be manipulated by the user).
        feature_ranges_dict: A dictionary of 'feature_name': np.arrays, that are
        the values which we will want to increment our sliders by.
        feature_starting_values: What value to start to the user-output at. By
        default this will be set to values of the selected recommendation.

    Returns: A dictionary of values captured from streamlit widgets.

    """
    features_list = [
        feature for feature in features_list if feature in feature_starting_values
    ]
    return {
        feature: streamlit_widget(
            label=feature + "-" + td.name(feature),
            min_value=min(
                feature_starting_values[feature],
                float(np.nanmin(feature_ranges_dict[feature])),
            ),
            max_value=max(
                feature_starting_values[feature],
                float(np.nanmax(feature_ranges_dict[feature])),
            ),
            value=feature_starting_values[feature],
        )
        for feature in features_list
    }


class ProjectContext(KedroContext):
    """The project context for usage within the streamlit app"""

    project_name = "model_front_end"
    project_version = "0.16.2"

    def _create_config_loader(self, conf_paths: Iterable[str]) -> TemplatedConfigLoader:
        """
        Returns templated config loader as per
        https://kedro.readthedocs.io/en/latest/04_user_guide/03_configuration.html#templating-configuration
        """
        return TemplatedConfigLoader(conf_paths, globals_pattern="*global.yml")

    def _get_pipelines(self):
        pass


def get_kedro_context() -> ProjectContext:
    """Gathers the Optimus kedro context for the app."""
    project_root = Path(__file__).parent.parent.parent.parent.parent.parent
    # see https://github.com/quantumblacklabs/private-kedro/issues/229
    return ProjectContext(project_root, env="base")


def main():  # pylint: disable=too-many-locals
    """
    Main application function. On every change of a streamlit widget, the
    main_function gets executed again. In fact, it's the entire script!

    Examples:
        To open up the streamlit app, navigate to the sensitivity modular pipeline
        folder, and enter the following from within your project.
        >> streamlit run path/to/app/streamlit_app.py
    Returns:

    """
    model = sys.argv[1]
    st.title(body="Interactive Sensitivity Chart")
    st.sidebar.title(body="Set your control values")
    features, _, recs, model, td, catalog, params, ts_col = load_data(model)

    features_list, controls_list = _get_features(td, params)
    target = _get_target(td, params)
    st.subheader(
        body="Select the shift, control variable, and target variable of primary "
        "interest"
    )
    # Add drop down for control selection
    plot_control = st.selectbox(
        label="Control Variable Selection",
        options=controls_list,
        format_func=lambda x: x + "-" + td.name(x),
    )

    # Add drop down for target selection
    plot_target = st.selectbox(label="Target Variable Selection", options=[target])

    # Add drop down for shift selection
    shift_selector = st.selectbox(
        label="Shift Selection", options=recs[ts_col].unique()
    )

    selected_shift_recs = (
        recs[recs[ts_col] == shift_selector].copy().reset_index(drop=True)
    )
    # numeric_inputs for controls
    feature_ranges_dict = {
        control: nodes.determine_control_range(td, control) for control in features_list
    }
    feature_starting_values = find_starting_values(selected_shift_recs)
    st.sidebar.subheader(body="Manipulate your control variables")
    feature_starting_values = {
        key: val for key, val in feature_starting_values.items() if val != None
    }
    control_numeric_input = get_widget_dict(
        st.sidebar.number_input,
        controls_list,
        feature_ranges_dict,
        feature_starting_values,
        td,
    )
    st.sidebar.subheader(body="Manipulate your non-control variables")
    # # sliders for non-controls features
    noncontrol_sliders = get_widget_dict(
        st.sidebar.slider,
        [x for x in features_list if x not in controls_list],
        feature_ranges_dict,
        feature_starting_values,
        td,
    )

    # Also update features where needed
    sens_features = features[features[ts_col] == shift_selector].copy().reset_index()
    input_vals = {**control_numeric_input, **noncontrol_sliders}
    for variable in input_vals.keys():
        sens_features.loc[0, variable] = input_vals[variable]
    # Treat slider values as the "recommendation"s
    new_sens_data = nodes.create_sensitivity_plot_data(
        # Update n_points in params here if you would like to dynamically change
        #  resolutions
        params,
        td,
        model,
        sens_features,
        selected_shift_recs,
    )["sensitivity_plot_df"]

    # Create sensitivity plot of selected control vs target variable
    plot_data = new_sens_data[
        (new_sens_data[ts_col] == shift_selector)
        & (new_sens_data.control_tag_id == plot_control)
        & (new_sens_data.target_tag_id == plot_target)
    ]

    try:
        current_rec_val = selected_shift_recs.loc[0, "controls"][plot_control][
            "suggested"
        ]
        vert_line = (
            alt.Chart(pd.DataFrame({plot_control: [current_rec_val]}))
            .mark_rule()
            .encode(x=alt.X(plot_control))
        )

        sensitivity_chart = create_altair_sensitivity_chart(
            plot_data, plot_control, plot_target
        )

        st.subheader(body=f"Sensitivity of {plot_target} to changes in {plot_control}")
        st.altair_chart(
            altair_chart=vert_line + sensitivity_chart, use_container_width=True
        )
    except:
        st.subheader(body=f"Variable {plot_control} not optimized in this shift")

    st.subheader(body="View the Tag Dict:")
    st.dataframe(td.to_frame())


if __name__ == "__main__":
    main()
