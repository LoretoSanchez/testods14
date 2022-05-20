from kedro.pipeline import Pipeline, node

from project_clisham.pipelines.data_quality.qa_nodes import go_or_no_go

from .curation_nodes import (
    replace_outliers_by_value,
    replace_outliers_by_nan,
    impute_missing_ffill,
)


def create_pipeline():
    pipe = Pipeline(
        [
            node(
                go_or_no_go,
                inputs=["data_qa_table", "tag_dict_master", "params:curation"],
                outputs="go_or_nogo",
                tags=["go_or_nogo"],
            ),
            node(
                replace_outliers_by_value,
                inputs=["data_clean_intermediate", "parameters", "go_or_nogo"],
                outputs=["data_corrected_by_val", "data_curation_stats"],
            ),
            node(
                replace_outliers_by_nan,
                inputs=[
                    "data_corrected_by_val",
                    "parameters",
                    "tag_dict_master",
                    "go_or_nogo",
                ],
                outputs=["data_corrected_nan", "data_curation_stats_nan"],
            ),
            node(
                impute_missing_ffill,
                inputs=[
                    "data_corrected_nan",
                    "parameters",
                ],
                outputs="data_corrected",
            ),
        ]
    )
    return pipe
