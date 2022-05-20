from kedro.pipeline import Pipeline, node

from .model_reporting_nodes import (
    get_all_metrics
)


def create_pipeline():
    """
    This pipeline should be run right after the official re-train process, 
    in order to store the metrics the exact day of training. We should be careful
    about running this, because it will store some data in a database, so
    it cannot be use frequently.
    """
    return Pipeline(
        [
            node(
                get_all_metrics,
                dict(parameters ="parameters"),
                outputs="model_report",
                name="model_report",
            ),
        ],
    )
