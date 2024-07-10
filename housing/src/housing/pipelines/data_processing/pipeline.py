from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_housing


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_housing,
                inputs="housing",
                outputs="preprocessed_housing",
                name="preprocess_housing_node",
            ),
        ]
    )
