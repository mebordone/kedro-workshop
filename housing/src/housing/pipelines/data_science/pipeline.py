from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs=["preprocessed_housing", "housing_labels"],
                outputs="regressor",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor", "X_test_preprocessed", "y_test"],
                name="evaluate_model_node",
                outputs="metrics",
            ),
        ]
    )
