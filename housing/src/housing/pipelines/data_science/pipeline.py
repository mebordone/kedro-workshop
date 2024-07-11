from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs=["preprocessed_housing",
                        "housing_labels"],
                outputs=["modelo_random_forest",
                         "final_model"],
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["final_model", "X_test_preprocessed", "y_test"],
                name="evaluate_model_node",
                outputs="metrics",
            ),
        ]
    )
