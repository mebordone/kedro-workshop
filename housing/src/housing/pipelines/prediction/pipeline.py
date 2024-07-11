from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_new_data, predict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_new_data,
            inputs=["new_data"],
            outputs="preprocessed_new_data",
            name="preprocess_new_data_node",
        ),
        node(
            func=predict,
            inputs=["final_prod_model", "preprocessed_new_data"],
            outputs="new_predictions",
            name="new_predict_node",
        ),
    ])
