from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_housing, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["housing", "params:model_options"],
                outputs=["X_train",  # , 
                         "X_test",  # X_test
                         "housing_labels",  # Y_train
                         "y_test"  # y_test
                         ],
                name="split_data_node",
            ),      
            node(
                func=preprocess_housing,
                inputs=["X_train"],
                outputs="preprocessed_housing",
                name="preprocess_housing_node",
            ),
            node(
                func=preprocess_housing,
                inputs=["X_test"],
                outputs="X_test_preprocessed",
                name="preprocess_X_test_node",
            ),


        ]
    )
