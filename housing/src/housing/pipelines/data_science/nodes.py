import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit


def split_data(housing: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    # X = data[parameters["features"]]
    # y = data["price"]
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    # )
    
    income_cat = pd.cut(housing["median_income"], bins=[0, 1.5, 3, 4.5, 6, 16], labels=[1, 2, 3, 4, 5])
    income_cat.head()



    split_object = StratifiedShuffleSplit(n_splits=1, 
                                          test_size=parameters["test_size"], 
                                          random_state=parameters["random_state"])
    gen_obj = split_object.split(housing, income_cat)
    train_ind, test_ind = next(gen_obj)
    strat_train_set = housing.loc[train_ind]
    strat_test_set = housing.loc[test_ind]
    
    
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(
    regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    me = max_error(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
    return {"r2_score": score, "mae": mae, "max_error": me}
