import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import sklearn
from sklearn.model_selection import StratifiedShuffleSplit



def train_model(housing_prepared: pd.DataFrame, 
                housing_labels: pd.Series) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    forest_reg = RandomForestRegressor()
    # Inputs necesarios para la funcion fit: hounsing_prepared, housing_labels
    # Donde está definido housing_labels?
    # Tienen los mismos nombres en el código?

    forest_reg.fit(housing_prepared, housing_labels)
    predictions = forest_reg.predict(housing_prepared)

    # Medir el error utilizando el RMSE con los datos de entrenamiento
    forest_rmse = mean_squared_error(
        housing_labels, predictions, squared=False)
    print("RMSE del modelo de Random Forest (datos de entrenamiento):", 
          forest_rmse)
    return forest_reg


def evaluate_model(
    forest_reg: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    # Obtener X_test y y_test del conjunto de prueba

    # Aplicar la transformación completa a X_test
    X_test_prepared = X_test

    final_model = forest_reg

    # Realizar predicciones sobre X_test_prepared utilizando el modelo final
    final_predictions = final_model.predict(X_test_prepared)

    # Calcular el RMSE entre las etiquetas reales (y_test) 
    # y las predicciones finales
    final_rmse = mean_squared_error(y_test, final_predictions, squared=False)

    print("RMSE final en el conjunto de prueba:", final_rmse)
    return {"RMSE": final_rmse}
