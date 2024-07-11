import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from typing import Dict, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# Importar la clase StandardScaler desde sklearn.preprocessing (escalado por estandarizacion de los datos)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import pandas as pd
# Combinar atributos - armando pipelines y transformadores manuales
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Calcula 'rooms_per_household'
        rooms_per_household = X[:, 3] / X[:, 6]
        # Calcula 'population_per_household'
        population_per_household = X[:, 5] / X[:, 6]

        if self.add_bedrooms_per_room:
            # Calcula 'bedrooms_per_room'
            bedrooms_per_room = X[:, 4] / X[:, 3]
            X_transformed = np.c_[X, rooms_per_household,
                                  population_per_household, bedrooms_per_room]
        else:
            X_transformed = np.c_[X, rooms_per_household,
                                  population_per_household]

        return X_transformed


def preprocess_new_data(new_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for houses.

    Args:
        housing: Raw data.
    Returns:
    """

    # Convertir X_new a un DataFrame con las columnas correspondientes
    columns = ['longitude', 'latitude', 'housing_median_age',
               'total_rooms', 'total_bedrooms', 'population',
               'households', 'median_income', 'ocean_proximity',
               'median_house_value']
    
    print(len(columns))
    X_new_df = pd.DataFrame([new_data], columns=columns)
    X_new_df.to_csv('new_data.csv')
    train_copy = X_new_df
    # Crea un objeto SimpleImputer con estrategia de imputación mediana
    imputer = SimpleImputer(strategy="median")

    # Variable que representa el número promedio de habitaciones por hogar
    train_copy["rooms_per_household"] = train_copy["total_rooms"] / \
        train_copy["households"]

    # Variable que representa el porcentaje de dormitorios en relación al número total de habitaciones
    train_copy["bedrooms_per_room"] = train_copy["total_bedrooms"] / \
        train_copy["total_rooms"]

    # Variable que representa la relación entre la población y el valor medio de la vivienda por hogar
    train_copy["population_per_household"] = train_copy["population"] / \
        train_copy["median_house_value"]

    train_data = train_copy

    # Elimina la columna 'ocean_proximity' del conjunto de datos de entrenamiento 'train_data' para trabajar solo con datos numéricos
    housing_num = train_data.drop("ocean_proximity", axis=1)

    # Ajusta el imputer utilizando los datos de entrenamiento 'housing_num' (calcula la mediana en este caso por cada columna)
    imputer.fit(housing_num)

    # Transforma los datos numéricos 'housing_num' imputando los valores nulos con la mediana
    out = imputer.transform(housing_num)

    # Crea un nuevo DataFrame 'housing_tr' con los datos transformados y las columnas originales de 'housing_num'
    housing_tr = pd.DataFrame(out, columns=housing_num.columns)

    # Ejemplo de uso del transformador personalizado
    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    housing_extra_attribs = attr_adder.transform(train_data.values)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(housing_num)

    # Encadenar transformadores (hacer un pipeline)
    from sklearn.pipeline import Pipeline
    num_pipeline = Pipeline([
        # completa los valores nulos
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),  # agrega columnas nuevas
        # escala los valores por estandarizacion
        ("std_scaler", StandardScaler())
    ])
    housing_num_tr = num_pipeline.fit_transform(housing_num)

    # Como Componer las salidas de todas las transformaciones

    # Definir el pipeline completo utilizando ColumnTransformer
    full_pipeline = ColumnTransformer([
        # Pipeline numérico definido anteriormente
        ("num", num_pipeline, list(housing_num)),
        # Transformador OneHotEncoder para la columna categórica
        ("cat", OneHotEncoder(), ["ocean_proximity"])
    ])

    # Aplicar el pipeline completo a los datos de entrenamiento 'train_data'
    housing_prepared = full_pipeline.fit_transform(train_copy)
    return housing_prepared


def predict(final_model, X_new_prepared) -> pd.DataFrame:

    # Hacer predicciones con el modelo cargado
    predictions = final_model.predict(X_new_prepared)

    print("Predicción:", predictions[0])
    return predictions
