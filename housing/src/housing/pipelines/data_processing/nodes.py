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

# Definir la clase del transformador personalizado para crear columnas nuevas
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


def preprocess_housing(housing: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Preprocesses the data for houses.

    Args:
        housing: Raw data.
    Returns:
    """
    train_copy = housing.copy()
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
            ('imputer',SimpleImputer(strategy="median")), # completa los valores nulos
            ('attribs_adder',CombinedAttributesAdder()), # agrega columnas nuevas
                ("std_scaler", StandardScaler()) # escala los valores por estandarizacion
                    ])
    housing_num_tr = num_pipeline.fit_transform(housing_num)

    # Como Componer las salidas de todas las transformaciones


    # Definir el pipeline completo utilizando ColumnTransformer
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, list(housing_num)),  # Pipeline numérico definido anteriormente
        ("cat", OneHotEncoder(), ["ocean_proximity"])  # Transformador OneHotEncoder para la columna categórica
    ])

    # Aplicar el pipeline completo a los datos de entrenamiento 'train_data'
    housing_prepared = full_pipeline.fit_transform(train_copy)
    return housing_prepared
