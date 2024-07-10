# kedro-workshop
Taller de kedro y jupyter

# Problema - California Housing
Este es el conjunto de datos utilizado en el segundo capítulo del reciente libro de Aurélien Géron 'Hands-On Machine learning with Scikit-Learn and TensorFlow'. Sirve como una excelente introducción a la implementación de algoritmos de aprendizaje automático, ya que requiere una limpieza de datos rudimentaria, tiene una lista de variables fácilmente comprensible y se encuentra en un tamaño óptimo entre ser demasiado de juguete y demasiado engorroso.

Los datos contienen información del censo de California de 1990. Así que, aunque puede que no le ayude a predecir los precios actuales de la vivienda como el conjunto de datos Zestimate de Zillow, proporciona un conjunto de datos introductorio accesible para enseñar a la gente los fundamentos del aprendizaje automático.

Contenido

Los datos se refieren a las casas que se encuentran en un determinado distrito de California y algunas estadísticas resumidas sobre ellas basadas en los datos del censo de 1990. Hay que tener en cuenta que los datos no están depurados, por lo que se requieren algunos pasos de preprocesamiento. Las columnas son las siguientes, sus nombres se explican por sí mismos:

- longitud
- latitud
- edad_media_vivienda
- total_habitaciones
- total_dormitorios
- población
- hogares
- ingresos_medianos
- proximidad_al_océano
- valor_medio_de_la_vivienda (valor que quiero predecir)

Descripción:
1. longitude: A measure of how far west a house is; a higher value is farther west
2. latitude: A measure of how far north a house is; a higher value is farther north
3. housingMedianAge: Median age of a house within a block; a lower number is a newer building
4. totalRooms: Total number of rooms within a block
5. totalBedrooms: Total number of bedrooms within a block
6. population: Total number of people residing within a block
7. households: Total number of households, a group of people residing within a home unit, for a block
8. medianIncome: Median income for households within a block of houses (measured in tens of thousands of US Dollars)
9. medianHouseValue: Median house value for households within a block (measured in US Dollars)
10. oceanProximity: Location of the house w.r.t ocean/sea

# Dataset Original:
https://github.com/ageron/handson-ml/tree/master/datasets/housing
