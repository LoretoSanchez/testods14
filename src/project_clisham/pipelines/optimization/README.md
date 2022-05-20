# Pipeline de optimizacion

Bienvenidos al pipeline de optimización

## Descripcion general
Este paquete incluye los pipelines que componen la optimizacion a continuación se encuentra una descripción de cada sub-pipeline y algunas aclaraciones:

### recommendation pipeline
Pipeline core de la optimización en este se definen los siguientes nodos:
  - filter_timestamp_optimization: filtra de la tabla master los timestamps que se van a optimizar
  - generate_uuid: genera un unique universal identifier para cada timestamp a optimizar
  - bulk_optimize: función que realiza la optimizacion
  - generate_recommendation_csv: transforma el output de la optimizacion en csv para poder ver las recomendaciones
  - create_bulk_result_tables: trasnforma el output de la optimizacion en diferentes csv usados en el reporte html
  - create_html_report: genera el reporte de uplift en html

### sensitivity pipeline
Pipeline de apoyo para generar los plots de sensibilidad
  - create_sensitivity_plot_data: genera los inputs para la aplicacion de streamlit que permite revisar la sensibilidad
  Para ejecutar la aplicacion de streamlit ejecutar: 
  ```
  make streamlit-show
  ```
### transform_models pipeline
Pipeline de apoyo para generar objetos de modelo que encapsulan la funcion objetivo
  hay un pipeline definido para cada funcion objetivo deseada por ahora se han definido las siguientes:
  - create_throughput_optimization: genera una funcion objetivo para maximizar el tph, definido como la suma de los tphs de una linea
  - create_cuf_optimization: genera una funcion objetivo que integra el tph con la recuperación para calcular el cufino usando los modelos entrenados

## Pipeline general
El pipeline general se debe definir para cada funcion que se quiere maximizar, esta funcion se identifica por el namespace definido, el cual se usa en los parametros, catalogo y para identificar el pipeline.
Adicionalmente, hay un nodo que se define especificamente para cada funcion objetivo y es el nodo:
- create_models_dictionary: este nodo recibe un diccionario de modelos y la idea es incluir todos los modelos necesarios para construir la funcion objetivo. El key del diccionario podria ser usado en la definicion de la funcion objetivo para identificar modelos de tph vs los de recuperacion, o las particularidades de cada linea

A continuacion se detallan algunas consideraciones que hay que tener en cuenta para agregar otra funcion de optimizacion: 
- definir el nombre de la funcion a optimizar: <objective_name>
- definir un nuevo pipeline de optimizacion: en el namespace usar el <objective_name>
- parameters_recommendation: Crear las siguientes entradas (basarse en ma2) y actualizar acorde los parametros
  <objective_name>.recommend
  <objective_name>.recommend_uptlift_report
- actualizar los parametros que entran a cada sub pipeline, mapeandolos a los correspondientes con <objective_name>
- catalogo:
  incluir las entradas necesarias en el catalogo, sirve como muestra los de ma2
- features: agregar como feature a la master el calculo de la funcion objetivo a partir de los tags que son target en los modelos
- parameters_global: agregar una nueva entrada de la foma:
  <objective_name>:
      opt_target: "<objective_name>_target"
- diccionario:
  - agregar al diccionario la feature que calcula la funcion objetivo
  - en la columna target, en la fila de la feature que calcula la funcion objetivo, agregar <objective_name>_target
  - asegurarse que todas las variables marcadas como control en los modelos que componen la funcion objetivo tengan los siguientes campos: tag_type, min_opt, max_opt
- El segundo pipeline en el pipeline global define la funcion objetivo a usar, llamar al pipeline indicado para el tipo de funcion objetivo o crear un nuevo pipeline que contenga el nodo para crear la funcion objetivo que se quiere definir

## Restricciones on/off
Las restricciones de tipo on/off en donde los controles entran o salen de la optimizacion dependiendo si el equipo esta prendido, se basan en features binarias que se crean en el pipeline. A continuacion los pasos para incluir este tipo de restricciones:
- Crear una variable on/off que a partir de un tag y un valor (x) se determina que un equipo esta prendido (1) si el valor del tag es mayor (o menor) a x, en caso contrario esta apagado (0)
- Agregar la variable al diccionario como una nueva fila, el tag_type de la variables es: on_off
- Para cada tag que debe ignorarse con la variable definida, agregar ese tag a la lista de tags en la columna on_off_dependencies