# LABORATORIO_PROCESAMIENTO_3

# OBJETIVOS DE LA PRACTICA

El objetivo de esta práctica fue analizar señales de voz mediante herramientas de procesamiento digital, utilizando el análisis espectral para identificar características como la frecuencia fundamental, la intensidad y la estabilidad vocal, y así comparar las diferencias existentes entre voces masculinas y femeninas.

# PARTE A

**1. Obtener las señales de voces**
En esta etapa se grabaron seis señales de voz entre hombre y mujeres pronunciando la misma frase, procurando mantener condiciones similares en la grabación. Despues, los audios se guardaron en formato .wav y se importaron en Python para su análisis y finalmente se graficaron.

**Señales voces de mujeres**


**Señales voces de hombres**


**2. Transformada y espectro de magnitud**

Se aplicó la Transformada de Fourier para obtener su espectro en frecuencia con el siguiente codigo en python.


**3. Calculos de las caracteristicas de la señal**

Por ultimo, se extrajeron parámetros como la frecuencia fundamental, frecuencia media, brillo e intensidad (energia), de la siguiente manera.

Los datos obtenidos fueron:

| SEÑAL      |    f0 (Hz)   |    f_media    |  Brillo    | Intensidad |
|------------|--------------|---------------|------------|------------|
| Mujer 1    |  253.259892  |  3731.791957  |  0.089506  |  10203768.0|
| Mujer 2    |  265.625000  |  3660.250096  |  0.051551  |  16567467.0|
| Mujer 3    |  470.181298  |  3690.630463  |  0.036946  |  17939960.0|
| Hombre 1   |  355.291193  |  2735.577073  |  0.017587  |  20006824.0|
| Hombre 2   |  277.529762  |  2317.599295  |  0.011562  |   9058843.0|
| Hombre 3   |  448.535156  |  1982.546529  |  0.029201  |  14756064.0|


**4. Diagrama de flujo**


# PARTE B
**1. Seleccion de voces**

Se seleccionó una señal de voz una de hombre y una de mujer

**2. Diseño de filtro pasa-banda**

Se diseño un filtro pasa-banda con el rango de la voz entre 80Hz y 400Hz para hombre y 150Hz y 500Hz para mujeres con el objetivo de eliminar ruido no deseado.

**3. Calculos de jitter y del shimmer**

Luego, se calcularon los valores de jitter, analizando la variación de los periodos de la señal, y de shimmer, evaluando la variación de la amplitud entre ciclos. Estos cálculos se realizaron para todas las grabaciones con el fin de analizar la estabilidad de las voces.


# PARTE C

En esta parte se compararon los parámetros obtenidos entre las voces, identificando diferencias en las caracteristicas obtenidas. Finalmente, se interpretaron los resultados, resaltando su importancia en aplicaciones como el análisis de la voz y posibles usos clínicos.

"Comparar los resultados obtenidos entre las voces masculinas y femeninas.
1. ¿Qué diferencias se observan en la frecuencia fundamental?
2. ¿Qué otras diferencias notan en términos de brillo, media o intensidad?
3. Redactar conclusiones sobre el comportamiento de la voz en hombres y mujeres a partir de los análisis realizados.
4. Discuta la importancia clínica del jitter y shimmer en el análisis de la voz."

# RESULTADOS 
# ANALISIS PARTE A 
# ANALISIS PARTE B
# ANALISIS PARTE C 
# ANALISIS PARTE D 

# ANALISIS GENERAL
# PREGUNTAS A DISCUCION

1. ¿Cómo es la frecuencia fundamental de la densidad espectral de potencia asociada a una señal de voz masculina con respecto a la que se obtiene a partir de una señal de voz femenina, mayor o menor? ¿Qué hay del valor RMS?
2. ¿Qué limitaciones plantea el uso de características como shimmer y jitter para la detección de patologías como disartrias y afasias? 

# CONCLUSIONES
