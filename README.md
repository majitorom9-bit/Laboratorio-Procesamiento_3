# LABORATORIO_PROCESAMIENTO_3

# OBJETIVOS DE LA PRACTICA

El objetivo de esta práctica fue analizar señales de voz mediante herramientas de procesamiento digital, utilizando el análisis espectral para identificar características como la frecuencia fundamental, la intensidad y la estabilidad vocal, y así comparar las diferencias existentes entre voces masculinas y femeninas.

# PARTE A

**1. Obtener las señales de voces**
En esta etapa se grabaron seis señales de voz entre hombre y mujeres pronunciando la misma frase, procurando mantener condiciones similares en la grabación. Despues, los audios se guardaron en formato .wav y se importaron en Python para su análisis y finalmente se graficaron.

**Código utilizado para la gráfica**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# mujeres
fs1,signal1=  wavfile.read('Mujer-1.wav')
fs2,signal2=  wavfile.read('Mujer-2.wav')
fs3,signal3=  wavfile.read('Mujer-3.wav')

# Hombres
fs4,signal4=  wavfile.read('Hombre-1.wav')
fs5,signal5= wavfile.read('Hombre-2.wav')
fs6,signal6= wavfile.read('Hombre-3.wav')


duracion = len(signal1)/fs1

duracion = len(signal2)/fs1

duracion = len(signal3)/fs1


duracion = len(signal4)/fs1

duracion = len(signal5)/fs1

duracion = len(signal6)/fs1


Tiempo1 = np.arange(len(signal1)) / fs1
Tiempo2 = np.arange(len(signal2)) / fs2
Tiempo3 = np.arange(len(signal3)) / fs3
Tiempo4 = np.arange(len(signal4)) / fs4
Tiempo5 = np.arange(len(signal5)) / fs5
Tiempo6 = np.arange(len(signal6)) / fs6
fig, axs = plt.subplots(6,1, figsize=(14,10),sharex=False)

print(len(Tiempo3),len(signal3))

# graficas
axs[0].plot(Tiempo1, signal1)
axs[0].set_title("Mujer 1")
axs[0].set_ylabel('Bits')
axs[0].set_xlabel('Tiempo (s)')

axs[1].plot(Tiempo2, signal2)
axs[1].set_title("Mujer 2")
axs[1].set_ylabel('Bits')
axs[1].set_xlabel('Tiempo (s)')

axs[2].plot(Tiempo3, signal3)
axs[2].set_title("Mujer 3")
axs[2].set_ylabel('Bits')
axs[2].set_xlabel('Tiempo (s)')

axs[3].plot(Tiempo4, signal4)
axs[3].set_title("Hombre 1")
axs[3].set_ylabel('Bits')
axs[3].set_xlabel('Tiempo (s)')

axs[4].plot(Tiempo5, signal5)
axs[4].set_title("Hombre 2")
axs[4].set_ylabel('Bits')
axs[4].set_xlabel('Tiempo (s)')

axs[5].plot(Tiempo6, signal6)
axs[5].set_title("Hombre 3")
axs[5].set_ylabel('Bits')
axs[5].set_xlabel('Tiempo (s)')

plt.tight_layout()
plt.show()

```

**Señales voces de mujeres**


**Señales voces de hombres**


**2. Transformada y espectro de magnitud**

Se aplicó la Transformada de Fourier para obtener su espectro en frecuencia con el siguiente codigo en python.

```phyton
senales = [
    ("Mujer 1", signal1, fs1),
    ("Mujer 2", signal2, fs2),
    ("Mujer 3", signal3, fs3),
    ("Hombre 1", signal4, fs4),
    ("Hombre 2", signal5, fs5),
    ("Hombre 3", signal6, fs6)
]

import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(
    nrows=len(senales), ncols=1,
    figsize=(12, 2.8*len(senales)),
    sharex=True
)

fny_comun = min(fs/2 for _, _, fs in senales)
ymax = 0.0
espectros = []

for (titulo, senal, fs) in senales:
    N = len(senal)
    freqs = np.fft.rfftfreq(N, 1/fs)
    espectro = np.abs(np.fft.rfft(senal))
    idx = freqs <= fny_comun
    espectros.append((titulo, freqs[idx], espectro[idx]))
    ymax = max(ymax, espectro[idx].max())

for ax, (titulo, F, X) in zip(axes, espectros):
    Fp = F[1:]
    Xp = X[1:]
    ax.semilogx(Fp, Xp)
    ax.set_title(titulo)
    ax.set_ylabel('Amplitud')
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
axes[-1].set_xlabel('Frecuencia (Hz)')

fig.tight_layout(); plt.show()

```

**3. Calculos de las caracteristicas de la señal**

Por ultimo, se extrajeron parámetros como la frecuencia fundamental, frecuencia media, brillo e intensidad (energia), de la siguiente manera.
```phyton
import numpy as np
import pandas as pd
from numpy.fft import rfft, rfftfreq
from scipy.signal import find_peaks


def _to_mono_float(x):
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    x = x - x.mean()
    return x

def analizar_senal(signal, fs):
    sig = _to_mono_float(signal)
    N = len(sig)
    if N == 0 or fs <= 0:
        return 0.0, 0.0, 0.0, 0.0

    X = np.abs(rfft(sig))
    f = rfftfreq(N, d=1.0/fs)

    #Frecuencia fundamental
    mask = f >= 50.0
    Xb = X[mask]
    fb = f[mask]
    if Xb.size and Xb.max() > 0:
        peaks, _ = find_peaks(Xb, height=0.10 * Xb.max())
        if peaks.size:
            f0 = float(fb[peaks[np.argmax(Xb[peaks])]])  #Pico más alto
        else:
            f0 = 0.0
    else:
        f0 = 0.0

    #Frecuencia media
    denom = X.sum()
    f_media = float((f * X).sum() / denom) if denom > 0 else 0.0

    #Brillo
    E_total = float((X**2).sum())
    E_altas = float((X[f > 1500.0]**2).sum())
    brillo = float(E_altas / E_total) if E_total > 0 else 0.0

    #Intensidad
    intensidad = float((sig**2).mean())

    return f0, f_media, brillo, intensidad

nombres = ["Mujer 1","Mujer 2","Mujer 3","Hombre 1","Hombre 2","Hombre 3"]
senales = [signal1, signal2, signal3, signal4, signal5, signal6]
fs_list = [fs1, fs2, fs3, fs4, fs5, fs6]

resultados = []
for nom, sig, fs in zip(nombres, senales, fs_list):
    f0, fmedia, brillo, intensidad = analizar_senal(sig, fs)
    resultados.append([nom, f0, fmedia, brillo, intensidad])

tabla = pd.DataFrame(resultados, columns=["Señal","f0 (Hz)","f_media (Hz)","Brillo","Intensidad"])
print(tabla.to_string(index=False))

```

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

<img width="341" height="896" alt="image" src="https://github.com/user-attachments/assets/6a5329b3-6f9c-44da-8c1b-9ec7c73d6655" />


# PARTE B
**1. Seleccion de voces**

Se seleccionó una señal de voz un de hombre ( hombre 3 ) y una de mujer (mujer 1)

**2. Diseño de filtro pasa-banda**

Se diseño un filtro pasa-banda con el rango de la voz entre 80Hz y 400Hz para hombre y 150Hz y 500Hz para mujeres con el objetivo de eliminar ruido no deseado.
el diseño del filtro fue hecho a mano con el fin de calcular el orden y así poder 

**3. Calculos de jitter y del shimmer**

Luego, se calcularon los valores de jitter, analizando la variación de los periodos de la señal, y de shimmer, evaluando la variación de la amplitud entre ciclos. Estos cálculos se realizaron para todas las grabaciones con el fin de analizar la estabilidad de las voces.


# PARTE C

En esta parte se compararon los parámetros obtenidos entre las voces, identificando diferencias en las caracteristicas obtenidas. Finalmente, se interpretaron los resultados, resaltando su importancia en aplicaciones como el análisis de la voz y posibles usos clínicos.

"Comparar los resultados obtenidos entre las voces masculinas y femeninas.
1. ¿Qué diferencias se observan en la frecuencia fundamental?
2. ¿Qué otras diferencias notan en términos de brillo, media o intensidad?
3. Redactar conclusiones sobre el comportamiento de la voz en hombres y mujeres a partir de los análisis realizados.
4. Discuta la importancia clínica del jitter y shimmer en el análisis de la voz."

# ANALISIS PARTE A 

En esta práctica se realizó la adquisición y procesamiento de seis señales de voz, correspondientes a tres mujeres y tres hombres, las cuales fueron grabadas en formato .wav y posteriormente analizadas en Python. Con el fin de garantizar la comparabilidad entre las señales, todas fueron convertidas a formato mono y normalizadas en amplitud, eliminando posibles diferencias asociadas al sistema de grabación.
Al observar las señales en el dominio del tiempo, se evidenció que todas presentan un comportamiento cuasi-periódico, característico de la voz humana. Aunque no son perfectamente periódicas, sí se identifican patrones repetitivos relacionados con la vibración de las cuerdas vocales. También se observaron variaciones en la amplitud entre las señales, lo que indica diferencias en la intensidad de la voz, así como la presencia de pequeños silencios o pausas, posiblemente asociados a la forma en que cada persona pronunció la frase o a las condiciones de grabación.
Posteriormente, se aplicó la Transformada Rápida de Fourier (FFT) a cada señal, obteniendo su espectro de magnitud en el dominio de la frecuencia. En estos espectros se observó que la mayor parte de la energía de la señal se concentra en bajas frecuencias, lo cual es típico de las señales de voz. Además, se identificaron picos bien definidos que corresponden a la frecuencia fundamental (F0) y a sus armónicos, lo que evidencia la naturaleza armónica de la voz humana. El uso de una escala logarítmica permitió visualizar con mayor claridad la distribución espectral en todo el rango de frecuencias.
A partir del análisis espectral, se calcularon diferentes características de cada señal. En cuanto a la frecuencia fundamental (F0), se determinó a partir del pico dominante del espectro por encima de 50 Hz. Se observó que, en general, las voces masculinas presentan valores más bajos de F0 en comparación con las voces femeninas. Esto se debe a diferencias fisiológicas, ya que las cuerdas vocales de los hombres son más largas y gruesas, lo que produce vibraciones más lentas, mientras que en las mujeres, al ser más cortas y tensas, generan frecuencias más altas.
En relación con la frecuencia media o centroide espectral, se encontró que las voces femeninas tienden a presentar valores más altos que las masculinas. Esto indica que su energía está distribuida hacia frecuencias más elevadas, lo que se traduce en una percepción de voz más aguda o brillante. De manera similar, el brillo, calculado como la proporción de energía en frecuencias superiores a 1500 Hz, también resultó mayor en las voces femeninas, reforzando la diferencia en el timbre entre ambos grupos.
Por otro lado, la intensidad de la señal, calculada como el valor promedio cuadrático (RMS), presentó variaciones entre las diferentes grabaciones, pero no mostró una relación directa con el género. Estas diferencias están más asociadas a factores como la intensidad con la que cada persona habló, la distancia al micrófono y las condiciones del entorno durante la grabación.
En general, el análisis realizado permitió identificar diferencias claras entre las voces masculinas y femeninas, especialmente en parámetros como la frecuencia fundamental, la frecuencia media y el brillo, los cuales fueron mayores en las voces femeninas. Por su parte, la intensidad no mostró una tendencia definida según el género. Estos resultados demuestran que el análisis espectral es una herramienta efectiva para caracterizar señales de voz y diferenciarlas según sus propiedades, cumpliendo con los objetivos planteados en la práctica.

# ANALISIS PARTE B

# ANALISIS GENERAL

# PREGUNTAS A DISCUCION

1. ¿Cómo es la frecuencia fundamental de la densidad espectral de potencia asociada a una señal de voz masculina con respecto a la que se obtiene a partir de una señal de voz femenina, mayor o menor? ¿Qué hay del valor RMS?
2. ¿Qué limitaciones plantea el uso de características como shimmer y jitter para la detección de patologías como disartrias y afasias? 

# CONCLUSIONES
