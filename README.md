# LABORATORIO_PROCESAMIENTO_3
En esta práctica se trabajaron conceptos fundamentales del procesamiento digital de señales aplicados al análisis de la voz humana. Primero, se adquirieron grabaciones de voz de hombres y mujeres, asegurando condiciones similares de muestreo y entorno. Luego, se aplicó la Transformada de Fourier para representar las señales en el dominio de la frecuencia y analizar su espectro, a partir de esto, se calcularon parámetros característicos como la frecuencia fundamental, el brillo, la intensidad, el jitter y el shimmer. Finalmente, se compararon los resultados entre voces masculinas y femeninas, identificando sus principales diferencias y comprendiendo la importancia de estas medidas en la evaluación y caracterización de la voz.

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

<img width="1339" height="458" alt="image" src="https://github.com/user-attachments/assets/9b41b0c2-ba11-43b4-a1c1-98ca9930fe50" />


**Señales voces de hombres**

<img width="1393" height="456" alt="image" src="https://github.com/user-attachments/assets/464cc2fa-294c-479b-b1e5-03bac1f2cf5e" />


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
Los espectros obtenidos como resultado fueron:

<img width="1046" height="681" alt="image" src="https://github.com/user-attachments/assets/f54161d9-4d0a-4e18-ab0e-9e351516f3c6" />


<img width="1023" height="718" alt="image" src="https://github.com/user-attachments/assets/91b1ca59-ecaa-4aa4-a64e-991797d5df2f" />


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

<img width="158" height="689" alt="image" src="https://github.com/user-attachments/assets/ecf6e875-b0f7-4b72-b724-c4d2c4191622" />



# PARTE B

Ahora bien, en esta parte se seleccionó una grabación de voz masculina y una femenina para realizar un análisis más detallado de estabilidad vocal. Primero, se aplicó un filtro pasa-banda en el rango correspondiente a cada género (80–400 Hz para hombres y 150–500 Hz para mujeres), con el fin de eliminar ruido externo y conservar únicamente las frecuencias relevantes de la voz.

Posteriormente, se calculó el jitter, que representa la variación en el periodo de la señal entre ciclos consecutivos; para ello, los periodos se estimaron mediante la detección de cruces por cero. Por otro lado, el shimmer, que mide la variación en la amplitud entre ciclos consecutivos, se obtuvo a partir de la detección de picos sucesivos en la señal.

A partir de estos valores, se calcularon tanto las medidas absolutas como relativas de jitter y shimmer. Finalmente, se registraron los resultados para todas las grabaciones, lo que permitió comparar la estabilidad vocal entre hombres y mujeres y analizar posibles diferencias en la regularidad de sus señales.

**1. Seleccion de voces**

Se seleccionó una señal de voz un de hombre ( hombre 3 ) y una de mujer (mujer 1)

**2. Diseño de filtro pasa-banda**

Se diseño un filtro pasa-banda con el rango de la voz entre 80Hz y 400Hz para hombre y 150Hz y 500Hz para mujeres con el objetivo de eliminar ruido no deseado.
el diseño del filtro fue hecho a mano con el fin de calcular el orden y así poder hacer su respectivo código para poder realizar el filtrado de las señales.

**Filtro pasa-banda a mano**
Pasa-Banda para hombres
<img width="1183" height="1600" alt="image" src="https://github.com/user-attachments/assets/f7587cc8-e896-4a1d-afbf-a06c15199f96" />

Pasa-Banda para mujeres

<img width="1201" height="1600" alt="image" src="https://github.com/user-attachments/assets/c99cc8a2-7390-498f-8387-e04b9700b5e9" />


**código utilizado para el filtro pasabandas**
```phyton
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def filtro_pasabanda(x, fs, lowcut, highcut, orden):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(orden, [low, high], btype='band')
    return filtfilt(b, a, x)

orden = 3

# Aplicar 
mujer1_filtrada = filtro_pasabanda(signal1, fs3, 80, 500, orden)
mujer2_filtrada = filtro_pasabanda(signal2, fs3, 80, 500, orden)
mujer3_filtrada = filtro_pasabanda(signal3, fs3, 80, 500, orden)
hombre1_filtrada = filtro_pasabanda(signal4, fs6, 80, 500, orden)
hombre2_filtrada = filtro_pasabanda(signal5, fs6, 80, 500, orden)
hombre3_filtrada = filtro_pasabanda(signal6, fs6, 80, 500, orden)

# Ejes de tiempo 
t_mujer = np.arange(len(signal1)) / fs3
t_hombre = np.arange(len(signal6)) / fs6

plt.figure(figsize=(12,6))

#  MUJER
plt.subplot(2,1,1)
plt.plot(t_mujer, signal1, label="Original", alpha=0.6)
plt.plot(t_mujer, mujer1_filtrada, label="Filtrada", linewidth=2)
plt.title(" Mujer 1")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid()

# HOMBRE
plt.subplot(2,1,2)
plt.plot(t_hombre, signal6, label="Original ", alpha=0.6)
plt.plot(t_hombre, hombre3_filtrada, label="Filtrada ", linewidth=2)
plt.title("Hombre 3")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
```
Dandonos como resultado las gráficas:

<img width="1014" height="507" alt="image" src="https://github.com/user-attachments/assets/9e373588-c877-4038-834a-57d0c0319240" />

**3. Ventana de 15 milisegundos**

Se implementó una ventana de 15 milisegundos en segmentos específicos de la señal donde se evidenciaba mayor presencia de voz, con el fin de aislar regiones cuasi-periódicas y garantizar un análisis más preciso del jitter y el shimmer, minimizando la influencia de ruido y de tramos no vocales.
El código utilizado para esta ventana fue:

```phyton
import numpy as np
import matplotlib.pyplot as plt


# Duración ventana

duracion = 0.015  # 15 ms


filtradas = [
    mujer1_filtrada, mujer2_filtrada, mujer3_filtrada,
    hombre1_filtrada, hombre2_filtrada, hombre3_filtrada
]

fs_list = [fs3, fs3, fs3, fs6, fs6, fs6]

nombres = [
    "Mujer 1", "Mujer 2", "Mujer 3",
    "Hombre 1", "Hombre 2", "Hombre 3"
]


#  zonas

tiempos_inicio = [1.0, 1.6, 1.0, 2.6, 2.3, 3.5]


# 4. Grafica de ventanas

plt.figure(figsize=(12,10))

for i, (x, fs, nombre, t_ini) in enumerate(zip(filtradas, fs_list, nombres, tiempos_inicio)):
    
    N = int(duracion * fs)
    inicio = int(t_ini * fs)
    
    segmento = x[inicio:inicio + N]
    t = np.arange(len(segmento)) / fs
    
    plt.subplot(3,2,i+1)
    plt.plot(t, segmento)
    plt.title(f"Ventana 15 ms - {nombre}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.grid()

plt.tight_layout()
plt.show()
```
Y las gráficas obtenidas fueron:

<img width="1024" height="548" alt="image" src="https://github.com/user-attachments/assets/0e629d09-bfe6-4f20-85c7-5b64471b3ede" />

<img width="1022" height="275" alt="image" src="https://github.com/user-attachments/assets/50d78225-59c0-4ec3-8738-34c1c8655b05" />


**4. Calculos de jitter y del shimmer**
Finalmente, se calcularon los valores de jitter a partir de la variación de los periodos estimados mediante cruces por cero entre ciclos consecutivos, y los valores de shimmer a partir de la variación de la amplitud obtenida mediante la detección de picos sucesivos de la señal. Estos cálculos se aplicaron a todas las grabaciones, permitiendo evaluar de manera más precisa la estabilidad vocal y la regularidad de las señales analizadas.

El código utilizado para el jitter fue:

```phyton
import numpy as np

def medir_jitter(señal, fs):
    # Convertir a mono si es necesario
    if señal.ndim > 1:
        señal = señal.mean(axis=1)

    # Normalizar para estabilidad
    señal = señal / np.max(np.abs(señal))

    # 1. Detectar cruces por cero positivos (de negativo a positivo)
    cruces = np.where((señal[:-1] < 0) & (señal[1:] >= 0))[0]

    # 2. Calcular periodos Ti (en segundos)
    tiempos_cruce = cruces / fs
    periodos = np.diff(tiempos_cruce)  # T_i = t_(i+1) - t_i

    if len(periodos) < 2:
        return 0.0, 0.0, 0  # no hay suficientes ciclos

    # 3. Calcular jitter absoluto
    jitter_abs = np.mean(np.abs(np.diff(periodos)))  # |T_(i+1) - T_i|

    # 4. Calcular jitter relativo (%)
    T_prom = np.mean(periodos)
    jitter_rel = (jitter_abs / T_prom) * 100

    return jitter_abs, jitter_rel, len(periodos)

# Mujer 3 filtrada
jitter_abs_m1, jitter_rel_m1, N_m1 = medir_jitter(mujer1_filtrada, fs1)
jitter_abs_m2, jitter_rel_m2, N_m2 = medir_jitter(mujer2_filtrada, fs2)
jitter_abs_m3, jitter_rel_m3, N_m3 = medir_jitter(mujer3_filtrada, fs3)

# Hombre 3 filtrada
jitter_abs_h1, jitter_rel_h1, N_h1 = medir_jitter(hombre1_filtrada, fs4)
jitter_abs_h2, jitter_rel_h2, N_h2 = medir_jitter(hombre2_filtrada, fs5)
jitter_abs_h3, jitter_rel_h3, N_h3 = medir_jitter(hombre3_filtrada, fs6)

# --- Imprimir resultados ---
print(f"Mujer 1  - Jitter absoluto: {jitter_abs_m1*1000:.3f} ms, Jitter relativo: {jitter_rel_m1:.2f} % con {N_m1} periodos")
print(f"Mujer 2  - Jitter absoluto: {jitter_abs_m2*1000:.3f} ms, Jitter relativo: {jitter_rel_m2:.2f} % con {N_m2} periodos")
print(f"Mujer 3  - Jitter absoluto: {jitter_abs_m3*1000:.3f} ms, Jitter relativo: {jitter_rel_m3:.2f} % con {N_m3} periodos")

print(f"Hombre 1  - Jitter absoluto: {jitter_abs_h1*1000:.3f} ms, Jitter relativo: {jitter_rel_h1:.2f} % con {N_h1} periodos")
print(f"Hombre 2  - Jitter absoluto: {jitter_abs_h2*1000:.3f} ms, Jitter relativo: {jitter_rel_h2:.2f} % con {N_h2} periodos")
print(f"Hombre 3  - Jitter absoluto: {jitter_abs_h3*1000:.3f} ms, Jitter relativo: {jitter_rel_h3:.2f} % con {N_h3} periodos")
```

Dando como resultado:

<img width="538" height="94" alt="image" src="https://github.com/user-attachments/assets/7058aaa2-c7fe-4144-a86b-232f6740c4a2" />

Y finalmente el código empleado para el cálculo del shimmer fue:

```phyton
import numpy as np
from scipy.signal import find_peaks

# FUNCIÓN SHIMMER (ABS + REL)

def medir_shimmer(segmento, fs):

    # normalizar
    x = segmento / np.max(np.abs(segmento))
    x = x - np.mean(x)

    # rango voz
    F0_min = 80
    F0_max = 300

    # distancia mínima entre picos
    dist_min = int(fs / F0_max)

    # detectar picos
    peaks, _ = find_peaks(x, distance=dist_min)

    if len(peaks) < 3:
        return None, None, len(peaks)

    # amplitudes
    A = x[peaks]

    # periodos
    T = np.diff(peaks) / fs

    mask = (T >= 1/F0_max) & (T <= 1/F0_min)

    if np.sum(mask) < 2:
        return None, None, len(peaks)

    valid_idx = np.where(mask)[0]
    A_valid = A[np.concatenate(([0], valid_idx + 1))]

    if len(A_valid) < 3:
        return None, None, len(A_valid)

 
    # SHIMMER ABSOLUTO

    shimmer_abs = np.mean(np.abs(np.diff(A_valid)))

  
    # SHIMMER RELATIVO (%)
 
    shimmer_rel = (shimmer_abs / np.mean(A_valid)) * 100

    return shimmer_abs, shimmer_rel, len(A_valid)

duracion = 0.015  # 15 ms

print("------ SHIMMER FINAL ------\n")

for x, fs, nombre, t_ini in zip(filtradas, fs_list, nombres, tiempos_inicio):

    N = int(duracion * fs)
    inicio = int(t_ini * fs)
    segmento = x[inicio:inicio + N]

    shimmer_abs, shimmer_rel, N_ciclos = medir_shimmer(segmento, fs)

    if shimmer_rel is None:
        print(f"{nombre}: ERROR (pocos ciclos: {N_ciclos})\n")
    else:
        print(f"{nombre}:")
        print(f"  Shimmer absoluto = {shimmer_abs:.4f}")
        print(f"  Shimmer relativo = {shimmer_rel:.2f}% ({N_ciclos} ciclos)\n")
```
Dando como resultado:

<img width="281" height="380" alt="image" src="https://github.com/user-attachments/assets/aa29f372-b4ea-4bd6-a3d9-9e4804786360" />


# PARTE C

En esta parte se compararon los parámetros obtenidos entre las voces, identificando diferencias en las caracteristicas obtenidas y en cuanto a la frecuencia fundamental, se observó que las voces femeninas tienen valores más altos que las masculinas. Esto se debe a que las cuerdas vocales de las mujeres vibran más rápido, mientras que en los hombres vibran más lento, produciendo sonidos más graves. Este resultado coincide con lo esperado teóricamente.

Respecto al brillo, la frecuencia media y la intensidad, se notó que las voces femeninas tienen más contenido en frecuencias altas, lo que hace que suenen más agudas o claras. En cambio, las voces masculinas tienen más energía en frecuencias bajas, por lo que suenan más graves. La intensidad no mostró una diferencia muy clara entre hombres y mujeres, ya que depende más de cómo se hizo la grabación y de cada persona.

A partir de estos resultados, se puede concluir que las voces femeninas, en general, son más agudas y presentan mayor contenido en frecuencias altas, mientras que las voces masculinas son más graves. Además, según los valores obtenidos de jitter y shimmer, las voces femeninas mostraron una menor variación en comparación con las masculinas, lo que indica que fueron un poco más estables en este análisis. Sin embargo, algunos valores fueron muy altos, lo que puede deberse a errores en el procesamiento o al ruido en la señal.

Finalmente, el jitter y el shimmer son importantes en el análisis de la voz porque permiten medir qué tan estable es. El jitter mide los cambios en la frecuencia y el shimmer los cambios en la amplitud. En el área clínica, estos parámetros ayudan a detectar posibles problemas en la voz, como enfermedades o alteraciones en las cuerdas vocales. Sin embargo, para que los resultados sean confiables, es importante tener una buena señal y un buen procesamiento.


# ANALISIS PARTE A 

En esta práctica se realizó la adquisición y procesamiento de seis señales de voz, correspondientes a tres mujeres y tres hombres, las cuales fueron grabadas en formato .wav y posteriormente analizadas en Python. Con el fin de garantizar la comparabilidad entre las señales, todas fueron convertidas a formato mono y normalizadas en amplitud, eliminando posibles diferencias asociadas al sistema de grabación.
Al observar las señales en el dominio del tiempo, se evidenció que todas presentan un comportamiento cuasi-periódico, característico de la voz humana. Aunque no son perfectamente periódicas, sí se identifican patrones repetitivos relacionados con la vibración de las cuerdas vocales. También se observaron variaciones en la amplitud entre las señales, lo que indica diferencias en la intensidad de la voz, así como la presencia de pequeños silencios o pausas, posiblemente asociados a la forma en que cada persona pronunció la frase o a las condiciones de grabación.
Posteriormente, se aplicó la Transformada Rápida de Fourier (FFT) a cada señal, obteniendo su espectro de magnitud en el dominio de la frecuencia. En estos espectros se observó que la mayor parte de la energía de la señal se concentra en bajas frecuencias, lo cual es típico de las señales de voz. Además, se identificaron picos bien definidos que corresponden a la frecuencia fundamental (F0) y a sus armónicos, lo que evidencia la naturaleza armónica de la voz humana. El uso de una escala logarítmica permitió visualizar con mayor claridad la distribución espectral en todo el rango de frecuencias.
A partir del análisis espectral, se calcularon diferentes características de cada señal. En cuanto a la frecuencia fundamental (F0), se determinó a partir del pico dominante del espectro por encima de 50 Hz. Se observó que, en general, las voces masculinas presentan valores más bajos de F0 en comparación con las voces femeninas. Esto se debe a diferencias fisiológicas, ya que las cuerdas vocales de los hombres son más largas y gruesas, lo que produce vibraciones más lentas, mientras que en las mujeres, al ser más cortas y tensas, generan frecuencias más altas.
En relación con la frecuencia media o centroide espectral, se encontró que las voces femeninas tienden a presentar valores más altos que las masculinas. Esto indica que su energía está distribuida hacia frecuencias más elevadas, lo que se traduce en una percepción de voz más aguda o brillante. De manera similar, el brillo, calculado como la proporción de energía en frecuencias superiores a 1500 Hz, también resultó mayor en las voces femeninas, reforzando la diferencia en el timbre entre ambos grupos.
Por otro lado, la intensidad de la señal, calculada como el valor promedio cuadrático (RMS), presentó variaciones entre las diferentes grabaciones, pero no mostró una relación directa con el género. Estas diferencias están más asociadas a factores como la intensidad con la que cada persona habló, la distancia al micrófono y las condiciones del entorno durante la grabación.
En general, el análisis realizado permitió identificar diferencias claras entre las voces masculinas y femeninas, especialmente en parámetros como la frecuencia fundamental, la frecuencia media y el brillo, los cuales fueron mayores en las voces femeninas. Por su parte, la intensidad no mostró una tendencia definida según el género. Estos resultados demuestran que el análisis espectral es una herramienta efectiva para caracterizar señales de voz y diferenciarlas según sus propiedades, cumpliendo con los objetivos planteados en la práctica.

# ANALISIS PARTE B

Se evaluó la estabilidad de las señales de voz mediante el cálculo de los parámetros jitter y shimmer, los cuales permiten analizar la variabilidad en la frecuencia y en la amplitud de la señal respectivamente. Para ello, inicialmente se aplicó un filtro pasa-banda a cada una de las grabaciones en el rango de 80 a 500 Hz, con el objetivo de eliminar ruido y conservar únicamente las componentes relevantes de la voz. Posteriormente, se seleccionaron segmentos de corta duración (15 ms) en cada señal, buscando trabajar con regiones aproximadamente periódicas y estacionarias que facilitaran el análisis.

En cuanto al jitter, este se calculó a partir de los cruces por cero positivos de la señal, lo que permitió estimar los periodos de vibración entre ciclos consecutivos. Los resultados obtenidos mostraron valores de jitter relativo entre 18.10% y 23.99% para las voces femeninas, y entre 30.60% y 33.43% para las voces masculinas. Estos valores son considerablemente superiores al rango típico esperado para voces sanas (menor al 1%), lo que indica una alta variabilidad en los periodos de la señal y, por tanto, una aparente inestabilidad en la frecuencia fundamental. Sin embargo, esta interpretación debe tomarse con precaución, ya que estos valores elevados probablemente no reflejan una condición fisiológica real, sino que están influenciados por el método de cálculo empleado. En particular, el uso de cruces por cero puede verse afectado por ruido, armónicos y pequeñas irregularidades en la señal, lo que genera una sobreestimación del jitter. Aun así, se observa que las voces masculinas presentan valores mayores que las femeninas, lo cual sugiere una menor estabilidad relativa en este grupo dentro del experimento.

Por otro lado, el shimmer se calculó a partir de la variación de amplitud entre picos consecutivos dentro de cada segmento analizado. Los resultados obtenidos muestran que las voces femeninas presentan valores de 6.16%, 3.08% y 4.16%, mientras que las voces masculinas presentan valores de 4.50%, 21.19% y 49.09%. Comparando estos valores con el rango típico de 3% a 5%, se observa que algunas señales (como Mujer 2, Mujer 3 y Hombre 1) se encuentran dentro o cerca del rango esperado, lo que indica una estabilidad aceptable en la amplitud. Sin embargo, otros casos, especialmente Hombre 2 y Hombre 3, presentan valores extremadamente altos, lo que sugiere una gran variabilidad en la intensidad de la señal. No obstante, estos resultados también deben analizarse con cautela, ya que el número de ciclos detectados en cada ventana fue muy bajo (entre 3 y 5 ciclos), lo que reduce significativamente la confiabilidad del cálculo. En este contexto, valores tan altos de shimmer no necesariamente representan una condición real de la voz, sino que pueden deberse a limitaciones del método, selección de ventanas poco adecuadas o errores en la detección de picos.

En conjunto, el análisis de jitter y shimmer permite evidenciar diferencias en la estabilidad de las señales de voz analizadas. En general, las voces femeninas presentan menor variabilidad tanto en frecuencia como en amplitud en comparación con las masculinas, lo que indica una mayor estabilidad relativa en este conjunto de datos. Sin embargo, los valores obtenidos, especialmente en jitter y en algunos casos de shimmer, están significativamente alejados de los rangos teóricos, lo que pone en evidencia limitaciones importantes en el procesamiento de la señal, como el uso de métodos sensibles al ruido, la corta duración de las ventanas de análisis y la detección imperfecta de ciclos. A pesar de estas limitaciones, la práctica permitió comprender la utilidad de estos parámetros en el análisis de la voz y la importancia de aplicar técnicas adecuadas de procesamiento para obtener resultados más precisos y confiables.

# ANALISIS GENERAL

En esta práctica se realizó el análisis de señales de voz tanto en el dominio del tiempo como en el dominio de la frecuencia, complementándolo con el estudio de parámetros de estabilidad como el jitter y el shimmer. En la Parte A, se evidenció que las señales de voz presentan un comportamiento cuasi-periódico, con patrones repetitivos asociados a la vibración de las cuerdas vocales. A partir de la Transformada de Fourier, se observó que la mayor parte de la energía se concentra en bajas frecuencias, además de la presencia de una frecuencia fundamental (F0) y sus armónicos, lo que confirma la naturaleza armónica de la voz.

Al comparar las voces masculinas y femeninas, se encontró que las mujeres presentan una frecuencia fundamental más alta, lo cual es coherente con la fisiología de las cuerdas vocales. Asimismo, las voces femeninas mostraron mayores valores de frecuencia media y brillo, indicando una mayor presencia de componentes en altas frecuencias, lo que se traduce en una percepción más aguda. Por otro lado, las voces masculinas presentaron mayor concentración de energía en bajas frecuencias, lo que explica su sonido más grave. En cuanto a la intensidad, no se observó una diferencia clara entre géneros, ya que este parámetro depende más de las condiciones de grabación que de características propias de la voz.

En la Parte B, el análisis se enfocó en la estabilidad de la señal mediante el cálculo de jitter y shimmer. Los resultados mostraron que las voces masculinas presentaron valores más altos de jitter en comparación con las femeninas, lo que indica una mayor variabilidad en la frecuencia. De manera similar, en el shimmer se observaron algunos valores elevados, especialmente en voces masculinas, lo que sugiere mayor variación en la amplitud. Sin embargo, estos valores fueron significativamente mayores a los rangos teóricos esperados, lo que indica que pueden estar influenciados por factores como el ruido, el método de cálculo o la selección de segmentos de análisis. A pesar de esto, se logró identificar que, en general, las voces femeninas presentaron una mayor estabilidad relativa que las masculinas en este experimento.

Integrando ambos análisis, se puede concluir que existen diferencias claras entre las voces masculinas y femeninas tanto en sus características espectrales como en su estabilidad. Las voces femeninas tienden a ser más agudas, con mayor contenido en altas frecuencias y menor variabilidad, mientras que las voces masculinas son más graves y presentan, en este caso, mayor variación en frecuencia y amplitud. Finalmente, el cálculo de jitter y shimmer tiene una gran importancia clínica, ya que permite evaluar la estabilidad de la voz y puede ser útil en la detección de posibles alteraciones o patologías vocales. No obstante, para obtener resultados confiables, es fundamental contar con señales de buena calidad y aplicar métodos adecuados de procesamiento.

# CONCLUSIONES

*El análisis permitió comprobar que la voz puede estudiarse de forma objetiva usando herramientas como la FFT y parámetros específicos, más allá de solo escucharla.

*Se identificaron diferencias entre las voces, lo que demuestra que las señales contienen información útil para caracterizarlas y compararlas.

*Los parámetros de jitter y shimmer evidencian la estabilidad de la señal, aunque su precisión depende mucho de la calidad del procesamiento.

*La práctica mostró que el procesamiento de señales de voz tiene aplicaciones importantes en áreas como la ingeniería y el análisis clínico.
