# Predicción de Fibrilación Auricular Paroxística Inminente mediante SVM


[![PhysioNet Challenge 2001](https://img.shields.io/badge/Dataset-PhysioNet%202001-green)](https://physionet.org/content/challenge-2001/)

> **Tesis de Ingeniería en Estadística — Universidad de Concepción, Chile (2025)**  
> Autor: Maximiliano Concha Sanhueza · Profesor guía: Patricio Sáez  
> Colaboradores: Guillermo Ferreira, Patricio Salas

---

## Descripción

Este proyecto desarrolla un sistema de predicción de **Fibrilación Auricular Paroxística (PAF) inminente** a partir de señales de electrocardiograma (ECG). El modelo está basado en **Máquinas de Vectores de Soporte (SVM)** con kernel RBF y fue entrenado y evaluado sobre la base de datos del [PhysioNet/Computing in Cardiology Challenge 2001](https://physionet.org/content/challenge-2001/).

### Resultado destacado

El modelo alcanzó una puntuación de **21/28 (75 %)** en el Evento 2 del desafío, superando a los ganadores oficiales de 2001 (Schreier et al., 20/28) y quedando sólo por debajo del mejor resultado no oficial (Zong & Mark, 22/28).

| Posición | Puntuación | Precisión | Participante |
|----------|-----------|-----------|--------------|
| 1 | 22/28 | 79 % | Zong & Mark — Harvard-MIT *(no oficial)* |
| **2** | **21/28** | **75 %** | **Maximiliano Concha — UdeC *(este trabajo)*** |
| 3 | 20/28 | 71 % | Schreier et al. — Austrian Research Centers *(ganadores oficiales)* |
| 4 | 19/28 | 68 % | de Chazal & Heneghan / Maier et al. |

---

## Pipeline

```
ECG crudo (30 min, 128 Hz)
        ↓
Downsampling → 64 Hz
        ↓
Segmentación en ventanas deslizantes de 5 min (50 % solapamiento → 11 ventanas)
        ↓
Detección de peaks R + Delineación de ondas P/QRS/T (NeuroKit2 + DWT)
        ↓
Extracción de características HRV (temporal, frecuencial, no lineal) + morfología onda P
        ↓
Vector aplanado por registro + features Δ entre ventanas
        ↓
Imputación (media) + Estandarización (StandardScaler)
        ↓
Selección de características: SelectKBest (k=40, ANOVA F-value)
        ↓
Optimización de hiperparámetros: GridSearchCV (C, γ, class_weight)
Validación cruzada: RepeatedStratifiedKFold (5×2)
        ↓
Predicción por pares (Evento 2): A/N según probabilidad relativa
```

---

## Características extraídas

- **HRV temporal:** MeanNN, SDNN, RMSSD, pNN50, SDSD, CVNN, MedianNN, MadNN, entre otras.
- **HRV frecuencial:** potencia LF, HF, ratio LF/HF, frecuencia pico.
- **HRV no lineal:** entropía aproximada, métricas de complejidad (SD1, SD2, SampEn, DFA).
- **Morfología onda P:** duración media y desviación estándar (via DWT con wavelet db6).
- **Features Δ:** diferencias entre ventanas consecutivas para capturar tendencias temporales.

---




> **Nota:** Los datos de PhysioNet no se incluyen en este repositorio. Ver sección [Datos](#datos) para obtenerlos.

---


### Dependencias principales

```
numpy
pandas
scikit-learn
neurokit2
joblib
optuna
matplotlib
scipy
```

---

## Datos

Los datos provienen del **PAF Prediction Challenge Database** disponible en PhysioNet:

```
https://physionet.org/content/challenge-2001/
```

Descarga los registros y colócalos en las rutas indicadas en `main.py`:

```python
DATA_DIR = "ruta/a/ECGS"          # Archivos .txt de señales ECG
ANSWERS_DIR = "ruta/a/respuestas" # Archivos event-1-answers.txt y event-2-answers.txt
```

La base contiene:
- **Grupo p (entrenamiento):** 25 pares de pacientes con PAF (50 registros).
- **Grupo n:** 25 pares de sujetos sin PAF (50 registros).
- **Grupo t (test):** 50 pares sin etiqueta explícita (100 registros).

---

## Uso

```bash
python main.py
```

El script ejecuta el pipeline completo para 10 semillas y reporta:

```
=== Resultados Evento 2 por Semilla ===
   seed  val_f1  precision_A
0     0   0.712        0.750
...

Promedio F1 = 0.703 ± 0.021
Promedio Precisión Grupo A = 0.696 ± 0.019
```

---

## Metodología resumida

### Preprocesamiento
Las señales ECG de 30 minutos a 128 Hz fueron remuestreadas a 64 Hz y segmentadas en ventanas deslizantes de 5 minutos con 50 % de solapamiento, generando 11 ventanas por registro. Esta duración es el estándar recomendado para análisis de VFC de corto plazo.

### Extracción de características
Para cada ventana se calcularon métricas de variabilidad de la frecuencia cardíaca (HRV) usando NeuroKit2, junto con la duración media y desviación estándar de la onda P mediante delineación por Transformada Wavelet Discreta (DWT, wavelet db6). Las características de las 11 ventanas se concatenaron en un único vector por registro.

### Modelo
Se entrenó una SVM con kernel RBF, optimizando los hiperparámetros C y γ mediante GridSearchCV con validación cruzada RepeatedStratifiedKFold (5×2) y F1-score como métrica. La selección de características se realizó con SelectKBest (k=40).

### Predicción por pares (Evento 2)
Para cada par de registros del grupo t, el modelo asigna una probabilidad de que el registro preceda inmediatamente a un episodio de PAF. El registro con mayor probabilidad recibe la etiqueta `A` y el otro `N`.

---

## Comparación de optimizadores

| Método | Overlap | Precisión Grupo A | Tiempo |
|--------|---------|-------------------|--------|
| **GridSearchCV** | **Sí** | **0.696 ± 0.019** | 7 min 37 s |
| GridSearchCV | No | 0.596 ± 0.045 | 4 min 28 s |
| Optuna | Sí | 0.629 ± 0.042 | 7 min 03 s |
| Hyperopt | Sí | 0.589 ± 0.078 | 7 min 05 s |

GridSearchCV con solapamiento del 50 % obtuvo el mejor rendimiento promedio y la menor varianza entre semillas.

---

## Limitaciones

- Conjunto de entrenamiento reducido (50 registros del grupo p), inherente al dataset de PhysioNet 2001.
- El modelo fue evaluado exclusivamente en este benchmark; la generalización a otros datasets requiere validación adicional.

---

## Citación

Si usas este trabajo, por favor cita:

```bibtex
@thesis{concha2025paf,
  author  = {Concha Sanhueza, Maximiliano},
  title   = {Predicción de Fibrilación Auricular Paroxística Inminente
             mediante Máquinas de Vectores de Soporte},
  school  = {Universidad de Concepción},
  year    = {2025},
  type    = {Tesis de Ingeniería en Estadística},
  address = {Concepción, Chile}
}
```

---

## Referencias clave

- Schreier G., Kastner P., Marko W. (2001). *An automatic ECG processing algorithm to identify patients prone to paroxysmal atrial fibrillation.* Computers in Cardiology.
- Zong W., Mark R.G. (2001). *A methodology for predicting PAF based on ECG arrhythmia feature analysis.* Computers in Cardiology.
- Makowski D. et al. (2021). *NeuroKit2: A Python toolbox for neurophysiological signal processing.* Behavior Research Methods.
- PhysioNet (2001). *PAF Prediction Challenge Database.* https://physionet.org/content/challenge-2001/

