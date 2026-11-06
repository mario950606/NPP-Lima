# NPP-Lima: modelado probabilístico de sismicidad mediante NPP-LSTM

**Proyecto académico de aprendizaje profundo | Lima, Perú**

Este repositorio implementa y evalúa un **Neural Point Process (NPP)** con una red **LSTM** para estudiar los intervalos temporales y las magnitudes de los sismos registrados en el catálogo utilizado del Instituto Geofísico del Perú (IGP). Incluye una adaptación de la arquitectura original, una mejora experimental de la rama de magnitud, ablaciones y una demo interactiva.

> **Importante:** este es un prototipo de investigación retrospectiva. **No permite predecir con certeza cuándo ocurrirá un terremoto**, ni constituye un servicio de pronóstico o alerta sísmica. Sus probabilidades no están calibradas para uso operativo.

## Objetivos

- Reproducir de forma adaptada el modelo NPP-LSTM de referencia, conservando su código original para trazabilidad.
- Estudiar cómo cambia el entrenamiento al sustituir **ReLU por Softplus** en la rama auxiliar de magnitud.
- Comparar historiales de **10, 20 y 30 eventos** mediante ablación.
- Evaluar el desempeño probabilístico del modelo elegido frente a **Poisson**, usando datos históricos posteriores al periodo de entrenamiento.
- Visualizar estimaciones temporales, distribuciones de magnitud y resultados históricos en **Gradio**.

## Datos y configuración

| Elemento | Configuración |
|---|---|
| Datos | Catálogo sísmico IGP incluido en `data/raw/` |
| Registros originales | 2 578 |
| Umbral de filtrado `Mcut` | Mw > 3.5 |
| Eventos después del filtro | 2 441 |
| Umbral objetivo `M0pred` | Mw ≥ 4.5 |
| Modelo neuronal | LSTM de 64 unidades, con ramas temporal y de magnitud |
| Historial seleccionado | 20 eventos (W20) |
| Entrenamiento | Adam, tasa 0.001, lote de 256, hasta 400 épocas, Early Stopping |
| División principal | Corte temporal de 450 000 horas desde el origen del catálogo |

La red utiliza como historial tanto los **intervalos entre sismos** como sus **magnitudes**. El umbral de 3.5 selecciona los eventos que forman parte del historial; el umbral de 4.5 define los eventos objetivo. La rama temporal modela el riesgo acumulado y la rama de magnitud modela una función acumulada de magnitud.

## Principales resultados

### Mejora de activación

| Experimento | Activación auxiliar de magnitud | Mejor pérdida de validación |
|---|---|---:|
| E0, W20 | ReLU | 12.1827 |
| E1, W20 | Softplus | **4.9968** |

En la evaluación histórica del modelo original se identificaron **117 eventos objetivo con densidad de magnitud casi nula**, correspondientes a Mw = 4.5. La versión Softplus eliminó esos casos en el conjunto evaluado y redujo la pérdida del modelo. Esto describe una **mejora de la función objetivo**, no un porcentaje de aciertos sísmicos.

### Ablación de longitud del historial

| Modelo Softplus | Mejor época | Mejor pérdida de validación |
|---|---:|---:|
| W10 | 182 | 5.0983 |
| **W20** | **282** | **4.9968** |
| W30 | 147 | 5.0939 |

**W20** fue seleccionado por tener la menor pérdida de validación en esta ejecución. Se utilizó una sola semilla, por lo que no se afirma superioridad estadística general.

### Comparación probabilística con Poisson

El notebook de comparación evalúa si ocurre **al menos un evento Mw ≥ 4.5** en los siguientes 1, 7 o 30 días. El **Brier Score** mide el error entre la probabilidad estimada y la ocurrencia observada; cuanto menor, mejor.

| Horizonte | Brier NPP Softplus W20 | Brier Poisson | Mejor |
|---|---:|---:|---|
| 1 día | 0.0805 | **0.0783** | Poisson |
| 7 días | 0.2871 | **0.2406** | Poisson |
| 30 días | 0.2289 | **0.2123** | Poisson |

El NPP muestra cierta capacidad de discriminación (ROC-AUC aproximada de **0.708, 0.641 y 0.528** para 1, 7 y 30 días), pero **Poisson obtuvo menor error probabilístico** en los tres horizontes. Una menor pérdida de entrenamiento de la red neuronal no implica una mejor calibración de sus probabilidades.

## Estructura del proyecto

```text
NPP-Lima/
├── data/raw/              # Catálogo utilizado
├── src/
│   ├── utils.py           # Funciones de preprocesamiento originales
│   ├── NeuralPP.py        # Código original conservado como referencia
│   ├── npp_model.py       # Arquitectura adaptada a TensorFlow moderno
│   ├── npp_training.py    # Ventanas, derivadas y pérdida NPP
│   └── npp_magnitude.py   # Visualización de distribución de magnitudes
├── notebooks/             # EDA, entrenamiento, evaluaciones y demo
├── models/                # Pesos entrenados (.weights.h5)
├── results/               # Historiales, métricas, figuras y predicciones
├── docs/                  # Notas para el equipo
├── requirements.txt
└── README.md
```

### Guía de notebooks

| Archivo | Contenido |
|---|---|
| `00_EDA_Catalogo_IGP.ipynb` | Análisis exploratorio |
| `01_Preparacion_Datos.ipynb` | Filtros, tiempos y secuencias |
| `02_Arquitectura_NPP_LSTM.ipynb` | Arquitectura y parámetros |
| `03_Entrenamiento_NPP.ipynb` | Entrenamiento original adaptado, E0 |
| `04_Evaluacion_NPP.ipynb` | Evaluación y diagnóstico de E0 |
| `05_Entrenamiento_NPP_Softplus.ipynb` | Entrenamiento E1 |
| `06_Evaluacion_Softplus.ipynb` | Evaluación E1 |
| `07_Ablacion_W10.ipynb` | Experimento W10 |
| `08_Ablacion_W30.ipynb` | Experimento W30 y comparación de ventanas |
| `09_Comparacion_NPP_Poisson.ipynb` | Pronóstico retrospectivo y métricas |
| `10_Diagnostico_Probabilidades.ipynb` | Análisis de errores, Accuracy y F1 |
| `11_Visualizacion_Metricas.ipynb` | Curvas ROC, PR, calibración y Brier |
| `12_Demo_NPP_Final.ipynb` | Interfaz interactiva en Gradio |

## Instalación y ejecución

Se desarrolló en **Python 3.12**, con TensorFlow, NumPy, pandas, Matplotlib, SciPy, scikit-learn, Jupyter y Gradio. Para reproducirlo en un entorno compatible:

```powershell
# Clonar el repositorio
git clone https://github.com/mario950606/NPP-Lima.git
cd NPP-Lima

# Crear y activar un entorno virtual en Windows
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

En VS Code, abre `notebooks/12_Demo_NPP_Final.ipynb`, selecciona como kernel `Python (.venv)` y ejecuta las celdas en orden. El notebook requiere el catálogo en `data/raw/` y los pesos **`models/NPP_softplus_W20.weights.h5`**. La aplicación Gradio se abre en el navegador local cuando se ejecuta su última celda.

Para reproducir el entrenamiento, sigue los notebooks en orden numérico. **No es necesario reentrenar para ejecutar la demo** si los pesos están presentes.

> GitHub almacena el código y los artefactos, pero **no aloja por sí mismo la aplicación Gradio en ejecución**. Para compartir una demo en vivo se requiere mantener una sesión activa o desplegarla en un servicio apropiado.

## Cómo interpretar la demo

La interfaz permite seleccionar una **fecha histórica** o un modo **desde hoy**, y un horizonte de 1, 7 o 30 días. Muestra las estimaciones del NPP y Poisson, el resultado histórico observable cuando el catálogo cubre todo el horizonte y una visualización de magnitudes.

- El porcentaje NPP es una **estimación experimental** derivada de la variación del riesgo acumulado, no una certeza.
- La visualización de magnitudes describe una distribución **normalizada dentro del rango mostrado**; no determina con exactitud la magnitud de un futuro sismo.
- El modo «desde hoy» depende de la **actualización del catálogo**. Un catálogo antiguo no permite asegurar que se conozcan todos los eventos recientes.
- Una clasificación basada en el 50 % es solo una regla ilustrativa y **no una alerta**.

## Limitaciones metodológicas

1. La adaptación actual **no es una reproducción numéricamente idéntica** de todos los archivos del proyecto original; hay diferencias de implementación en TensorFlow/Keras y se incorporó Softplus como modificación experimental.
2. La normalización de los experimentos iniciales se calculó sobre el bloque histórico completo, antes de separar la validación interna; esta decisión puede introducir **fuga de información de validación**.
3. La inspección del conjunto de prueba durante el desarrollo motivó el diagnóstico de ReLU. Las comparaciones posteriores deben describirse como **exploratorias**, no como una evaluación completamente ciega.
4. Los horizontes evaluados pueden solaparse; las referencias históricas no son necesariamente observaciones independientes.
5. La probabilidad NPP para horizontes futuros supone un historial fijo y requiere más trabajo de **calibración y validación probabilística**.
6. **ETAS no se incorporó todavía** como baseline ejecutado en este repositorio; la comparación implementada es NPP frente a Poisson.

## Procedencia y reconocimiento

El proyecto parte de una implementación de Neural Point Processes proporcionada por un integrante del equipo, basada en la investigación de **Stockman, Lawson y Werner (2023)** sobre pronóstico sísmico con procesos puntuales neuronales. Los archivos originales conservados permiten rastrear la procedencia del diseño, mientras que los notebooks y módulos adicionales documentan la adaptación, los experimentos con el catálogo de Lima y las evaluaciones.

---

**Uso exclusivamente académico y de investigación. No utilizar para decisiones de protección civil ni emisión de alertas.**
