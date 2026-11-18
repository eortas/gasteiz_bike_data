# 🚲 BiciVitoria: Sistema de Alertas Predictivas, MLOps y Redistribución Inteligente de Bicicletas

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![LightGBM](https://img.shields.io/badge/Machine%20Learning-LightGBM-green.svg)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)
![OpenStreetMap](https://img.shields.io/badge/Routing-OpenStreetMap%20OSRM-orange.svg)
![MLOps](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black.svg)
![Deploy](https://img.shields.io/badge/Hosting-Render-purple.svg)

Sistema de inteligencia logística para la red pública de bicicletas de Vitoria-Gasteiz (**BiciVitoria / Mugibike**). El proyecto combina **Machine Learning (LightGBM)**, **Meteorología en tiempo real (Open-Meteo)**, **Calendario de festivos/eventos locales** y **Optimización de Rutas Multiparada (OpenStreetMap OSRM)** para predecir vaciados de estaciones a 30 minutos vista y recomendar circuitos de redistribución óptimos en furgoneta.

---

## 📊 Métricas de Rendimiento del Modelo ML (LightGBM)

El modelo predictivo `LGBMRegressor` se evalúa sobre el dataset histórico completo con una división temporal estricta por timestamp (el 20% más reciente queda como test). Las métricas se exportan automáticamente a [metricas_modelo.json](metricas_modelo.json) y el pipeline de CI bloquea la publicación del modelo si no supera los umbrales de calidad.

| Métrica | Valor Obtenido | Descripción | Significado Práctico en Producción |
|---|---|---|---|
| **MAE** *(Mean Absolute Error)* | **0.85 bicicletas** | Error promedio absoluto | El modelo se equivoca por **menos de 1 bici** al predecir a 30 minutos vista. |
| **RMSE** *(Root Mean Squared Error)* | **1.27 bicicletas** | Raíz del error cuadrático | Confirma la ausencia de desviaciones graves en periodos de alta volatilidad. |
| **R² Score** | **0.87** | Varianza explicada | Explica el **87% del comportamiento dinámico** de la red de transporte. |
| **Recall (Sensibilidad)** | **81.2%** | Detección de vaciados | Anticipa **8 de cada 10 vaciados de estación** con 30 minutos de margen. |

La validación cruzada temporal (5 pliegues) da un MAE medio de 0.82 bicicletas, así que el rendimiento es estable en el tiempo y no depende del corte elegido.

### 🧠 Importancia de las Variables Explicativas (*Feature Importance*)
Las variables de estado, temporales y meteorológicas se posicionan como los principales factores determinantes de la movilidad en Vitoria-Gasteiz:

```
 1. Bicicletas Actuales      [1348]  ████████████████████████████████
 2. Temperatura (°C)         [1305]  ███████████████████████████████
 3. Hora del día             [1283]  ██████████████████████████████
 4. Tendencia 30 min         [1030]  ████████████████████████
 5. Estación ID              [1013]  ████████████████████████
 6. Viento (km/h)            [ 933]  ██████████████████████
 7. Anclajes disponibles     [ 450]  ███████████
 8. Día de la semana         [ 449]  ███████████
 9. % Ocupación              [ 303]  ███████
10. Tendencia 15 min         [ 250]  ██████
11. Capacidad                [ 244]  ██████
12. Vacaciones UPV/EHU       [ 208]  █████
13. Festivo / La Blanca      [  78]  ██
14. Llueve                   [  27]  █
```

`es_finde` no aparece en la lista porque su importancia es 0: es determinista a partir de `dia_semana`, así que LightGBM la ignora y no aporta información nueva. La mantenemos por compatibilidad con los artefactos ya publicados.

El desglose de MAE por estación está disponible en `metricas_modelo.json` (`metricas_por_estacion`): las estaciones de mayor uso (Leizaola, Ballester, Antso Jakituna) concentran el error, mientras que las de menor actividad se predicen con gran precisión.

---

## 🏗️ Arquitectura General del Sistema

```text
┌────────────────────────────────┐
│ Mugibike API Scraper (5 min)   │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐      ┌─────────────────────────────┐
│ CSV Mensual (datos/YYYY-MM)    ├─────►│ Open-Meteo Weather API      │
└───────────────┬────────────────┘      └──────────────┬──────────────┘
                │                                      │
                ▼                                      ▼
┌─────────────────────────────────────────────────────────────┐
│ GitHub Actions MLOps (Reentrenamiento diario nocturno 03:00) │
│ - Convierte a Parquet comprimido (Ahorro del 97.8%)         │
│ - Reentrena modelo LightGBM (modelo_lightgbm.joblib)        │
│ - Precalcula demanda histórica (demanda_estaciones.csv)     │
│ - Puerta de calidad: bloquea el push si el recall baja      │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ Algoritmo de Rutas Multiparada Furgoneta (VRP Dynamic)      │
│ - Puntuación continua: criticidad + demanda + déficit       │
│   menos coste de desplazamiento (prioriza estaciones usadas)│
│ - Relleno dinámico según salida neta esperada por hora      │
│ - Matriz OSRM calibrada (16 km/h velocidad efectiva urbana) │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ Dashboard Streamlit en Vivo (Desplegado en Render.com)      │
│ - Hora, calendario y tendencias recalculados al instante    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Módulos del Proyecto

### 1. 🗜️ Data Engineering y Parquet Híbrido ([convertir_a_parquet.py](convertir_a_parquet.py))
- **Particionamiento Mensual**: Almacenamiento en `datos/historico_YYYY-MM.csv` para mantener los diffs de Git por debajo de 1 KB.
- **Compresión Parquet**: Conversión automatizada a `datos/historico_YYYY-MM.parquet` consiguiendo una **reducción de peso del 97.8%** (de 44.2 MB en CSV a 0.97 MB en Parquet).

### 2. 🤖 Modelo Predictivo LightGBM ([modelo_prediccion.py](modelo_prediccion.py))
- Inferencia a **30 minutos vista** con división temporal estricta por timestamp (el 20% más reciente queda como test).
- **Catálogo de estaciones fijado con el modelo**: `id_estacion` se codifica siempre con las mismas categorías, incluso si una estación falta en una instantánea, evitando cruces silenciosos.
- **Lags temporales reales**: tendencias a 15/30 min y target calculados por tiempo (no por número de filas), robustos ante la cadencia irregular del scraper.
- Integración de variables meteorológicas (Open-Meteo) y de calendario local ([obtener_eventos.py](obtener_eventos.py): Fiestas de La Blanca, San Prudencio, Día del Blusa y receso académico UPV/EHU), con calendario autonómico anual vía `holidays`.
- Métricas desglosadas por estación en `metricas_modelo.json` y reentrenado final con el 100% de los datos.

### 3. 🚐 Planificador de Rutas Multiparada Furgoneta ([optimizar_ruta_multiparada.py](optimizar_ruta_multiparada.py))
- **Priorización continua por uso**: cada destino se puntúa como `criticidad + 30·demanda + 20·déficit − 40·tiempo_llegada`, de modo que entre estaciones igual de críticas gana la de mayor demanda histórica en esa franja, sin ignorar una estación cercana.
- **Relleno dinámico**: la meta de bicis al llegar escala con la salida neta esperada de la hora (`meta = clamp(salida_neta · 1.5, 5, capacidad − 2)`), en lugar de fijarse siempre en 5.
- **Doble Restricción de Seguridad**: la estación origen mantiene una reserva de 5 bicicletas predichas y el destino nunca supera su capacidad. La última descarga se realiza en la estación más cercana con hueco real.
- **Matriz OSRM**: Basada en la red viaria real de OpenStreetMap calibrada a la velocidad media urbana efectiva de Vitoria-Gasteiz ([matriz_distancias_estaciones.csv](matriz_distancias_estaciones.csv)).
- **Comparativa A/B medida**: `simular_redistribucion.py` reproduce el histórico completo con los dos enfoques y guarda los resultados en `resumen_simulacion_impacto.json` (demanda) y `resumen_simulacion_impacto_clasico.json` (clásico).

Resultados sobre el histórico completo, contabilizando solo el horario operativo (06:00–23:00):

| Enfoque | Indisponibilidad simulada | Mejora frente a la red real | Bicis redistribuidas |
|---|---|---|---|
| Red real (sin sistema) | 15.69% | — | — |
| Priorización clásica | 14.37% | 8.4% | 3.748 |
| Priorización por demanda | **12.77%** | **18.6%** | 4.290 |

La priorización por uso más que duplica la mejora del enfoque clásico (18.6% frente a 8.4%) con una jornada de furgoneta similar (60.5% frente a 59.8%). El desglose por estación (`resumen_simulacion_estaciones.csv`) muestra 9 de 12 estaciones mejoradas; las tres excepciones (Ballester −40.7%, Ibaiondo −9.5% y Universidad −2.5%) son estaciones de las que el motor retira bicis como origen, y quedan como línea de trabajo abierta.

### 4. 📈 Demanda Histórica por Estación ([calcular_demanda_estaciones.py](calcular_demanda_estaciones.py))
- Precalcula `demanda_estaciones.csv` con la actividad media y la **salida neta esperada por estación, hora y tipo de día**.
- Es la señal que alimenta la priorización de rutas y el dimensionado del relleno.

### 5. 🖥️ Dashboard Interactivo ([dashboard.py](dashboard.py))
- Panel de control gráfico construido en **Streamlit**.
- **Estado de inferencia en vivo** ([inferencia.py](inferencia.py)): la hora, el calendario y las tendencias se recalculan con las lecturas más recientes del scraper, en lugar de reutilizar la instantánea nocturna.
- Desplegado en **Render** con pipeline automatizado de **Keep-Alive (cron-job.org)** para servicio 24/7/365 sin tiempos de reposo.

---

## 🚀 Instalación y Ejecución Local

### 1. Clonar el repositorio e instalar dependencias
```bash
git clone https://github.com/eortas/gasteiz_bike_data.git
cd gasteiz_bike_data
pip install -r requirements.txt
```

### 2. Generar matriz de distancias OSRM, features y demanda
```bash
python generar_matriz_distancias.py
python obtener_meteo.py
python preparar_features.py
python modelo_prediccion.py
python calcular_demanda_estaciones.py
```

### 3. Lanzar el Dashboard local de Streamlit
```bash
streamlit run dashboard.py
```
Abre tu navegador en `http://localhost:8501`.

---

## 🔄 Pipeline MLOps Automático (CI/CD)

El archivo [.github/workflows/retrain_ml.yml](.github/workflows/retrain_ml.yml) ejecuta un reentrenamiento nocturno cada día a las 03:00 AM UTC:
1. Extrae los nuevos datos del scraper y actualiza la compresión Parquet.
2. Regenera las features con lags temporales reales y el histórico meteorológico completo.
3. Reentrena el modelo LightGBM y evalúa sus métricas.
4. Ejecuta la puerta de calidad (`verificar_calidad.py`): si el recall o el MAE empeoran de los umbrales, el modelo no se publica.
5. Precalcula la demanda por estación y lanza la simulación A/B (priorización por demanda vs lógica clásica).
6. Pushea los artefactos actualizados a `main`, desencadenando el auto-despliegue en **Render.com**.