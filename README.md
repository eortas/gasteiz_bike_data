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

El modelo predictivo `LGBMRegressor` ha sido evaluado sobre un dataset histórico de **+121.000 lecturas** (junio-julio 2026), obteniendo los siguientes resultados de precisión:

| Métrica | Valor Obtenido | Descripción | Significado Práctico en Producción |
|---|---|---|---|
| **MAE** *(Mean Absolute Error)* | **0.95 bicicletas** | Error promedio absoluto | El modelo se equivoca por **menos de 1 bici** al predecir a 30 minutos vista. |
| **RMSE** *(Root Mean Squared Error)* | **1.59 bicicletas** | Raíz del error cuadrático | Confirma la ausencia de desviaciones graves en periodos de alta volatilidad. |
| **$R^2$ Score** | **0.91 (91%)** | Varianza explicada | Explica el **91% del comportamiento dinámico** de la red de transporte. |
| **Recall (Sensibilidad)** | **91.5%** | Detección de vaciados | Anticipa **9 de cada 10 vaciados de estación** con 30 minutos de margen. |

### 🧠 Importancia de las Variables Explicativas (*Feature Importance*)
Las variables meteorológicas y temporales se posicionan como los principales factores determinantes de la movilidad en Vitoria-Gasteiz:

```
 1. Temperatura (°C)         [1513]  ████████████████████████████████
 2. Bicicletas Actuales      [1479]  ███████████████████████████████
 3. Hora del día             [1381]  █████████████████████████████
 4. Viento (km/h)            [1296]  ███████████████████████████
 5. Estación ID              [ 844]  █████████████████
 6. Tendencia 30 min         [ 652]  █████████████
 7. Día de la semana         [ 642]  █████████████
 8. Vacaciones UPV/EHU       [ 154]  ███
```

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
│ - Convierte a Parquet comprimido (Ahorro del 97.9%)         │
│ - Reentrena modelo LightGBM (modelo_lightgbm.joblib)        │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ Algoritmo de Rutas Multiparada Furgoneta (VRP Dynamic)      │
│ - Priorización estricta: 🔴 CRÍTICA -> 🟡 PRECAUCIÓN        │
│ - Matriz OSRM calibrada (16 km/h velocidad efectiva urbana) │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ Dashboard Streamlit en Vivo (Desplegado en Render.com)      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Módulos del Proyecto

### 1. 🗜️ Data Engineering y Parquet Híbrido ([convertir_a_parquet.py](convertir_a_parquet.py))
- **Particionamiento Mensual**: Almacenamiento en `datos/historico_YYYY-MM.csv` para mantener los diffs de Git por debajo de 1 KB.
- **Compresión Parquet**: Conversión automatizada a `datos/historico_YYYY-MM.parquet` consiguiendo una **reducción de peso del 97.9%** (de 10.6 MB en CSV a solo 0.23 MB en Parquet).

### 2. 🤖 Modelo Predictivo LightGBM ([modelo_prediccion.py](modelo_prediccion.py))
- Inferencia a **30 minutos vista**.
- Soporte categórico nativo para `id_estacion`.
- Integración de variables meteorológicas (Open-Meteo) y de calendario local ([obtener_eventos.py](obtener_eventos.py): Fiestas de La Blanca, San Prudencio, Día del Blusa y receso académico UPV/EHU).

### 3. 🚐 Planificador de Rutas Multiparada Furgoneta ([optimizar_ruta_multiparada.py](optimizar_ruta_multiparada.py))
- **Priorización de Alertas**: Atiende obligatoriamente las estaciones en **🔴 CRÍTICA** (0-1 bicis) antes de abordar cualquier estación en **🟡 PRECAUCIÓN**.
- **Doble Restricción de Seguridad**: Garantiza que la estación origen mantiene una reserva superior o igual a 5 bicicletas predichas y que el destino no supera su capacidad.
- **Matriz OSRM**: Basada en la red viaria real de OpenStreetMap calibrada a la velocidad media urbana efectiva de Vitoria-Gasteiz ([matriz_distancias_estaciones.csv](matriz_distancias_estaciones.csv)).
- **Resultados de Simulación Mensual**: Reduce la indisponibilidad de la red del **13.0% al 4.2%** (mejora neta del **67.5%** en reducción de fallos), redistribuyendo **295 bicicletas/mes** con una ocupación de furgoneta de solo el **37.0%** de la jornada.

### 4. 🖥️ Dashboard Interactivo ([dashboard.py](dashboard.py))
- Panel de control gráfico construido en **Streamlit**.
- Desplegado en **Render** con pipeline automatizado de **Keep-Alive (cron-job.org)** para servicio 24/7/365 sin tiempos de reposo.

---

## 🚀 Instalación y Ejecución Local

### 1. Clonar el repositorio e instalar dependencias
```bash
git clone https://github.com/eortas/gasteiz_bike_data.git
cd gasteiz_bike_data
pip install -r requirements.txt
```

### 2. Generar matriz de distancias OSRM y features
```bash
python generar_matriz_distancias.py
python obtener_meteo.py
python preparar_features.py
python modelo_prediccion.py
```

### 3. Lanzar el Dashboard local de Streamlit
```bash
streamlit run dashboard.py
```
Abre tu navegador en `http://localhost:8501`.

---

## 🔄 Pipeline MLOps Automático (CI/CD)

El archivo [.github/workflows/retrain_ml.yml](.github/workflows/retrain_ml.yml) ejecuta un reentrenamiento nocturno cada día a las 03:00 AM UTC:
1. Extrae los nuevos datos del scraper.
2. Actualiza la compresión Parquet.
3. Reentrena el modelo LightGBM y evalúa sus métricas.
4. Pushea los artefactos actualizados a `main`, desencadenando el auto-despliegue en **Render.com**.