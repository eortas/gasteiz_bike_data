# 🚲 BiciVitoria: predicción y redistribución inteligente

Sistema de apoyo a la operación de la red pública de bicicletas de Vitoria-Gasteiz (Mugibike). Combina datos históricos y en tiempo real para anticipar incidencias en las estaciones y proponer rutas de redistribución.

## Qué hace

- Consulta la disponibilidad de bicicletas y anclajes de Mugibike en tiempo real.
- Incorpora meteorología de Open-Meteo y el calendario local de Vitoria-Gasteiz.
- Predice el número de bicicletas disponibles en cada estación a 30 minutos vista con LightGBM.
- Clasifica alertas de vaciado y saturación en niveles crítico, precaución y normal.
- Calcula rutas multiparada para una furgoneta de redistribución mediante una matriz de distancias OSRM.
- Incluye análisis de inactividad, simulación histórica y auditoría de la flota mínima de 50 bicicletas.

## Arquitectura

```text
Mugibike + Open-Meteo + calendario local
                 ↓
       Histórico y preparación de variables
                 ↓
       Modelo LightGBM (predicción a 30 min)
                 ↓
 Alertas, ruta de redistribución y auditorías
                 ↓
  Dashboard Streamlit / web estática + API Vercel
```

## Resultados actuales del modelo

Las métricas se regeneran en cada reentrenamiento a partir de una división temporal 80/20 y validación cruzada `TimeSeriesSplit` de 5 pliegues.

| Métrica | Resultado |
| --- | ---: |
| MAE en test | 0,77 bicicletas |
| RMSE en test | 1,25 bicicletas |
| Recall de alertas críticas | 83,3 % |
| MAE medio en validación cruzada | 0,77 bicicletas |
| RMSE medio en validación cruzada | 1,20 bicicletas |

Las variables con mayor importancia en el modelo actual son la hora, las bicicletas disponibles, la temperatura, el viento y la estación.

## Aplicaciones

### Dashboard Streamlit

Ejecuta el panel de análisis, alertas y auditorías:

```bash
streamlit run dashboard.py
```

### Interfaz web y API

La interfaz web está formada por `index.html`, `style.css` y `app.js`. Consume el endpoint `GET /api/calcular`, que devuelve el estado, las alertas, la ruta de la furgoneta, la simulación y la auditoría de flota. El archivo `vercel.json` permite desplegar esta versión en Vercel.

Para probarla localmente:

```bash
python server_local.py
```

Después abre `http://localhost:8000`.

## Instalación

```bash
git clone https://github.com/eortas/gasteiz_bike_data.git
cd gasteiz_bike_data
python -m pip install -r requirements.txt
```

## Flujo de datos y entrenamiento

```bash
# Reconstruye el histórico a partir del historial de Git
python reconstruir.py

# Genera Parquet y variables para el modelo
python convertir_a_parquet.py
python preparar_features.py

# Entrena el modelo y guarda sus métricas
python modelo_prediccion.py
```

Los principales artefactos generados son:

- `modelo_lightgbm.joblib`: modelo entrenado.
- `metricas_modelo.json`: métricas e importancia de variables.
- `features_historico.csv`: datos preparados para entrenamiento e inferencia.
- `matriz_distancias_estaciones.csv`: tiempos y distancias entre estaciones.

## Análisis operativos

```bash
# Inactividad por estación
python analisis_estaciones.py

# Simulación de redistribución en horario diurno y 24 horas
python simular_redistribucion.py

# Auditoría de flota a las 04:00 y en ventana temporal
python analisis_flota_4am.py
python analizar_todos_dias_ventana.py
```

La última simulación histórica disponible debe interpretarse como una evaluación del algoritmo, no como una mejora demostrada: en el periodo analizado, la indisponibilidad simulada fue del 13,16 % frente al 12,75 % real. Este resultado sirve para ajustar la estrategia de redistribución antes de su uso operativo.

## Automatización

El workflow [`.github/workflows/retrain_ml.yml`](.github/workflows/retrain_ml.yml) se ejecuta cada día a las 03:00 UTC o manualmente desde GitHub Actions. Reconstruye el histórico, prepara los datos, reentrena el modelo, ejecuta los análisis y publica los artefactos actualizados cuando hay cambios.

## Tecnologías

- Python 3.11, Pandas, NumPy y scikit-learn
- LightGBM
- Streamlit
- Open-Meteo
- OpenStreetMap / OSRM
- GitHub Actions y Vercel
