# Mugibike Vitoria-Gasteiz: predicción y redistribución inteligente de bicicletas

Documento de presentación del proyecto. Explica qué problema resuelve, cómo funciona, qué resultados se han medido y por qué es una base sólida para escalar.

---

## 1. Resumen ejecutivo

Mugibike es el sistema de bicicleta pública de Vitoria-Gasteiz: 12 estaciones que se vacían y se llenan según la hora, el día de la semana, el clima y el calendario de la ciudad. El problema operativo es doble: quedarse sin bicicletas cuando hay demanda y quedarse sin anclajes cuando la gente quiere devolverlas.

Este proyecto convierte el histórico de uso en tres capacidades:

1. **Predecir** la ocupación de cada estación a 30 minutos vista con un error medio de ~1 bicicleta.
2. **Priorizar** qué estaciones atender primero según su uso real, no solo según su urgencia del momento.
3. **Planificar** la ruta de la furgoneta de redistribución y **medir** su impacto frente a la operación sin sistema.

Cifras clave, todas medidas sobre datos reales:

| Indicador | Valor |
|---|---|
| Error medio de predicción a 30 min (MAE) | **0,85 bicicletas** |
| Varianza del comportamiento de la red explicada (R²) | **0,87** |
| Vaciados críticos detectados con antelación | **8 de cada 10** |
| Indisponibilidad de la red con el sistema | **12,77%** del tiempo operativo |
| Indisponibilidad sin sistema (histórico real) | **15,69%** |
| Mejora frente a la lógica clásica de reparto | **18,6% vs 8,4%** |

En términos operativos: el sistema recupera unas **895 horas-estación de servicio** en los cinco meses analizados, es decir, unos **6 horas-estación al día** que hoy se pierden por estaciones vacías o saturadas.

---

## 2. El problema, con datos

### 2.1 La indisponibilidad no es un caso aislado

Durante el periodo analizado (19 de junio a 18 de noviembre de 2026), una estación media estuvo inutilizable **el 15,69% del horario operativo** (06:00 a 23:00), ya sea por falta de bicicletas o por falta de anclajes.

El reparto no es homogéneo. Sobre el total de horas registradas (incluyendo la madrugada):

| Estación | Tiempo inutilizable | Causa dominante |
|---|---|---|
| Mendizorrotza | 28,4% | Sin bicicletas |
| Universidad | 20,3% | Sin bicicletas |
| AUO - HUA | 17,8% | Sin bicicletas |

### 2.2 El patrón está lejos de ser aleatorio

La demanda sigue una estructura muy marcada: horas punta de entrada y salida al trabajo, sábados distintos de domingos, festivos locales (San Prudencio, La Blanca), periodos vacacionales de la universidad, lluvia y viento. Ese patrón es exactamente lo que un modelo puede aprender, y también lo que una operación manual basada en la urgencia del momento no puede anticipar.

### 2.3 El coste de decidir tarde

Cuando la furgoneta decide a dónde ir mirando solo el estado actual, llega después de que el problema haya ocurrido. Y cuando reparte con reglas rígidas ("primero las críticas, luego las demás"), tiende a atender la estación más cercana o la más urgente, no la que más usuarios va a dejar sin servicio.

---

## 3. La solución: tres capas que se encadenan

### Capa 1. Predicción (modelo LightGBM)

Un modelo de gradient boosting estima cuántas bicicletas habrá en cada estación dentro de 30 minutos, usando:

- estado actual (bicicletas, anclajes, capacidad, porcentaje de ocupación);
- tendencia de los últimos 15 y 30 minutos;
- hora, día de la semana y tipo de día;
- calendario local (festivos autonómicos y locales, La Blanca, vacaciones de la UPV/EHU);
- clima (temperatura, lluvia, viento).

### Capa 2. Priorización por uso real

Es la mejora central del proyecto. Cada estación recibe una **puntuación continua** en lugar de una etiqueta de urgencia:

```
score = criticidad + 30 · demanda_histórica + 20 · déficit − 40 · tiempo_de_llegada
```

- **criticidad**: 100 si ya está vacía o llena, 50 si el modelo prevé que lo estará en 30 minutos.
- **demanda_histórica**: cuánta actividad genera esa estación en esa franja horaria y tipo de día (0 a 1).
- **déficit**: cuánto le falta para su objetivo de relleno.
- **tiempo_de_llegada**: coste normalizado de desplazamiento.

El efecto es directo: entre dos estaciones igual de críticas gana la que más gente usa, sin ignorar por completo una estación cercana.

Además, el objetivo de relleno ya no es fijo (5 bicicletas), sino que **escala con la salida neta esperada de esa hora**:

```
meta = clamp(salida_neta_hora · 1,5, 5, capacidad − 2)
```

Una estación que pierde 8 bicicletas por hora se rellena más que una que pierde 1.

### Capa 3. Ruta de la furgoneta y medición

El planificador construye un circuito multiparada que carga en estaciones con excedente y descarga en estaciones con déficit, respetando la capacidad de la furgoneta y la capacidad física de cada estación, y cerrando la ruta en una estación con hueco real. La simulación reproduce después el histórico completo con y sin sistema para medir el impacto.

---

## 4. Cómo funciona el sistema de punta a punta

```
Scraper de Mugibike (cada ~5 min)
        │
        ▼
Histórico mensual + Parquet comprimido (97,8% menos peso)
        │
        ├──► Open-Meteo (clima horario histórico y en vivo)
        ├──► Calendario de Vitoria-Gasteiz (festivos y eventos)
        │
        ▼
Features: estado, tendencias reales a 15/30 min, clima y calendario
        │
        ▼
Modelo LightGBM (predicción a 30 min) ──► Puerta de calidad en CI
        │
        ├──► Tabla de demanda histórica por estación, hora y tipo de día
        │
        ▼
Motor de rutas con priorización por demanda
        │
        ├──► Alertas (crítica / precaución / normal)
        ├──► Ruta multiparada para la furgoneta
        └──► Dashboard (Streamlit) y API (Vercel)
```

Todo el pipeline se reentrena automáticamente cada noche a las 03:00 UTC en GitHub Actions y se publica solo si supera los umbrales de calidad.

---

## 5. Qué cambia respecto a la versión anterior

| Aspecto | Antes | Ahora |
|---|---|---|
| Prioridad de reparto | Urgencia binaria (crítica / resto) y desempate por distancia | Puntuación continua que incorpora el uso real de cada estación |
| Relleno de estación | 5 bicicletas fijas | Escala con la salida neta esperada de la hora |
| Cálculo de tendencias | Desplazamiento por número de filas | Desplazamiento por tiempo real (±5 min de tolerancia) |
| División train/test | Por índice de fila (una misma hora podía caer en ambos lados) | Por marca de tiempo única |
| Código de estación | Categoría recalculada en cada ejecución | Catálogo fijado con el modelo |
| Estado en producción | Hora y calendario de la última captura nocturna | Hora, calendario y tendencias del instante real |
| Cobertura del clima | 19-jun a 24-jul (el resto, valores de relleno) | 3.648 horas completas, sin huecos |
| Alertas | Solo avisaban de estaciones vacías | Simétricas: también recomiendan descargar en saturadas |
| Calidad del modelo | Sin control automático | El pipeline se bloquea si empeoran recall, MAE o RMSE |
| Impacto | Cifras sin simulación reproducible | Comparativa A/B medida y versionada |

---

## 6. Resultados medidos

### 6.1 Modelo predictivo

Evaluación sobre el 20% de datos más reciente, con división temporal estricta:

| Métrica | Valor | Lectura |
|---|---|---|
| MAE | 0,85 bicicletas | El error típico es de menos de una bicicleta a 30 minutos vista |
| RMSE | 1,27 bicicletas | Sin desviaciones graves en momentos volátiles |
| R² | 0,87 | Explica el 87% del comportamiento de la red |
| Recall en alertas críticas | 81,2% | Detecta 8 de cada 10 vaciados antes de que ocurran |
| Validación cruzada temporal (5 pliegues) | MAE 0,82 | El rendimiento es estable en el tiempo, no un golpe de suerte |

### 6.2 Impacto de la redistribución

Simulación sobre el histórico completo, contabilizando solo el horario operativo:

| Enfoque | Indisponibilidad | Mejora | Bicis redistribuidas |
|---|---|---|---|
| Red real (sin sistema) | 15,69% | — | — |
| Lógica clásica | 14,37% | 8,4% | 3.748 |
| **Priorización por demanda** | **12,77%** | **18,6%** | 4.290 |

La mejora del enfoque nuevo **más que duplica** la del clásico, con una ocupación de la furgoneta prácticamente idéntica (60,5% frente a 59,8%). Es decir: no se gana trabajando más horas, se gana eligiendo mejor dónde ir.

Con contabilidad de 24 horas el patrón se mantiene: 16,8% → 13,93% (mejora del 17,0%).

---

## 7. Por qué este proyecto es defendible

Un modelo de bici pública es fácil de vender con números bonitos y difícil de sostener con datos reales. Aquí hay cuatro decisiones que marcan la diferencia:

1. **División temporal honesta.** El test es siempre el futuro del entrenamiento, nunca una mezcla aleatoria. Sin esto, el modelo "aprende" del futuro y las métricas se inflan.
2. **Tendencias medidas en tiempo real.** El scraper no llega con cadencia exacta; desplazar por filas etiquetaba huecos de horas como si fueran 15 minutos. Ahora se busca la lectura más cercana en el tiempo (98,79% de los casos dentro de la ventana de 25 a 35 minutos).
3. **Puerta de calidad en el pipeline.** Si un reentrenamiento baja el recall por debajo del 65%, sube el MAE por encima de 1,5 o el RMSE por encima de 2,5, el modelo no se publica. La calidad deja de depender de que alguien mire una tabla.
4. **Comparativa A/B reproducible.** El impacto no es una estimación de folleto: hay dos ejecuciones sobre el mismo histórico, con los mismos datos y solo distinta lógica de decisión.

También decimos lo que aún no funciona: en el desglose por estación, 9 de 12 mejoran, pero Ballester (−40,7%), Ibaiondo (−9,5%) y Universidad (−2,5%) empeoran porque el motor les retira bicicletas como origen. Documentarlo es parte del método; resolverlo es la siguiente línea de trabajo.

---

## 8. Para quién es útil

| Perfil | Qué obtiene |
|---|---|
| Operador del servicio | Menos viajes en vacío, rutas priorizadas por impacto real y alertas con 30 minutos de margen |
| Ayuntamiento | Argumentos medibles para dimensionar flota, estaciones y campañas |
| Equipo de datos | Pipeline reproducible, métricas versionadas y control de calidad automático |
| Consultoría o integrador | Base reutilizable para cualquier red de bici pública con histórico de ocupación |

Las tres palancas de valor son siempre las mismas: **menos fricción para el usuario**, **menos kilómetros improductivos** y **decisiones basadas en datos en lugar de urgencias**.

---

## 9. Escalabilidad

El diseño está separado en capas y no depende de Vitoria-Gasteiz salvo en dos puntos explícitos: el calendario local y el catálogo de estaciones.

- **Más estaciones o más ciudades**: basta con ampliar el catálogo y la matriz de distancias. El modelo se reentrena solo.
- **Otra fuente de datos**: el scraper está aislado en un módulo; cualquier API compatible con el mismo esquema entra sin tocar el resto.
- **Otra flota**: la capacidad de la furgoneta y los tiempos de manipulación son parámetros de la simulación, no constantes enterradas.
- **Otro calendario**: los festivos autonómicos se calculan por año con la librería `holidays` y los locales fijos en una lista de dos líneas.

---

## 10. Hoja de ruta

| Prioridad | Trabajo | Por qué |
|---|---|---|
| Alta | Evitar que el motor vacíe estaciones de alta demanda como origen | Corrige las tres regresiones por estación |
| Alta | Completar un año de histórico (hoy hay cinco meses) | Más fiabilidad en festivos y estacionalidad |
| Media | Optimización de ruta con ventanas de tiempo | Rutas más realistas en hora punta |
| Media | Predicción de demanda de devoluciones, no solo de ocupación | Anticipar saturación con la misma precisión |
| Baja | Panel de simulación interactivo | Probar escenarios de flota sin tocar código |

---

## 11. Ficha técnica

| Elemento | Detalle |
|---|---|
| Datos de entrada | 517.188 registros de 12 estaciones (19-jun a 18-nov-2026), cadencia ~5 min |
| Clima | 3.648 horas horarias de Open-Meteo, sin huecos |
| Demanda | 576 combinaciones estación × hora × tipo de día |
| Modelo | LightGBM, 16 variables, horizonte de 30 minutos |
| Almacenamiento | CSV mensual + Parquet comprimido (97,8% de ahorro) |
| Automatización | GitHub Actions, reentrenamiento diario 03:00 UTC con puerta de calidad |
| Interfaz | Dashboard Streamlit (Render) y API serverless (Vercel) |
| Dependencias | pandas, numpy, scikit-learn, lightgbm, streamlit, joblib, pyarrow, requests, holidays |

---

## 12. Preguntas frecuentes

**¿Por qué 30 minutos y no 10?**
Porque 30 minutos es el horizonte útil para que una furgoneta llegue a tiempo. Predecir a 10 minutos deja poco margen de actuación y a 60 minutos la incertidumbre crece sin mejorar la decisión.

**¿El modelo funciona con estaciones nuevas sin histórico?**
Predecirá peor al principio, porque el código de estación no tiene patrones aprendidos. La estructura de features (hora, clima, tendencia) sigue aportando, y el reentrenamiento diario lo incorpora en cuanto haya datos.

**¿Qué pasa si falla la API en vivo?**
El sistema cae a las últimas lecturas del scraper y, si tampoco hay, a la última instantánea histórica. La predicción se sigue generando, marcada como modo histórico en el dashboard.

**¿Por qué no usar aprendizaje por refuerzo para la ruta?**
Porque con cinco meses de datos y 12 estaciones no hay volumen para entrenar una política fiable. La puntuación ponderada es interpretable, auditable y ya mejora un 18,6%. El refuerzo tendría sentido con años de operación y muchas más estaciones.

**¿Cómo sé que la mejora del 18,6% no es casualidad?**
Porque las dos lógicas se ejecutan sobre exactamente el mismo histórico, con el mismo modelo, la misma furgoneta y los mismos tiempos. La única diferencia es el criterio de decisión.

---

## 13. Reproducirlo en cinco comandos

```bash
pip install -r requirements.txt
python preparar_features.py
python calcular_demanda_estaciones.py
python modelo_prediccion.py
python simular_redistribucion.py
```

El último comando genera los tres escenarios (priorización por demanda, lógica clásica y contabilidad 24 horas) y deja los resultados en `resumen_simulacion_impacto.json` y `resumen_simulacion_estaciones.csv`.

Para la vista operativa:

```bash
python alertas.py                    # estado y alertas por estación
python optimizar_ruta_multiparada.py # ruta de la furgoneta
streamlit run dashboard.py           # panel interactivo
```
