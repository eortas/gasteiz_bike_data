# Mugibike Vitoria-Gasteiz: predicción y redistribución inteligente de bicicletas

Documento de presentación del proyecto. Explica qué problema resuelve, cómo funciona, qué resultados se han medido y por qué es una base sólida para escalar.

---

## 1. Resumen ejecutivo

Mugibike es el sistema de bicicleta pública de Vitoria-Gasteiz: 12 estaciones que se vacían y se llenan según la hora, el día de la semana, el clima y el calendario de la ciudad. El problema operativo es doble: quedarse sin bicicletas cuando hay demanda y quedarse sin anclajes cuando la gente quiere devolverlas.

Este proyecto convierte el histórico de uso en tres capacidades:

1. **Predecir** la ocupación de cada estación a 30 minutos vista con un error medio de ~1 bicicleta.
2. **Priorizar** qué estaciones atender primero según su uso real, no solo según su urgencia del momento.
3. **Planificar** la ruta de la furgoneta de redistribución y **medir** su impacto frente a dos referencias: la operación actual y un escenario sin ninguna furgoneta.

Cifras clave, todas medidas sobre datos reales:

| Indicador | Valor |
|---|---|
| Error medio de predicción a 30 min (MAE) | **0,85 bicicletas** |
| Varianza del comportamiento de la red explicada (R²) | **0,87** |
| Vaciados críticos detectados con antelación | **8 de cada 10** |
| Indisponibilidad de la red con el sistema | **12,95%** del tiempo operativo |
| Indisponibilidad de la red sin ninguna furgoneta | **16,55%** |
| Indisponibilidad de la operación real observada | **15,69%** |
| Mejora frente a la lógica clásica de reparto | **21,8% vs 16,5%** |

En términos operativos: el sistema recupera unas **1.104 horas-estación de servicio** en los cinco meses analizados, es decir, unos **7 horas-estación al día** que hoy se pierden por estaciones vacías o saturadas.

Un detalle importante de método: la red ya tiene una furgoneta operando, así que el histórico mezcla movimientos de clientes con intervenciones del operario. El proyecto detecta esas intervenciones y las descuenta (ver sección 7), de modo que las cifras anteriores comparan el sistema con una red que no recibe ninguna ayuda externa.

---

## 2. El problema, con datos

### 2.1 La indisponibilidad no es un caso aislado

Durante el periodo analizado (19 de junio a 18 de noviembre de 2026), una estación media estuvo inutilizable **el 15,69% del horario operativo** (06:00 a 23:00), ya sea por falta de bicicletas o por falta de anclajes. Y eso es **con la furgoneta actual ya trabajando**: sin ella, la reconstrucción del mismo periodo da un 16,55%.

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

El planificador construye un circuito multiparada que carga en estaciones con excedente y descarga en estaciones con déficit, respetando la capacidad de la furgoneta y la capacidad física de cada estación, y cerrando la ruta en una estación con hueco real. La simulación reproduce después el histórico completo en tres escenarios: sin furgoneta, operación real observada y sistema propuesto.

---

## 4. Cómo funciona el sistema de punta a punta

```
Datos de Mugibike (cada ~5 min)
        │
        ▼
Histórico mensual + Parquet comprimido (97,8% menos peso)
        │
        ├──► Open-Meteo (clima horario histórico y en vivo)
        ├──► Calendario de Vitoria-Gasteiz (festivos y eventos)
        │
        ▼
Detección de intervenciones del operario (intervenciones.py)
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
        ├──► Simulación A/B contra la operación real y sin furgoneta
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
| Furgoneta que ya opera | Sus movimientos se contaban como demanda de clientes | Se detectan y descuentan; el impacto se mide contra un contrafactual sin furgoneta |
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

Simulación sobre el histórico completo, contabilizando solo el horario operativo. Las dos referencias son la red sin ninguna furgoneta (contrafactual reconstruido) y la operación que realmente ocurrió:

| Escenario | Indisponibilidad | Mejora vs sin furgoneta | Mejora vs operación real | Bicis redistribuidas |
|---|---|---|---|---|
| Sin furgoneta (contrafactual) | 16,55% | — | — | — |
| Operación real observada | 15,69% | 5,2% | — | — |
| Lógica clásica | 13,82% | 16,5% | 11,9% | 3.243 |
| **Priorización por demanda** | **12,95%** | **21,8%** | **17,4%** | 3.624 |

La mejora del enfoque nuevo **supera en un tercio** la del clásico, con una ocupación de la furgoneta prácticamente idéntica (55,8% frente a 54,3%). Es decir: no se gana trabajando más horas, se gana eligiendo mejor dónde ir.

Las dos referencias responden a preguntas distintas. Frente a la operación real, el 17,4% es el valor incremental de sustituir la planificación actual. Frente al escenario sin furgoneta, el 21,8% es el valor total de tener una furgoneta bien dirigida; la diferencia entre ambos (5,2%) es lo que ya aporta hoy la operación manual.

Con contabilidad de 24 horas el patrón se mantiene: 17,56% sin furgoneta → 14,09% con el sistema (mejora del 19,7%, y del 16,0% frente a la operación real).

### 6.3 La furgoneta que ya opera en la red

Este es el punto que más cuidado exige, porque afecta a la validez de todas las cifras anteriores. Mugibike ya redistribuye bicicletas con una furgoneta, pero no hay registro público de su horario ni de su rutina. Si se ignora, el histórico parece demanda natural cuando en realidad es demanda **después de la intervención**: la estación que el operario rellena "se llena sola", su salida neta parece pequeña y el modelo aprende a no esperar problemas donde sí los hay.

La detección es heurística y conservadora. Solo se marca una intervención cuando el salto es casi imposible que venga de clientes:

- **3 o más bicicletas** entre lecturas separadas por 10 minutos o menos, en horario de bajo uso (23:00 a 07:00).
- **5 o más bicicletas** a cualquier hora.

| Medida | Valor |
|---|---|
| Actividad total observada | 83.528 bicis movidas |
| Intervenciones de alta confianza | 1.230 eventos, **5.377 bicis (6,4% del total)** |
| Ritmo detectado | 35,4 bicis al día |
| Noches con intervención | 135 de 152 |
| Techo si contamos todos los saltos de ≥3 | **12,8% de la actividad** |

Las cifras honestas son un rango, no un número: entre el 6,4% confirmado y el 12,8% como techo. Para corregir los datos usamos solo el 6,4%, porque borrar demanda real es peor que dejar pasar alguna intervención.

Sobre esa base se construye un contrafactual anclado a cada hora natural: dentro de cada hora se aplican solo los movimientos de clientes. El anclaje es imprescindible, porque reconstruir la serie completa desde el inicio del histórico acumula el recorte en cero y acaba con una red irrealmente vacía (84% de indisponibilidad, un artefacto aritmético, no un escenario). El resultado es una **cota inferior** del daño real: no puede recuperar la demanda que se perdió por no haber bicicletas.

---

## 7. Por qué este proyecto es defendible

Un modelo de bici pública es fácil de vender con números bonitos y difícil de sostener con datos reales. Aquí hay cinco decisiones que marcan la diferencia:

1. **División temporal honesta.** El test es siempre el futuro del entrenamiento, nunca una mezcla aleatoria. Sin esto, el modelo "aprende" del futuro y las métricas se inflan.
2. **Tendencias medidas en tiempo real.** El scraper no llega con cadencia exacta; desplazar por filas etiquetaba huecos de horas como si fueran 15 minutos. Ahora se busca la lectura más cercana en el tiempo (98,79% de los casos dentro de la ventana de 25 a 35 minutos).
3. **Puerta de calidad en el pipeline.** Si un reentrenamiento baja el recall por debajo del 65%, sube el MAE por encima de 1,5 o el RMSE por encima de 2,5, el modelo no se publica. La calidad deja de depender de que alguien mire una tabla.
4. **Contabilidad de la furgoneta que ya opera.** El histórico no es demanda pura: se detectan y descuentan las intervenciones del operario, y el impacto se mide contra un contrafactual sin furgoneta además de contra la operación real. Sin este paso, el sistema se atribuiría el trabajo que ya hace el vehículo actual.
5. **Comparativa A/B reproducible.** El impacto no es una estimación de folleto: hay dos ejecuciones sobre el mismo histórico, con los mismos datos y solo distinta lógica de decisión.

También decimos lo que aún no funciona: en el desglose por estación, 11 de 12 mejoran frente al contrafactual y 10 de 12 frente a la operación real, pero Ballester empeora un 39,1% porque el motor le retira bicicletas como origen siendo una estación muy usada. Documentarlo es parte del método; resolverlo es la siguiente línea de trabajo.

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
| Alta | Evitar que el motor vacíe estaciones de alta demanda como origen | Corrige la regresión de Ballester |
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
| Demanda | 576 combinaciones estación × hora × tipo de día, solo con movimientos de clientes |
| Intervenciones del operario | 1.230 eventos de alta confianza (6,4% de la actividad) |
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
Porque con cinco meses de datos y 12 estaciones no hay volumen para entrenar una política fiable. La puntuación ponderada es interpretable, auditable y ya mejora un 21,8%. El refuerzo tendría sentido con años de operación y muchas más estaciones.

**¿Cómo sé que la mejora del 21,8% no es casualidad?**
Porque las dos lógicas se ejecutan sobre exactamente el mismo histórico, con el mismo modelo, la misma furgoneta y los mismos tiempos. La única diferencia es el criterio de decisión.

**¿Y si la furgoneta actual hiciera más de lo que detectáis?**
Entonces el contrafactual sería todavía peor y la mejora del sistema mayor. Nuestro sesgo va en la dirección conservadora: preferimos subestimar el impacto antes que atribuirnos el trabajo del operario.

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
