# Informe técnico — Predicción y redistribución inteligente de Mugibike

Documento técnico del sistema: qué mide, cómo lo mide y qué resultados se han obtenido sobre cinco meses de operación real en Vitoria-Gasteiz.

---

## 1. Resumen

El sistema predice la ocupación de las 12 estaciones de Mugibike a 30 minutos vista y planifica la ruta de la furgoneta de redistribución con un criterio de prioridad basado en el uso real de cada estación.

| Indicador | Valor |
|---|---|
| Error medio de predicción a 30 minutos (MAE) | **0,85 bicicletas** |
| RMSE | **1,27 bicicletas** |
| Varianza explicada (R²) | **0,87** |
| Recall en alertas críticas | **81,2%** |
| MAE en validación cruzada temporal (5 pliegues) | **0,82 bicicletas** |
| Indisponibilidad de la red sin ninguna furgoneta | **16,55%** del horario operativo |
| Indisponibilidad de la operación observada | **15,69%** |
| Indisponibilidad con el sistema | **12,95%** |
| Mejora frente a la planificación clásica | **21,8% frente a 16,5%** |

En términos operativos, el sistema recupera **1.104 horas-estación de servicio** en 152 días (7,3 al día) frente a un escenario sin furgoneta, y **837 horas** frente a la operación que se realiza hoy.

---

## 2. Contexto

La indisponibilidad de una estación tiene dos causas simétricas: no hay bicicletas cuando alguien quiere retirar, o no hay anclajes cuando alguien quiere devolver. En el periodo analizado, una estación media estuvo inutilizable el **15,69% del horario operativo** (06:00 a 23:00), y esa cifra ya incluye el trabajo de la furgoneta que opera actualmente.

El reparto no es homogéneo. Sobre el total de horas registradas, incluyendo la madrugada:

| Estación | Tiempo inutilizable | Causa dominante |
|---|---|---|
| Mendizorrotza | 28,4% | Sin bicicletas |
| Universidad | 20,3% | Sin bicicletas |
| AUO - HUA | 17,8% | Sin bicicletas |

La demanda sigue un patrón marcado: entradas y salidas al trabajo, sábados distintos de domingos, festivos locales y autonómicos, periodos vacacionales de la UPV/EHU, lluvia y viento. Ese patrón es aprendible y, sobre todo, anticipable.

---

## 3. Datos y arquitectura

| Elemento | Detalle |
|---|---|
| Lecturas de ocupación | 517.188 registros de 12 estaciones (19-jun a 18-nov-2026) |
| Cadencia | ~5 minutos; mediana del horizonte de 6 lecturas = 30,0 min |
| Calidad de la cadencia | 98,79% de los horizontes dentro de 25-35 minutos |
| Clima | 3.648 horas horarias de Open-Meteo, sin huecos |
| Calendario | Festivos autonómicos por año y locales de Vitoria-Gasteiz |
| Almacenamiento | CSV mensual + Parquet comprimido (44,23 MB → 0,97 MB, 97,8% de ahorro) |
| Automatización | Reentrenamiento diario a las 03:00 UTC con puerta de calidad |
| Interfaces | Dashboard Streamlit y API serverless |

Flujo de punta a punta:

```
Scraper de Mugibike (~5 min)
        │
        ▼
Histórico mensual + Parquet comprimido
        │
        ├──► Open-Meteo (clima histórico y en vivo)
        ├──► Calendario local (festivos, eventos, UPV/EHU)
        │
        ▼
Detección de intervenciones del operario
        │
        ▼
Features: estado, tendencias reales a 15/30 min, clima y calendario
        │
        ▼
Modelo LightGBM (30 min vista) ──► Puerta de calidad
        │
        ├──► Demanda histórica por estación, hora y tipo de día
        │
        ▼
Motor de rutas con priorización por demanda
        │
        ├──► Alertas (crítica / precaución / normal)
        ├──► Ruta multiparada
        ├──► Simulación A/B con tres referencias
        └──► Dashboard y API
```

---

## 4. Metodología

### 4.1 Predicción a 30 minutos

Modelo de gradient boosting (LightGBM) con 16 variables: estado actual (bicicletas, anclajes, capacidad, ocupación), tendencia de los últimos 15 y 30 minutos medida en tiempo real, hora, día de la semana, tipo de día, calendario local y clima.

Cuatro decisiones sostienen la validez de las métricas:

1. **División temporal estricta.** El test es siempre el futuro del entrenamiento. La frontera se sitúa en una marca de tiempo única, de modo que ninguna instantánea cae a ambos lados.
2. **Tendencias medidas en tiempo real.** El scraper no tiene cadencia exacta; los desplazamientos se resuelven con `merge_asof` sobre el instante real con tolerancia de ±5 minutos, no por número de filas.
3. **Catálogo de estaciones fijado con el modelo.** La codificación de `id_estacion` se impone en entrenamiento y en las seis rutas de inferencia, de modo que una estación ausente en una instantánea no desplaza los códigos del resto.
4. **Puerta de calidad automática.** El pipeline no publica un modelo si el recall baja del 65%, el MAE supera 1,5 o el RMSE supera 2,5.

La inferencia en producción reconstruye el estado con la hora real de Madrid, el calendario del día y las tendencias de las últimas lecturas del scraper, con una caché corta para no golpear la API en cada petición.

### 4.2 Priorización por uso real

Cada estación recibe una puntuación continua en lugar de una etiqueta de urgencia:

```
score = criticidad + 30 · demanda_histórica + 20 · déficit − 40 · tiempo_de_llegada
```

- **criticidad**: 100 si la estación está vacía o llena, 50 si el modelo prevé que lo estará en 30 minutos.
- **demanda_histórica**: actividad de esa estación en esa franja horaria y tipo de día, normalizada entre 0 y 1.
- **déficit**: distancia al objetivo de relleno.
- **tiempo_de_llegada**: coste normalizado de desplazamiento sobre la red viaria real.

El objetivo de relleno tampoco es fijo: escala con la salida neta esperada de la hora.

```
meta = clamp(salida_neta_hora · 1,5, 5, capacidad − 2)
```

Una estación que pierde ocho bicicletas por hora se rellena más que una que pierde una.

### 4.3 Ruta y medición

El planificador construye un circuito multiparada que carga en estaciones con excedente y descarga en estaciones con déficit, respeta la capacidad del vehículo y la capacidad física de cada estación, mantiene una reserva de seguridad en el origen y cierra el circuito en una estación con hueco real. La simulación reproduce después el histórico completo con los mismos tiempos de conducción y manipulación.

### 4.4 Tratamiento de la operación existente

La red ya cuenta con una furgoneta redistribuyendo bicicletas, y no existe un registro público de su horario ni de su rutina. Ignorarlo sesgaría las tres piezas del sistema: la estación que el operario rellena parecería tener menos salida neta, el modelo aprendería como natural un comportamiento inducido, y la simulación sumaría su propia furgoneta sobre el trabajo de la real.

El sistema separa ambas cosas con una regla conservadora. Se marca una intervención cuando el salto entre lecturas separadas por 10 minutos o menos es:

- de 3 o más bicicletas en horario de bajo uso (23:00 a 07:00);
- de 5 o más bicicletas a cualquier hora.

| Medida | Valor |
|---|---|
| Actividad total observada | 83.528 bicicletas movidas |
| Intervenciones de alta confianza | 1.230 eventos, **5.377 bicicletas** |
| Peso sobre la actividad | **6,4%** |
| Ritmo | 35,4 bicicletas al día |
| Noches con intervención | 135 de 152 |
| Techo si se cuentan todos los saltos de ≥3 | **12,8%** |

Solo se corrige con el conjunto de alta confianza; el techo se publica como rango. Sobre esa base se reconstruye una serie sin furgoneta que ancla cada hora natural al valor observado y aplica únicamente los movimientos de clientes. Ese anclaje mantiene el contrafactual dentro de un rango físicamente plausible durante todo el periodo.

---

## 5. Resultados

### 5.1 Modelo predictivo

Evaluación sobre el 20% de datos más reciente, con división temporal estricta.

| Métrica | Valor | Lectura |
|---|---|---|
| MAE | 0,85 bicicletas | Error típico inferior a una bicicleta a 30 minutos |
| RMSE | 1,27 bicicletas | Sin desviaciones graves en momentos volátiles |
| R² | 0,87 | Explica el 87% de la varianza de la red |
| Recall en alertas críticas | 81,2% | Anticipa 8 de cada 10 vaciados |
| MAE en validación cruzada | 0,82 bicicletas | Rendimiento estable en el tiempo |

Las variables más relevantes son el estado actual, la temperatura, la hora del día, la tendencia de 30 minutos y la estación. El clima aporta señal propia a través de la temperatura y el viento, y el calendario lo hace a través de las vacaciones universitarias y los festivos.

### 5.2 Impacto de la redistribución

Simulación sobre el histórico completo, contabilizando el horario operativo (06:00-23:00). Las dos referencias son la red sin ninguna furgoneta y la operación que realmente ocurrió.

| Escenario | Indisponibilidad | Mejora vs sin furgoneta | Mejora vs operación observada | Bicicletas redistribuidas |
|---|---|---|---|---|
| Sin furgoneta (contrafactual) | 16,55% | — | — | — |
| Operación observada | 15,69% | 5,2% | — | — |
| Planificación clásica | 13,82% | 16,5% | 11,9% | 3.243 |
| **Priorización por demanda** | **12,95%** | **21,8%** | **17,4%** | 3.624 |

La ocupación del vehículo es prácticamente idéntica en los dos enfoques (55,8% frente a 54,3%): la diferencia no proviene de trabajar más horas, sino de elegir mejor el destino.

Las dos referencias responden a preguntas distintas. Frente a la operación observada, el 17,4% es el valor incremental de mejorar la planificación actual. Frente al escenario sin furgoneta, el 21,8% es el valor total de una furgoneta bien dirigida.

Con contabilidad de 24 horas el patrón se mantiene: 17,56% sin furgoneta, 16,78% en la operación observada y 14,09% con el sistema (19,7% y 16,0% de mejora).

### 5.3 Desglose por estación

| Estación | Sin furgoneta | Operación observada | Con sistema | Mejora vs observada |
|---|---|---|---|---|
| Mendizorrotza | 29,85% | 28,76% | 19,89% | +30,8% |
| Universidad | 21,97% | 20,93% | 18,93% | +9,6% |
| AUO - HUA | 18,25% | 17,42% | 11,40% | +34,5% |
| Salburua | 18,16% | 16,93% | 14,36% | +15,2% |
| Zabalgana | 17,83% | 16,81% | 14,47% | +13,9% |
| Tren geltokia | 17,54% | 16,98% | 16,61% | +2,2% |
| Antso Jakituna | 17,07% | 16,11% | 12,84% | +20,3% |
| Ibaiondo | 15,40% | 14,41% | 8,61% | +40,2% |
| San Martín | 14,38% | 13,47% | 8,32% | +38,3% |
| Leizaola | 12,64% | 11,53% | 12,04% | −4,4% |
| Autobus geltokia | 8,27% | 7,83% | 7,82% | +0,1% |
| Ballester | 7,29% | 7,04% | 10,14% | −44,0% |

Once de doce estaciones mejoran frente al contrafactual y diez frente a la operación observada. Los mayores beneficios se concentran en las estaciones que combinan alta rotación con episodios de vaciado: Ibaiondo, San Martín, AUO - HUA y Mendizorrotza.

Ballester y Leizaola son estaciones con excedente habitual, de modo que el circuito las utiliza como origen. Calibrar el criterio de origen con el uso esperado de la hora siguiente es la siguiente iteración prevista y el punto donde el margen de mejora es mayor.

---

## 6. Supuestos y alcance

- **Contrafactual conservador.** La reconstrucción no recupera la demanda que se perdió por falta de bicicletas, así que el impacto real es igual o superior al medido.
- **Detección heurística.** Sin la rutina del vehículo, el porcentaje atribuible al operario se expresa como rango: 6,4% confirmado, 12,8% como techo.
- **Horizonte de 30 minutos.** Es el margen útil para que un vehículo llegue a tiempo; horizontes más cortos dejan poca capacidad de actuación y más largos aumentan la incertidumbre sin mejorar la decisión.
- **Cinco meses de histórico.** Suficiente para hora y día de la semana; la estacionalidad completa requiere un ciclo anual.
- **Un vehículo.** Los resultados corresponden a una furgoneta de 10 bicicletas operando de 06:00 a 23:00.

---

## 7. Siguientes pasos

| Prioridad | Trabajo | Efecto esperado |
|---|---|---|
| Alta | Criterio de origen ponderado por uso esperado | Cierra la desviación en estaciones con excedente |
| Alta | Target de mínimo del horizonte o probabilidad de vaciado | Alinea la predicción con la decisión operativa |
| Media | Ampliación a un ciclo anual de histórico | Más fiabilidad en festivos y estacionalidad |
| Media | Ventanas de tiempo en la optimización de ruta | Rutas realistas en hora punta |
| Media | Predicción de devoluciones | Anticipar saturación con la misma precisión |
| Baja | Panel de simulación interactivo | Evaluar escenarios de flota sin tocar código |

---

## 8. Reproducibilidad

```bash
pip install -r requirements.txt
python preparar_features.py
python calcular_demanda_estaciones.py
python modelo_prediccion.py
python verificar_calidad.py
python simular_redistribucion.py
```

El último comando genera los tres escenarios sobre el contrafactual sin furgoneta y deja los resultados en `resumen_simulacion_impacto.json`, `resumen_simulacion_clasico_impacto.json`, `resumen_simulacion_24h_impacto.json` y sus desgloses por estación en CSV. El diagnóstico de intervenciones del operario se guarda dentro de cada JSON.

Para la operación diaria:

```bash
python alertas.py
python optimizar_ruta_multiparada.py
streamlit run dashboard.py
```
