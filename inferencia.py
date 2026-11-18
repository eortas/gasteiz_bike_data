"""Utilidades compartidas de inferencia.

Aquí centralizamos dos piezas que antes estaban duplicadas (y desincronizadas)
en dashboard.py, api/calcular.py y alertas.py:

1. El catálogo de categorías de id_estacion con el que se entrenó el modelo,
   para que una estación ausente en una instantánea no se cruce en silencio
   con el código de otra estación.
2. La construcción del estado de inferencia con la hora, el calendario y las
   tendencias reales del instante actual, en lugar de reutilizar las de la
   última instantánea nocturna de features_historico.csv.
"""

import glob
import os
import time

import numpy as np
import pandas as pd

from config import FEATURE_COLS
from obtener_eventos import es_festivo, es_fiestas_la_blanca, es_vacaciones_universidad

# Tolerancia para localizar una lectura a 15/30 minutos vista (la cadencia del
# scraper es irregular, así que buscamos la lectura más cercana dentro de ±5 min).
TOLERANCIA_LAG_MIN = 5

# Caché ligera de las lecturas recientes para no releer los CSV en cada rerun.
_CACHE_LECTURAS = {}


def aplicar_catalogo_estaciones(df, modelo):
    """Fijamos las categorías de id_estacion al catálogo aprendido por el modelo."""
    catalogo = getattr(modelo, 'categorias_estacion_', None)
    if catalogo is None:
        df['id_estacion'] = df['id_estacion'].astype('category')
    else:
        df['id_estacion'] = pd.Categorical(df['id_estacion'], categories=list(catalogo))
    return df


def predecir_30m(modelo, df_estado):
    """Predecimos las bicis a 30 minutos y recortamos al rango [0, capacidad]."""
    X = df_estado[FEATURE_COLS].copy()
    X = aplicar_catalogo_estaciones(X, modelo)
    return np.clip(modelo.predict(X), 0, df_estado['capacidad'].values)


def _cargar_lecturas_recientes(ttl_segundos=60, directorio='.'):
    """Leemos las últimas lecturas del scraper con una caché corta en memoria."""
    ahora = time.time()
    cache = _CACHE_LECTURAS.get(directorio)
    if cache is not None and ahora - cache['momento'] < ttl_segundos:
        return cache['df']

    archivos = sorted(glob.glob(os.path.join(directorio, 'datos', 'historico_*.csv')))[-2:]
    if not archivos:
        ruta_raiz = os.path.join(directorio, 'historico.csv')
        if os.path.exists(ruta_raiz):
            archivos = [ruta_raiz]
        else:
            return None

    columnas = ['timestamp', 'id_estacion', 'nombre_estacion', 'bicis_disponibles', 'anclajes_disponibles']
    partes = [pd.read_csv(f, header=None, names=columnas) for f in archivos]
    lecturas = pd.concat(partes, ignore_index=True)
    lecturas = lecturas[lecturas['id_estacion'].astype(str).str.startswith('st_')].copy()
    lecturas['timestamp'] = pd.to_datetime(lecturas['timestamp'], format='ISO8601', utc=True)
    # Trabajamos en hora local de Madrid sin zona horaria, igual que el resto del proyecto.
    lecturas['timestamp'] = lecturas['timestamp'].dt.tz_convert('Europe/Madrid').dt.tz_localize(None)
    lecturas['timestamp'] = lecturas['timestamp'].astype('datetime64[ns]')
    lecturas = lecturas.drop_duplicates(subset=['timestamp', 'id_estacion'], keep='last')
    lecturas = lecturas.sort_values(['id_estacion', 'timestamp']).reset_index(drop=True)

    _CACHE_LECTURAS[directorio] = {'momento': ahora, 'df': lecturas}
    return lecturas


def _ultima_lectura_por_estacion(lecturas, ahora):
    """Devolvemos la última lectura conocida de cada estación a la hora indicada."""
    recientes = lecturas[lecturas['timestamp'] <= ahora]
    if recientes.empty:
        return None
    return (recientes.sort_values('timestamp')
            .groupby('id_estacion', as_index=False)
            .tail(1)[['id_estacion', 'bicis_disponibles', 'anclajes_disponibles']]
            .rename(columns={'bicis_disponibles': 'bicis_recientes', 'anclajes_disponibles': 'anclajes_recientes'})
            .reset_index(drop=True))


def _valor_hace(lecturas, minutos):
    """Buscamos el valor de bicis en t - minutos con la lectura más cercana."""
    consulta = (lecturas.sort_values('timestamp')
                .groupby('id_estacion', as_index=False)
                .tail(1)[['id_estacion', 'timestamp']])
    objetivo = lecturas[['id_estacion', 'timestamp', 'bicis_disponibles']].copy()
    objetivo['timestamp'] = objetivo['timestamp'] + pd.Timedelta(minutes=minutos)
    objetivo = objetivo.rename(columns={'bicis_disponibles': 'valor'}).sort_values('timestamp')
    consulta = consulta.sort_values('timestamp')
    resultado = pd.merge_asof(
        consulta, objetivo, on='timestamp', by='id_estacion',
        direction='nearest', tolerance=pd.Timedelta(minutes=TOLERANCIA_LAG_MIN)
    )
    return resultado.drop(columns=['timestamp'])


def _tendencias_desde_lecturas(lecturas):
    """Calculamos tendencia_15m y tendencia_30m con las lecturas reales del scraper."""
    actuales = (lecturas.sort_values('timestamp')
                .groupby('id_estacion', as_index=False)
                .tail(1)[['id_estacion', 'bicis_disponibles']]
                .rename(columns={'bicis_disponibles': 'bicis_actual'}))
    resultado = actuales.reset_index(drop=True)
    for minutos in (15, 30):
        pasado = _valor_hace(lecturas, minutos)
        resultado = resultado.merge(pasado.rename(columns={'valor': f'hace_{minutos}m'}),
                                    on='id_estacion', how='left')
    resultado['tendencia_15m'] = resultado['bicis_actual'] - resultado['hace_15m']
    resultado['tendencia_30m'] = resultado['bicis_actual'] - resultado['hace_30m']
    return resultado[['id_estacion', 'tendencia_15m', 'tendencia_30m']]


def construir_estado_inferencia(df_features, df_live=None, meteo_actual=None, ahora=None,
                                directorio_datos='.'):
    """Construimos el estado operativo con la información real del instante actual.

    Partimos de la última lectura conocida de cada estación (no solo de las filas
    del último timestamp global, para no perder estaciones que se saltaron una
    lectura), sustituimos bicis y anclajes por los datos en vivo si los hay y
    recalculamos hora, día de la semana, calendario y tendencias con el momento
    actual.
    """
    if ahora is None:
        ahora = pd.Timestamp.now(tz='Europe/Madrid').tz_localize(None)
    ahora = pd.Timestamp(ahora).as_unit('ns')

    if df_features is None or df_features.empty:
        return pd.DataFrame()

    base = (df_features.sort_values('timestamp')
            .groupby('id_estacion', as_index=False)
            .tail(1)[['id_estacion', 'nombre_estacion', 'capacidad', 'bicis_disponibles',
                      'anclajes_disponibles', 'tendencia_15m', 'tendencia_30m']]
            .reset_index(drop=True))
    tendencias_historicas = base.set_index('id_estacion')[['tendencia_15m', 'tendencia_30m']]

    lecturas = _cargar_lecturas_recientes(directorio=directorio_datos)

    # Si la API en vivo no responde, al menos usamos las lecturas más recientes
    # del scraper, que son bastante más frescas que la instantánea nocturna.
    if lecturas is not None:
        recientes = _ultima_lectura_por_estacion(lecturas, ahora)
        if recientes is not None:
            base = base.merge(recientes, on='id_estacion', how='left')
            base['bicis_disponibles'] = base['bicis_recientes'].fillna(base['bicis_disponibles'])
            base['anclajes_disponibles'] = base['anclajes_recientes'].fillna(base['anclajes_disponibles'])

    # Los datos de la API en vivo tienen prioridad sobre cualquier histórico.
    if df_live is not None and not df_live.empty and 'id_estacion' in df_live.columns:
        live = df_live.drop_duplicates('id_estacion').set_index('id_estacion')
        for idx in base.index:
            id_estacion = base.loc[idx, 'id_estacion']
            if id_estacion not in live.index:
                continue
            bicis = live.at[id_estacion, 'bicis_disponibles']
            anclajes = live.at[id_estacion, 'anclajes_disponibles']
            capacidad = base.loc[idx, 'capacidad']
            if pd.isna(anclajes):
                anclajes = max(capacidad - bicis, 0)
            base.loc[idx, 'bicis_disponibles'] = float(bicis)
            base.loc[idx, 'anclajes_disponibles'] = float(anclajes)

    # Recalculamos las variables temporales con el instante actual.
    base['timestamp'] = ahora
    base['hora'] = ahora.hour
    base['dia_semana'] = ahora.weekday()
    base['es_finde'] = int(ahora.weekday() >= 5)
    base['capacidad'] = base['capacidad'].clip(lower=1)
    base['pct_ocupacion'] = base['bicis_disponibles'] / base['capacidad']

    # Tendencias reales; si no hay lecturas suficientes, conservamos las históricas.
    if lecturas is not None:
        tendencias = _tendencias_desde_lecturas(lecturas)
        base = base.drop(columns=['tendencia_15m', 'tendencia_30m'], errors='ignore')
        base = base.merge(tendencias, on='id_estacion', how='left')
        base['tendencia_15m'] = base['tendencia_15m'].fillna(
            base['id_estacion'].map(tendencias_historicas['tendencia_15m']))
        base['tendencia_30m'] = base['tendencia_30m'].fillna(
            base['id_estacion'].map(tendencias_historicas['tendencia_30m']))
    base['tendencia_15m'] = base['tendencia_15m'].fillna(0.0)
    base['tendencia_30m'] = base['tendencia_30m'].fillna(0.0)

    if meteo_actual is None:
        meteo_actual = {'temperatura': 18.0, 'llueve': 0, 'viento_kmh': 10.0}
    base['temperatura'] = float(meteo_actual.get('temperatura', 18.0))
    base['llueve'] = int(meteo_actual.get('llueve', 0))
    base['viento_kmh'] = float(meteo_actual.get('viento_kmh', 10.0))

    base['es_festivo'] = es_festivo(ahora)
    base['es_la_blanca'] = es_fiestas_la_blanca(ahora)
    base['es_vacaciones_upv'] = es_vacaciones_universidad(ahora)

    return base.reset_index(drop=True)
