"""Demanda histórica por estación, franja horaria y tipo de día.

El motor de redistribución necesitaba una señal de uso real de cada estación.
Aquí la precalculamos a partir del histórico: actividad media (retiradas más
devoluciones) y salida neta esperada por hora, que después usamos para
priorizar destinos y dimensionar el relleno de cada estación.
"""

import os

import pandas as pd

RUTA_DEMANDA = 'demanda_estaciones.csv'

COLUMNAS_DEMANDA = [
    'id_estacion', 'nombre_estacion', 'hora', 'es_finde',
    'actividad_media', 'demanda_norm', 'salida_neta_hora'
]


def calcular_demanda_historica(df_features):
    """Calculamos la demanda por estación, hora y tipo de día desde features_historico."""
    df = df_features.copy()
    if df.empty:
        return pd.DataFrame(columns=COLUMNAS_DEMANDA)

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    df = df.sort_values(['id_estacion', 'timestamp']).reset_index(drop=True)
    df['hora'] = df['timestamp'].dt.hour
    df['fecha'] = df['timestamp'].dt.date
    if 'es_finde' not in df.columns:
        df['es_finde'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
    df['es_finde'] = df['es_finde'].astype(int)

    # Actividad = valor absoluto del cambio entre lecturas consecutivas.
    df['delta'] = df.groupby('id_estacion')['bicis_disponibles'].diff()
    df['actividad'] = df['delta'].abs()
    actividad = (df.dropna(subset=['actividad'])
                 .groupby(['id_estacion', 'nombre_estacion', 'hora', 'es_finde'])['actividad']
                 .mean()
                 .reset_index()
                 .rename(columns={'actividad': 'actividad_media'}))

    # Salida neta por hora y día: cuántas bicis pierde de media la estación en esa franja.
    agrupado = df.sort_values('timestamp').groupby(
        ['id_estacion', 'nombre_estacion', 'fecha', 'hora', 'es_finde'])
    inicio = agrupado['bicis_disponibles'].first()
    fin = agrupado['bicis_disponibles'].last()
    neto = (inicio - fin).reset_index().rename(columns={'bicis_disponibles': 'neto'})
    salida = (neto.groupby(['id_estacion', 'nombre_estacion', 'hora', 'es_finde'])['neto']
              .mean()
              .reset_index()
              .rename(columns={'neto': 'salida_neta_hora'}))

    demanda = pd.merge(actividad, salida,
                       on=['id_estacion', 'nombre_estacion', 'hora', 'es_finde'],
                       how='outer')

    # Normalizamos la actividad dentro de cada franja para comparar estaciones.
    maximo_franja = demanda.groupby(['hora', 'es_finde'])['actividad_media'].transform('max').replace(0, 1)
    demanda['demanda_norm'] = (demanda['actividad_media'] / maximo_franja).clip(0, 1)
    demanda['salida_neta_hora'] = demanda['salida_neta_hora'].fillna(0.0)

    return demanda[COLUMNAS_DEMANDA].sort_values(['hora', 'es_finde', 'demanda_norm'],
                                                 ascending=[True, True, False]).reset_index(drop=True)


def cargar_demanda_estaciones(ruta=RUTA_DEMANDA):
    """Cargamos la demanda precalculada; si no existe, devolvemos un DataFrame vacío."""
    if not os.path.exists(ruta):
        return pd.DataFrame(columns=COLUMNAS_DEMANDA)
    demanda = pd.read_csv(ruta)
    for columna in ('hora', 'es_finde'):
        demanda[columna] = demanda[columna].astype(int)
    return demanda


def tabla_franja(df_demanda, hora, es_finde):
    """Devolvemos la demanda de la franja actual indexada por nombre de estación.

    Si la franja exacta no existe (por ejemplo, un festivo con poca historia),
    caemos primero al tipo de día y después al agregado global.
    """
    if df_demanda is None or df_demanda.empty:
        return pd.DataFrame(columns=['demanda_norm', 'salida_neta_hora'])

    franja = df_demanda[(df_demanda['hora'] == int(hora)) & (df_demanda['es_finde'] == int(es_finde))]
    if franja.empty:
        franja = df_demanda[df_demanda['es_finde'] == int(es_finde)]
    if franja.empty:
        franja = df_demanda

    return franja.groupby('nombre_estacion')[['demanda_norm', 'salida_neta_hora']].mean()


def valor_franja(tabla, estacion, columna, defecto):
    """Leemos un valor de la tabla de franja con un valor de reserva si falta."""
    if tabla is None or tabla.empty or estacion not in tabla.index:
        return defecto
    valor = tabla.at[estacion, columna]
    return defecto if pd.isna(valor) else float(valor)
