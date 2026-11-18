"""Demanda histórica por estación, franja horaria y tipo de día.

El motor de redistribución necesitaba una señal de uso real de cada estación.
Aquí la precalculamos a partir del histórico: actividad media (retiradas más
devoluciones) y salida neta esperada por hora, que después usamos para
priorizar destinos y dimensionar el relleno de cada estación.

El cálculo se hace sobre la serie sin intervenciones del operario: si la
furgoneta ya rellena una estación, ese movimiento no es demanda de clientes y
no debe inflar la actividad ni disimular la salida neta.
"""

import os

import pandas as pd

from intervenciones import serie_sin_intervenciones

RUTA_DEMANDA = 'demanda_estaciones.csv'

COLUMNAS_DEMANDA = [
    'id_estacion', 'nombre_estacion', 'hora', 'es_finde',
    'actividad_media', 'demanda_norm', 'salida_neta_hora'
]


def calcular_demanda_historica(df_features):
    """Calculamos la demanda por estación, hora y tipo de día desde features_historico."""
    if df_features is None or df_features.empty:
        return pd.DataFrame(columns=COLUMNAS_DEMANDA)

    df = df_features[['id_estacion', 'nombre_estacion', 'timestamp',
                      'bicis_disponibles', 'capacidad']].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')

    # Reconstruimos la ocupación sin los movimientos atribuibles al operario.
    limpia = serie_sin_intervenciones(df)
    limpia = limpia.merge(df[['id_estacion', 'timestamp', 'nombre_estacion']],
                          on=['id_estacion', 'timestamp'], how='left')

    limpia['hora'] = limpia['timestamp'].dt.hour
    limpia['fecha'] = limpia['timestamp'].dt.date
    limpia['es_finde'] = (limpia['timestamp'].dt.dayofweek >= 5).astype(int)

    # Actividad = valor absoluto del cambio de clientes entre lecturas.
    limpia['actividad'] = limpia['delta_cliente'].abs()
    actividad = (limpia.dropna(subset=['actividad'])
                 .groupby(['id_estacion', 'nombre_estacion', 'hora', 'es_finde'])['actividad']
                 .mean()
                 .reset_index()
                 .rename(columns={'actividad': 'actividad_media'}))

    # Salida neta por hora y día: cuántas bicis pierde de media la estación en esa franja.
    agrupado = limpia.sort_values('timestamp').groupby(
        ['id_estacion', 'nombre_estacion', 'fecha', 'hora', 'es_finde'])
    inicio = agrupado['bicis_sin_van'].first()
    fin = agrupado['bicis_sin_van'].last()
    neto = (inicio - fin).reset_index().rename(columns={'bicis_sin_van': 'neto'})
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
