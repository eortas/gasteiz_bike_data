import os
import glob
import numpy as np
import pandas as pd

from cargar_historico import cargar_historico_completo
from obtener_meteo import obtener_meteo_historica
from obtener_eventos import enriquecer_eventos_calendario

# Tolerancia al buscar la lectura más cercana a 15/30 minutos (cadencia irregular del scraper).
TOLERANCIA_LAG_MIN = 5


def _calcular_lags(df, columna, lags, tolerancia_min=TOLERANCIA_LAG_MIN):
    """Añadimos columnas con el valor de `columna` desplazado en tiempo real.

    `lags` es una lista de pares (minutos, nombre). Los minutos negativos buscan
    valores futuros (por ejemplo, el target a 30 minutos). Usamos merge_asof con
    la lectura más cercana dentro de la tolerancia, en lugar de shift() posicional,
    para que los huecos de cobertura no etiqueten horizontes de horas como 15/30 min.
    """
    trabajo = df[['id_estacion', 'timestamp', columna]].copy()
    trabajo['_orden'] = np.arange(len(trabajo))
    base = trabajo.sort_values('timestamp').reset_index(drop=True)
    resultado = base[['_orden', 'id_estacion', 'timestamp']].copy()

    for minutos, nombre in lags:
        origen = base[['id_estacion', 'timestamp', columna]].copy()
        origen['timestamp'] = origen['timestamp'] + pd.Timedelta(minutes=minutos)
        origen = origen.rename(columns={columna: nombre}).sort_values('timestamp')
        resultado = pd.merge_asof(
            resultado, origen, on='timestamp', by='id_estacion',
            direction='nearest', tolerance=pd.Timedelta(minutes=tolerancia_min)
        )

    resultado = resultado.sort_values('_orden')
    return resultado.drop(columns=['_orden', 'timestamp']).reset_index(drop=True)


def _asegurar_meteo(df):
    """Descargamos el histórico meteorológico si la caché no cubre el periodo de datos."""
    ruta = 'meteo_historica.csv'
    fecha_ini = df['timestamp'].min().normalize()
    fecha_fin = df['timestamp'].max().normalize()

    df_meteo = pd.DataFrame()
    cubre = False
    if os.path.exists(ruta):
        df_meteo = pd.read_csv(ruta)
        if not df_meteo.empty and 'timestamp_hora' in df_meteo.columns:
            horas = pd.to_datetime(df_meteo['timestamp_hora'])
            # Toleramos un par de días de retraso del archivo de Open-Meteo.
            cubre = horas.min() <= fecha_ini and horas.max() >= fecha_fin - pd.Timedelta(days=2)

    if not cubre:
        print("Actualizando histórico meteorológico (la caché no cubre el periodo de datos)...")
        df_meteo = obtener_meteo_historica(
            fecha_inicio=fecha_ini.strftime('%Y-%m-%d'),
            fecha_fin=fecha_fin.strftime('%Y-%m-%d')
        )
        if not df_meteo.empty:
            df_meteo.to_csv(ruta, index=False)
            print(f"Guardadas {len(df_meteo)} lecturas horarias en {ruta}")
    return df_meteo


def main():
    datos_dir = 'datos'
    archivos_parquet = glob.glob(os.path.join(datos_dir, '*.parquet'))

    if archivos_parquet:
        print(f"Cargando dataset optimizado desde {len(archivos_parquet)} particiones Parquet...")
        dfs = [pd.read_parquet(f) for f in sorted(archivos_parquet)]
        df = pd.concat(dfs, ignore_index=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)
        df = df[df['id_estacion'].astype(str).str.startswith('st_')].copy()
    else:
        print("Cargando dataset desde los CSV históricos (raíz y particiones mensuales)...")
        df = cargar_historico_completo()

    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Madrid').dt.tz_localize(None)
    # Unificamos la precisión para que merge_asof no rechace claves us/ns distintas.
    df['timestamp'] = pd.to_datetime(df['timestamp']).astype('datetime64[ns]')

    df['id_estacion'] = df['id_estacion'].astype(str)
    df = (df.drop_duplicates(subset=['timestamp', 'id_estacion'], keep='last')
            .sort_values(by=['id_estacion', 'timestamp'])
            .reset_index(drop=True))

    df['capacidad'] = (df['bicis_disponibles'] + df['anclajes_disponibles']).clip(lower=1)

    # Extraemos características temporales.
    df['hora'] = df['timestamp'].dt.hour
    df['dia_semana'] = df['timestamp'].dt.dayofweek
    df['es_finde'] = (df['dia_semana'] >= 5).astype(int)

    # Porcentaje de ocupación actual.
    df['pct_ocupacion'] = df['bicis_disponibles'] / df['capacidad']

    # Tendencias a 15 y 30 minutos y target a 30 minutos, calculados en tiempo real.
    lags = _calcular_lags(
        df,
        'bicis_disponibles',
        [(15, 'bicis_hace_15m'), (30, 'bicis_hace_30m'), (-30, 'target_bicis_30m')]
    )
    df = pd.concat([df, lags], axis=1)
    df['tendencia_15m'] = df['bicis_disponibles'] - df['bicis_hace_15m']
    df['tendencia_30m'] = df['bicis_disponibles'] - df['bicis_hace_30m']

    # Integración de festivos y eventos de Vitoria-Gasteiz.
    print("Integrando festivos y eventos locales de Vitoria-Gasteiz...")
    df = enriquecer_eventos_calendario(df)

    # Integración de variables meteorológicas (Open-Meteo).
    print("Integrando datos meteorológicos de Open-Meteo...")
    df_meteo = _asegurar_meteo(df)

    if not df_meteo.empty:
        df_meteo['timestamp_hora'] = pd.to_datetime(df_meteo['timestamp_hora'])
        if df_meteo['timestamp_hora'].dt.tz is not None:
            df_meteo['timestamp_hora'] = df_meteo['timestamp_hora'].dt.tz_convert('Europe/Madrid').dt.tz_localize(None)
        else:
            df_meteo['timestamp_hora'] = df_meteo['timestamp_hora'].dt.tz_localize(None)
        df_meteo = df_meteo.drop_duplicates(subset=['timestamp_hora'], keep='last')
        df_meteo = df_meteo[['timestamp_hora', 'temperatura', 'precipitacion', 'viento_kmh', 'llueve']]

        df['timestamp_hora'] = df['timestamp'].dt.floor('h')
        df = pd.merge(df, df_meteo, on='timestamp_hora', how='left')
        df['temperatura'] = df['temperatura'].fillna(18.0)
        df['precipitacion'] = df['precipitacion'].fillna(0.0)
        df['llueve'] = df['llueve'].fillna(0).astype(int)
        df['viento_kmh'] = df['viento_kmh'].fillna(10.0)
    else:
        df['temperatura'] = 18.0
        df['precipitacion'] = 0.0
        df['llueve'] = 0
        df['viento_kmh'] = 10.0

    # Conservamos las últimas lecturas sin tendencia de 30 min (primera media hora de cada estación).
    df_clean = df.dropna(subset=['tendencia_30m']).reset_index(drop=True)
    df_clean['id_estacion'] = df_clean['id_estacion'].astype('category')

    print(f"Dataset de features con eventos y clima generado: {len(df_clean)} filas procesadas.")
    df_clean.to_csv('features_historico.csv', index=False)
    print("Guardado en features_historico.csv")


if __name__ == '__main__':
    main()
