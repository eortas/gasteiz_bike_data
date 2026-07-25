import os
import glob
import pandas as pd
from obtener_meteo import obtener_meteo_historica
from obtener_eventos import enriquecer_eventos_calendario

def main():
    datos_dir = 'datos'
    archivos_parquet = glob.glob(os.path.join(datos_dir, '*.parquet'))
    
    if archivos_parquet:
        print(f"Cargando dataset optimizado desde {len(archivos_parquet)} particiones Parquet...")
        dfs = [pd.read_parquet(f) for f in sorted(archivos_parquet)]
        df = pd.concat(dfs, ignore_index=True)
    elif os.path.exists('historico.parquet'):
        print("Cargando dataset optimizado desde historico.parquet...")
        df = pd.read_parquet('historico.parquet')
    else:
        print("Cargando dataset desde archivos CSV...")
        columnas = ['timestamp', 'id_estacion', 'nombre_estacion', 'bicis_disponibles', 'anclajes_disponibles']
        archivos_csv = glob.glob(os.path.join(datos_dir, '*.csv')) or ['historico.csv']
        dfs = [pd.read_csv(f, header=None, names=columnas) for f in sorted(archivos_csv)]
        df = pd.concat(dfs, ignore_index=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    df = df.sort_values(by=['id_estacion', 'timestamp']).reset_index(drop=True)
    df['capacidad'] = df['bicis_disponibles'] + df['anclajes_disponibles']
    
    # Extraemos características temporales
    df['hora'] = df['timestamp'].dt.hour
    df['dia_semana'] = df['timestamp'].dt.dayofweek
    df['es_finde'] = df['dia_semana'].apply(lambda d: 1 if d >= 5 else 0)
    
    # Porcentaje de ocupación actual
    df['pct_ocupacion'] = df['bicis_disponibles'] / df['capacidad']
    
    # Tendencias de cambio a 15 y 30 minutos
    df['bicis_hace_15m'] = df.groupby('id_estacion')['bicis_disponibles'].shift(3)
    df['bicis_hace_30m'] = df.groupby('id_estacion')['bicis_disponibles'].shift(6)
    
    df['tendencia_15m'] = df['bicis_disponibles'] - df['bicis_hace_15m']
    df['tendencia_30m'] = df['bicis_disponibles'] - df['bicis_hace_30m']
    
    # Integración de festivos y eventos de Vitoria-Gasteiz
    print("Integrando festivos y eventos locales de Vitoria-Gasteiz...")
    df = enriquecer_eventos_calendario(df)
    
    # Integración de variables meteorológicas (Open-Meteo)
    print("Integrando datos meteorológicos de Open-Meteo...")
    fecha_ini = df['timestamp'].min().strftime('%Y-%m-%d')
    fecha_fin = df['timestamp'].max().strftime('%Y-%m-%d')
    
    if not os.path.exists('meteo_historica.csv'):
        df_meteo = obtener_meteo_historica(fecha_inicio=fecha_ini, fecha_fin=fecha_fin)
        if not df_meteo.empty:
            df_meteo.to_csv('meteo_historica.csv', index=False)
    else:
        df_meteo = pd.read_csv('meteo_historica.csv')
        
    if not df_meteo.empty:
        df_meteo['timestamp_hora'] = pd.to_datetime(df_meteo['timestamp_hora']).dt.tz_localize(None)
        df['timestamp_hora'] = df['timestamp'].dt.tz_localize(None).dt.floor('h')
        
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
        
    # Target: bicis a 30 minutos
    df['target_bicis_30m'] = df.groupby('id_estacion')['bicis_disponibles'].shift(-6)
    
    # Para inferencia en tiempo real conservamos los últimos 30 min donde target_bicis_30m aún no se conoce
    df_clean = df.dropna(subset=['tendencia_30m']).reset_index(drop=True)
    df_clean['id_estacion'] = df_clean['id_estacion'].astype('category')
    
    print(f"Dataset de features con eventos y clima generado: {len(df_clean)} filas procesadas.")
    df_clean.to_csv('features_historico.csv', index=False)
    print("Guardado en features_historico.csv")

if __name__ == '__main__':
    main()
