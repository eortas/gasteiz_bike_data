import os
import glob
import pandas as pd

def main():
    datos_dir = 'datos'
    archivos_parquet = glob.glob(os.path.join(datos_dir, '*.parquet'))
    
    if archivos_parquet:
        # Ordenamos los archivos cronológicamente para cargar todos los parquets disponibles
        archivos_parquet_ordenados = sorted(archivos_parquet)
        print(f"Cargando dataset completo desde {len(archivos_parquet_ordenados)} particiones Parquet...")
        dfs = [pd.read_parquet(f) for f in archivos_parquet_ordenados]
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
    
    # Ordenamos de forma cronológica por estación y fecha
    df = df.sort_values(by=['id_estacion', 'timestamp']).reset_index(drop=True)
    
    # Calculamos la capacidad total de cada estación
    df['capacidad'] = df['bicis_disponibles'] + df['anclajes_disponibles']
    
    # Extraemos características temporales
    df['hora'] = df['timestamp'].dt.hour
    df['dia_semana'] = df['timestamp'].dt.dayofweek
    df['es_finde'] = df['dia_semana'].apply(lambda d: 1 if d >= 5 else 0)
    
    # Calculamos el porcentaje de ocupación actual
    df['pct_ocupacion'] = df['bicis_disponibles'] / df['capacidad']
    
    # Calculamos las tendencias de cambio en 15 min (3 lecturas) y 30 min (6 lecturas)
    df['bicis_hace_15m'] = df.groupby('id_estacion')['bicis_disponibles'].shift(3)
    df['bicis_hace_30m'] = df.groupby('id_estacion')['bicis_disponibles'].shift(6)
    
    df['tendencia_15m'] = df['bicis_disponibles'] - df['bicis_hace_15m']
    df['tendencia_30m'] = df['bicis_disponibles'] - df['bicis_hace_30m']
    
    # Definimos el objetivo (target): cuántas bicis habrá en 30 minutos (6 lecturas a futuro)
    df['target_bicis_30m'] = df.groupby('id_estacion')['bicis_disponibles'].shift(-6)
    
    # Eliminamos las filas con valores nulos generadas por los desplazamientos temporales
    df_clean = df.dropna().reset_index(drop=True)
    
    # Convertimos id_estacion a tipo categoría para soporte nativo en LightGBM
    df_clean['id_estacion'] = df_clean['id_estacion'].astype('category')
    
    print(f"Dataset de features generado con éxito: {len(df_clean)} filas procesadas de todos los parquets.")
    df_clean.to_csv('features_historico.csv', index=False)
    print("Guardado en features_historico.csv")

if __name__ == '__main__':
    main()
