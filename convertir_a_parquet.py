import os
import pandas as pd

def main():
    csv_path = 'historico.csv'
    parquet_path = 'historico.parquet'
    
    print(f"Cargando dataset desde {csv_path}...")
    columnas = ['timestamp', 'id_estacion', 'nombre_estacion', 'bicis_disponibles', 'anclajes_disponibles']
    
    # Cargamos el archivo CSV
    df = pd.read_csv(csv_path, header=None, names=columnas)
    
    # Convertimos los tipos de datos a formatos eficientes en memoria
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    df['id_estacion'] = df['id_estacion'].astype('category')
    df['nombre_estacion'] = df['nombre_estacion'].astype('category')
    df['bicis_disponibles'] = df['bicis_disponibles'].astype('int16')
    df['anclajes_disponibles'] = df['anclajes_disponibles'].astype('int16')
    
    print(f"Guardando {len(df)} filas comprimidas en formato Parquet...")
    # Guardamos en formato parquet con compresión snappy por defecto
    df.to_parquet(parquet_path, index=False, compression='snappy')
    
    # Comparamos tamaños de los archivos
    tam_csv = os.path.getsize(csv_path) / (1024 * 1024)
    tam_parquet = os.path.getsize(parquet_path) / (1024 * 1024)
    reduccion = (1 - (tam_parquet / tam_csv)) * 100
    
    print("\n==================================================================================")
    print("RESULTADO DE LA CONVERSIÓN A PARQUET")
    print("==================================================================================")
    print(f"Tamaño original (historico.csv): {tam_csv:.2f} MB")
    print(f"Tamaño comprimido (historico.parquet): {tam_parquet:.2f} MB")
    print(f"Reducción de espacio conseguida: {reduccion:.1f}%")
    print("==================================================================================")

if __name__ == '__main__':
    main()
