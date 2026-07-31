import os
import sys
import glob
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    datos_dir = 'datos'
    os.makedirs(datos_dir, exist_ok=True)
    
    archivos_csv = glob.glob(os.path.join(datos_dir, '*.csv'))
    if not archivos_csv and os.path.exists('historico.csv'):
        archivos_csv = ['historico.csv']
        
    print(f"Encontrados {len(archivos_csv)} archivos CSV mensuales para optimizar...")
    
    total_tam_csv = 0.0
    total_tam_parquet = 0.0
    
    columnas = ['timestamp', 'id_estacion', 'nombre_estacion', 'bicis_disponibles', 'anclajes_disponibles']
    
    for csv_file in archivos_csv:
        df = pd.read_csv(csv_file, header=None, names=columnas)
        df = df[df['id_estacion'].astype(str).str.startswith('st_')].copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)
        df['id_estacion'] = df['id_estacion'].astype('category')
        df['nombre_estacion'] = df['nombre_estacion'].astype('category')
        df['bicis_disponibles'] = df['bicis_disponibles'].astype('int16')
        df['anclajes_disponibles'] = df['anclajes_disponibles'].astype('int16')
        
        parquet_file = csv_file.replace('.csv', '.parquet')
        df.to_parquet(parquet_file, index=False, compression='snappy')
        
        s_csv = os.path.getsize(csv_file) / (1024 * 1024)
        s_parq = os.path.getsize(parquet_file) / (1024 * 1024)
        
        total_tam_csv += s_csv
        total_tam_parquet += s_parq
        
        print(f" [OK] {os.path.basename(csv_file)} ({s_csv:.2f} MB) -> {os.path.basename(parquet_file)} ({s_parq:.2f} MB)")
        
    reduccion = (1 - (total_tam_parquet / max(total_tam_csv, 0.001))) * 100
    print("\n==================================================================================")
    print("CONVERSION MENSUAL A PARQUET COMPLETADA")
    print("==================================================================================")
    print(f"Tamaño total CSVs: {total_tam_csv:.2f} MB")
    print(f"Tamaño total Parquet: {total_tam_parquet:.2f} MB")
    print(f"Ahorro global: {reduccion:.1f}%")
    print("==================================================================================")

if __name__ == '__main__':
    main()
