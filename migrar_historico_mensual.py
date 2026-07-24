import os
import pandas as pd

def main():
    repo_dir = r"c:\Users\ortas\OneDrive\Documentos\gasteiz_bike_data-1"
    csv_path = os.path.join(repo_dir, "historico.csv")
    datos_dir = os.path.join(repo_dir, "datos")
    
    # Creamos la carpeta datos si no existe
    os.makedirs(datos_dir, exist_ok=True)
    
    print(f"Cargando dataset desde {csv_path}...")
    columnas = ['timestamp', 'id_estacion', 'nombre_estacion', 'bicis_disponibles', 'anclajes_disponibles']
    
    # Cargamos el archivo CSV original
    df = pd.read_csv(csv_path, header=None, names=columnas)
    
    # Extraemos el formato año-mes (YYYY-MM) del timestamp
    df['year_month'] = df['timestamp'].astype(str).str.slice(0, 7)
    
    # Separamos y guardamos cada mes en su archivo correspondiente dentro de datos/
    for mes, grupo in df.groupby('year_month'):
        df_mes = grupo.drop(columns=['year_month'])
        ruta_salida = os.path.join(datos_dir, f'historico_{mes}.csv')
        df_mes.to_csv(ruta_salida, index=False, header=False)
        print(f"OK Creado {ruta_salida} con {len(df_mes)} registros.")
        
    print("\n¡Migración mensual completada con éxito!")

if __name__ == '__main__':
    main()
