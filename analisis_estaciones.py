import pandas as pd

def cargar_y_preparar_datos(ruta_csv):
    # Definimos los nombres de las columnas ya que el CSV no tiene cabecera
    columnas = ['timestamp', 'id_estacion', 'nombre_estacion', 'bicis_disponibles', 'anclajes_disponibles']
    
    # Cargamos el archivo CSV con pandas
    df = pd.read_csv(ruta_csv, header=None, names=columnas)
    
    # Convertimos la columna timestamp a tipo fecha (datetime)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ordenamos los datos por estación y fecha/hora
    df = df.sort_values(by=['id_estacion', 'timestamp']).reset_index(drop=True)
    
    # Calculamos la duración en minutos de cada medición usando el registro siguiente de esa estación
    df['siguiente_timestamp'] = df.groupby('id_estacion')['timestamp'].shift(-1)
    df['duracion_minutos'] = (df['siguiente_timestamp'] - df['timestamp']).dt.total_seconds() / 60.0
    
    # Si la última muestra de una estación no tiene siguiente registro, asignamos la mediana del intervalo
    mediana_intervalo = df['duracion_minutos'].median()
    df['duracion_minutos'] = df['duracion_minutos'].fillna(mediana_intervalo)
    
    # Identificamos los estados problemáticos
    df['sin_bicis'] = df['bicis_disponibles'] == 0
    df['sin_anclajes'] = df['anclajes_disponibles'] == 0
    df['inutilizada'] = df['sin_bicis'] | df['sin_anclajes']
    
    # Calculamos los minutos de cada estado para facilitar las sumas
    df['min_sin_bicis'] = df['duracion_minutos'] * df['sin_bicis']
    df['min_sin_anclajes'] = df['duracion_minutos'] * df['sin_anclajes']
    df['min_inutilizada'] = df['duracion_minutos'] * df['inutilizada']
    
    # Mapeamos los días de la semana al español
    dias_espanol = {
        'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
        'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }
    df['dia_semana'] = df['timestamp'].dt.day_name().map(dias_espanol)
    
    # Clasificamos el día entre laborable y fin de semana
    df['tipo_dia'] = df['dia_semana'].apply(lambda dia: 'Fin de semana' if dia in ['Sábado', 'Domingo'] else 'Laborable')
    
    return df

def calcular_resumen_general(df):
    # Agrupamos por estación para calcular los acumulados totales
    resumen = df.groupby(['id_estacion', 'nombre_estacion']).agg(
        total_minutos=('duracion_minutos', 'sum'),
        minutos_sin_bicis=('min_sin_bicis', 'sum'),
        minutos_sin_anclajes=('min_sin_anclajes', 'sum'),
        minutos_inutilizada=('min_inutilizada', 'sum')
    ).reset_index()
    
    # Calculamos los porcentajes de indisponibilidad
    resumen['pct_sin_bicis'] = (resumen['minutos_sin_bicis'] / resumen['total_minutos']) * 100
    resumen['pct_sin_anclajes'] = (resumen['minutos_sin_anclajes'] / resumen['total_minutos']) * 100
    resumen['pct_inutilizada'] = (resumen['minutos_inutilizada'] / resumen['total_minutos']) * 100
    
    # Ordenamos de mayor a menor tiempo inutilizada
    resumen = resumen.sort_values(by='minutos_inutilizada', ascending=False).reset_index(drop=True)
    return resumen

def calcular_resumen_por_dias(df):
    # Agrupamos por estación, día de la semana y tipo de día
    resumen_dias = df.groupby(['nombre_estacion', 'dia_semana']).agg(
        total_minutos=('duracion_minutos', 'sum'),
        minutos_sin_bicis=('min_sin_bicis', 'sum'),
        minutos_sin_anclajes=('min_sin_anclajes', 'sum'),
        minutos_inutilizada=('min_inutilizada', 'sum')
    ).reset_index()
    
    resumen_dias['pct_inutilizada'] = (resumen_dias['minutos_inutilizada'] / resumen_dias['total_minutos']) * 100
    
    # Agrupamos por tipo de día (Laborable vs Fin de semana)
    resumen_tipo_dia = df.groupby(['nombre_estacion', 'tipo_dia']).agg(
        total_minutos=('duracion_minutos', 'sum'),
        minutos_sin_bicis=('min_sin_bicis', 'sum'),
        minutos_sin_anclajes=('min_sin_anclajes', 'sum'),
        minutos_inutilizada=('min_inutilizada', 'sum')
    ).reset_index()
    
    resumen_tipo_dia['pct_inutilizada'] = (resumen_tipo_dia['minutos_inutilizada'] / resumen_tipo_dia['total_minutos']) * 100
    
    return resumen_dias, resumen_tipo_dia

def imprimir_informe(resumen_general, resumen_dias, resumen_tipo_dia):
    print("=" * 80)
    print("INFORME DE ESTACIONES INUTILIZADAS (SIN BICIS / LLENAS)")
    print("=" * 80)
    print()
    
    print("--- 1. RESUMEN GENERAL POR ESTACIÓN ---")
    tabla = resumen_general[['nombre_estacion', 'minutos_sin_bicis', 'minutos_sin_anclajes', 'minutos_inutilizada', 'pct_inutilizada']].copy()
    tabla.columns = ['Estación', 'Min. Sin Bicis', 'Min. Llena', 'Total Min. Inutil', '% Inutilizada']
    tabla['% Inutilizada'] = tabla['% Inutilizada'].round(2)
    tabla['Min. Sin Bicis'] = tabla['Min. Sin Bicis'].round(1)
    tabla['Min. Llena'] = tabla['Min. Llena'].round(1)
    tabla['Total Min. Inutil'] = tabla['Total Min. Inutil'].round(1)
    print(tabla.to_string(index=False))
    print()
    
    print("--- 2. COMPARATIVA LABORABLE VS FIN DE SEMANA ---")
    tabla_tipo = resumen_tipo_dia[['nombre_estacion', 'tipo_dia', 'minutos_inutilizada', 'pct_inutilizada']].copy()
    tabla_tipo.columns = ['Estación', 'Tipo de Día', 'Min. Inutilizada', '% Inutilizada']
    tabla_tipo['% Inutilizada'] = tabla_tipo['% Inutilizada'].round(2)
    tabla_tipo['Min. Inutilizada'] = tabla_tipo['Min. Inutilizada'].round(1)
    print(tabla_tipo.to_string(index=False))
    print()

    print("--- 3. DESGLOSE POR DÍAS DE LA SEMANA ---")
    tabla_dias = resumen_dias[['nombre_estacion', 'dia_semana', 'minutos_inutilizada', 'pct_inutilizada']].copy()
    tabla_dias.columns = ['Estación', 'Día', 'Min. Inutilizada', '% Inutilizada']
    tabla_dias['% Inutilizada'] = tabla_dias['% Inutilizada'].round(2)
    tabla_dias['Min. Inutilizada'] = tabla_dias['Min. Inutilizada'].round(1)
    print(tabla_dias.to_string(index=False))
    print()

def main():
    ruta_archivo = 'historico.csv'
    
    # Cargamos y preparamos los datos
    df = cargar_y_preparar_datos(ruta_archivo)
    
    # Calculamos los resúmenes
    resumen_general = calcular_resumen_general(df)
    resumen_dias, resumen_tipo_dia = calcular_resumen_por_dias(df)
    
    # Mostramos los resultados por consola
    imprimir_informe(resumen_general, resumen_dias, resumen_tipo_dia)
    
    # Exportamos los resúmenes a archivos CSV
    resumen_general.to_csv('resumen_estaciones_inutilizadas.csv', index=False)
    resumen_dias.to_csv('resumen_por_dias.csv', index=False)
    resumen_tipo_dia.to_csv('resumen_laborable_vs_finde.csv', index=False)
    
    print("Los resultados se han guardado en los archivos CSV:")
    print(" - resumen_estaciones_inutilizadas.csv")
    print(" - resumen_por_dias.csv")
    print(" - resumen_laborable_vs_finde.csv")

if __name__ == '__main__':
    main()
