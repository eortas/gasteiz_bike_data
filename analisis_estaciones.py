import pandas as pd

# Definimos la franja horaria nocturna sin servicio (de 23:00 a 06:00)
HORA_INICIO_NOCHE = 23
HORA_FIN_NOCHE = 6

def cargar_y_preparar_datos(ruta_csv):
    columnas = ['timestamp', 'id_estacion', 'nombre_estacion', 'bicis_disponibles', 'anclajes_disponibles']

    # Cargamos las columnas necesarias del histórico de features actualizado.
    df = pd.read_csv(ruta_csv, usecols=columnas)
    
    # Convertimos la columna timestamp a tipo fecha (datetime)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ordenamos los datos por estación y fecha/hora
    df = df.sort_values(by=['id_estacion', 'timestamp']).reset_index(drop=True)
    
    # Identificamos las mediciones que ocurren en horario nocturno (23:00 a 06:00)
    df['es_noche'] = df['timestamp'].dt.hour.apply(lambda hora: hora >= HORA_INICIO_NOCHE or hora < HORA_FIN_NOCHE)
    
    # Calculamos la duración en minutos de cada medición usando el registro siguiente de esa estación
    df['siguiente_timestamp'] = df.groupby('id_estacion')['timestamp'].shift(-1)
    df['duracion_minutos'] = (df['siguiente_timestamp'] - df['timestamp']).dt.total_seconds() / 60.0
    
    # Evitamos contabilizar como inactividad los huecos sin mediciones.
    intervalos_validos = df['duracion_minutos'].between(0, 30, inclusive='right')
    mediana_intervalo = df.loc[intervalos_validos, 'duracion_minutos'].median()
    df.loc[~intervalos_validos, 'duracion_minutos'] = mediana_intervalo
    df['duracion_minutos'] = df['duracion_minutos'].fillna(mediana_intervalo)
    
    # Definimos el tiempo operativo durante el día (fuera del horario nocturno de 23:00 a 06:00)
    df['es_dia'] = ~df['es_noche']
    df['duracion_operativa_minutos'] = df['duracion_minutos'] * df['es_dia'].astype(int)
    
    # Identificamos los estados problemáticos
    df['sin_bicis'] = df['bicis_disponibles'] == 0
    df['sin_anclajes'] = df['anclajes_disponibles'] == 0
    df['inutilizada'] = df['sin_bicis'] | df['sin_anclajes']
    
    # Calculamos los minutos de cada estado dentro del horario operativo diurno
    df['min_sin_bicis'] = df['duracion_operativa_minutos'] * df['sin_bicis']
    df['min_sin_anclajes'] = df['duracion_operativa_minutos'] * df['sin_anclajes']
    df['min_inutilizada'] = df['duracion_operativa_minutos'] * df['inutilizada']
    
    # Mapeamos los días de la semana al español
    dias_espanol = {
        'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
        'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }
    df['dia_semana'] = df['timestamp'].dt.day_name().map(dias_espanol)
    
    # Clasificamos el día entre laborable y fin de semana
    df['tipo_dia'] = df['dia_semana'].apply(lambda dia: 'Fin de semana' if dia in ['Sábado', 'Domingo'] else 'Laborable')
    
    return df

def determinar_tipo_indisponibilidad(row):
    pct_bicis = row['pct_sin_bicis']
    pct_anclajes = row['pct_sin_anclajes']
    
    if pct_bicis == 0 and pct_anclajes == 0:
        return 'Disponible'
    elif pct_anclajes == 0:
        return 'Sin bicis'
    elif pct_bicis == 0:
        return 'Sin hueco (Llena)'
    else:
        proporcion_bicis = pct_bicis / (pct_bicis + pct_anclajes)
        if proporcion_bicis >= 0.8:
            return 'Sin bicis'
        elif proporcion_bicis <= 0.2:
            return 'Sin hueco (Llena)'
        else:
            return 'Mixta (Sin bicis / Sin hueco)'

def calcular_resumen_general(df):
    # Agrupamos por estación para calcular los acumulados en horario operativo
    resumen = df.groupby(['id_estacion', 'nombre_estacion']).agg(
        total_minutos=('duracion_operativa_minutos', 'sum'),
        minutos_sin_bicis=('min_sin_bicis', 'sum'),
        minutos_sin_anclajes=('min_sin_anclajes', 'sum'),
        minutos_inutilizada=('min_inutilizada', 'sum')
    ).reset_index()
    
    # Calculamos los porcentajes de indisponibilidad sobre el tiempo operativo
    resumen['pct_sin_bicis'] = (resumen['minutos_sin_bicis'] / resumen['total_minutos']) * 100
    resumen['pct_sin_anclajes'] = (resumen['minutos_sin_anclajes'] / resumen['total_minutos']) * 100
    resumen['pct_inutilizada'] = (resumen['minutos_inutilizada'] / resumen['total_minutos']) * 100
    
    # Determinamos la causa principal de indisponibilidad
    resumen['tipo_indisponibilidad'] = resumen.apply(determinar_tipo_indisponibilidad, axis=1)
    
    # Ordenamos de mayor a menor tiempo inutilizada
    resumen = resumen.sort_values(by='minutos_inutilizada', ascending=False).reset_index(drop=True)
    return resumen

def calcular_resumen_por_dias(df):
    # Agrupamos por estación y día de la semana
    resumen_dias = df.groupby(['nombre_estacion', 'dia_semana']).agg(
        total_minutos=('duracion_operativa_minutos', 'sum'),
        minutos_sin_bicis=('min_sin_bicis', 'sum'),
        minutos_sin_anclajes=('min_sin_anclajes', 'sum'),
        minutos_inutilizada=('min_inutilizada', 'sum')
    ).reset_index()
    
    resumen_dias['pct_inutilizada'] = (resumen_dias['minutos_inutilizada'] / resumen_dias['total_minutos']) * 100
    
    # Agrupamos por tipo de día (Laborable vs Fin de semana)
    resumen_tipo_dia = df.groupby(['nombre_estacion', 'tipo_dia']).agg(
        total_minutos=('duracion_operativa_minutos', 'sum'),
        minutos_sin_bicis=('min_sin_bicis', 'sum'),
        minutos_sin_anclajes=('min_sin_anclajes', 'sum'),
        minutos_inutilizada=('min_inutilizada', 'sum')
    ).reset_index()
    
    resumen_tipo_dia['pct_inutilizada'] = (resumen_tipo_dia['minutos_inutilizada'] / resumen_tipo_dia['total_minutos']) * 100
    
    return resumen_dias, resumen_tipo_dia

def main():
    ruta_archivo = 'features_historico.csv'
    
    # Cargamos y preparamos los datos
    df = cargar_y_preparar_datos(ruta_archivo)
    
    # Calculamos los resúmenes
    resumen_general = calcular_resumen_general(df)
    resumen_dias, resumen_tipo_dia = calcular_resumen_por_dias(df)
    
    # Exportamos los resúmenes a archivos CSV
    resumen_general.to_csv('resumen_estaciones_inutilizadas.csv', index=False)
    resumen_dias.to_csv('resumen_por_dias.csv', index=False)
    resumen_tipo_dia.to_csv('resumen_laborable_vs_finde.csv', index=False)
    
    print("Los resultados se han guardado en los archivos CSV con la clasificación del tipo de indisponibilidad.")

if __name__ == '__main__':
    main()
