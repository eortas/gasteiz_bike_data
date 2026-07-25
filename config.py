# Archivo de configuración centralizado para el proyecto gasteiz_bike_data

# Lista centralizada de variables explicativas (features) del modelo de predicción
FEATURE_COLS = [
    'id_estacion', 'hora', 'dia_semana', 'es_finde', 'capacidad',
    'bicis_disponibles', 'anclajes_disponibles', 'pct_ocupacion',
    'tendencia_15m', 'tendencia_30m',
    'temperatura', 'llueve', 'viento_kmh',
    'es_festivo', 'es_la_blanca', 'es_vacaciones_upv'
]

# Zona horaria predeterminada para el proyecto (Vitoria-Gasteiz)
DEFAULT_TIMEZONE = 'Europe/Madrid'
