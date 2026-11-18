import pandas as pd

try:
    import holidays as _holidays
    # Festivos nacionales y del País Vasco calculados por año (sin caducidad manual).
    _FESTIVOS_PV = _holidays.Spain(subdiv='PV', years=range(2024, 2036))
except Exception:  # Si la librería no está disponible, usamos la lista manual de reserva.
    _FESTIVOS_PV = None

# Festivos locales y eventos clave de Vitoria-Gasteiz que no cubre el calendario autonómico.
FESTIVOS_VITORIA = [
    '2026-01-01', # Año Nuevo
    '2026-01-06', # Reyes
    '2026-04-02', # Jueves Santo
    '2026-04-03', # Viernes Santo
    '2026-04-06', # Lunes de Pascua
    '2026-04-28', # San Prudencio (Patrón de Álava)
    '2026-05-01', # Fiesta del Trabajo
    '2026-07-25', # Día del Blusa y de la Neska / Santiago
    '2026-08-05', # Virgen Blanca (Día Grande de La Blanca)
    '2026-10-12', # Fiesta Nacional
    '2026-11-01', # Todos los Santos
    '2026-12-06', # Constitución
    '2026-12-08', # Inmaculada
    '2026-12-25'  # Navidad
]

# Festivos locales con fecha fija que no cubre el calendario autonómico.
FESTIVOS_LOCALES_ANUALES = [(4, 28), (7, 25)]

def es_festivo(fecha_dt):
    """Devuelve 1 si la fecha es un festivo oficial o domingo en Vitoria-Gasteiz."""
    fecha_dt = pd.to_datetime(fecha_dt)
    fecha_str = fecha_dt.strftime('%Y-%m-%d')
    # Comprobamos si es domingo (weekday 6) o festivo oficial registrado.
    if fecha_dt.weekday() == 6:
        return 1
    if _FESTIVOS_PV is not None and fecha_dt.date() in _FESTIVOS_PV:
        return 1
    if (fecha_dt.month, fecha_dt.day) in FESTIVOS_LOCALES_ANUALES:
        return 1
    if fecha_str in FESTIVOS_VITORIA:
        return 1
    return 0

def es_fiestas_la_blanca(fecha_dt):
    """Devuelve 1 si la fecha está entre el 4 y el 9 de agosto (Fiestas de La Blanca)."""
    mes = fecha_dt.month
    dia = fecha_dt.day
    return 1 if (mes == 8 and 4 <= dia <= 9) else 0

def es_vacaciones_universidad(fecha_dt):
    """Devuelve 1 si coincide con el periodo de receso universitario (julio y agosto)."""
    mes = fecha_dt.month
    return 1 if mes in [7, 8] else 0

def enriquecer_eventos_calendario(df):
    """Añade las columnas de eventos y festivos al DataFrame."""
    if 'timestamp' in df.columns:
        dt_col = pd.to_datetime(df['timestamp'])
        df['es_festivo'] = dt_col.apply(es_festivo)
        df['es_la_blanca'] = dt_col.apply(es_fiestas_la_blanca)
        df['es_vacaciones_upv'] = dt_col.apply(es_vacaciones_universidad)
    return df

def main():
    print("Probando módulo de eventos y festivos de Vitoria-Gasteiz...")
    fecha_test = pd.to_datetime('2026-07-25')
    print(f"Fecha {fecha_test.date()}: Festivo={es_festivo(fecha_test)}, La Blanca={es_fiestas_la_blanca(fecha_test)}, Vacaciones UPV={es_vacaciones_universidad(fecha_test)}")

if __name__ == '__main__':
    main()
