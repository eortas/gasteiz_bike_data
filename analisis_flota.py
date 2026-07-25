import sys
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def analizar_flota_historica():
    print("Cargando datos históricos para análisis de disponibilidad de la flota...")
    try:
        df = pd.read_parquet('historico.parquet')
    except Exception:
        df = pd.read_csv('features_historico.csv')
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'capacidad' not in df.columns:
        df['capacidad'] = df['bicis_disponibles'] + df['anclajes_disponibles']
        
    # Agrupamos por marca de tiempo para ver la foto global de la red en cada momento
    red_por_ts = df.groupby('timestamp').agg(
        total_bicis_estaciones=('bicis_disponibles', 'sum'),
        total_anclajes_libres=('anclajes_disponibles', 'sum'),
        capacidad_total_red=('capacidad', 'sum'),
        num_estaciones=('id_estacion', 'count')
    ).reset_index()
    
    # Asumimos que la flota contratada total son 50 bicicletas
    FLOTA_CONTRATADA = 50.0
    
    # Bicicletas en uso/tránsito o mantenimiento = Flota de referencia - Bicis ancladas en estaciones
    total_max_registrado = red_por_ts['total_bicis_estaciones'].max()
    flota_referencia = max(FLOTA_CONTRATADA, total_max_registrado)
    
    red_por_ts['pct_disponibilidad_estaciones'] = (red_por_ts['total_bicis_estaciones'] / flota_referencia) * 100
    red_por_ts['bicis_en_uso_o_taller'] = np.maximum(0, flota_referencia - red_por_ts['total_bicis_estaciones'])
    
    red_por_ts['hora'] = red_por_ts['timestamp'].dt.hour
    red_por_ts['dia_semana'] = red_por_ts['timestamp'].dt.day_name()
    red_por_ts['es_noche'] = (red_por_ts['hora'] >= 23) | (red_por_ts['hora'] < 6)
    
    # Métricas globales
    media_bicis_estacionadas = red_por_ts['total_bicis_estaciones'].mean()
    mediana_bicis_estacionadas = red_por_ts['total_bicis_estaciones'].median()
    min_bicis_estacionadas = red_por_ts['total_bicis_estaciones'].min()
    max_bicis_estacionadas = red_por_ts['total_bicis_estaciones'].max()
    
    media_pct_disponibilidad = red_por_ts['pct_disponibilidad_estaciones'].mean()
    pct_tiempo_bajo_85 = (red_por_ts['pct_disponibilidad_estaciones'] < 85.0).mean() * 100
    pct_tiempo_bajo_70 = (red_por_ts['pct_disponibilidad_estaciones'] < 70.0).mean() * 100
    
    # Análisis por franja horaria (Diurno vs Nocturno)
    diurno = red_por_ts[~red_por_ts['es_noche']]
    nocturno = red_por_ts[red_por_ts['es_noche']]
    
    # Perfil horario de disponibilidad
    perfil_horario = red_por_ts.groupby('hora').agg(
        bicis_promedio=('total_bicis_estaciones', 'mean'),
        bicis_minimas=('total_bicis_estaciones', 'min'),
        pct_disponibilidad_media=('pct_disponibilidad_estaciones', 'mean'),
        bicis_en_uso_promedio=('bicis_en_uso_o_taller', 'mean')
    ).reset_index()
    
    print("\n=======================================================")
    print("   ESTUDIO DE DISPONIBILIDAD DE FLOTA DE BICICLETAS")
    print("=======================================================")
    print(f"Flota de referencia evaluada: {flota_referencia:.0f} bicicletas")
    print(f"Capacidad total sumada de las estaciones: {red_por_ts['capacidad_total_red'].mode()[0]:.0f} anclajes")
    print(f"Total de timestamps analizados: {len(red_por_ts):,} mediciones")
    print("-------------------------------------------------------")
    print(f"• Bicis promedio disponibles en estaciones: {media_bicis_estacionadas:.2f} bicis ({media_pct_disponibilidad:.1f}% de la flota)")
    print(f"• Mínimo de bicis simultáneas en estaciones: {min_bicis_estacionadas:.0f} bicis ({(min_bicis_estacionadas/flota_referencia)*100:.1f}%)")
    print(f"• Máximo de bicis simultáneas en estaciones: {max_bicis_estacionadas:.0f} bicis ({(max_bicis_estacionadas/flota_referencia)*100:.1f}%)")
    print(f"• % del tiempo que la flota en estaciones estuvo por debajo del 85%: {pct_tiempo_bajo_85:.1f}%")
    print(f"• % del tiempo que la flota en estaciones estuvo por debajo del 70%: {pct_tiempo_bajo_70:.1f}%")
    print("-------------------------------------------------------")
    print(f"• Promedio Diurno (06h - 23h): {diurno['total_bicis_estaciones'].mean():.2f} bicis en estaciones ({(diurno['total_bicis_estaciones'].mean()/flota_referencia)*100:.1f}%)")
    print(f"• Promedio Nocturno (23h - 06h): {nocturno['total_bicis_estaciones'].mean():.2f} bicis en estaciones ({(nocturno['total_bicis_estaciones'].mean()/flota_referencia)*100:.1f}%)")
    print("-------------------------------------------------------")
    print("\nPERFIL HORARIO PROMEDIO (Bicis en Estaciones vs Bicis en Uso/Tránsito):")
    for _, row in perfil_horario.iterrows():
        hora = int(row['hora'])
        bicis_est = row['bicis_promedio']
        bicis_uso = row['bicis_en_uso_promedio']
        pct_disp = row['pct_disponibilidad_media']
        bar = '█' * int(pct_disp / 5)
        print(f"Hora {hora:02d}:00 | Estaciones: {bicis_est:5.1f} bicis | En uso/taller: {bicis_uso:4.1f} | Disponibilidad: {pct_disp:5.1f}% | {bar}")

    # Guardamos los resultados resumidos en CSV
    perfil_horario.to_csv('resumen_disponibilidad_flota_horaria.csv', index=False)
    print("\n✓ Resultados exportados a resumen_disponibilidad_flota_horaria.csv")

if __name__ == '__main__':
    analizar_flota_historica()
