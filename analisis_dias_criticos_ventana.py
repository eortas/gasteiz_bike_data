import sys
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def analizar_ventana_dias_criticos():
    print("Analizando ventana temporal (04:00 AM ± 8 horas) para días críticos...")
    try:
        df = pd.read_parquet('historico.parquet')
    except Exception:
        df = pd.read_csv('features_historico.csv')
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is not None:
        df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Madrid').dt.tz_localize(None)
    else:
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        
    # Agrupamos por marca de tiempo para ver la foto global de la red
    red_por_ts = df.groupby('timestamp').agg(
        total_bicis_estaciones=('bicis_disponibles', 'sum')
    ).reset_index()
    
    # Identificamos los días críticos (donde a las 4am había < 43 bicis)
    df_4am = red_por_ts[red_por_ts['timestamp'].dt.hour == 4].copy()
    df_4am['fecha'] = df_4am['timestamp'].dt.date
    flota_4am = df_4am.groupby('fecha')['total_bicis_estaciones'].median().reset_index()
    
    FLOTA_LICITADA = 50.0
    UMBRAL_85 = 42.5
    
    dias_criticos = flota_4am[flota_4am['total_bicis_estaciones'] < UMBRAL_85]['fecha'].tolist()
    
    resultados_criticos = []
    
    for fecha in dias_criticos:
        # Timestamp central: 04:00 AM del día crítico
        ts_central = pd.Timestamp(fecha) + pd.Timedelta(hours=4)
        
        # Ventanas: -8h (20:00 PM día anterior), 04:00 AM central, +8h (12:00 PM mediodía)
        ts_minus_8h = ts_central - pd.Timedelta(hours=8)
        ts_plus_8h = ts_central + pd.Timedelta(hours=8)
        
        # Obtenemos las bicis más cercanas a esas horas exactas
        idx_minus = (red_por_ts['timestamp'] - ts_minus_8h).abs().idxmin()
        idx_central = (red_por_ts['timestamp'] - ts_central).abs().idxmin()
        idx_plus = (red_por_ts['timestamp'] - ts_plus_8h).abs().idxmin()
        
        bicis_minus_8h = red_por_ts.loc[idx_minus, 'total_bicis_estaciones']
        bicis_4am = red_por_ts.loc[idx_central, 'total_bicis_estaciones']
        bicis_plus_8h = red_por_ts.loc[idx_plus, 'total_bicis_estaciones']
        
        es_incumplimiento_sostenido = (bicis_minus_8h < UMBRAL_85) and (bicis_4am < UMBRAL_85) and (bicis_plus_8h < UMBRAL_85)
        
        resultados_criticos.append({
            'Fecha Crítica': fecha.strftime('%Y-%m-%d'),
            '20:00 PM (-8h)': bicis_minus_8h,
            '04:00 AM (Central)': bicis_4am,
            '12:00 PM (+8h)': bicis_plus_8h,
            'Promedio Ventana (16h)': round((bicis_minus_8h + bicis_4am + bicis_plus_8h)/3.0, 1),
            'Diagnóstico 24h': 'Deficit Sostenido en Taller' if es_incumplimiento_sostenido else 'Variación de Uso Puntual'
        })
        
    df_res = pd.DataFrame(resultados_criticos)
    
    print("\n=========================================================================")
    print("  VERIFICACIÓN DE SEGURIDAD EN DÍAS CRÍTICOS (VENTANA 04:00 AM ± 8 HORAS)")
    print("=========================================================================")
    print(f"Base de licitación: {FLOTA_LICITADA:.0f} bicis | Umbral 85%: {UMBRAL_85:.1f} bicis\n")
    print(df_res.to_string(index=False))
    print("=========================================================================\n")
    
    df_res.to_csv('resumen_dias_criticos_ventana_8h.csv', index=False)
    print("✓ Resultados exportados a resumen_dias_criticos_ventana_8h.csv")

if __name__ == '__main__':
    analizar_ventana_dias_criticos()
