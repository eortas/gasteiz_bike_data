import sys
import pandas as pd
import numpy as np
from cargar_historico import cargar_historico_completo

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def analizar_todos_los_dias_ventana():
    print("Analizando todos los días del dataset con la ventana de seguridad (20:00 - 04:00 - 12:00)...")
    df = cargar_historico_completo()
    df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Madrid').dt.tz_localize(None)
        
    red_por_ts = df.groupby('timestamp').agg(
        total_bicis_estaciones=('bicis_disponibles', 'sum')
    ).reset_index()
    
    red_por_ts['fecha'] = red_por_ts['timestamp'].dt.date
    fechas_unicas = sorted(red_por_ts['fecha'].unique())
    
    FLOTA_LICITADA = 50.0
    UMBRAL_85 = 42.5  # < 43 bicis
    
    registros_dias = []
    
    for fecha in fechas_unicas:
        ts_central = pd.Timestamp(fecha) + pd.Timedelta(hours=4)
        ts_minus_8h = ts_central - pd.Timedelta(hours=8)
        ts_plus_8h = ts_central + pd.Timedelta(hours=8)
        
        # Verificar que existan datos en la ventana
        if ts_minus_8h < red_por_ts['timestamp'].min() or ts_plus_8h > red_por_ts['timestamp'].max():
            continue
            
        idx_minus = (red_por_ts['timestamp'] - ts_minus_8h).abs().idxmin()
        idx_central = (red_por_ts['timestamp'] - ts_central).abs().idxmin()
        idx_plus = (red_por_ts['timestamp'] - ts_plus_8h).abs().idxmin()
        
        # Limitamos el cómputo al máximo de 50 bicicletas de la referencia contractual.
        bicis_20pm = min(red_por_ts.loc[idx_minus, 'total_bicis_estaciones'], FLOTA_LICITADA)
        bicis_4am = min(red_por_ts.loc[idx_central, 'total_bicis_estaciones'], FLOTA_LICITADA)
        bicis_12pm = min(red_por_ts.loc[idx_plus, 'total_bicis_estaciones'], FLOTA_LICITADA)
        
        promedio_ventana = (bicis_20pm + bicis_4am + bicis_12pm) / 3.0
        
        es_critico_4am = bicis_4am < UMBRAL_85
        es_critico_promedio_ventana = promedio_ventana < UMBRAL_85
        es_deficit_sostenido_3_puntos = (bicis_20pm < UMBRAL_85) and (bicis_4am < UMBRAL_85) and (bicis_12pm < UMBRAL_85)
        
        registros_dias.append({
            'Fecha': fecha.strftime('%Y-%m-%d'),
            '20:00 PM (-8h)': bicis_20pm,
            '04:00 AM (Central)': bicis_4am,
            '12:00 PM (+8h)': bicis_12pm,
            'Promedio Ventana': round(promedio_ventana, 1),
            'Critico 4 AM': es_critico_4am,
            'Critico Promedio Ventana': es_critico_promedio_ventana,
            'Deficit Sostenido (3 Puntos)': es_deficit_sostenido_3_puntos
        })
        
    df_eval = pd.DataFrame(registros_dias)
    
    total_dias = len(df_eval)
    dias_criticos_4am = df_eval['Critico 4 AM'].sum()
    dias_criticos_promedio = df_eval['Critico Promedio Ventana'].sum()
    dias_deficit_sostenido = df_eval['Deficit Sostenido (3 Puntos)'].sum()
    
    print("\n=========================================================================")
    print("   EVALUACIÓN GLOBAL DE DÍAS CRÍTICOS EN TODO EL PERÍODO (50 BICIS)")
    print("=========================================================================")
    print(f"Total de días completos analizados con ventana de 16h: {total_dias} días")
    print(f"Base de licitación: 50 bicis | Umbral 85%: {UMBRAL_85:.1f} bicis (mínimo 43 bicis)")
    print("-------------------------------------------------------------------------")
    print(f"1. Días con < 43 bicis a las 04:00 AM puntual: {dias_criticos_4am} días ({(dias_criticos_4am/total_dias)*100:.1f}%)")
    print(f"2. Días con PROMEDIO DE VENTANA < 43 bicis: {dias_criticos_promedio} días ({(dias_criticos_promedio/total_dias)*100:.1f}%)")
    print(f"3. Días con DÉFICIT SOSTENIDO en los 3 puntos (20h, 4h y 12h): {dias_deficit_sostenido} días ({(dias_deficit_sostenido/total_dias)*100:.1f}%)")
    print("=========================================================================\n")
    
    print("DETALLE DE DÍAS EVALUADOS COMO CRÍTICOS SEGÚN EL PROMEDIO DE VENTANA:")
    df_solos_criticos = df_eval[df_eval['Critico Promedio Ventana'] | df_eval['Critico 4 AM']]
    print(df_solos_criticos.to_string(index=False))

    df_eval.to_csv('resumen_evaluacion_ventana_todos_dias.csv', index=False)
    print("\n✓ Informe completo guardado en resumen_evaluacion_ventana_todos_dias.csv")

if __name__ == '__main__':
    analizar_todos_los_dias_ventana()
