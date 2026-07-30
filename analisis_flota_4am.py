import sys
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def analizar_flota_4am_50_bicis():
    print("Cargando datos para el estudio de flota operativa a las 04:00 AM sobre las 50 bicis licitadas...")
    df = pd.read_csv('features_historico.csv')
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Agrupamos por marca de tiempo para obtener la foto completa de la red
    red_por_ts = df.groupby('timestamp').agg(
        total_bicis_estaciones=('bicis_disponibles', 'sum'),
        num_estaciones=('id_estacion', 'count')
    ).reset_index()
    
    red_por_ts['fecha'] = red_por_ts['timestamp'].dt.date
    red_por_ts['hora'] = red_por_ts['timestamp'].dt.hour
    red_por_ts['minuto'] = red_por_ts['timestamp'].dt.minute
    
    # Filtramos la franja nocturna de mínimo uso: entre 03:30 y 04:30.
    minuto_del_dia = red_por_ts['hora'] * 60 + red_por_ts['minuto']
    df_4am = red_por_ts[minuto_del_dia.between(210, 270)].copy()
    
    # Calculamos la foto de flota operativa por cada día a las 04:00 AM
    flota_diaria_4am = df_4am.groupby('fecha').agg(
        bicis_observadas_4am=('total_bicis_estaciones', 'median'),
        bicis_min_observadas_4am=('total_bicis_estaciones', 'min'),
        bicis_max_observadas_4am=('total_bicis_estaciones', 'max')
    ).reset_index()
    
    # FLOTA TOTAL LICITADA EN EL CONTRATO = 50 BICICLETAS EXACTAS
    FLOTA_LICITADA = 50.0
    UMBRAL_85_PCT = 0.85 * FLOTA_LICITADA  # 42.5 bicis (mínimo 43 bicis para cumplir)

    # Para la auditoría contractual, el stock computable no puede superar la flota licitada.
    flota_diaria_4am['bicis_operativas_4am'] = flota_diaria_4am['bicis_observadas_4am'].clip(upper=FLOTA_LICITADA)
    flota_diaria_4am['bicis_min_4am'] = flota_diaria_4am['bicis_min_observadas_4am'].clip(upper=FLOTA_LICITADA)
    flota_diaria_4am['bicis_max_4am'] = flota_diaria_4am['bicis_max_observadas_4am'].clip(upper=FLOTA_LICITADA)
    flota_diaria_4am['pct_flota_operativa'] = (flota_diaria_4am['bicis_operativas_4am'] / FLOTA_LICITADA) * 100
    flota_diaria_4am['cumple_85_pct'] = flota_diaria_4am['bicis_operativas_4am'] >= UMBRAL_85_PCT
    flota_diaria_4am['bicis_fuera_de_servicio'] = np.maximum(0, FLOTA_LICITADA - flota_diaria_4am['bicis_operativas_4am'])
    
    # Estadísticas resumidas a las 4:00 AM
    media_4am = flota_diaria_4am['bicis_operativas_4am'].mean()
    mediana_4am = flota_diaria_4am['bicis_operativas_4am'].median()
    min_4am = flota_diaria_4am['bicis_operativas_4am'].min()
    max_4am = flota_diaria_4am['bicis_operativas_4am'].max()
    
    pct_dias_cumplen_85 = flota_diaria_4am['cumple_85_pct'].mean() * 100
    dias_totales = len(flota_diaria_4am)
    dias_cumplen = int(flota_diaria_4am['cumple_85_pct'].sum())
    dias_incumplen = dias_totales - dias_cumplen
    
    print("\n=========================================================================")
    print("   ESTUDIO DE FLOTA OPERATIVA A LAS 04:00 AM (SOBRE 50 BICIS LICITADAS)")
    print("=========================================================================")
    print(f"Base de licitación oficial: {FLOTA_LICITADA:.0f} bicicletas")
    print(f"Umbral exigido del 85%: {UMBRAL_85_PCT:.1f} bicicletas (mínimo 43 bicis ancladas)")
    print(f"Total de días evaluados a las 04:00 AM: {dias_totales} días")
    print("-------------------------------------------------------------------------")
    print(f"• Bicis promedio ancladas a las 04:00 AM: {media_4am:.2f} bicis de 50 ({(media_4am/FLOTA_LICITADA)*100:.1f}% de la licitación)")
    print(f"• Mediana de bicis ancladas a las 04:00 AM: {mediana_4am:.1f} bicis de 50 ({(mediana_4am/FLOTA_LICITADA)*100:.1f}%)")
    print(f"• Mínimo de bicis a las 04:00 AM (peor día): {min_4am:.0f} bicis ({(min_4am/FLOTA_LICITADA)*100:.1f}%)")
    print(f"• Máximo de bicis a las 04:00 AM (mejor día): {max_4am:.0f} bicis ({(max_4am/FLOTA_LICITADA)*100:.1f}%)")
    print("-------------------------------------------------------------------------")
    print(f"• Días que CUMPLEN el 85% a las 04:00 AM (≥ 43 bicis): {dias_cumplen} de {dias_totales} días ({pct_dias_cumplen_85:.1f}%)")
    print(f"• Días que INCUMPLEN el 85% a las 04:00 AM (< 43 bicis): {dias_incumplen} de {dias_totales} días ({100 - pct_dias_cumplen_85:.1f}%)")
    print("=========================================================================\n")
    
    print("MUESTRA DEL ESTADO DE LA FLOTA LICITADA A LAS 04:00 AM POR DÍAS (Primeros 15 días):")
    print(flota_diaria_4am[['fecha', 'bicis_operativas_4am', 'pct_flota_operativa', 'bicis_fuera_de_servicio', 'cumple_85_pct']].head(15).to_string(index=False))

    flota_diaria_4am.to_csv('resumen_flota_operativa_4am_50bicis.csv', index=False)
    print("\n✓ Datos exportados a resumen_flota_operativa_4am_50bicis.csv")

if __name__ == '__main__':
    analizar_flota_4am_50_bicis()
