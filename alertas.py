import sys
import pandas as pd
import numpy as np
import joblib

from obtener_meteo import obtener_meteo_actual
from obtener_eventos import es_festivo, es_fiestas_la_blanca, es_vacaciones_universidad

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def cargar_modelo_y_matriz():
    modelo = joblib.load('modelo_lightgbm.joblib')
    df_distancias = pd.read_csv('matriz_distancias_estaciones.csv')
    return modelo, df_distancias

def obtener_ultimo_estado():
    df = pd.read_csv('features_historico.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    ultimo_ts = df['timestamp'].max()
    df_ultimo = df[df['timestamp'] == ultimo_ts].copy()
    return df_ultimo

def generar_predicciones_y_alertas(df_estado, modelo):
    meteo_actual = obtener_meteo_actual()
    df_estado['temperatura'] = meteo_actual['temperatura']
    df_estado['llueve'] = meteo_actual['llueve']
    df_estado['viento_kmh'] = meteo_actual['viento_kmh']
    
    dt_actual = df_estado['timestamp'].iloc[0]
    df_estado['es_festivo'] = es_festivo(dt_actual)
    df_estado['es_la_blanca'] = es_fiestas_la_blanca(dt_actual)
    df_estado['es_vacaciones_upv'] = es_vacaciones_universidad(dt_actual)
    
    feature_cols = [
        'id_estacion', 'hora', 'dia_semana', 'es_finde', 'capacidad',
        'bicis_disponibles', 'anclajes_disponibles', 'pct_ocupacion',
        'tendencia_15m', 'tendencia_30m',
        'temperatura', 'llueve', 'viento_kmh',
        'es_festivo', 'es_la_blanca', 'es_vacaciones_upv'
    ]
    
    X = df_estado[feature_cols].copy()
    X['id_estacion'] = X['id_estacion'].astype('category')
    
    df_estado['prediccion_30m'] = np.clip(modelo.predict(X), 0, df_estado['capacidad'])
    df_estado['prediccion_30m_round'] = df_estado['prediccion_30m'].round(1)
    
    def clasificar_alerta(row):
        pred = row['prediccion_30m']
        cap = row['capacidad']
        bicis_actuales = row['bicis_disponibles']
        anclajes_actuales = row['anclajes_disponibles']

        # El estado actual es más urgente que cualquier predicción.
        if bicis_actuales <= 1:
            return 'CRITICA (Sin bicicletas ahora)'
        elif anclajes_actuales <= 1:
            return 'CRITICA (Sin anclajes ahora)'
        elif pred <= 1.0:
            return 'CRITICA (Vaciado inminente)'
        elif pred >= cap - 1.0:
            return 'CRITICA (Saturacion inminente)'
        elif pred <= 2.5:
            return 'PRECAUCION (Vaciado cercano)'
        elif pred >= cap - 2.5:
            return 'PRECAUCION (Saturacion cercana)'
        else:
            return 'NORMAL'
            
    df_estado['nivel_alerta'] = df_estado.apply(clasificar_alerta, axis=1)
    return df_estado

def calcular_recomendaciones_redistribucion(df_estado, df_distancias):
    recomendaciones = []
    destinos_necesitados = df_estado[df_estado['prediccion_30m'] <= 2.5].copy()
    origenes_disponibles = df_estado[df_estado['prediccion_30m'] >= 5.0].copy()
    
    for _, destino in destinos_necesitados.iterrows():
        nombre_dest = destino['nombre_estacion']
        pred_dest = destino['prediccion_30m']
        cap_dest = destino['capacidad']
        
        for _, origen in origenes_disponibles.iterrows():
            nombre_orig = origen['nombre_estacion']
            pred_orig = origen['prediccion_30m']
            
            filtro_dist = (df_distancias['estacion_origen'] == nombre_orig) & (df_distancias['estacion_destino'] == nombre_dest)
            match = df_distancias[filtro_dist]
            
            if not match.empty:
                tiempo_min = match.iloc[0]['tiempo_conduccion_min']
                dist_km = match.iloc[0]['distancia_km']
                
                max_ceder_origen = int(np.floor(pred_orig - 5.0))
                max_recibir_destino = int(np.floor(cap_dest - pred_dest - 2.0))
                bicis_a_mover = min(max_ceder_origen, max_recibir_destino)
                
                if bicis_a_mover > 0:
                    recomendaciones.append({
                        'estacion_origen': nombre_orig,
                        'estacion_destino': nombre_dest,
                        'bicis_a_mover': bicis_a_mover,
                        'tiempo_conduccion_min': tiempo_min,
                        'distancia_km': dist_km,
                        'prediccion_origen_despues': round(pred_orig - bicis_a_mover, 1),
                        'prediccion_destino_despues': round(pred_dest + bicis_a_mover, 1)
                    })
    
    df_recom = pd.DataFrame(recomendaciones)
    if not df_recom.empty:
        df_recom = df_recom.sort_values(by=['tiempo_conduccion_min']).reset_index(drop=True)
    return df_recom

def main():
    print("Cargando modelo LightGBM enriquecido con festivos y eventos...")
    modelo, df_distancias = cargar_modelo_y_matriz()
    
    print("Obteniendo estado actual e integrando calendario en vivo...")
    df_estado = obtener_ultimo_estado()
    df_estado = generar_predicciones_y_alertas(df_estado, modelo)
    
    print("\n==================================================================================")
    print("ESTADO PREDICTIVO A 30 MINUTOS EN LAS ESTACIONES")
    print("==================================================================================")
    resumen_alertas = df_estado[['nombre_estacion', 'bicis_disponibles', 'capacidad', 'prediccion_30m_round', 'nivel_alerta']].copy()
    resumen_alertas.columns = ['Estación', 'Bicis Actuales', 'Capacidad', 'Predicción (30 min)', 'Estado Alerta']
    print(resumen_alertas.to_string(index=False))

if __name__ == '__main__':
    main()
