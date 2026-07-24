import sys
import pandas as pd
import numpy as np
import joblib

# Configuramos la salida por consola para soportar caracteres UTF-8 en Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def cargar_modelo_y_matriz():
    modelo = joblib.load('modelo_lightgbm.joblib')
    df_distancias = pd.read_csv('matriz_distancias_estaciones.csv')
    return modelo, df_distancias

def obtener_ultimo_estado():
    # Cargamos el dataset con las características procesadas más recientes
    df = pd.read_csv('features_historico.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    
    # Tomamos la última lectura registrada para cada una de las 12 estaciones
    ultimo_ts = df['timestamp'].max()
    df_ultimo = df[df['timestamp'] == ultimo_ts].copy()
    return df_ultimo

def generar_predicciones_y_alertas(df_estado, modelo):
    feature_cols = [
        'id_estacion', 'hora', 'dia_semana', 'es_finde', 'capacidad',
        'bicis_disponibles', 'anclajes_disponibles', 'pct_ocupacion',
        'tendencia_15m', 'tendencia_30m'
    ]
    
    X = df_estado[feature_cols].copy()
    X['id_estacion'] = X['id_estacion'].astype('category')
    
    # Predicción del número de bicicletas a 30 minutos vista con LightGBM
    df_estado['prediccion_30m'] = np.clip(modelo.predict(X), 0, df_estado['capacidad'])
    df_estado['prediccion_30m_round'] = df_estado['prediccion_30m'].round(1)
    
    # Clasificamos el estado de alerta
    def clasificar_alerta(row):
        pred = row['prediccion_30m']
        cap = row['capacidad']
        if pred <= 1.0:
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
    
    # Identificamos estaciones destino que necesitan bicicletas (predicción a 30 min <= 2.5)
    destinos_necesitados = df_estado[df_estado['prediccion_30m'] <= 2.5].copy()
    
    # Identificamos estaciones origen que pueden ceder bicicletas (predicción a 30 min > 4.0)
    origenes_disponibles = df_estado[df_estado['prediccion_30m'] > 4.0].copy()
    
    for _, destino in destinos_necesitados.iterrows():
        nombre_dest = destino['nombre_estacion']
        pred_dest = destino['prediccion_30m']
        cap_dest = destino['capacidad']
        
        # Buscamos opciones de origen ordenadas por tiempo de conducción en furgoneta
        for _, origen in origenes_disponibles.iterrows():
            nombre_orig = origen['nombre_estacion']
            pred_orig = origen['prediccion_30m']
            
            # Consultamos el tiempo de viaje real en carretera
            filtro_dist = (df_distancias['estacion_origen'] == nombre_orig) & (df_distancias['estacion_destino'] == nombre_dest)
            match = df_distancias[filtro_dist]
            
            if not match.empty:
                tiempo_min = match.iloc[0]['tiempo_conduccion_min']
                dist_km = match.iloc[0]['distancia_km']
                
                # Calculamos el número seguro de bicis a transferir (sin dejar comprometido el origen)
                max_ceder_origen = int(np.floor(pred_orig - 3.0)) # El origen mantendrá al menos 3 bicis predichas
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
    
    # Convertimos a DataFrame y ordenamos por menor tiempo de desplazamiento
    df_recom = pd.DataFrame(recomendaciones)
    if not df_recom.empty:
        df_recom = df_recom.sort_values(by=['tiempo_conduccion_min']).reset_index(drop=True)
    return df_recom

def main():
    print("Cargando modelo LightGBM y matriz de tiempos precargada...")
    modelo, df_distancias = cargar_modelo_y_matriz()
    
    print("Obteniendo último estado registrado de las estaciones...")
    df_estado = obtener_ultimo_estado()
    
    print("Generando predicciones a 30 minutos...")
    df_estado = generar_predicciones_y_alertas(df_estado, modelo)
    
    print("\n==================================================================================")
    print("ESTADO PREDICTIVO A 30 MINUTOS EN LAS ESTACIONES")
    print("==================================================================================")
    resumen_alertas = df_estado[['nombre_estacion', 'bicis_disponibles', 'capacidad', 'prediccion_30m_round', 'nivel_alerta']].copy()
    resumen_alertas.columns = ['Estación', 'Bicis Actuales', 'Capacidad', 'Predicción (30 min)', 'Estado Alerta']
    print(resumen_alertas.to_string(index=False))
    
    print("\n==================================================================================")
    print("RECOMENDACIONES DE REDISTRIBUCIÓN INTELIGENTE (ORIGEN -> DESTINO)")
    print("==================================================================================")
    df_recom = calcular_recomendaciones_redistribucion(df_estado, df_distancias)
    if not df_recom.empty:
        print(df_recom[['estacion_origen', 'estacion_destino', 'bicis_a_mover', 'tiempo_conduccion_min', 'distancia_km']].to_string(index=False))
    else:
        print("Todas las estaciones están en equilibrio o no se requieren movimientos inmediatos.")

if __name__ == '__main__':
    main()
