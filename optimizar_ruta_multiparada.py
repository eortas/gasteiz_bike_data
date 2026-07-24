import sys
import pandas as pd
import numpy as np
import joblib

# Reconfiguramos la salida de la consola a UTF-8 para Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def cargar_datos():
    modelo = joblib.load('modelo_lightgbm.joblib')
    df_distancias = pd.read_csv('matriz_distancias_estaciones.csv')
    df_features = pd.read_csv('features_historico.csv')
    df_features['timestamp'] = pd.to_datetime(df_features['timestamp'], format='ISO8601')
    return modelo, df_distancias, df_features

def obtener_necesidades_estaciones(modelo, df_features):
    ultimo_ts = df_features['timestamp'].max()
    df_estado = df_features[df_features['timestamp'] == ultimo_ts].copy()
    
    feature_cols = [
        'id_estacion', 'hora', 'dia_semana', 'es_finde', 'capacidad',
        'bicis_disponibles', 'anclajes_disponibles', 'pct_ocupacion',
        'tendencia_15m', 'tendencia_30m'
    ]
    X = df_estado[feature_cols].copy()
    X['id_estacion'] = X['id_estacion'].astype('category')
    
    # Predicción del número de bicicletas a 30 minutos vista
    df_estado['prediccion_30m'] = np.clip(modelo.predict(X), 0, df_estado['capacidad'])
    
    necesidades = {}
    for _, row in df_estado.iterrows():
        nombre = row['nombre_estacion']
        pred = row['prediccion_30m']
        cap = row['capacidad']
        
        if pred <= 2.5:
            # Asignamos nivel de urgencia: 100 para CRÍTICA (<=1.0 bici), 10 para PRECAUCIÓN (<=2.5 bicis)
            urgencia = 100 if pred <= 1.0 else 10
            meta = min(5, cap - 2)
            necesidad = int(np.ceil(meta - pred))
            if necesidad > 0:
                necesidades[nombre] = {'tipo': 'DESTINO', 'cantidad': necesidad, 'urgencia': urgencia}
        elif pred > 4.0:
            cedible = int(np.floor(pred - 3.0))
            if cedible > 0:
                necesidades[nombre] = {'tipo': 'ORIGEN', 'cantidad': cedible, 'urgencia': 0}
                
    return necesidades

def obtener_tiempo(df_distancias, orig, dest):
    if orig == dest:
        return 0.0
    filtro = (df_distancias['estacion_origen'] == orig) & (df_distancias['estacion_destino'] == dest)
    match = df_distancias[filtro]
    if not match.empty:
        return match.iloc[0]['tiempo_conduccion_min']
    return 10.0

def obtener_distancia(df_distancias, orig, dest):
    if orig == dest:
        return 0.0
    filtro = (df_distancias['estacion_origen'] == orig) & (df_distancias['estacion_destino'] == dest)
    match = df_distancias[filtro]
    if not match.empty:
        return match.iloc[0]['distancia_km']
    return 2.0

def calcular_ruta_multiparada_optima(necesidades, df_distancias, capacidad_furgoneta=10):
    if not necesidades:
        return [], 0.0, 0.0
        
    origenes = {k: v['cantidad'] for k, v in necesidades.items() if v['tipo'] == 'ORIGEN'}
    destinos = {k: {'cantidad': v['cantidad'], 'urgencia': v['urgencia']} for k, v in necesidades.items() if v['tipo'] == 'DESTINO'}
    
    if not origenes or not destinos:
        return [], 0.0, 0.0
        
    # La furgoneta empieza en la estación con mayor excedente
    estacion_actual = max(origenes, key=origenes.get)
    bicis_en_furgoneta = 0
    
    ruta_pasos = []
    tiempo_total = 0.0
    distancia_total = 0.0
    
    # Cargar en la primera estación
    cargar = min(origenes[estacion_actual], capacidad_furgoneta - bicis_en_furgoneta)
    bicis_en_furgoneta += cargar
    origenes[estacion_actual] -= cargar
    
    ruta_pasos.append({
        'paso': 1,
        'estacion': estacion_actual,
        'accion': f"CARGAR {cargar} bicis",
        'bicis_en_furgoneta': bicis_en_furgoneta,
        'tiempo_tramo_min': 0.0,
        'distancia_tramo_km': 0.0
    })
    
    paso_num = 2
    max_iteraciones = 15
    iter_count = 0
    
    while iter_count < max_iteraciones:
        iter_count += 1
        
        # Si la furgoneta tiene bicis a bordo, atiende primero las estaciones CRÍTICAS (Urgencia 100)
        destinos_pendientes = [d for d, info in destinos.items() if info['cantidad'] > 0]
        
        if bicis_en_furgoneta > 0 and destinos_pendientes:
            max_urgencia = max(destinos[d]['urgencia'] for d in destinos_pendientes)
            candidatos_urgentes = [d for d in destinos_pendientes if destinos[d]['urgencia'] == max_urgencia]
            
            # Entre las estaciones con máxima urgencia, elegimos la más cercana
            siguiente = min(candidatos_urgentes, key=lambda d: obtener_tiempo(df_distancias, estacion_actual, d))
            
            t_tramo = obtener_tiempo(df_distancias, estacion_actual, siguiente)
            d_tramo = obtener_distancia(df_distancias, estacion_actual, siguiente)
            
            descargar = min(bicis_en_furgoneta, destinos[siguiente]['cantidad'])
            bicis_en_furgoneta -= descargar
            destinos[siguiente]['cantidad'] -= descargar
            
            tiempo_total += t_tramo
            distancia_total += d_tramo
            
            ruta_pasos.append({
                'paso': paso_num,
                'estacion': siguiente,
                'accion': f"DESCARGAR {descargar} bicis",
                'bicis_en_furgoneta': bicis_en_furgoneta,
                'tiempo_tramo_min': t_tramo,
                'distancia_tramo_km': d_tramo
            })
            estacion_actual = siguiente
            paso_num += 1
            
        # Si la furgoneta está vacía o necesita cargar más bicis para cubrir las urgencias
        elif bicis_en_furgoneta < capacidad_furgoneta and any(cant > 0 for cant in origenes.values()):
            candidatos_origen = [o for o, cant in origenes.items() if cant > 0]
            siguiente = min(candidatos_origen, key=lambda o: obtener_tiempo(df_distancias, estacion_actual, o))
            
            t_tramo = obtener_tiempo(df_distancias, estacion_actual, siguiente)
            d_tramo = obtener_distancia(df_distancias, estacion_actual, siguiente)
            
            cargas_posibles = capacidad_furgoneta - bicis_en_furgoneta
            cargar = min(origenes[siguiente], cargas_posibles)
            bicis_en_furgoneta += cargar
            origenes[siguiente] -= cargar
            
            tiempo_total += t_tramo
            distancia_total += d_tramo
            
            ruta_pasos.append({
                'paso': paso_num,
                'estacion': siguiente,
                'accion': f"CARGAR {cargar} bicis",
                'bicis_en_furgoneta': bicis_en_furgoneta,
                'tiempo_tramo_min': t_tramo,
                'distancia_tramo_km': d_tramo
            })
            estacion_actual = siguiente
            paso_num += 1
        else:
            break
            
    return ruta_pasos, round(tiempo_total, 1), round(distancia_total, 2)

def main():
    print("Cargando datos y modelo predictivo LightGBM...")
    modelo, df_distancias, df_features = cargar_datos()
    
    print("Calculando necesidades predictivas con priorización de alertas...")
    necesidades = obtener_necesidades_estaciones(modelo, df_features)
    
    print("Optimizando circuito multiparada de la furgoneta de reparto...")
    pasos_ruta, tiempo_total, distancia_total = calcular_ruta_multiparada_optima(necesidades, df_distancias, capacidad_furgoneta=10)
    
    print("\n==================================================================================")
    print("CIRCUITO MULTIPARADA OPTIMO (PRIORIDAD: CRITICA -> PRECAUCION)")
    print("==================================================================================")
    
    if pasos_ruta:
        df_pasos = pd.DataFrame(pasos_ruta)
        df_pasos.columns = ['Paso', 'Estación Parada', 'Acción Recomendada', 'Bicis en Furgoneta', 'Tiempo Tramo (min)', 'Distancia (km)']
        print(df_pasos.to_string(index=False))
        
        print("\n----------------------------------------------------------------------------------")
        print(f"TIEMPO TOTAL EN CARRETERA: {tiempo_total} minutos")
        print(f"DISTANCIA TOTAL RECORRIDA: {distancia_total} km")
        print("----------------------------------------------------------------------------------")
    else:
        print("No se requiere circuito de redistribución en este momento.")

if __name__ == '__main__':
    main()
