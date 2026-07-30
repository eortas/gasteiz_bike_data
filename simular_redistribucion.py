import sys
import time
import json
import pandas as pd
import numpy as np
import joblib

from config import FEATURE_COLS

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Constantes de tiempos operativos para la simulación
TIEMPO_BASE_PARADA_MIN = 2.0      # Minutos fijos de maniobra (aparcamiento, apertura, verificación)
TIEMPO_POR_BICI_MIN = 1.5         # Minutos por cada bici (desanclar/anclar, transportar a furgoneta y amarrar)

def cargar_recursos():
    modelo = joblib.load('modelo_lightgbm.joblib')
    df_distancias = pd.read_csv('matriz_distancias_estaciones.csv')
    df_features = pd.read_csv('features_historico.csv')
    df_features['timestamp'] = pd.to_datetime(df_features['timestamp'], format='ISO8601')
    return modelo, df_distancias, df_features

def obtener_tiempo_conduccion(df_distancias, orig, dest):
    if orig == dest:
        return 0.0
    filtro = (df_distancias['estacion_origen'] == orig) & (df_distancias['estacion_destino'] == dest)
    match = df_distancias[filtro]
    if not match.empty:
        return match.iloc[0]['tiempo_conduccion_min']
    return 10.0

def ejecutar_simulacion(tiempo_base_parada=2.0, tiempo_por_bici=1.5, capacidad_furgoneta=10, evaluar_24h=False):
    """
    Ejecuta la simulación de redistribución de bicicletas.
    - evaluar_24h = False: Contabiliza indisponibilidad solo en horario operativo (06:00 a 23:00).
    - evaluar_24h = True: Contabiliza indisponibilidad las 24h del día, pero la furgoneta solo opera de 06:00 a 23:00.
    """
    t_inicio = time.time()
    modelo, df_distancias, df_features = cargar_recursos()
    
    # Pre-calculamos las predicciones del modelo para todo el dataset usando FEATURE_COLS
    X_all = df_features[FEATURE_COLS].copy()
    X_all['id_estacion'] = X_all['id_estacion'].astype('category')
    df_features['prediccion_30m'] = np.clip(modelo.predict(X_all), 0, df_features['capacidad'])
    
    # Ordenamos por timestamp e id_estacion
    df_features = df_features.sort_values(by=['timestamp', 'id_estacion']).reset_index(drop=True)
    
    timestamps_unicos = df_features['timestamp'].unique()
    estaciones = df_features['nombre_estacion'].unique()
    capacidades = df_features.groupby('nombre_estacion')['capacidad'].first().to_dict()
    
    # Agrupamos las filas por marca de tiempo
    grupos_por_timestamp = []
    for ts, sub_df in df_features.groupby('timestamp', sort=False):
        dt = pd.to_datetime(ts)
        filas = sub_df[['nombre_estacion', 'bicis_disponibles', 'capacidad', 'prediccion_30m']].to_dict('records')
        grupos_por_timestamp.append((dt, dt.hour, filas))
        
    # Estado inicial simulado de bicicletas por estación
    df_primer_ts = df_features[df_features['timestamp'] == timestamps_unicos[0]]
    bicis_simuladas = df_primer_ts.set_index('nombre_estacion')['bicis_disponibles'].to_dict()
    bicis_reales_prev = bicis_simuladas.copy()
    
    # Estado de la furgoneta y del operario
    bicis_en_furgoneta = 0
    minutos_operacion_restantes = 0
    accion_en_curso = None
    estacion_actual_furgoneta = estaciones[0]
    
    # Acumuladores globales
    minutos_operativos_totales = 0
    minutos_jornada_furgoneta = 0
    min_indisponible_real = 0
    min_indisponible_simulado = 0
    
    minutos_furgoneta_activa = 0
    minutos_conduccion_totales = 0
    minutos_manipulacion_totales = 0
    suma_bicis_en_furgoneta = 0
    pasos_simulacion = 0
    bicis_redistribuidas_total = 0
    
    # Acumuladores por estación
    indisponible_real_est = {est: 0.0 for est in estaciones}
    indisponible_sim_est = {est: 0.0 for est in estaciones}
    minutos_totales_est = {est: 0.0 for est in estaciones}
    
    prev_dt = None
    
    for dt, hora, filas in grupos_por_timestamp:
        es_operativo_furgoneta = (hora >= 6 and hora < 23)
        es_contabilizable = True if evaluar_24h else es_operativo_furgoneta
        
        if prev_dt is None:
            duracion_min = 5.0
        else:
            duracion_min = (dt - prev_dt).total_seconds() / 60.0
            if duracion_min <= 0 or duracion_min > 30:
                duracion_min = 5.0
        prev_dt = dt
        
        # 1. Actualizamos la demanda real del público (ocurre las 24h)
        for r in filas:
            est = r['nombre_estacion']
            b_real = r['bicis_disponibles']
            b_prev = bicis_reales_prev.get(est, b_real)
            
            delta_cliente = b_real - b_prev
            bicis_reales_prev[est] = b_real
            
            cap = capacidades[est]
            bicis_simuladas[est] = int(np.clip(bicis_simuladas[est] + delta_cliente, 0, cap))
            
        # 2. Contabilizamos la indisponibilidad en la red
        if es_contabilizable:
            minutos_operativos_totales += duracion_min * len(estaciones)
            pasos_simulacion += 1
            suma_bicis_en_furgoneta += bicis_en_furgoneta
            
            for r in filas:
                est = r['nombre_estacion']
                b_real = r['bicis_disponibles']
                cap = r['capacidad']
                minutos_totales_est[est] += duracion_min
                if b_real == 0 or b_real == cap:
                    min_indisponible_real += duracion_min
                    indisponible_real_est[est] += duracion_min
                    
            for est, b_sim in bicis_simuladas.items():
                cap = capacidades[est]
                if b_sim == 0 or b_sim == cap:
                    min_indisponible_simulado += duracion_min
                    indisponible_sim_est[est] += duracion_min
                    
        if es_operativo_furgoneta:
            minutos_jornada_furgoneta += duracion_min
            
        # 3. Simulación de la furgoneta (solo opera de 06:00 a 23:00)
        if es_operativo_furgoneta:
            if minutos_operacion_restantes > 0:
                minutos_operacion_restantes -= duracion_min
                minutos_furgoneta_activa += duracion_min
                
                if minutos_operacion_restantes <= 0:
                    if accion_en_curso:
                        tipo = accion_en_curso['tipo']
                        est = accion_en_curso['estacion']
                        cant = accion_en_curso['cantidad']
                        cap = capacidades[est]
                        
                        if tipo == 'CARGAR':
                            cargadas = min(cant, bicis_simuladas[est], capacidad_furgoneta - bicis_en_furgoneta)
                            bicis_simuladas[est] -= cargadas
                            bicis_en_furgoneta += cargadas
                        elif tipo == 'DESCARGAR':
                            descargadas = min(cant, bicis_en_furgoneta, cap - bicis_simuladas[est])
                            bicis_simuladas[est] += descargadas
                            bicis_en_furgoneta -= descargadas
                            bicis_redistribuidas_total += descargadas
                            
                        estacion_actual_furgoneta = est
                        
                    accion_en_curso = None
                    minutos_operacion_restantes = 0
            else:
                destinos = {}
                origenes = {}
                
                for r in filas:
                    est = r['nombre_estacion']
                    pred = r['prediccion_30m']
                    cap = r['capacidad']
                    b_sim = bicis_simuladas[est]

                    nivel_seguro = min(b_sim, pred)
                    if b_sim <= 1 or pred <= 2.5:
                        meta = min(5, cap - 2)
                        necesidad = min(5, int(np.ceil(meta - nivel_seguro)))
                        if necesidad > 0:
                            urgencia = 100 if b_sim <= 1 or pred <= 1.0 else 10
                            destinos[est] = {
                                'cantidad': necesidad,
                                'urgencia': urgencia
                            }
                    elif b_sim >= 5 and pred >= 5:
                        sobrante = min(int(np.floor(nivel_seguro - 5)), 4)
                        if sobrante > 0:
                            origenes[est] = sobrante
                            
                if bicis_en_furgoneta > 0 and destinos:
                    max_urgencia = max(info['urgencia'] for info in destinos.values())
                    destinos_urgentes = [
                        est for est, info in destinos.items()
                        if info['urgencia'] == max_urgencia
                    ]
                    est_destino = min(
                        destinos_urgentes,
                        key=lambda d: obtener_tiempo_conduccion(
                            df_distancias,
                            estacion_actual_furgoneta,
                            d
                        )
                    )
                    cant_b = min(bicis_en_furgoneta, destinos[est_destino]['cantidad'])
                    
                    t_conduccion = obtener_tiempo_conduccion(df_distancias, estacion_actual_furgoneta, est_destino)
                    t_manipulacion = tiempo_base_parada + (cant_b * tiempo_por_bici)
                    t_total_op = t_conduccion + t_manipulacion
                    
                    minutos_operacion_restantes = t_total_op
                    minutos_conduccion_totales += t_conduccion
                    minutos_manipulacion_totales += t_manipulacion
                    
                    accion_en_curso = {
                        'tipo': 'DESCARGAR',
                        'estacion': est_destino,
                        'cantidad': cant_b
                    }
                elif bicis_en_furgoneta < capacidad_furgoneta and origenes and destinos:
                    est_origen = min(origenes.keys(), key=lambda o: obtener_tiempo_conduccion(df_distancias, estacion_actual_furgoneta, o))
                    cargas_posibles = capacidad_furgoneta - bicis_en_furgoneta
                    cant_b = min(origenes[est_origen], cargas_posibles)
                    
                    t_conduccion = obtener_tiempo_conduccion(df_distancias, estacion_actual_furgoneta, est_origen)
                    t_manipulacion = tiempo_base_parada + (cant_b * tiempo_por_bici)
                    t_total_op = t_conduccion + t_manipulacion
                    
                    minutos_operacion_restantes = t_total_op
                    minutos_conduccion_totales += t_conduccion
                    minutos_manipulacion_totales += t_manipulacion
                    
                    accion_en_curso = {
                        'tipo': 'CARGAR',
                        'estacion': est_origen,
                        'cantidad': cant_b
                    }

    promedio_bicis_furgoneta = suma_bicis_en_furgoneta / max(pasos_simulacion, 1)
    horas_totales_servicio_red = (minutos_operativos_totales / 60.0) / len(estaciones)
    
    horas_indisponible_real = min_indisponible_real / 60.0
    horas_indisponible_sim = min_indisponible_simulado / 60.0
    
    pct_real = (min_indisponible_real / minutos_operativos_totales) * 100
    pct_sim = (min_indisponible_simulado / minutos_operativos_totales) * 100
    mejora_pct = ((min_indisponible_real - min_indisponible_simulado) / max(min_indisponible_real, 1)) * 100
    
    # Porcentaje de ocupación respecto a la jornada laboral de la furgoneta (06:00 a 23:00)
    pct_furgoneta_ocupada = (minutos_furgoneta_activa / max(minutos_jornada_furgoneta, 1.0)) * 100

    filas_est = []
    for est in estaciones:
        t_total_e = max(minutos_totales_est[est], 1.0)
        h_real = indisponible_real_est[est] / 60.0
        h_sim = indisponible_sim_est[est] / 60.0
        p_real = (indisponible_real_est[est] / t_total_e) * 100
        p_sim = (indisponible_sim_est[est] / t_total_e) * 100
        mejora_e = ((indisponible_real_est[est] - indisponible_sim_est[est]) / max(indisponible_real_est[est], 1.0)) * 100
        
        filas_est.append({
            'Estación': est,
            'Horas Inútil (Sin Sistema)': round(h_real, 1),
            'Horas Inútil (Con ML)': round(h_sim, 1),
            '% Inactiva (Sin Sistema)': round(p_real, 2),
            '% Inactiva (Con ML)': round(p_sim, 2),
            '% Mejora': round(mejora_e, 1)
        })
        
    df_estaciones_comp = pd.DataFrame(filas_est).sort_values(by='Horas Inútil (Sin Sistema)', ascending=False).reset_index(drop=True)
    tiempo_ejecucion_seg = round(time.time() - t_inicio, 2)

    resumen = {
        'evaluar_24h': evaluar_24h,
        'horas_totales_servicio_red': round(horas_totales_servicio_red, 1),
        'horas_indisponible_real': round(horas_indisponible_real, 1),
        'horas_indisponible_sim': round(horas_indisponible_sim, 1),
        'pct_real': round(pct_real, 2),
        'pct_sim': round(pct_sim, 2),
        'mejora_pct': round(mejora_pct, 1),
        'bicis_redistribuidas_total': bicis_redistribuidas_total,
        'horas_conduccion': round(minutos_conduccion_totales / 60.0, 1),
        'horas_manipulacion': round(minutos_manipulacion_totales / 60.0, 1),
        'horas_operario_total': round(minutos_furgoneta_activa / 60.0, 1),
        'pct_furgoneta_ocupada': round(pct_furgoneta_ocupada, 1),
        'promedio_bicis_furgoneta': round(promedio_bicis_furgoneta, 2),
        'df_estaciones_comp': df_estaciones_comp,
        'tiempo_ejecucion_seg': tiempo_ejecucion_seg
    }
    
    return resumen

def guardar_resumen_precalculado():
    print("Ejecutando simulación estándar (horario operativo 06:00 - 23:00)...")
    resumen_diurno = ejecutar_simulacion(evaluar_24h=False)
    
    # Exportamos los ficheros por defecto para el dashboard (horario operativo)
    df_est = resumen_diurno.pop('df_estaciones_comp')
    df_est.to_csv('resumen_simulacion_estaciones.csv', index=False)
    
    with open('resumen_simulacion_impacto.json', 'w', encoding='utf-8') as f:
        json.dump(resumen_diurno, f, indent=4, ensure_ascii=False)
        
    print("Ejecutando simulación extra (24 horas globales con furgoneta operando de 06:00 a 23:00)...")
    resumen_24h = ejecutar_simulacion(evaluar_24h=True)
    
    df_est_24h = resumen_24h.pop('df_estaciones_comp')
    df_est_24h.to_csv('resumen_simulacion_24h_estaciones.csv', index=False)
    
    with open('resumen_simulacion_24h_impacto.json', 'w', encoding='utf-8') as f:
        json.dump(resumen_24h, f, indent=4, ensure_ascii=False)
        
    print("\n✓ Ambas simulaciones (Diurna 6-23h y 24h Global) completadas y guardadas correctamente.")

def main():
    guardar_resumen_precalculado()

if __name__ == '__main__':
    main()
