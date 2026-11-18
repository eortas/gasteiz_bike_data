import sys
import pandas as pd
import numpy as np
import joblib

from demanda import cargar_demanda_estaciones, tabla_franja, valor_franja
from inferencia import construir_estado_inferencia, predecir_30m
from priorizacion import meta_relleno, score_con_tiempo, score_destino

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def cargar_datos():
    modelo = joblib.load('modelo_lightgbm.joblib')
    df_distancias = pd.read_csv('matriz_distancias_estaciones.csv')
    df_features = pd.read_csv('features_historico.csv')
    df_features['timestamp'] = pd.to_datetime(df_features['timestamp'], format='ISO8601')
    df_demanda = cargar_demanda_estaciones()
    return modelo, df_distancias, df_features, df_demanda


def obtener_necesidades_estaciones(modelo, df_features, df_estado_actual=None, df_demanda=None):
    """Calculamos qué estaciones necesitan o ceden bicis, ya priorizadas por uso.

    La urgencia deja de ser binaria: combinamos criticidad, demanda histórica de
    la franja y déficit previsto. La meta de relleno también escala con la salida
    neta esperada de la estación en esa hora.
    """
    if df_estado_actual is not None:
        df_estado = df_estado_actual.copy()
        if 'prediccion_30m' not in df_estado.columns:
            df_estado['prediccion_30m'] = predecir_30m(modelo, df_estado)
    else:
        # Sin estado explícito reconstruimos el estado real con la hora y las
        # tendencias del instante actual, no con la instantánea nocturna.
        df_estado = construir_estado_inferencia(df_features)
        df_estado['prediccion_30m'] = predecir_30m(modelo, df_estado)

    if df_demanda is None:
        df_demanda = cargar_demanda_estaciones()

    hora_actual = int(df_estado['hora'].iloc[0])
    es_finde_actual = int(df_estado['es_finde'].iloc[0])
    franja = tabla_franja(df_demanda, hora_actual, es_finde_actual)

    necesidades = {}
    for _, row in df_estado.iterrows():
        nombre = row['nombre_estacion']
        pred = row['prediccion_30m']
        capacidad = int(row['capacidad'])
        bicis_actuales = float(row['bicis_disponibles'])
        nivel_seguro = min(bicis_actuales, pred)

        demanda_norm = valor_franja(franja, nombre, 'demanda_norm', 0.5)
        salida_neta = valor_franja(franja, nombre, 'salida_neta_hora', 0.0)

        if bicis_actuales <= 1 or pred <= 2.5:
            criticidad = 100.0 if (bicis_actuales <= 1 or pred <= 1.0) else 50.0
            meta = meta_relleno(capacidad, salida_neta)
            deficit = max(0.0, (meta - nivel_seguro) / meta) if meta > 0 else 0.0
            cantidad = int(np.ceil(meta - nivel_seguro))
            if cantidad > 0:
                necesidades[nombre] = {
                    'tipo': 'DESTINO',
                    'cantidad': cantidad,
                    'urgencia': criticidad,
                    'demanda': round(demanda_norm, 3),
                    'score_base': score_destino(criticidad, demanda_norm, deficit),
                    'bicis_actuales': bicis_actuales,
                    'capacidad': capacidad,
                    'prediccion_30m': float(pred),
                    'meta': meta
                }
        elif bicis_actuales >= 5.0 and pred >= 5.0:
            cedible = int(np.floor(nivel_seguro - 5.0))
            if cedible > 0:
                necesidades[nombre] = {
                    'tipo': 'ORIGEN',
                    'cantidad': cedible,
                    'urgencia': 0.0,
                    'demanda': round(demanda_norm, 3),
                    'bicis_actuales': bicis_actuales,
                    'capacidad': capacidad,
                    'prediccion_30m': float(pred)
                }

    return necesidades


def obtener_tiempo(df_distancias, orig, dest):
    if orig == dest:
        return 0.0
    filtro = (df_distancias['estacion_origen'] == orig) & (df_distancias['estacion_destino'] == dest)
    match = df_distancias[filtro]
    if not match.empty:
        return float(match.iloc[0]['tiempo_conduccion_min'])
    print(f"Aviso: no hay tiempo de conducción para {orig} -> {dest}; usamos 10 min.")
    return 10.0


def obtener_distancia(df_distancias, orig, dest):
    if orig == dest:
        return 0.0
    filtro = (df_distancias['estacion_origen'] == orig) & (df_distancias['estacion_destino'] == dest)
    match = df_distancias[filtro]
    if not match.empty:
        return float(match.iloc[0]['distancia_km'])
    print(f"Aviso: no hay distancia para {orig} -> {dest}; usamos 2 km.")
    return 2.0


def calcular_ruta_multiparada_optima(necesidades, df_distancias, capacidad_furgoneta=10,
                                     tiempo_max_ruta_min=None):
    """Construimos el circuito multiparada con priorización continua por uso.

    En cada paso elegimos el destino con mejor puntuación (criticidad, demanda y
    déficit menos coste de desplazamiento) y verificamos que haya hueco real.
    """
    if not necesidades:
        return [], 0.0, 0.0

    origenes = {k: dict(v) for k, v in necesidades.items() if v['tipo'] == 'ORIGEN'}
    destinos = {k: dict(v) for k, v in necesidades.items() if v['tipo'] == 'DESTINO'}
    if not origenes or not destinos:
        return [], 0.0, 0.0

    tiempo_max_red = max(float(df_distancias['tiempo_conduccion_min'].max()), 1.0)
    capacidades = {n: int(v.get('capacidad', 0)) for n, v in necesidades.items()}
    estado = {n: int(v.get('bicis_actuales', 0)) for n, v in necesidades.items()}

    # Arrancamos por el origen más próximo a la red de destinos, no por el que
    # más bicis cede: evitamos cruzar la ciudad antes de la primera parada.
    estacion_actual = min(
        origenes.keys(),
        key=lambda origen: min(obtener_tiempo(df_distancias, origen, d) for d in destinos)
    )

    demanda_total = sum(info['cantidad'] for info in destinos.values())
    cargar = min(origenes[estacion_actual]['cantidad'], capacidad_furgoneta, demanda_total)
    if cargar <= 0:
        return [], 0.0, 0.0

    bicis_en_furgoneta = cargar
    origenes[estacion_actual]['cantidad'] -= cargar
    estado[estacion_actual] -= cargar

    ruta_pasos = [{
        'paso': 1,
        'estacion': estacion_actual,
        'accion': f"CARGAR {cargar} bicis",
        'bicis_en_furgoneta': bicis_en_furgoneta,
        'tiempo_tramo_min': 0.0,
        'distancia_tramo_km': 0.0
    }]

    tiempo_total = 0.0
    distancia_total = 0.0
    paso_num = 2

    for _ in range(15):
        if tiempo_max_ruta_min is not None and tiempo_total >= tiempo_max_ruta_min:
            break

        pendientes = [d for d, info in destinos.items() if info['cantidad'] > 0]

        if bicis_en_furgoneta > 0 and pendientes:
            candidatos_destino = [d for d in pendientes if capacidades.get(d, 0) - estado.get(d, 0) > 0]
            if not candidatos_destino:
                break

            def prioridad_destino(nombre):
                tiempo = obtener_tiempo(df_distancias, estacion_actual, nombre)
                return score_con_tiempo(destinos[nombre]['score_base'], tiempo, tiempo_max_red)

            siguiente = max(candidatos_destino, key=prioridad_destino)
            t_tramo = obtener_tiempo(df_distancias, estacion_actual, siguiente)
            d_tramo = obtener_distancia(df_distancias, estacion_actual, siguiente)

            hueco = capacidades[siguiente] - estado[siguiente]
            descargar = min(bicis_en_furgoneta, destinos[siguiente]['cantidad'], hueco)
            if descargar <= 0:
                break

            bicis_en_furgoneta -= descargar
            destinos[siguiente]['cantidad'] -= descargar
            estado[siguiente] += descargar

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

        elif bicis_en_furgoneta < capacidad_furgoneta:
            demanda_pendiente = sum(destinos[d]['cantidad'] for d in pendientes)
            if demanda_pendiente <= bicis_en_furgoneta:
                break
            candidatos_origen = [o for o, info in origenes.items() if info['cantidad'] > 0]
            if not candidatos_origen:
                break

            siguiente = min(
                candidatos_origen,
                key=lambda origen: obtener_tiempo(df_distancias, estacion_actual, origen)
            )
            t_tramo = obtener_tiempo(df_distancias, estacion_actual, siguiente)
            d_tramo = obtener_distancia(df_distancias, estacion_actual, siguiente)

            espacio = capacidad_furgoneta - bicis_en_furgoneta
            cargar = min(origenes[siguiente]['cantidad'], espacio, demanda_pendiente - bicis_en_furgoneta)
            if cargar <= 0:
                break

            bicis_en_furgoneta += cargar
            origenes[siguiente]['cantidad'] -= cargar
            estado[siguiente] -= cargar

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

    # Garantizamos que la furgoneta no se quede con bicicletas al terminar la ruta:
    # descargamos en la estación más cercana CON HUECO, nunca a ciegas en la actual.
    if bicis_en_furgoneta > 0:
        con_hueco = [n for n in capacidades if capacidades[n] - estado[n] >= bicis_en_furgoneta]
        if not con_hueco:
            con_hueco = [n for n in capacidades if capacidades[n] - estado[n] > 0]

        if con_hueco:
            cierre = min(con_hueco, key=lambda n: obtener_tiempo(df_distancias, estacion_actual, n))
            t_tramo = obtener_tiempo(df_distancias, estacion_actual, cierre)
            d_tramo = obtener_distancia(df_distancias, estacion_actual, cierre)
            hueco = capacidades[cierre] - estado[cierre]
            descargar = min(bicis_en_furgoneta, hueco)

            if descargar > 0:
                if cierre != estacion_actual:
                    tiempo_total += t_tramo
                    distancia_total += d_tramo
                estado[cierre] += descargar
                bicis_en_furgoneta -= descargar
                ruta_pasos.append({
                    'paso': paso_num,
                    'estacion': cierre,
                    'accion': f"DESCARGAR {descargar} bicis (cierre de ruta)",
                    'bicis_en_furgoneta': bicis_en_furgoneta,
                    'tiempo_tramo_min': t_tramo if cierre != estacion_actual else 0.0,
                    'distancia_tramo_km': d_tramo if cierre != estacion_actual else 0.0
                })
                estacion_actual = cierre
                paso_num += 1

        if bicis_en_furgoneta > 0:
            ruta_pasos.append({
                'paso': paso_num,
                'estacion': estacion_actual,
                'accion': f"RETENER {bicis_en_furgoneta} bicis (sin hueco en la red)",
                'bicis_en_furgoneta': bicis_en_furgoneta,
                'tiempo_tramo_min': 0.0,
                'distancia_tramo_km': 0.0
            })

    return ruta_pasos, round(tiempo_total, 1), round(distancia_total, 2)


def main():
    print("Cargando datos y modelo predictivo LightGBM...")
    modelo, df_distancias, df_features, df_demanda = cargar_datos()
    necesidades = obtener_necesidades_estaciones(modelo, df_features, None, df_demanda)
    pasos_ruta, tiempo_total, distancia_total = calcular_ruta_multiparada_optima(
        necesidades, df_distancias, capacidad_furgoneta=10
    )

    if pasos_ruta:
        df_pasos = pd.DataFrame(pasos_ruta)
        df_pasos.columns = ['Paso', 'Estación Parada', 'Acción Recomendada',
                            'Bicis en Furgoneta', 'Tiempo Tramo (min)', 'Distancia (km)']
        print(df_pasos.to_string(index=False))
        print(f"\nTiempo total: {tiempo_total} min | Distancia total: {distancia_total} km")


if __name__ == '__main__':
    main()
