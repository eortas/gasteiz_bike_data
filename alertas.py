import sys
import pandas as pd
import numpy as np
import joblib

from config import FEATURE_COLS
from demanda import cargar_demanda_estaciones, tabla_franja, valor_franja
from inferencia import construir_estado_inferencia, predecir_30m
from obtener_meteo import obtener_meteo_actual

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def cargar_modelo_y_matriz():
    modelo = joblib.load('modelo_lightgbm.joblib')
    df_distancias = pd.read_csv('matriz_distancias_estaciones.csv')
    return modelo, df_distancias


def obtener_ultimo_estado():
    """Reconstruimos el estado con la hora, calendario y tendencias actuales."""
    df = pd.read_csv('features_historico.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    return construir_estado_inferencia(df, meteo_actual=obtener_meteo_actual())


def generar_predicciones_y_alertas(df_estado, modelo):
    df_estado['prediccion_30m'] = predecir_30m(modelo, df_estado)
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


def calcular_recomendaciones_redistribucion(df_estado, df_distancias, df_demanda=None):
    """Emparejamos estaciones que necesitan bicis con estaciones que pueden cederlas.

    Incluimos como orígenes también las estaciones saturadas (sin anclajes), que
    antes recibían alerta pero nunca una recomendación de descarga. Ordenamos por
    demanda histórica de la franja y, después, por tiempo de conducción.
    """
    if df_demanda is None:
        df_demanda = cargar_demanda_estaciones()

    hora_actual = int(df_estado['hora'].iloc[0])
    es_finde_actual = int(df_estado['es_finde'].iloc[0])
    franja = tabla_franja(df_demanda, hora_actual, es_finde_actual)

    destinos_necesitados = df_estado[
        (df_estado['prediccion_30m'] <= 2.5) | (df_estado['bicis_disponibles'] <= 1)
    ].copy()
    origenes_disponibles = df_estado[
        ((df_estado['prediccion_30m'] >= 5.0) & (df_estado['bicis_disponibles'] >= 5))
        | (df_estado['anclajes_disponibles'] <= 2)
    ].copy()

    recomendaciones = []
    for _, destino in destinos_necesitados.iterrows():
        nombre_dest = destino['nombre_estacion']
        pred_dest = destino['prediccion_30m']
        cap_dest = destino['capacidad']

        for _, origen in origenes_disponibles.iterrows():
            nombre_orig = origen['nombre_estacion']
            if nombre_orig == nombre_dest:
                continue
            pred_orig = origen['prediccion_30m']

            filtro_dist = (df_distancias['estacion_origen'] == nombre_orig) & (df_distancias['estacion_destino'] == nombre_dest)
            match = df_distancias[filtro_dist]
            if match.empty:
                continue

            tiempo_min = float(match.iloc[0]['tiempo_conduccion_min'])
            dist_km = float(match.iloc[0]['distancia_km'])

            max_ceder_origen = int(np.floor(pred_orig - 5.0))
            max_recibir_destino = int(np.floor(cap_dest - pred_dest - 2.0))
            bicis_a_mover = min(max_ceder_origen, max_recibir_destino)

            if bicis_a_mover > 0:
                recomendaciones.append({
                    'estacion_origen': nombre_orig,
                    'estacion_destino': nombre_dest,
                    'bicis_a_mover': bicis_a_mover,
                    'demanda_destino': round(valor_franja(franja, nombre_dest, 'demanda_norm', 0.5), 2),
                    'tiempo_conduccion_min': tiempo_min,
                    'distancia_km': dist_km,
                    'prediccion_origen_despues': round(pred_orig - bicis_a_mover, 1),
                    'prediccion_destino_despues': round(pred_dest + bicis_a_mover, 1)
                })

    df_recom = pd.DataFrame(recomendaciones)
    if not df_recom.empty:
        df_recom = df_recom.sort_values(
            by=['demanda_destino', 'tiempo_conduccion_min'], ascending=[False, True]
        ).reset_index(drop=True)
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
    resumen_alertas = df_estado[['nombre_estacion', 'bicis_disponibles', 'capacidad',
                                 'prediccion_30m_round', 'nivel_alerta']].copy()
    resumen_alertas.columns = ['Estación', 'Bicis Actuales', 'Capacidad', 'Predicción (30 min)', 'Estado Alerta']
    print(resumen_alertas.to_string(index=False))


if __name__ == '__main__':
    main()
