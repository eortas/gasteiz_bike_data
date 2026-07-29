import os
import sys
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler
import pandas as pd
import numpy as np
import joblib

# Añadimos el directorio raíz al path para importar módulos locales
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import FEATURE_COLS
from obtener_meteo import obtener_meteo_actual
from obtener_eventos import es_festivo, es_fiestas_la_blanca, es_vacaciones_universidad
from obtener_mugibike_realtime import obtener_estaciones_mugibike_realtime
from optimizar_ruta_multiparada import (
    obtener_necesidades_estaciones,
    calcular_ruta_multiparada_optima
)

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            url_path = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(url_path.query)
            capacidad_van = int(query.get('capacidad_van', [10])[0])

            # Cargar modelo y matrices
            ruta_modelo = os.path.join(BASE_DIR, 'modelo_lightgbm.joblib')
            ruta_features = os.path.join(BASE_DIR, 'features_historico.csv')
            ruta_distancias = os.path.join(BASE_DIR, 'matriz_distancias_estaciones.csv')

            modelo = joblib.load(ruta_modelo)
            df_features = pd.read_csv(ruta_features)
            df_features['timestamp'] = pd.to_datetime(df_features['timestamp'], format='ISO8601')
            df_distancias = pd.read_csv(ruta_distancias)

            # Cargar clima y Mugibike en vivo
            meteo_actual = obtener_meteo_actual()
            df_live, exito_live = obtener_estaciones_mugibike_realtime()

            ultimo_ts = df_features['timestamp'].max()
            df_estado = df_features[df_features['timestamp'] == ultimo_ts].copy()

            if exito_live and not df_live.empty:
                mapping_bicis = df_live.set_index('nombre_estacion')['bicis_disponibles'].to_dict()
                mapping_anclajes = df_live.set_index('nombre_estacion')['anclajes_disponibles'].to_dict()

                for idx in df_estado.index:
                    est_nombre = df_estado.loc[idx, 'nombre_estacion']
                    if est_nombre in mapping_bicis:
                        b_live = mapping_bicis[est_nombre]
                        a_live = mapping_anclajes[est_nombre]
                        df_estado.loc[idx, 'bicis_disponibles'] = b_live
                        df_estado.loc[idx, 'anclajes_disponibles'] = a_live
                        cap = df_estado.loc[idx, 'capacidad']
                        df_estado.loc[idx, 'pct_ocupacion'] = b_live / max(cap, 1)

            df_estado['temperatura'] = meteo_actual['temperatura']
            df_estado['llueve'] = meteo_actual['llueve']
            df_estado['viento_kmh'] = meteo_actual['viento_kmh']

            dt_actual = df_estado['timestamp'].iloc[0]
            festivo_hoy = es_festivo(dt_actual)
            blanca_hoy = es_fiestas_la_blanca(dt_actual)
            upv_vacaciones = es_vacaciones_universidad(dt_actual)

            df_estado['es_festivo'] = festivo_hoy
            df_estado['es_la_blanca'] = blanca_hoy
            df_estado['es_vacaciones_upv'] = upv_vacaciones

            X = df_estado[FEATURE_COLS].copy()
            X['id_estacion'] = X['id_estacion'].astype('category')

            df_estado['prediccion_30m'] = np.clip(modelo.predict(X), 0, df_estado['capacidad'])
            df_estado['prediccion_30m_round'] = df_estado['prediccion_30m'].round(1)

            def clasificar_alerta(row):
                pred = row['prediccion_30m']
                cap = row['capacidad']
                if pred <= 1.0:
                    return '🔴 CRÍTICA (Vaciado inminente)'
                elif pred >= cap - 1.0:
                    return '🔴 CRÍTICA (Saturación inminente)'
                elif pred <= 2.5:
                    return '🟡 PRECAUCIÓN (Vaciado cercano)'
                elif pred >= cap - 2.5:
                    return '🟡 PRECAUCIÓN (Saturación cercana)'
                else:
                    return '🟢 NORMAL'

            df_estado['nivel_alerta'] = df_estado.apply(clasificar_alerta, axis=1)

            # Cálculo de la ruta de la furgoneta
            necesidades = obtener_necesidades_estaciones(modelo, df_features)
            pasos_ruta, t_total, d_total = calcular_ruta_multiparada_optima(necesidades, df_distancias, capacidad_furgoneta=capacidad_van)

            # Formatear lista de estaciones
            estaciones_lista = []
            for _, row in df_estado.iterrows():
                estaciones_lista.append({
                    "nombre_estacion": row['nombre_estacion'],
                    "bicis_disponibles": int(row['bicis_disponibles']),
                    "capacidad": int(row['capacidad']),
                    "prediccion_30m": float(row['prediccion_30m_round']),
                    "nivel_alerta": row['nivel_alerta']
                })

            total_estaciones = len(df_estado)
            total_bicis = int(df_estado['bicis_disponibles'].sum())
            alertas_criticas = int((df_estado['nivel_alerta'].str.contains('CRÍTICA')).sum())
            alertas_precaucion = int((df_estado['nivel_alerta'].str.contains('PRECAUCIÓN')).sum())
            estado_calendario = "🎉 Fiestas La Blanca" if blanca_hoy else ("🎈 Festivo" if festivo_hoy else "📅 Día Laborable")

            # Cargar resúmenes adicionales precalculados si existen
            inactividad_datos = []
            ruta_inutil = os.path.join(BASE_DIR, 'resumen_estaciones_inutilizadas.csv')
            if os.path.exists(ruta_inutil):
                df_inutil = pd.read_csv(ruta_inutil)
                df_inutil['horas_sin_bicis'] = (df_inutil['minutos_sin_bicis'] / 60.0).round(1)
                df_inutil['horas_sin_anclajes'] = (df_inutil['minutos_sin_anclajes'] / 60.0).round(1)
                df_inutil['horas_inutilizada'] = (df_inutil['minutos_inutilizada'] / 60.0).round(1)
                df_inutil['pct_sin_bicis'] = df_inutil['pct_sin_bicis'].round(2)
                df_inutil['pct_inutilizada'] = df_inutil['pct_inutilizada'].round(2)
                inactividad_datos = df_inutil[['nombre_estacion', 'horas_sin_bicis', 'horas_sin_anclajes', 'horas_inutilizada', 'pct_sin_bicis', 'pct_inutilizada', 'tipo_indisponibilidad']].to_dict(orient='records')

            simulacion_datos = None
            ruta_sim_json = os.path.join(BASE_DIR, 'resumen_simulacion_impacto.json')
            ruta_sim_csv = os.path.join(BASE_DIR, 'resumen_simulacion_estaciones.csv')
            if os.path.exists(ruta_sim_json) and os.path.exists(ruta_sim_csv):
                with open(ruta_sim_json, 'r', encoding='utf-8') as f:
                    simulacion_datos = json.load(f)
                df_sim_comp = pd.read_csv(ruta_sim_csv)
                simulacion_datos['df_estaciones_comp'] = df_sim_comp.to_dict(orient='records')

            flota_audit_data = None
            ruta_4am = os.path.join(BASE_DIR, 'resumen_flota_operativa_4am_50bicis.csv')
            ruta_ventana = os.path.join(BASE_DIR, 'resumen_evaluacion_ventana_todos_dias.csv')
            if os.path.exists(ruta_4am) and os.path.exists(ruta_ventana):
                df_4am = pd.read_csv(ruta_4am)
                df_vent = pd.read_csv(ruta_ventana)
                dias_tot = len(df_4am)
                cumplen = int(df_4am['cumple_85_pct'].sum())
                pct_cumplimiento = float((cumplen / max(dias_tot, 1)) * 100)
                flota_audit_data = {
                    "pct_cumplimiento": round(pct_cumplimiento, 1),
                    "dias_cumplen": cumplen,
                    "dias_totales": dias_tot,
                    "dias_deficit_sostenido": int(df_vent['Deficit Sostenido (3 Puntos)'].sum()),
                    "historico_4am": df_4am.to_dict(orient='records'),
                    "historico_ventana": df_vent.to_dict(orient='records')
                }

            respuesta = {
                "exito": True,
                "modo_realtime": exito_live and not df_live.empty,
                "resumen": {
                    "total_estaciones": total_estaciones,
                    "total_bicis": total_bicis,
                    "alertas_criticas": alertas_criticas,
                    "alertas_precaucion": alertas_precaucion,
                    "temperatura": meteo_actual['temperatura'],
                    "viento_kmh": meteo_actual['viento_kmh'],
                    "estado_calendario": estado_calendario
                },
                "estaciones": estaciones_lista,
                "ruta_furgoneta": {
                    "pasos": pasos_ruta,
                    "tiempo_total_min": t_total,
                    "distancia_total_km": d_total,
                    "num_paradas": len(pasos_ruta) if pasos_ruta else 0
                },
                "inactividad": inactividad_datos,
                "simulacion_impacto": simulacion_datos,
                "auditoria_flota": flota_audit_data
            }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(respuesta, ensure_ascii=False).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            err_payload = {"exito": False, "error": str(e)}
            self.wfile.write(json.dumps(err_payload, ensure_ascii=False).encode('utf-8'))
