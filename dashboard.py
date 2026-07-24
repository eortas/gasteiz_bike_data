import streamlit as st
import pandas as pd
import numpy as np
import joblib

from obtener_meteo import obtener_meteo_actual
from obtener_eventos import es_festivo, es_fiestas_la_blanca, es_vacaciones_universidad
from optimizar_ruta_multiparada import (
    obtener_necesidades_estaciones,
    calcular_ruta_multiparada_optima
)

st.set_page_config(
    page_title="BiciVitoria - Redistribución Inteligente ML",
    page_icon="🚲",
    layout="wide"
)

@st.cache_resource
def cargar_modelo():
    return joblib.load('modelo_lightgbm.joblib')

@st.cache_data
def cargar_datos():
    df_features = pd.read_csv('features_historico.csv')
    df_features['timestamp'] = pd.to_datetime(df_features['timestamp'], format='ISO8601')
    df_distancias = pd.read_csv('matriz_distancias_estaciones.csv')
    return df_features, df_distancias

def main():
    st.title("🚲 BiciVitoria: Predicción ML con Clima y Calendario de Eventos")
    st.markdown("Optimización logística con **LightGBM**, **Meteorología en Tiempo Real**, **Festivos / Fiestas de La Blanca** y **Ruta VRP Multiparada**.")
    
    modelo = cargar_modelo()
    df_features, df_distancias = cargar_datos()
    
    # Obtenemos meteorología en vivo
    meteo_actual = obtener_meteo_actual()
    
    ultimo_ts = df_features['timestamp'].max()
    df_estado = df_features[df_features['timestamp'] == ultimo_ts].copy()
    
    # Inyectamos clima y eventos de la fecha actual
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
    
    # Tarjetas superiores incluyendo meteorología y calendario
    col1, col2, col3, col4, col5 = st.columns(5)
    total_estaciones = len(df_estado)
    total_bicis = int(df_estado['bicis_disponibles'].sum())
    alertas_criticas = len(df_estado[df_estado['nivel_alerta'].str.contains('CRÍTICA')])
    alertas_precaucion = len(df_estado[df_estado['nivel_alerta'].str.contains('PRECAUCIÓN')])
    
    estado_calendario = "🎉 Fiestas La Blanca" if blanca_hoy else ("🎈 Festivo" if festivo_hoy else ("🎓 Vacaciones UPV" if upv_vacaciones else "📅 Día Laborable"))
    
    col1.metric("Estaciones", f"{total_estaciones}")
    col2.metric("Bicicletas Activas", f"{total_bicis}")
    col3.metric("Alertas Críticas", f"{alertas_criticas}", delta_color="inverse")
    col4.metric("Clima en Vivo", f"{meteo_actual['temperatura']} °C", delta=f"💨 {meteo_actual['viento_kmh']} km/h")
    col5.metric("Calendario Vitoria", estado_calendario)
    
    st.divider()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚐 Circuito Multiparada Furgoneta",
        "🚨 Alertas Predictivas a 30 min",
        "🤖 Modelo ML & Importancia Variables",
        "🗺️ Tiempos & Rutas OSM"
    ])
    
    with tab1:
        st.subheader("1. Circuito Óptimo Multiparada para Furgoneta de Reparto")
        st.caption("Priorización estricta: Las alertas 🔴 CRÍTICAS son atendidas inmediatamente.")
        
        capacidad_van = st.slider("Capacidad máxima de la furgoneta de reparto (bicicletas):", 5, 20, 10)
        
        necesidades = obtener_necesidades_estaciones(modelo, df_features)
        pasos_ruta, t_total, d_total = calcular_ruta_multiparada_optima(necesidades, df_distancias, capacidad_furgoneta=capacidad_van)
        
        if pasos_ruta:
            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Tiempo Total en Carretera", f"{t_total} min")
            col_r2.metric("Distancia Total Recorrida", f"{d_total} km")
            col_r3.metric("Total de Paradas", f"{len(pasos_ruta)}")
            
            df_pasos = pd.DataFrame(pasos_ruta)
            df_pasos.columns = ['Paso', 'Estación Parada', 'Acción Recomendada', 'Bicis en Furgoneta', 'Tiempo Tramo (min)', 'Distancia (km)']
            st.dataframe(df_pasos, use_container_width=True)
        else:
            st.success("Toda la red está equilibrada. No se requiere circuito de reabastecimiento en este momento.")

    with tab2:
        st.subheader("Estado Predictivo a 30 Minutos con Contexto de Calendario y Clima")
        df_mostrar_estado = df_estado[['nombre_estacion', 'bicis_disponibles', 'capacidad', 'prediccion_30m_round', 'nivel_alerta']].copy()
        df_mostrar_estado.columns = ['Estación', 'Bicis Actuales', 'Capacidad Total', 'Predicción a 30 min', 'Nivel de Alerta']
        st.dataframe(df_mostrar_estado, use_container_width=True)

    with tab3:
        st.subheader("Evaluación e Importancia de las Variables Explicativas (LightGBM)")
        c_m1, c_m2 = st.columns(2)
        c_m1.metric("MAE (Error Medio Absoluto)", "0.44 bicis")
        c_m2.metric("RMSE (Error Cuadrático Medio)", "0.74 bicis")
        
        st.markdown("#### Importancia de Variables en el Modelo (Bicis, Hora, Temperatura y Viento)")
        importancia_data = pd.DataFrame({
            'Variable': ['Bicis Actuales', 'Hora del día', 'Temperatura (°C)', 'Viento (km/h)', 'Estación ID', 'Tendencia 30 min', 'Día de la semana', 'Anclajes libres', 'Pct Ocupación', 'Tendencia 15 min', 'Vacaciones UPV', 'Capacidad', 'Lluvia'],
            'Importancia': [1500, 1381, 1371, 1296, 841, 652, 642, 457, 279, 255, 154, 154, 18]
        })
        st.bar_chart(importancia_data.set_index('Variable'))

    with tab4:
        st.subheader("Consulta de Distancias y Tiempos OSM")
        col_o, col_d = st.columns(2)
        est_orig = col_o.selectbox("Origen:", df_distancias['estacion_origen'].unique())
        est_dest = col_d.selectbox("Destino:", df_distancias['estacion_destino'].unique())
        if est_orig != est_dest:
            m = df_distancias[(df_distancias['estacion_origen'] == est_orig) & (df_distancias['estacion_destino'] == est_dest)]
            if not m.empty:
                st.info(f"🚗 Tiempo en furgoneta: **{m.iloc[0]['tiempo_conduccion_min']} min** | 📏 Distancia: **{m.iloc[0]['distancia_km']} km**")

if __name__ == '__main__':
    main()
