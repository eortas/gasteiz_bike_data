import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from config import FEATURE_COLS
from obtener_meteo import obtener_meteo_actual
from obtener_eventos import es_festivo, es_fiestas_la_blanca, es_vacaciones_universidad
from optimizar_ruta_multiparada import (
    obtener_necesidades_estaciones,
    calcular_ruta_multiparada_optima
)

st.set_page_config(
    page_title="Mugibike - Redistribución Inteligente ML",
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

@st.cache_data(ttl=300)
def cargar_meteo():
    return obtener_meteo_actual()

@st.cache_data
def cargar_resumenes_inactividad():
    df_inutil = pd.read_csv('resumen_estaciones_inutilizadas.csv')
    df_finde = pd.read_csv('resumen_laborable_vs_finde.csv')
    return df_inutil, df_finde

@st.cache_data
def cargar_resumen_simulacion_precalculado():
    ruta_json = 'resumen_simulacion_impacto.json'
    ruta_csv = 'resumen_simulacion_estaciones.csv'
    if os.path.exists(ruta_json) and os.path.exists(ruta_csv):
        with open(ruta_json, 'r', encoding='utf-8') as f:
            resumen = json.load(f)
        resumen['df_estaciones_comp'] = pd.read_csv(ruta_csv)
        return resumen
    return None

def main():
    st.title("🚲 Mugibike: Redistribución Inteligente y Análisis de Red")
    st.markdown("Sistema de gestión inteligente para **Mugibike**: predice la ocupación a 30 minutos vista (considerando clima en tiempo real y calendario de eventos) y **calcula rutas óptimas de reparto** para evitar estaciones vacías o saturadas, garantizando que toda la red permanezca operativa.")
    
    modelo = cargar_modelo()
    df_features, df_distancias = cargar_datos()
    
    # Obtenemos meteorología en vivo con caché de 5 minutos
    meteo_actual = cargar_meteo()
    
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
    
    # Tarjetas superiores incluyendo meteorología y calendario
    col1, col2, col3, col4, col5 = st.columns(5)
    total_estaciones = len(df_estado)
    total_bicis = int(df_estado['bicis_disponibles'].sum())
    alertas_criticas = len(df_estado[df_estado['nivel_alerta'].str.contains('CRÍTICA')])
    alertas_precaucion = len(df_estado[df_estado['nivel_alerta'].str.contains('PRECAUCIÓN')])
    
    estado_calendario = "🎉 Fiestas La Blanca" if blanca_hoy else ("🎈 Festivo" if festivo_hoy else "📅 Día Laborable")
    
    col1.metric("Estaciones", f"{total_estaciones}")
    col2.metric("Bicicletas Activas", f"{total_bicis}")
    col3.metric("Alertas Críticas", f"{alertas_criticas}", delta_color="inverse")
    
    titulo_clima = "Clima (Estimado)" if meteo_actual.get('es_fallback', False) else "Clima en Vivo"
    col4.metric(titulo_clima, f"{meteo_actual['temperatura']} °C", delta=f"💨 {meteo_actual['viento_kmh']} km/h")
    col5.metric("Calendario Vitoria", estado_calendario)
    
    st.divider()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚨 Alertas Predictivas a 30 min",
        "🚲 Redistribución bicicletas ML",
        "⚠️ Análisis de Tiempo Inactivo por Estación",
        "📈 Simulación & Evaluación de Impacto",
        "📊 Auditoría de Flota (50 Bicis & Ventana 24h)"
    ])
    
    with tab1:
        st.subheader("Estado Predictivo a 30 Minutos con Contexto de Calendario y Clima")
        df_mostrar_estado = df_estado[['nombre_estacion', 'bicis_disponibles', 'capacidad', 'prediccion_30m_round', 'nivel_alerta']].copy()
        df_mostrar_estado.columns = ['Estación', 'Bicis Actuales', 'Capacidad Total', 'Predicción a 30 min', 'Nivel de Alerta']
        st.dataframe(df_mostrar_estado, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Redistribución de Bicicletas con ML (Circuito Óptimo Multiparada)")
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
            st.dataframe(df_pasos, use_container_width=True, hide_index=True)
        else:
            st.success("Toda la red está equilibrada. No se requiere circuito de reabastecimiento en este momento.")

    with tab3:
        st.subheader("Análisis de Tiempo Inactivo por Falta de Bicicletas o Anclajes")
        st.caption("Contabilizado únicamente dentro del horario operativo diurno (excluyendo el horario nocturno de 23:00 a 06:00).")
        
        df_inutil, df_finde = cargar_resumenes_inactividad()
        
        df_inutil['horas_sin_bicis'] = (df_inutil['minutos_sin_bicis'] / 60.0).round(1)
        df_inutil['horas_sin_anclajes'] = (df_inutil['minutos_sin_anclajes'] / 60.0).round(1)
        df_inutil['horas_inutilizada'] = (df_inutil['minutos_inutilizada'] / 60.0).round(1)
        df_inutil['pct_sin_bicis'] = df_inutil['pct_sin_bicis'].round(2)
        
        top_sin_bicis = df_inutil.sort_values(by='pct_sin_bicis', ascending=False).iloc[0]
        promedio_pct_inactiva = df_inutil['pct_inutilizada'].mean()
        total_estaciones_afectadas = len(df_inutil[df_inutil['pct_inutilizada'] > 0])
        
        c_i1, c_i2, c_i3 = st.columns(3)
        c_i1.metric("Estación con Más Tiempo Sin Bicis", top_sin_bicis['nombre_estacion'], f"{top_sin_bicis['pct_sin_bicis']:.1f}% del tiempo", delta_color="inverse")
        c_i2.metric("Promedio Inactividad Red", f"{promedio_pct_inactiva:.1f}% del tiempo operativo")
        c_i3.metric("Estaciones Afectadas por Inactividad", f"{total_estaciones_afectadas} de {len(df_inutil)}")
        
        st.divider()
        
        st.markdown("#### 📊 Top 10 Estaciones con Mayor Porcentaje de Tiempo Vacías (% Sin Bicicletas)")
        top10_sin_bicis = df_inutil.sort_values(by='pct_sin_bicis', ascending=False).head(10)
        st.bar_chart(top10_sin_bicis.set_index('nombre_estacion')['pct_sin_bicis'])
        
        st.divider()
        
        st.markdown("#### 📋 Detalle de Inactividad e Indisponibilidad por Estación")
        filtro_tipo = st.selectbox(
            "Filtrar por Diagnóstico de Indisponibilidad:",
            options=["Todas", "Sin bicis", "Sin hueco (Llena)", "Mixta (Sin bicis / Sin hueco)", "Disponible"]
        )
        
        df_filtrado = df_inutil.copy()
        if filtro_tipo != "Todas":
            df_filtrado = df_filtrado[df_filtrado['tipo_indisponibilidad'] == filtro_tipo]
            
        df_tabla_inactiva = df_filtrado[['nombre_estacion', 'horas_sin_bicis', 'horas_sin_anclajes', 'horas_inutilizada', 'pct_inutilizada', 'tipo_indisponibilidad']].copy()
        df_tabla_inactiva.columns = ['Estación', 'Horas Sin Bicis', 'Horas Sin Hueco', 'Horas Inútil Total', '% Tiempo Inoperativa', 'Diagnóstico']
        df_tabla_inactiva['% Tiempo Inoperativa'] = df_tabla_inactiva['% Tiempo Inoperativa'].round(2)
        
        st.dataframe(df_tabla_inactiva, use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("📈 Evaluación de Impacto y Mejora de Disponibilidad (Backtest Mensual)")
        st.markdown("Estudio de impacto basado en los **datos históricos de todo el mes**, incorporando **tiempos reales de desplazamiento**, maniobra base del operario en cada parada y **tiempo de manipulación física por bicicleta**.")
        
        # Leemos exclusivamente los datos estáticos precalculados
        resumen_sim = cargar_resumen_simulacion_precalculado()
        
        if resumen_sim:
            st.info("⏱️ **Parámetros del estudio operativo**: Parada base = 2,0 min | Manipulación = 1,5 min/bici | Capacidad furgoneta = 10 bicis")
            
            st.divider()
            
            # Tarjetas principales de impacto
            c_sim1, c_sim2, c_sim3, c_sim4 = st.columns(4)
            c_sim1.metric("Indisponibilidad Histórica Real", f"{resumen_sim['horas_indisponible_real']} h", f"{resumen_sim['pct_real']}% del tiempo", delta_color="inverse")
            c_sim2.metric("Indisponibilidad Con Sistema ML", f"{resumen_sim['horas_indisponible_sim']} h", f"{resumen_sim['pct_sim']}% del tiempo", delta_color="normal")
            c_sim3.metric("Mejora Neto Disponibilidad", f"-{resumen_sim['mejora_pct']}%", "Reducción de fallos", delta_color="normal")
            c_sim4.metric("Bicicletas Redistribuidas", f"{resumen_sim['bicis_redistribuidas_total']} bicis/mes")
            
            st.divider()
            
            # Desglose del operario y uso de furgoneta
            st.markdown("#### ⏱️ Análisis del Tiempo de Servicio y Carga del Operario")
            c_op1, c_op2, c_op3, c_op4 = st.columns(4)
            c_op1.metric("Tiempo Conducción", f"{resumen_sim['horas_conduccion']} h/mes")
            c_op2.metric("Tiempo Maniobra & Carga", f"{resumen_sim['horas_manipulacion']} h/mes")
            c_op3.metric("Tiempo Activo Operario", f"{resumen_sim['horas_operario_total']} h/mes", f"{resumen_sim['pct_furgoneta_ocupada']}% de la jornada")
            c_op4.metric("Carga Media en Furgoneta", f"{resumen_sim['promedio_bicis_furgoneta']} bicis", "Inventario en tránsito")
            
            st.caption(f"💡 **Conclusión Operativa**: Las bicicletas **no se quedan retenidas** en la furgoneta (promedio < 1 bici). El operario permanece activo en ruta el **{resumen_sim['pct_furgoneta_ocupada']}%** de la jornada, por lo que la operativa se cubre holgadamente con una única furgoneta.")
            
            st.divider()
            
            # Gráfico comparativo por estación
            st.markdown("#### 📊 Comparativa de Indisponibilidad por Estación: Histórico Real vs Simulación ML")
            df_comp = resumen_sim['df_estaciones_comp']
            
            chart_data = df_comp.set_index('Estación')[['% Inactiva (Sin Sistema)', '% Inactiva (Con ML)']]
            st.bar_chart(chart_data)
            
            st.markdown("#### 📋 Detalle Comparativo por Estación")
            st.dataframe(df_comp, use_container_width=True, hide_index=True)
        else:
            st.warning("No se encontraron los datos precalculados de simulación.")

    with tab5:
        st.subheader("📋 Auditoría de Disponibilidad de Flota de Bicicletas (Licitación Municipal: 50 Bicis)")
        st.caption("Control automático del cumplimiento del umbral mínimo del 85% de flota operativa exigido por el Ayuntamiento de Vitoria-Gasteiz.")
        
        ruta_csv_4am = 'resumen_flota_operativa_4am_50bicis.csv'
        ruta_csv_ventana = 'resumen_evaluacion_ventana_todos_dias.csv'
        
        if os.path.exists(ruta_csv_4am) and os.path.exists(ruta_csv_ventana):
            df_4am_audit = pd.read_csv(ruta_csv_4am)
            df_ventana_audit = pd.read_csv(ruta_csv_ventana)
            
            dias_totales = len(df_4am_audit)
            dias_cumplen_85 = df_4am_audit['cumple_85_pct'].sum()
            pct_cumplimiento = (dias_cumplen_85 / dias_totales) * 100 if dias_totales > 0 else 0
            
            dias_criticos_promedio = df_ventana_audit['Critico Promedio Ventana'].sum()
            dias_deficit_sostenido = df_ventana_audit['Deficit Sostenido (3 Puntos)'].sum()
            
            c_f1, c_f2, c_f3, c_f4 = st.columns(4)
            c_f1.metric("Flota Licitada Oficial", "50 bicis", help="Número de bicicletas eléctricas del contrato municipal")
            c_f2.metric("Umbral 85% Exigido", "42.5 bicis", help="Mínimo 43 bicicletas ancladas listas para el servicio a las 04:00 AM")
            c_f3.metric("% Cumplimiento 4 AM", f"{pct_cumplimiento:.1f}%", f"{dias_cumplen_85}/{dias_totales} días")
            c_f4.metric("Días Críticos Sostenidos (16h)", f"{dias_deficit_sostenido} días", help="Días con déficit ininterrumpido a las 20h, 4h y 12h")
            
            st.markdown("---")
            st.markdown("#### 🌙 1. Censo Nocturno Diario a las 04:00 AM (Stock de Flota en Reposo)")
            st.caption("A las 04:00 AM el uso ciudadano es nulo. Todas las bicicletas operativas deben estar ancladas en alguna estación. Las bicis no presentes corresponden a averías en taller o mantenimiento.")
            
            # Gráfico de barras/línea del stock a las 04:00 AM
            df_chart_4am = df_4am_audit.set_index('fecha')[['bicis_operativas_4am']]
            st.line_chart(df_chart_4am)
            
            st.dataframe(df_4am_audit, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.markdown("#### 🛡️ 2. Auditoría de Seguridad con Ventana Temporal (20:00 PM → 04:00 AM → 12:00 PM)")
            st.caption("Filtro anti-falsos positivos: Analiza las 8 horas previas y posteriores para diferenciar picos nocturnos de uso (ej. fiestas de San Juan o la Blanca) frente a fallos reales de taller.")
            
            st.dataframe(df_ventana_audit, use_container_width=True, hide_index=True)
            
            st.info("💡 **Metodología de Auditoría**: Si la flota vuelve a superar las 43 bicicletas al mediodía siguiente (12:00 PM), se considera una variación de uso puntual. Si permanece por debajo de 43 bicicletas en los 3 momentos (-8h, 4AM, +8h), se ratifica una falta grave de flota operativa.")
        else:
            st.warning("No se encontraron los datos precalculados de auditoría de flota. Ejecuta `python analisis_flota_4am.py` y `python analizar_todos_dias_ventana.py` para generarlos.")

if __name__ == '__main__':
    main()
