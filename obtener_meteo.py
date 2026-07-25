import sys
import json
import pandas as pd
import requests

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Coordenadas de Vitoria-Gasteiz
LAT = 42.8467
LON = -2.6716

def obtener_meteo_actual():
    """Obtiene la temperatura, lluvia y viento actuales en Vitoria-Gasteiz desde Open-Meteo."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&current=temperature_2m,precipitation,rain,wind_speed_10m&timezone=Europe%2FMadrid"
    headers = {'User-Agent': 'BiciVitoriaBot/1.0'}
    
    try:
        # Hacemos la petición HTTP a la API de Open-Meteo
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        current = data.get('current', {})
        
        # Extraemos variables comprobando posibles nombres alternativos de la API
        temp = current.get('temperature_2m', 18.0)
        precip = current.get('precipitation', 0.0)
        viento = current.get('wind_speed_10m', current.get('windspeed_10m', 10.0))
        
        return {
            'temperatura': float(temp),
            'precipitacion': float(precip),
            'llueve': 1 if float(precip) > 0.1 else 0,
            'viento_kmh': float(viento),
            'es_fallback': False
        }
    except Exception as e:
        print(f"Nota: No se pudo conectar con Open-Meteo ({e}). Usamos valores meteorológicos estimativos por defecto.")
        return {
            'temperatura': 18.0,
            'precipitacion': 0.0,
            'llueve': 0,
            'viento_kmh': 10.0,
            'es_fallback': True
        }

def obtener_meteo_historica(fecha_inicio="2026-06-19", fecha_fin="2026-07-24"):
    """Obtiene el historial horario meteorológico para Vitoria-Gasteiz."""
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}&start_date={fecha_inicio}&end_date={fecha_fin}&hourly=temperature_2m,precipitation,rain,wind_speed_10m&timezone=Europe%2FMadrid"
    headers = {'User-Agent': 'BiciVitoriaBot/1.0'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        hourly = data.get('hourly', {})
        viento_col = hourly.get('wind_speed_10m', hourly.get('windspeed_10m', []))
        
        df_meteo = pd.DataFrame({
            'timestamp_hora': pd.to_datetime(hourly['time']),
            'temperatura': hourly['temperature_2m'],
            'precipitacion': hourly['precipitation'],
            'viento_kmh': viento_col
        })
        df_meteo['llueve'] = (df_meteo['precipitacion'] > 0.1).astype(int)
        return df_meteo
    except Exception as e:
        print(f"Error al descargar meteo histórica: {e}")
        return pd.DataFrame()

def main():
    print("Probando conexión con API Open-Meteo para Vitoria-Gasteiz...")
    meteo_actual = obtener_meteo_actual()
    print("\n--- METEOROLOGÍA ACTUAL EN VITORIA-GASTEIZ ---")
    print(f"Temperatura: {meteo_actual['temperatura']} °C")
    print(f"Precipitación: {meteo_actual['precipitacion']} mm (Llueve: {meteo_actual['llueve']})")
    print(f"Viento: {meteo_actual['viento_kmh']} km/h")
    
    print("\nDescargando historial meteorológico de junio-julio 2026...")
    df_meteo = obtener_meteo_historica()
    if not df_meteo.empty:
        print(f"✓ Descargadas {len(df_meteo)} lecturas horarias de temperatura, lluvia y viento.")
        df_meteo.to_csv('meteo_historica.csv', index=False)
        print("Guardado en meteo_historica.csv")

if __name__ == '__main__':
    main()
