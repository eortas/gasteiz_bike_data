import sys
import urllib.request
import json
import time
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

API_URL = "https://mugibike.eus/api/client/entities"

def obtener_estaciones_mugibike_realtime():
    """
    Consulta la API en vivo de Mugibike (https://mugibike.eus/api/client/entities)
    Retorna un DataFrame con la disponibilidad actual por estación y un booleano de éxito.
    """
    req = urllib.request.Request(API_URL)
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0')
    req.add_header('Accept', 'application/json')
    
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            stations = data.get('data', {}).get('stations', [])
            anclajes_api_validos = any(
                s.get('availableSlots') not in (None, 0)
                for s in stations
            )
            
            filas = []
            for s in stations:
                b_disp = s.get('availableBikes', 0)
                a_disp = s.get('availableSlots') if anclajes_api_validos else None
                cap = b_disp + a_disp if a_disp is not None else None
                pct_oc = b_disp / cap if cap else None
                
                filas.append({
                    'id_estacion': s.get('id'),
                    'nombre_estacion': s.get('label'),
                    'bicis_disponibles': b_disp,
                    'anclajes_disponibles': a_disp,
                    'capacidad': cap,
                    'pct_ocupacion': pct_oc,
                    'is_closed': s.get('isClosed', False),
                    'latitud': s.get('coordinates', [0, 0])[0] if s.get('coordinates') else 0.0,
                    'longitud': s.get('coordinates', [0, 0])[1] if s.get('coordinates') else 0.0,
                    'timestamp_api': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())
                })
            
            df_live = pd.DataFrame(filas)
            return df_live, True
            
    except Exception as e:
        print(f"Nota: No se pudo consultar la API de Mugibike ({e}). Se usarán datos históricos como respaldo.")
        return pd.DataFrame(), False

def main():
    print("Probando consulta en tiempo real a la API de Mugibike (https://mugibike.eus/api/client/entities)...")
    df_live, exito = obtener_estaciones_mugibike_realtime()
    
    if exito and not df_live.empty:
        print(f"✓ Éxito: Obtenidas {len(df_live)} estaciones en tiempo real de Mugibike Vitoria-Gasteiz.\n")
        print(df_live[['id_estacion', 'nombre_estacion', 'bicis_disponibles', 'anclajes_disponibles', 'capacidad']].to_string(index=False))
    else:
        print("✗ No se pudo obtener respuesta de la API.")

if __name__ == '__main__':
    main()
