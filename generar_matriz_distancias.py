import pandas as pd
import json
import urllib.request

# Coordenadas geográficas exactas de las 12 estaciones en Vitoria-Gasteiz
estaciones_coords = {
    '1 Autobus geltokia - Estación de autobuses': {'lat': 42.8596, 'lon': -2.6806},
    '2 Leizaola plaza - Plaza de Leizaola': {'lat': 42.8453, 'lon': -2.6738},
    '3 Ballester apezpikua - Obispo Ballester': {'lat': 42.8530, 'lon': -2.6610},
    '4 Antso Jakituna - Sancho el Sabio': {'lat': 42.8420, 'lon': -2.6775},
    '5 Mendizorrotza - Mendizorroza': {'lat': 42.8375, 'lon': -2.6860},
    '6 Unibertsitatea - Universidad': {'lat': 42.8390, 'lon': -2.6720},
    '7 Salburuko gizarte-etxea - C.C. Salburua': {'lat': 42.8535, 'lon': -2.6480},
    '8 Tren geltokia - Estación de Tren': {'lat': 42.8425, 'lon': -2.6680},
    '9 Ibaiondoko gizarte-etxea - C.C. Ibaiondo': {'lat': 42.8680, 'lon': -2.6955},
    '10 AUO - HUA': {'lat': 42.8565, 'lon': -2.6890},
    '11 San Martin (Udala) - Ayuntamiento San Martín': {'lat': 42.8475, 'lon': -2.6865},
    '12 Zabalganeko gizarte-etxea - CC Zabalgana': {'lat': 42.8450, 'lon': -2.7010}
}

def obtener_matriz_osrm(nombres, coords):
    coords_str = ";".join([f"{c['lon']},{c['lat']}" for c in coords])
    url = f"http://router.project-osrm.org/table/v1/driving/{coords_str}?annotations=distance,duration"
    
    print("Consultando OSRM y ajustando con la velocidad media urbana real en Vitoria-Gasteiz...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'BiciVitoriaBot/1.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        durations = data['durations']
        distances = data['distances']
        return durations, distances
    except Exception as e:
        print(f"Error al conectar con OSRM ({e}). Usando cálculo de reserva...")
        import math
        n = len(nombres)
        durations = [[0.0]*n for _ in range(n)]
        distances = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                c1, c2 = coords[i], coords[j]
                dlat = math.radians(c2['lat'] - c1['lat'])
                dlon = math.radians(c2['lon'] - c1['lon'])
                a = math.sin(dlat/2)**2 + math.cos(math.radians(c1['lat'])) * math.cos(math.radians(c2['lat'])) * math.sin(dlon/2)**2
                dist_km = 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)) * 1.3
                distances[i][j] = dist_km * 1000
                durations[i][j] = (dist_km / 16.0) * 3600
        return durations, distances

def main():
    nombres_estaciones = list(estaciones_coords.keys())
    coords_list = [estaciones_coords[nombre] for nombre in nombres_estaciones]
    
    durations_sec, distances_m = obtener_matriz_osrm(nombres_estaciones, coords_list)
    
    # En Vitoria-Gasteiz (zonas 30 km/h, semáforos, tranvía y giro comercial),
    # la velocidad media de circulación real en furgoneta es de ~16 km/h.
    # Calibramos con un factor de 2.2x sobre la duración teórica de flujo libre de OSRM.
    FACTOR_TRAFICO_REAL = 2.2
    
    registros = []
    for i, origen in enumerate(nombres_estaciones):
        for j, destino in enumerate(nombres_estaciones):
            if origen != destino:
                dist_km = round(distances_m[i][j] / 1000.0, 2)
                # Calculamos el tiempo real considerando la velocidad efectiva en ciudad (~16 km/h)
                tiempo_real_min = round((durations_sec[i][j] * FACTOR_TRAFICO_REAL) / 60.0, 1)
                registros.append({
                    'estacion_origen': origen,
                    'estacion_destino': destino,
                    'distancia_km': dist_km,
                    'tiempo_conduccion_min': tiempo_real_min
                })
    
    df_matriz = pd.DataFrame(registros)
    archivo_salida = 'matriz_distancias_estaciones.csv'
    df_matriz.to_csv(archivo_salida, index=False)
    
    print(f"Matriz de tiempos ajustada y guardada en {archivo_salida}")
    
    # Mostramos rutas de referencia para verificación
    ruta_tren = df_matriz[(df_matriz['estacion_origen'].str.contains('Autobus')) & (df_matriz['estacion_destino'].str.contains('Tren'))]
    print("\nRuta Autobuses -> Estación de Tren (Verificación):")
    print(ruta_tren.to_string(index=False))

if __name__ == '__main__':
    main()
