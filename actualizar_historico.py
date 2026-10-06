import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


API_URL = "https://mugibike.eus/api/client/entities"


# Capacidades nominales conocidas de las estaciones de Mugibike Vitoria-Gasteiz
CAPACIDADES_CONOCIDAS = {
    "st_d4sl1494gpvs73agno80": 24,
    "st_d4sl14d0ol7c73bfvs4g": 24,
    "st_d4sl14emk29c73efcpf0": 24,
    "st_d4sl14ffimdc73fprpdg": 24,
    "st_d4sl14gvnoqc739efe8g": 13,
    "st_d4sl14gvnoqc739efea0": 17,
    "st_d4sl14gvnoqc739efebg": 23,
    "st_d4sl14gvnoqc739efed0": 12,
    "st_d4sl14h4gpvs73agnodg": 10,
    "st_d4sl14p4gpvs73agnoi0": 11,
    "st_d51sccd4faec738epl3g": 10,
    "st_d7p43v8g4a6c73a823dg": 11,
}


def obtener_capacidades(estaciones_api=None):
    capacidades = dict(CAPACIDADES_CONOCIDAS)

    # Si detectamos estaciones en la API que no están en el mapa conocido, consultamos el histórico local
    if estaciones_api:
        desconocidas = {
            e.get("id") for e in estaciones_api
            if e.get("id") and e.get("id") not in capacidades
        }
        if desconocidas:
            archivos = sorted(Path("datos").glob("historico_*.csv"), reverse=True)
            if Path("historico.csv").exists():
                archivos.append(Path("historico.csv"))

            for archivo in archivos:
                with archivo.open(encoding="utf-8") as csv_file:
                    for fila in csv.reader(csv_file):
                        if len(fila) == 5 and fila[1] in desconocidas:
                            try:
                                cap = int(float(fila[3])) + int(float(fila[4]))
                                capacidades[fila[1]] = max(capacidades.get(fila[1], 0), cap)
                            except ValueError:
                                continue
    return capacidades



def obtener_lecturas():
    request = Request(
        API_URL,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        },
    )

    with urlopen(request, timeout=20) as response:
        datos_api = json.loads(response.read().decode("utf-8"))

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    estaciones = datos_api.get("data", {}).get("stations", [])
    capacidades = obtener_capacidades(estaciones)
    lecturas = []

    for estacion in estaciones:
        id_estacion = estacion.get("id")
        if not id_estacion or not str(id_estacion).startswith("st_"):
            continue

        bicis_disponibles = estacion.get("availableBikes", 0)
        capacidad = capacidades.get(id_estacion)
        if capacidad is None:
            raise ValueError(f"No hay capacidad histórica para la estación {id_estacion}")

        lecturas.append([
            timestamp,
            id_estacion,
            estacion.get("label", ""),
            bicis_disponibles,
            max(capacidad - bicis_disponibles, 0),
        ])

    if not lecturas:
        raise ValueError("La API no ha devuelto estaciones válidas")

    return timestamp, lecturas


def guardar_lecturas(timestamp, lecturas):
    periodo = timestamp[:7]
    carpeta_datos = Path("datos")
    carpeta_datos.mkdir(exist_ok=True)
    archivo = carpeta_datos / f"historico_{periodo}.csv"

    with archivo.open("a", newline="", encoding="utf-8") as csv_file:
        escritor = csv.writer(csv_file)
        escritor.writerows(lecturas)

    print(f"Guardadas {len(lecturas)} lecturas en {archivo}")


if __name__ == "__main__":
    timestamp, lecturas = obtener_lecturas()
    guardar_lecturas(timestamp, lecturas)
