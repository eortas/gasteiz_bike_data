import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


API_URL = "https://mugibike.eus/api/client/entities"


def obtener_capacidades():
    capacidades = {}
    archivos = [Path("historico.csv")]
    archivos.extend(Path("datos").glob("historico_*.csv"))

    for archivo in archivos:
        if not archivo.exists():
            continue

        with archivo.open(encoding="utf-8") as csv_file:
            for fila in csv.reader(csv_file):
                if len(fila) != 5 or not fila[1].startswith("st_"):
                    continue

                try:
                    capacidad = int(float(fila[3])) + int(float(fila[4]))
                except ValueError:
                    continue

                capacidades[fila[1]] = max(capacidades.get(fila[1], 0), capacidad)

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
    capacidades = obtener_capacidades()
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
