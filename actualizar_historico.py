import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


API_URL = "https://mugibike.eus/api/client/entities"


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
    lecturas = []

    for estacion in estaciones:
        id_estacion = estacion.get("id")
        if not id_estacion or not str(id_estacion).startswith("st_"):
            continue

        lecturas.append([
            timestamp,
            id_estacion,
            estacion.get("label", ""),
            estacion.get("availableBikes", 0),
            estacion.get("availableSlots", 0),
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
