import subprocess
import os
import pandas as pd

repo_dir = r"c:\Users\ortas\OneDrive\Documentos\gasteiz_bike_data-1"
historico_path = os.path.join(repo_dir, "historico.csv")

print("Obteniendo la lista de commits...")
blobs = subprocess.check_output(
    ["git", "log", "--format=%H", "--", "historico.csv"],
    cwd=repo_dir
).decode("utf-8").splitlines()

lines_set = set()
print(f"Procesando {len(blobs)} commits...")
for c in blobs:
    out = subprocess.check_output(
        ["git", "show", f"{c}:historico.csv"],
        cwd=repo_dir
    ).decode("utf-8")
    for line in out.splitlines():
        line_clean = line.strip()
        if line_clean and not line_clean.startswith("Timestamp"):
            lines_set.add(line_clean)

print(f"Total líneas únicas encontradas: {len(lines_set)}")

data = [line.split(",") for line in lines_set if len(line.split(",")) == 5]
df = pd.DataFrame(data, columns=["timestamp", "id_estacion", "nombre_estacion", "bicis_disponibles", "anclajes_disponibles"])
df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(by=["timestamp_dt", "id_estacion"]).drop(columns=["timestamp_dt"])

print(f"Escribiendo {len(df)} filas en {historico_path}...")
df.to_csv(historico_path, index=False, header=False)
print("¡Completado con éxito!")
