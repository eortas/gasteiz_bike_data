import subprocess
import os
import pandas as pd

# Definimos la ruta del repositorio y del archivo histórico
repo_dir = r"c:\Users\ortas\OneDrive\Documentos\gasteiz_bike_data-1"
historico_path = os.path.join(repo_dir, "historico.csv")

print("Extrayendo historial completo usando diffs de git log...")
# Leemos el historial completo de diffs en un solo comando de git
out = subprocess.check_output(
    ["git", "log", "-U0", "-p", "--", "historico.csv"],
    cwd=repo_dir
).decode("utf-8", errors="ignore")

# Guardamos en un conjunto para evitar duplicados
conjunto_lineas = set()
for linea in out.splitlines():
    if linea.startswith("+202"):
        linea_limpia = linea[1:].strip()
        if len(linea_limpia.split(",")) == 5:
            conjunto_lineas.add(linea_limpia)

print(f"Total de líneas únicas encontradas: {len(conjunto_lineas)}")

# Convertimos las líneas en un DataFrame de pandas
datos = [linea.split(",") for linea in conjunto_lineas]
df = pd.DataFrame(datos, columns=["timestamp", "id_estacion", "nombre_estacion", "bicis_disponibles", "anclajes_disponibles"])

# Convertimos el timestamp a tipo datetime para ordenar los datos cronológicamente
df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(by=["timestamp_dt", "id_estacion"]).drop(columns=["timestamp_dt"])

print(f"Escribiendo {len(df)} filas en {historico_path}...")
# Guardamos el histórico completo sin cabecera ni índice
df.to_csv(historico_path, index=False, header=False)
print("¡Completado con éxito!")
