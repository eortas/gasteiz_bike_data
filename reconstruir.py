import subprocess
import os
import pandas as pd

repo_dir = os.path.dirname(os.path.abspath(__file__))
datos_dir = os.path.join(repo_dir, "datos")
os.makedirs(datos_dir, exist_ok=True)

print("Extrayendo historial completo usando diffs de git log...")
# Leemos el historial completo de diffs de todos los archivos CSV
out = subprocess.check_output(
    ["git", "log", "-U0", "-p", "--", "*.csv", "datos/*.csv"],
    cwd=repo_dir
).decode("utf-8", errors="ignore")

conjunto_lineas = set()
for linea in out.splitlines():
    if linea.startswith("+202"):
        linea_limpia = linea[1:].strip()
        if len(linea_limpia.split(",")) == 5:
            conjunto_lineas.add(linea_limpia)

print(f"Total de líneas únicas encontradas: {len(conjunto_lineas)}")

datos = [linea.split(",") for linea in conjunto_lineas]
df = pd.DataFrame(datos, columns=["timestamp", "id_estacion", "nombre_estacion", "bicis_disponibles", "anclajes_disponibles"])
df["timestamp_dt"] = pd.to_datetime(df["timestamp"], format='ISO8601')
df = df.sort_values(by=["timestamp_dt", "id_estacion"])

# Particionamos por año y mes
df['periodo'] = df['timestamp_dt'].dt.to_period('M')

for periodo, grupo in df.groupby('periodo'):
    archivo_mes = os.path.join(datos_dir, f"historico_{periodo}.csv")
    df_mes = grupo.drop(columns=['timestamp_dt', 'periodo'])
    print(f"Escribiendo {len(df_mes)} filas en {archivo_mes}...")
    df_mes.to_csv(archivo_mes, index=False, header=False)

# Guardamos también una copia consolidada en historico.csv para retrocompatibilidad
historico_path = os.path.join(repo_dir, "historico.csv")
df.drop(columns=['timestamp_dt', 'periodo']).to_csv(historico_path, index=False, header=False)

print("¡Completado con éxito!")
