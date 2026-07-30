from pathlib import Path

import pandas as pd


COLUMNAS_HISTORICO = [
    'timestamp',
    'id_estacion',
    'nombre_estacion',
    'bicis_disponibles',
    'anclajes_disponibles'
]


def cargar_historico_completo():
    rutas = [Path('historico.csv')]
    rutas.extend(sorted(Path('datos').glob('historico_*.csv')))

    partes = []
    for ruta in rutas:
        if ruta.exists():
            partes.append(pd.read_csv(ruta, header=None, names=COLUMNAS_HISTORICO))

    historico = pd.concat(partes, ignore_index=True)
    historico['timestamp'] = pd.to_datetime(
        historico['timestamp'], format='ISO8601', utc=True
    )
    historico = historico.drop_duplicates(
        subset=['timestamp', 'id_estacion'], keep='last'
    )
    return historico.sort_values(['timestamp', 'id_estacion']).reset_index(drop=True)
