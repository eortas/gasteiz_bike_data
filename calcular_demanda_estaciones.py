"""Precalculamos la demanda histórica por estación para el motor de redistribución."""

import sys

import pandas as pd

from demanda import RUTA_DEMANDA, calcular_demanda_historica

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def main():
    print("Calculando demanda histórica por estación, hora y tipo de día...")
    df_features = pd.read_csv('features_historico.csv')
    demanda = calcular_demanda_historica(df_features)

    if demanda.empty:
        print("No hay filas suficientes para calcular la demanda.")
        return

    demanda.to_csv(RUTA_DEMANDA, index=False)

    resumen = (demanda.groupby('nombre_estacion')['demanda_norm']
               .mean().sort_values(ascending=False))
    print(f"\nDemanda calculada para {demanda['nombre_estacion'].nunique()} estaciones "
          f"y {len(demanda)} combinaciones estación-hora-tipo de día.")
    print(f"Guardado en {RUTA_DEMANDA}")
    print("\nÍndice medio de demanda por estación (0-1):")
    print(resumen.round(2).to_string())


if __name__ == '__main__':
    main()
