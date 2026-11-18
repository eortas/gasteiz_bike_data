"""Puerta de calidad del modelo en CI.

Bloqueamos la publicación del modelo si el recall de alertas críticas o el MAE
empeoran por debajo de los umbrales definidos. Las métricas se leen de
metricas_modelo.json, que genera modelo_prediccion.py.
"""

import json
import sys

UMBRAL_RECALL_MIN = 65.0
UMBRAL_MAE_MAX = 1.5
UMBRAL_RMSE_MAX = 2.5


def main():
    with open('metricas_modelo.json', 'r', encoding='utf-8') as archivo:
        metricas = json.load(archivo)

    recall = float(metricas.get('recall', 0.0))
    mae = float(metricas.get('mae', 99.0))
    rmse = float(metricas.get('rmse', 99.0))

    print(f"Recall alertas críticas: {recall:.1f}% (mínimo {UMBRAL_RECALL_MIN:.1f}%)")
    print(f"MAE: {mae:.2f} bicis (máximo {UMBRAL_MAE_MAX:.2f})")
    print(f"RMSE: {rmse:.2f} bicis (máximo {UMBRAL_RMSE_MAX:.2f})")

    errores = []
    if recall < UMBRAL_RECALL_MIN:
        errores.append(f"recall {recall:.1f}% < {UMBRAL_RECALL_MIN:.1f}%")
    if mae > UMBRAL_MAE_MAX:
        errores.append(f"MAE {mae:.2f} > {UMBRAL_MAE_MAX:.2f}")
    if rmse > UMBRAL_RMSE_MAX:
        errores.append(f"RMSE {rmse:.2f} > {UMBRAL_RMSE_MAX:.2f}")

    if errores:
        print("\nModelo RECHAZADO: " + "; ".join(errores))
        return 1

    print("\nModelo ACEPTADO: cumple los umbrales de calidad.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
