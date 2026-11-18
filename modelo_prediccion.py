import json
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import joblib

from config import FEATURE_COLS


def _crear_modelo():
    return LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )


def main():
    print("Cargando dataset procesado desde features_historico.csv...")
    df = pd.read_csv('features_historico.csv')
    df = df.dropna(subset=['target_bicis_30m']).reset_index(drop=True)

    # Ordenamos por timestamp para garantizar una división temporal real (pasado vs futuro).
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # Fijamos un catálogo único de estaciones y lo aplicamos a TODO el dataset.
    # Así el modelo aprende un código estable por estación y la inferencia puede
    # reutilizarlo aunque falte alguna estación en una instantánea concreta.
    catalogo_estaciones = sorted(df['id_estacion'].astype(str).unique())
    df['id_estacion'] = pd.Categorical(df['id_estacion'].astype(str), categories=catalogo_estaciones)

    target_col = 'target_bicis_30m'
    X = df[FEATURE_COLS].copy()
    y = df[target_col]

    # División temporal por timestamps únicos (evitamos repartir una misma instantánea
    # entre train y test, como ocurría al cortar por índice de fila).
    timestamps_unicos = np.sort(df['timestamp'].unique())
    corte_ts = timestamps_unicos[int(len(timestamps_unicos) * 0.8)]
    es_train = df['timestamp'] < corte_ts

    X_train, X_test = X[es_train].copy(), X[~es_train].copy()
    y_train, y_test = y[es_train], y[~es_train]

    print(f"Train: {len(X_train)} filas hasta {corte_ts} | Test: {len(X_test)} filas posteriores")

    # Validación cruzada temporal (TimeSeriesSplit con 5 pliegues).
    print("\nEjecutando validación cruzada temporal (TimeSeriesSplit con 5 pliegues)...")
    tscv = TimeSeriesSplit(n_splits=5)
    maes_cv = []
    rmses_cv = []

    for fold, (train_cv_idx, val_cv_idx) in enumerate(tscv.split(X_train), 1):
        X_tr_cv, X_val_cv = X_train.iloc[train_cv_idx], X_train.iloc[val_cv_idx]
        y_tr_cv, y_val_cv = y_train.iloc[train_cv_idx], y_train.iloc[val_cv_idx]

        modelo_cv = _crear_modelo()
        modelo_cv.fit(X_tr_cv, y_tr_cv)
        preds_cv = modelo_cv.predict(X_val_cv)

        mae_fold = mean_absolute_error(y_val_cv, preds_cv)
        rmse_fold = root_mean_squared_error(y_val_cv, preds_cv)
        maes_cv.append(mae_fold)
        rmses_cv.append(rmse_fold)
        print(f"  - Pliegue {fold}: MAE = {mae_fold:.2f} bicis, RMSE = {rmse_fold:.2f} bicis")

    print(f"Resultado CV (promedio): MAE = {np.mean(maes_cv):.2f} bicis, RMSE = {np.mean(rmses_cv):.2f} bicis")

    # Entrenamos un modelo de evaluación sobre el 80% y medimos en el 20% futuro.
    print(f"\nEntrenando modelo de evaluación LightGBM ({len(X_train)} muestras)...")
    modelo_eval = _crear_modelo()
    modelo_eval.fit(X_train, y_train)

    predicciones = modelo_eval.predict(X_test)
    mae_test = mean_absolute_error(y_test, predicciones)
    rmse_test = root_mean_squared_error(y_test, predicciones)
    r2_test = r2_score(y_test, predicciones)

    # Calculamos el Recall para situaciones críticas de alerta en test
    # (vacío <= 1 o lleno >= capacidad - 1).
    capacidad_test = X_test['capacidad']
    es_critico_real = (y_test <= 1.0) | (y_test >= capacidad_test - 1.0)
    es_critico_pred = (predicciones <= 1.5) | (predicciones >= capacidad_test - 1.5)

    alertas_reales = np.sum(es_critico_real)
    alertas_detectadas = np.sum(es_critico_real & es_critico_pred)
    recall = (alertas_detectadas / alertas_reales * 100) if alertas_reales > 0 else 100.0

    print("\n--- EVALUACIÓN FINAL EN CONJUNTO DE TEST ---")
    print(f"Error Medio Absoluto (MAE): {mae_test:.2f} bicicletas")
    print(f"Error Cuadrático Medio (RMSE): {rmse_test:.2f} bicicletas")
    print(f"R² (varianza explicada): {r2_test:.2f}")
    print(f"Recall en alertas críticas: {recall:.1f}% ({alertas_detectadas}/{alertas_reales})")

    # Métricas desglosadas por estación para detectar dónde predice peor el modelo.
    evaluacion = pd.DataFrame({
        'estacion': df.loc[~es_train, 'nombre_estacion'].values,
        'real': y_test.values,
        'prediccion': predicciones
    })
    metricas_por_estacion = []
    for estacion, grupo in evaluacion.groupby('estacion'):
        metricas_por_estacion.append({
            'estacion': estacion,
            'mae': round(float(mean_absolute_error(grupo['real'], grupo['prediccion'])), 2),
            'n': int(len(grupo))
        })
    metricas_por_estacion.sort(key=lambda fila: fila['mae'], reverse=True)

    print("\nMAE por estación en test (peores primero):")
    for fila in metricas_por_estacion:
        print(f"  {fila['estacion'][:45]:45s} MAE = {fila['mae']:.2f} ({fila['n']} filas)")

    # Reentrenamos el modelo final con TODOS los datos disponibles antes de guardarlo.
    print(f"\nReentrenando modelo final LightGBM con el 100% de los datos ({len(X)} muestras)...")
    modelo_final = _crear_modelo()
    modelo_final.fit(X, y)
    # Guardamos el catálogo con el modelo para que la inferencia use los mismos códigos.
    modelo_final.categorias_estacion_ = catalogo_estaciones

    importancia_df = pd.DataFrame({
        'Variable': FEATURE_COLS,
        'Importancia': modelo_final.feature_importances_
    }).sort_values(by='Importancia', ascending=False)

    print("\nImportancia de las variables explicativas:")
    print(importancia_df.to_string(index=False))

    archivo_modelo = 'modelo_lightgbm.joblib'
    joblib.dump(modelo_final, archivo_modelo)
    print(f"\nModelo guardado correctamente en {archivo_modelo}")

    metricas = {
        'mae': round(float(mae_test), 2),
        'rmse': round(float(rmse_test), 2),
        'r2': round(float(r2_test), 3),
        'recall': round(float(recall), 1),
        'mae_cv_promedio': round(float(np.mean(maes_cv)), 2),
        'rmse_cv_promedio': round(float(np.mean(rmses_cv)), 2),
        'fecha_corte_test': str(corte_ts),
        'catalogo_estaciones': catalogo_estaciones,
        'metricas_por_estacion': metricas_por_estacion,
        'importancias': importancia_df.to_dict('records')
    }

    with open('metricas_modelo.json', 'w', encoding='utf-8') as f:
        json.dump(metricas, f, indent=4, ensure_ascii=False)
    print("Métricas del modelo exportadas a metricas_modelo.json")


if __name__ == '__main__':
    main()
