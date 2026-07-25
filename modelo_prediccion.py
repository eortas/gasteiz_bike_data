import json
import pandas as pd
import numpy as np

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import joblib

from config import FEATURE_COLS

def main():
    print("Cargando dataset procesado desde features_historico.csv...")
    df = pd.read_csv('features_historico.csv')
    df = df.dropna(subset=['target_bicis_30m']).reset_index(drop=True)
    
    # Ordenamos por timestamp para garantizar una división temporal real (pasado vs futuro)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    target_col = 'target_bicis_30m'
    
    X = df[FEATURE_COLS].copy()
    y = df[target_col]
    
    # División temporal (80% entrenamiento, 20% prueba)
    corte_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:corte_idx].copy(), X.iloc[corte_idx:].copy()
    y_train, y_test = y.iloc[:corte_idx], y.iloc[corte_idx:]
    
    # Convertimos id_estacion a tipo categoría después de la división
    X_train['id_estacion'] = X_train['id_estacion'].astype('category')
    X_test['id_estacion'] = X_test['id_estacion'].astype('category')
    
    # Validación cruzada temporal (TimeSeriesSplit con 5 pliegues)
    print("\nEjecutando validación cruzada temporal (TimeSeriesSplit con 5 pliegues)...")
    tscv = TimeSeriesSplit(n_splits=5)
    maes_cv = []
    rmses_cv = []
    
    for fold, (train_cv_idx, val_cv_idx) in enumerate(tscv.split(X_train), 1):
        X_tr_cv, X_val_cv = X_train.iloc[train_cv_idx], X_train.iloc[val_cv_idx]
        y_tr_cv, y_val_cv = y_train.iloc[train_cv_idx], y_train.iloc[val_cv_idx]
        
        modelo_cv = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
        modelo_cv.fit(X_tr_cv, y_tr_cv)
        preds_cv = modelo_cv.predict(X_val_cv)
        
        mae_fold = mean_absolute_error(y_val_cv, preds_cv)
        rmse_fold = root_mean_squared_error(y_val_cv, preds_cv)
        maes_cv.append(mae_fold)
        rmses_cv.append(rmse_fold)
        print(f"  - Pliegue {fold}: MAE = {mae_fold:.2f} bicis, RMSE = {rmse_fold:.2f} bicis")
        
    print(f"Resultado CV (promedio): MAE = {np.mean(maes_cv):.2f} bicis, RMSE = {np.mean(rmses_cv):.2f} bicis")
    
    # Entrenamos el modelo final con todo el conjunto de entrenamiento
    print(f"\nEntrenando modelo final LightGBM ({len(X_train)} muestras)...")
    modelo_final = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    modelo_final.fit(X_train, y_train)
    
    # Evaluación en el conjunto de test (último 20%)
    predicciones = modelo_final.predict(X_test)
    mae_test = mean_absolute_error(y_test, predicciones)
    rmse_test = root_mean_squared_error(y_test, predicciones)
    
    # Calculamos el Recall para situaciones críticas de alerta en test (vacío <= 1 o lleno >= capacidad - 1)
    capacidad_test = X_test['capacidad']
    es_critico_real = (y_test <= 1.0) | (y_test >= capacidad_test - 1.0)
    es_critico_pred = (predicciones <= 1.5) | (predicciones >= capacidad_test - 1.5)
    
    alertas_reales = np.sum(es_critico_real)
    alertas_detectadas = np.sum(es_critico_real & es_critico_pred)
    recall = (alertas_detectadas / alertas_reales * 100) if alertas_reales > 0 else 100.0
    
    print(f"\n--- EVALUACIÓN FINAL EN CONJUNTO DE TEST ---")
    print(f"Error Medio Absoluto (MAE): {mae_test:.2f} bicicletas")
    print(f"Error Cuadrático Medio (RMSE): {rmse_test:.2f} bicicletas")
    print(f"Recall en alertas críticas: {recall:.1f}% ({alertas_detectadas}/{alertas_reales})")
    
    # Extraemos la importancia de las variables
    importancia_df = pd.DataFrame({
        'Variable': FEATURE_COLS,
        'Importancia': modelo_final.feature_importances_
    }).sort_values(by='Importancia', ascending=False)
    
    print("\nImportancia de las variables explicativas:")
    print(importancia_df.to_string(index=False))
    
    # Guardamos el modelo entrenado
    archivo_modelo = 'modelo_lightgbm.joblib'
    joblib.dump(modelo_final, archivo_modelo)
    print(f"\nModelo guardado correctamente en {archivo_modelo}")
    
    # Guardamos las métricas dinámicas en JSON para el dashboard
    metricas = {
        'mae': round(float(mae_test), 2),
        'rmse': round(float(rmse_test), 2),
        'recall': round(float(recall), 1),
        'mae_cv_promedio': round(float(np.mean(maes_cv)), 2),
        'rmse_cv_promedio': round(float(np.mean(rmses_cv)), 2),
        'importancias': importancia_df.to_dict('records')
    }
    
    with open('metricas_modelo.json', 'w', encoding='utf-8') as f:
        json.dump(metricas, f, indent=4, ensure_ascii=False)
    print("Métricas del modelo exportadas a metricas_modelo.json")

if __name__ == '__main__':
    main()
