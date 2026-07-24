import pandas as pd
import numpy as np

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import joblib

def main():
    print("Cargando dataset procesado desde features_historico.csv...")
    df = pd.read_csv('features_historico.csv')
    df['id_estacion'] = df['id_estacion'].astype('category')
    
    # Lista de variables explicativas incluyendo clima y eventos de Vitoria-Gasteiz
    feature_cols = [
        'id_estacion', 'hora', 'dia_semana', 'es_finde', 'capacidad',
        'bicis_disponibles', 'anclajes_disponibles', 'pct_ocupacion',
        'tendencia_15m', 'tendencia_30m',
        'temperatura', 'llueve', 'viento_kmh',
        'es_festivo', 'es_la_blanca', 'es_vacaciones_upv'
    ]
    
    target_col = 'target_bicis_30m'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # División temporal (80% entrenamiento, 20% prueba)
    corte_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:corte_idx], X.iloc[corte_idx:]
    y_train, y_test = y.iloc[:corte_idx], y.iloc[corte_idx:]
    
    print(f"Entrenando modelo LightGBM con festivos y eventos ({len(X_train)} muestras)...")
    
    modelo = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    
    modelo.fit(X_train, y_train)
    
    predicciones = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, predicciones)
    rmse = root_mean_squared_error(y_test, predicciones)
    
    print(f"\n--- EVALUACIÓN DEL MODELO LIGHTGBM CON EVENTOS Y CLIMA ---")
    print(f"Error Medio Absoluto (MAE): {mae:.2f} bicicletas")
    print(f"Error Cuadrático Medio (RMSE): {rmse:.2f} bicicletas")
    
    importancia = pd.DataFrame({
        'Feature': feature_cols,
        'Importancia': modelo.feature_importances_
    }).sort_values(by='Importancia', ascending=False)
    
    print("\nImportancia de las variables explicativas:")
    print(importancia.to_string(index=False))
    
    archivo_modelo = 'modelo_lightgbm.joblib'
    joblib.dump(modelo, archivo_modelo)
    print(f"\nModelo guardado correctamente en {archivo_modelo}")

if __name__ == '__main__':
    main()
