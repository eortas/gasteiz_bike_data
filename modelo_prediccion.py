import pandas as pd
import numpy as np

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import joblib

def main():
    print("Cargando dataset procesado desde features_historico.csv...")
    df = pd.read_csv('features_historico.csv')
    
    # Convertimos id_estacion a tipo categoría para LightGBM
    df['id_estacion'] = df['id_estacion'].astype('category')
    
    # Definimos la lista de variables explicativas (features)
    feature_cols = [
        'id_estacion', 'hora', 'dia_semana', 'es_finde', 'capacidad',
        'bicis_disponibles', 'anclajes_disponibles', 'pct_ocupacion',
        'tendencia_15m', 'tendencia_30m'
    ]
    
    target_col = 'target_bicis_30m'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Realizamos una división temporal (80% entrenamiento, 20% prueba)
    corte_idx = int(len(df) * 0.8)
    
    X_train, X_test = X.iloc[:corte_idx], X.iloc[corte_idx:]
    y_train, y_test = y.iloc[:corte_idx], y.iloc[corte_idx:]
    
    print(f"Entrenando modelo LightGBM con {len(X_train)} muestras...")
    
    # Configuramos el modelo LightGBM
    modelo = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    
    # Entrenamos el modelo
    modelo.fit(X_train, y_train)
    
    # Evaluamos en el conjunto de test
    predicciones = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, predicciones)
    rmse = root_mean_squared_error(y_test, predicciones)
    
    print(f"\n--- EVALUACIÓN DEL MODELO LIGHTGBM ---")
    print(f"Error Medio Absoluto (MAE): {mae:.2f} bicicletas")
    print(f"Error Cuadrático Medio (RMSE): {rmse:.2f} bicicletas")
    
    # Importancia de las variables
    importancia = pd.DataFrame({
        'Feature': feature_cols,
        'Importancia': modelo.feature_importances_
    }).sort_values(by='Importancia', ascending=False)
    
    print("\nVariables más importantes en la predicción:")
    print(importancia.to_string(index=False))
    
    # Serializamos el modelo para producción
    archivo_modelo = 'modelo_lightgbm.joblib'
    joblib.dump(modelo, archivo_modelo)
    print(f"\nModelo guardado correctamente en {archivo_modelo}")

if __name__ == '__main__':
    main()
