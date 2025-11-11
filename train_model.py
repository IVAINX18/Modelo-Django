import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "StudentPerformanceFactors.csv")
MODEL_PATH = os.path.join(BASE_DIR, "prediccion_estudiantes", "modelo_prediccion.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "prediccion_estudiantes", "scaler.pkl")

df = pd.read_csv(CSV_PATH, sep=";")

columnas = [
    'Hours_Studied', 'Attendance', 'Parental_Involvement',
    'Access_to_Resources', 'Extracurricular_Activities',
    'Sleep_Hours', 'Previous_Scores', 'Motivation_Level'
]

X = df[columnas].copy()
y = df['Exam_Score']

mapeo = {
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'Yes': 1,
    'No': 0
}

X = X.replace(mapeo)
X = X.apply(pd.to_numeric, errors='coerce')

for col in X.columns:
    if X[col].isnull().any():
        X[col].fillna(X[col].median(), inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=columnas)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Entrenando modelo con hiperparametros optimizados...")

modelo = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)

modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)
mae = mean_absolute_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)
rmse = np.sqrt(mean_squared_error(y_test, predicciones))

print("\n" + "="*60)
print("RESULTADOS DEL ENTRENAMIENTO")
print("="*60)
print(f"Error Absoluto Medio (MAE):     {mae:.3f}")
print(f"Error Cuadratico Medio (RMSE):  {rmse:.3f}")
print(f"Coeficiente R2:                 {r2:.4f}")
print(f"Precision del modelo:           {r2*100:.2f}%")
print("="*60)

importancias = pd.DataFrame({
    'Caracteristica': columnas,
    'Importancia': modelo.feature_importances_
}).sort_values('Importancia', ascending=False)

print("\nIMPORTANCIA DE CARACTERISTICAS:")
print(importancias.to_string(index=False))
print("="*60)

joblib.dump(modelo, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"\nModelo guardado en: {MODEL_PATH}")
print(f"Scaler guardado en: {SCALER_PATH}")
print("\nEntrenamiento completado exitosamente.")
